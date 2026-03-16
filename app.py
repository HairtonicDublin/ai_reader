import os
import sys
import sqlite3
import threading
import re
import logging
import traceback
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from markupsafe import escape as html_escape
import json as json_module
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# --- 库检测 ---
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# --- 路径处理（云端适配）---
def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(relative_path):
    return os.path.join(get_base_path(), relative_path)

def get_data_dir():
    # 优先使用环境变量指定的目录（用于 Railway Volume）
    data_dir = os.environ.get('DATA_DIR')
    if data_dir:
        try:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            # 测试是否可写
            test_file = os.path.join(data_dir, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logging.info(f"Using DATA_DIR: {data_dir}")
            return data_dir
        except Exception as e:
            logging.error(f"DATA_DIR not writable: {e}")
    
    # 尝试在 /data 目录（Railway Volume 默认位置）
    try:
        if os.path.exists('/data'):
            test_file = '/data/.test'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logging.info("Using /data directory")
            return '/data'
    except:
        pass
    
    # 其次使用项目目录下的 data 文件夹
    try:
        data_dir = os.path.join(get_base_path(), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        logging.info(f"Using local data dir: {data_dir}")
        return data_dir
    except Exception as e:
        logging.error(f"Local data dir failed: {e}")
    
    # 最后使用 /tmp 目录
    tmp_dir = '/tmp/ai_reader_data'
    os.makedirs(tmp_dir, exist_ok=True)
    logging.info(f"Using tmp dir: {tmp_dir}")
    return tmp_dir

app = Flask(__name__, template_folder=get_resource_path('templates'))

# 从环境变量获取 secret key，或使用持久化的密钥（避免重启后 session 失效）
def _get_or_create_secret_key():
    env_key = os.environ.get('SECRET_KEY')
    if env_key:
        return env_key
    key_file = os.path.join(get_data_dir(), '.secret_key')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    new_key = secrets.token_hex(32)
    try:
        with open(key_file, 'w') as f:
            f.write(new_key)
    except Exception:
        pass
    return new_key

app.secret_key = _get_or_create_secret_key()

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(traceback.format_exc())
    return jsonify({'error': str(e)}), 500

# 加载 .env 文件
_env_loaded = {}
for _try_path in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
    os.path.join(os.getcwd(), '.env'),
]:
    if os.path.exists(_try_path):
        logging.info(f"加载 .env 文件: {_try_path}")
        with open(_try_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _key, _val = _line.split('=', 1)
                    _key, _val = _key.strip(), _val.strip()
                    os.environ[_key] = _val  # 强制覆盖，确保 .env 中的值生效
                    _env_loaded[_key] = _val
        break

# AI 配置：Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
OPENAI_API_KEY = GEMINI_API_KEY
OPENAI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
AI_MODEL = os.environ.get('AI_MODEL', 'gemini-2.0-flash')

logging.info(f"Gemini API Key 状态: {'已设置 (' + GEMINI_API_KEY[:8] + '...)' if GEMINI_API_KEY else '未设置'}")
logging.info(f"使用模型: {AI_MODEL}")
if not GEMINI_API_KEY:
    logging.warning("⚠️ GEMINI_API_KEY 未设置，AI 功能将不可用。请在 .env 中设置 GEMINI_API_KEY")

UPLOAD_FOLDER = os.path.join(get_data_dir(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 上传限制

lemmatizer = None

def init_nltk():
    global lemmatizer
    try:
        # 设置 NLTK 数据目录
        nltk_data_dir = os.path.join(get_base_path(), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.insert(0, nltk_data_dir)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
            nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
        
        lemmatizer = WordNetLemmatizer()
        wordnet.synsets('hello')
        logging.info("NLTK 初始化成功")
    except Exception as e:
        logging.error(f"NLTK 初始化失败: {e}")

def get_db_path():
    return os.path.join(get_data_dir(), 'study.db')

# ============================================
# 🔐 用户认证系统
# ============================================

def hash_password(password):
    """密码加密 - 使用 werkzeug 安全哈希 (pbkdf2:sha256)"""
    return generate_password_hash(password, method='pbkdf2:sha256')

def verify_password(stored_hash, password):
    """验证密码 - 兼容旧版 SHA256 哈希，成功后自动升级"""
    if stored_hash.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_hash, password)
    # 兼容旧版 SHA256 裸哈希
    return stored_hash == hashlib.sha256(password.encode()).hexdigest()

def init_db():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
    except:
        pass
    
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        is_guest INTEGER DEFAULT 0,
        oauth_provider TEXT,
        oauth_id TEXT,
        avatar_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )''')

    # 兼容旧数据库：逐列添加新字段
    for col, col_type, default in [
        ('is_guest', 'INTEGER', '0'),
        ('oauth_provider', 'TEXT', None),
        ('oauth_id', 'TEXT', None),
        ('avatar_url', 'TEXT', None),
    ]:
        try:
            if default is not None:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type} DEFAULT {default}")
            else:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
        except:
            pass
    
    # 书籍表
    c.execute('''CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT,
        content TEXT,
        total_vocab INTEGER DEFAULT 0,
        new_vocab INTEGER DEFAULT 0,
        cumulative_vocab INTEGER DEFAULT 0,
        word_count INTEGER DEFAULT 0,
        folder_id INTEGER DEFAULT 0,
        source_type TEXT DEFAULT 'upload',
        author TEXT,
        chapters TEXT,
        file_format TEXT,
        page_count INTEGER DEFAULT 0,
        last_read_at TIMESTAMP,
        read_position INTEGER DEFAULT 0,
        read_percent REAL DEFAULT 0,
        current_chapter TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, title)
    )''')

    # 兼容旧 books 表：逐列添加新字段
    for col, col_type, default in [
        ('source_type', 'TEXT', "'upload'"),
        ('author', 'TEXT', None),
        ('chapters', 'TEXT', None),
        ('file_format', 'TEXT', None),
        ('page_count', 'INTEGER', '0'),
        ('last_read_at', 'TIMESTAMP', None),
        ('read_position', 'INTEGER', '0'),
        ('read_percent', 'REAL', '0'),
        ('current_chapter', 'TEXT', None),
    ]:
        try:
            if default is not None:
                c.execute(f"ALTER TABLE books ADD COLUMN {col} {col_type} DEFAULT {default}")
            else:
                c.execute(f"ALTER TABLE books ADD COLUMN {col} {col_type}")
        except:
            pass

    # 样书库表（公版书籍，全局共享）
    c.execute('''CREATE TABLE IF NOT EXISTS sample_books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE NOT NULL,
        author TEXT,
        description TEXT,
        content TEXT,
        difficulty TEXT DEFAULT 'intermediate',
        category TEXT,
        word_count INTEGER DEFAULT 0,
        cover_emoji TEXT DEFAULT '📖',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # 文件夹表
    c.execute('''CREATE TABLE IF NOT EXISTS folders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, name)
    )''')
    
    # 生词本表
    c.execute('''CREATE TABLE IF NOT EXISTS notebook (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        book_id INTEGER,
        text TEXT,
        explanation TEXT,
        type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        next_review TIMESTAMP,
        interval INTEGER DEFAULT 0,
        ease_factor REAL DEFAULT 2.5,
        repetitions INTEGER DEFAULT 0,
        quiz_correct_count INTEGER DEFAULT 0,
        word_index INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(book_id) REFERENCES books(id)
    )''')
    
    # 忽略词表
    c.execute('''CREATE TABLE IF NOT EXISTS ignored_words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        word TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, word)
    )''')
    
    # 用户设置表
    c.execute('''CREATE TABLE IF NOT EXISTS user_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        key TEXT,
        value TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, key)
    )''')
    
    # 阅读会话记录表
    c.execute('''CREATE TABLE IF NOT EXISTS reading_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        book_id INTEGER NOT NULL,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        duration_seconds INTEGER DEFAULT 0,
        start_position INTEGER DEFAULT 0,
        end_position INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(book_id) REFERENCES books(id)
    )''')

    # 密码重置令牌表
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        used INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # 创建索引
    try:
        c.execute("CREATE INDEX IF NOT EXISTS idx_books_user ON books(user_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_notebook_user ON notebook(user_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_notebook_book ON notebook(book_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_notebook_created ON notebook(created_at);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reset_token ON password_reset_tokens(token);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reading_sessions_user ON reading_sessions(user_id, book_id);")
    except:
        pass
    
    conn.commit()
    conn.close()
    logging.info("数据库初始化成功")

def get_user_ignore_set(user_id):
    """获取用户的忽略词集合"""
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("SELECT word FROM ignored_words WHERE user_id = ?", (user_id,))
        ignore_set = set(row[0] for row in c.fetchall())
        conn.close()
        return ignore_set
    except:
        return set()

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': 'unauthorized', 'message': '请先登录'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user_id():
    return session.get('user_id')

# 初始化
threading.Thread(target=init_nltk, daemon=True).start()
init_db()

# --- 核心逻辑 ---
def is_valid_word(word, ignore_set):
    word = word.lower()
    if word in ignore_set:
        return False
    if len(word) < 2 and word not in ['a', 'i']:
        return False
    if not re.search(r'[aeiouy]', word):
        return False
    return True

def extract_words_list(text, user_id, ignore_set=None):
    if not text:
        return []
    if ignore_set is None:
        ignore_set = get_user_ignore_set(user_id)
    return [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if is_valid_word(w, ignore_set)]

def extract_words(text, user_id, ignore_set=None):
    return set(extract_words_list(text, user_id, ignore_set))

def recalculate_chain(user_id):
    ignore_set = get_user_ignore_set(user_id)
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id, content FROM books WHERE user_id = ? ORDER BY id ASC", (user_id,))
    books = c.fetchall()
    global_vocab = set()
    for bid, content in books:
        word_list = extract_words_list(content, user_id, ignore_set)
        words = set(word_list)
        new_count = len(words - global_vocab)
        global_vocab.update(words)
        c.execute("UPDATE books SET total_vocab=?, new_vocab=?, cumulative_vocab=?, word_count=? WHERE id=?",
                  (len(words), new_count, len(global_vocab), len(word_list), bid))
    conn.commit()
    conn.close()

def analyze_vocabulary_task(bid, text, user_id):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE books SET content = ? WHERE id = ?", (text, bid))
    conn.commit()
    conn.close()
    recalculate_chain(user_id)

def _get_api_key():
    """动态获取 API Key（优先全局变量，回退环境变量）"""
    return OPENAI_API_KEY or os.environ.get('GEMINI_API_KEY', '')

def call_openai(prompt, max_tokens=200):
    """调用 Gemini API（兼容 OpenAI Chat Completions 格式）"""
    api_key = _get_api_key()
    if not api_key:
        return "请在 .env 中设置 GEMINI_API_KEY 以使用 AI 功能"
    if not HAS_REQUESTS:
        return "Error: requests library missing."
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        resp = requests.post(
            OPENAI_API_URL,
            json=data,
            headers=headers,
            timeout=15
        )
        if resp.status_code == 200:
            result = resp.json()
            return result['choices'][0]['message']['content']
        else:
            error_detail = resp.json().get('error', {}).get('message', f'Status {resp.status_code}')
            return f"API Error: {error_detail}"
    except Exception as e:
        return f"Net Error: {e}"

def call_openai_chat(messages, max_tokens=800):
    """多轮对话版本，接受完整 messages 列表"""
    api_key = _get_api_key()
    if not api_key:
        return "请在 .env 中设置 GEMINI_API_KEY 以使用 AI 功能"
    if not HAS_REQUESTS:
        return "Error: requests library missing."
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": AI_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.5
        }
        resp = requests.post(
            OPENAI_API_URL,
            json=data,
            headers=headers,
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            return result['choices'][0]['message']['content']
        else:
            error_detail = resp.json().get('error', {}).get('message', f'Status {resp.status_code}')
            return f"API Error: {error_detail}"
    except Exception as e:
        return f"Net Error: {e}"


def clean_block_text(text):
    text = re.sub(r'(\w)-\n\s*(\w)', r'\1\2', text)
    text = text.replace('\n', ' ').strip()
    return re.sub(r'\s+', ' ', text)

def is_header_footer(text):
    """判断文本是否是页眉页脚（页码、章节标题重复等）"""
    text = text.strip()
    # 纯数字（页码）
    if re.match(r'^\d{1,4}$', text):
        return True
    # 常见页眉页脚模式
    if re.match(r'^(page|第)\s*\d+\s*(页|of|/)\s*\d*', text, re.IGNORECASE):
        return True
    # 版权声明等短文本
    if len(text) < 5:
        return True
    # 全大写且很短的标题重复（页眉）
    if text.isupper() and len(text) < 50:
        return True
    return False

def extract_pdf_chapters(doc):
    """从 PDF 目录（TOC）提取章节信息，无内嵌目录时自动检测"""
    chapters = []
    # 第一优先：使用 PDF 内嵌目录
    try:
        toc = doc.get_toc()
        for level, title, page in toc:
            chapters.append({
                'level': level,
                'title': title.strip(),
                'page': page
            })
    except:
        pass

    if chapters:
        return chapters

    # 第二优先：通过字体大小和文本模式自动检测章节标题
    chapters = auto_detect_pdf_chapters(doc)
    return chapters

def auto_detect_pdf_chapters(doc):
    """当 PDF 无内嵌目录时，通过字体大小和文本模式自动检测章节"""
    chapters = []
    # 章节标题的正则模式（中英文）
    chapter_patterns = [
        re.compile(r'^(chapter|chap\.?)\s+(\d+|[ivxlcdm]+)', re.IGNORECASE),
        re.compile(r'^(part|section)\s+(\d+|[ivxlcdm]+)', re.IGNORECASE),
        re.compile(r'^第[一二三四五六七八九十百千\d]+[章节篇部回卷]', re.IGNORECASE),
        re.compile(r'^\d+[\.\)]\s+[A-Z\u4e00-\u9fff]'),  # "1. Title" 或 "1) 标题"
    ]

    # 收集所有页面中的文本块及其字体信息
    all_spans = []
    for page_idx, page in enumerate(doc):
        try:
            blocks = page.get_text("dict", flags=0)["blocks"]
        except Exception:
            continue
        for block in blocks:
            if block.get("type", 0) != 0:  # 非文本块
                continue
            for line in block.get("lines", []):
                line_text = ""
                max_size = 0
                is_bold = False
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    size = span.get("size", 0)
                    if size > max_size:
                        max_size = size
                    flags = span.get("flags", 0)
                    if flags & 2 ** 4:  # bold flag
                        is_bold = True
                line_text = line_text.strip()
                if line_text and max_size > 0:
                    all_spans.append({
                        'text': line_text,
                        'size': max_size,
                        'bold': is_bold,
                        'page': page_idx + 1
                    })

    if not all_spans:
        return chapters

    # 计算正文的常见字体大小（众数）
    size_counts = {}
    for s in all_spans:
        rounded = round(s['size'], 1)
        size_counts[rounded] = size_counts.get(rounded, 0) + 1
    body_size = max(size_counts, key=size_counts.get) if size_counts else 12
    # 标题阈值：比正文大 20% 以上
    title_size_threshold = body_size * 1.2

    seen_pages = set()
    for s in all_spans:
        text = s['text']
        is_title = False
        level = 1

        # 方法1：字体大小显著大于正文
        if s['size'] >= title_size_threshold and len(text) < 100:
            is_title = True
            if s['size'] >= body_size * 1.6:
                level = 1  # 大标题
            else:
                level = 2  # 小标题

        # 方法2：匹配章节模式
        if not is_title:
            for pattern in chapter_patterns:
                if pattern.match(text):
                    is_title = True
                    level = 1
                    break

        # 方法3：粗体 + 短文本（可能是小节标题）
        if not is_title and s['bold'] and len(text) < 80 and s['size'] >= body_size:
            # 只有在文本看起来像标题时才标记（非纯数字、非太短）
            if len(text) > 3 and not re.match(r'^\d+$', text):
                is_title = True
                level = 2

        if is_title:
            # 避免同一页重复添加（大标题优先）
            page_key = (s['page'], text[:30])
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                chapters.append({
                    'level': level,
                    'title': text[:120],
                    'page': s['page']
                })

    # 如果检测到的章节太多（可能误判），只保留 level 1
    if len(chapters) > 80:
        chapters = [c for c in chapters if c['level'] == 1]

    # 如果还是太多，放弃自动检测
    if len(chapters) > 100:
        return []

    return chapters

def extract_pdf_smart(path):
    """增强版 PDF 提取：过滤页眉页脚、提取目录、支持 OCR"""
    try:
        import fitz
        doc = fitz.open(path)
        total_pages = len(doc)
        chapters = extract_pdf_chapters(doc)

        all_text = []
        # 第一遍：收集所有块的位置信息，用于检测页眉页脚区域
        page_heights = []
        for p in doc:
            rect = p.rect
            page_heights.append(rect.height)

        for page_idx, p in enumerate(doc):
            page_height = page_heights[page_idx] if page_idx < len(page_heights) else 800
            header_zone = page_height * 0.08  # 顶部 8% 为页眉区域
            footer_zone = page_height * 0.92  # 底部 8% 为页脚区域

            blocks = p.get_text("blocks")
            page_text = []

            for b in blocks:
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                if b[6] == 1:  # 图片块跳过
                    continue

                y_top = b[1]
                y_bottom = b[3]
                raw = clean_block_text(b[4])

                # 过滤页眉页脚区域
                if y_top < header_zone and len(raw) < 60:
                    continue
                if y_bottom > footer_zone and len(raw) < 60:
                    continue

                # 过滤明显的页眉页脚内容
                if is_header_footer(raw):
                    continue

                if len(raw) > 3:
                    page_text.append(raw)

            if page_text:
                all_text.append('\n'.join(page_text))

        text = '\n\n'.join(all_text)

        # 如果提取到的文本太少，尝试 OCR（扫描版 PDF）
        if len(text.split()) < 50 and total_pages > 0:
            ocr_text = extract_pdf_ocr(doc)
            if ocr_text and len(ocr_text.split()) > len(text.split()):
                text = ocr_text
                logging.info(f"PDF OCR 提取成功: {len(text)} 字符")

        # 提取作者信息
        author = ''
        try:
            metadata = doc.metadata
            if metadata:
                author = metadata.get('author', '') or ''
        except:
            pass

        doc.close()
        return {
            'text': text,
            'chapters': chapters,
            'page_count': total_pages,
            'author': author,
            'format': 'pdf'
        }
    except Exception as e:
        logging.error(f"PDF 提取失败: {e}")
        return {'text': f"PDF 读取失败: {str(e)}", 'chapters': [], 'page_count': 0, 'author': '', 'format': 'pdf'}

def preprocess_image_for_ocr(img):
    """对图片进行预处理以提高 OCR 识别率"""
    try:
        from PIL import ImageFilter, ImageOps
        # 转为灰度
        img = img.convert('L')
        # 增强对比度（自适应直方图均衡）
        img = ImageOps.autocontrast(img, cutoff=2)
        # 轻微锐化
        img = img.filter(ImageFilter.SHARPEN)
        # 二值化（大津阈值近似）
        threshold = 128
        img = img.point(lambda x: 255 if x > threshold else 0, '1')
        img = img.convert('L')
        return img
    except Exception:
        return img

def extract_pdf_ocr(doc):
    """对扫描版 PDF 进行 OCR 识别，支持多引擎回退"""
    ocr_texts = []
    max_pages = min(len(doc), 100)
    ocr_engine = None  # 'pymupdf', 'pytesseract', or None

    # 检测可用的 OCR 引擎：优先尝试 PyMuPDF 内置 OCR
    if len(doc) > 0:
        try:
            test_page = doc[0]
            tp = test_page.get_textpage_ocr(flags=0, language='eng', dpi=200)
            ocr_engine = 'pymupdf'
            logging.info("使用 PyMuPDF 内置 OCR 引擎")
        except Exception:
            pass

    if not ocr_engine:
        try:
            from PIL import Image
            import pytesseract
            # 快速测试 pytesseract 是否可用
            pytesseract.get_tesseract_version()
            ocr_engine = 'pytesseract'
            logging.info("使用 pytesseract OCR 引擎")
        except Exception:
            pass

    if not ocr_engine:
        logging.info("OCR 不可用：需要安装 pytesseract 和 Tesseract-OCR，或使用支持 OCR 的 PyMuPDF 版本")
        return ''

    try:
        for page_idx in range(max_pages):
            page = doc[page_idx]
            text = ''

            if ocr_engine == 'pymupdf':
                try:
                    tp = page.get_textpage_ocr(flags=0, language='eng', dpi=200)
                    text = page.get_text("text", textpage=tp).strip()
                except Exception as e:
                    logging.warning(f"PyMuPDF OCR 第 {page_idx + 1} 页失败: {e}")

            if ocr_engine == 'pytesseract' or (ocr_engine == 'pymupdf' and not text):
                try:
                    from PIL import Image
                    import pytesseract
                    import io
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    # 图片预处理提高识别率
                    img = preprocess_image_for_ocr(img)
                    # 同时尝试中英文识别
                    try:
                        text = pytesseract.image_to_string(img, lang='eng+chi_sim')
                    except Exception:
                        text = pytesseract.image_to_string(img, lang='eng')
                    text = text.strip()
                except ImportError:
                    logging.info("OCR 需要安装 pytesseract 和 Pillow: pip install pytesseract Pillow")
                    return ''
                except Exception as e:
                    logging.warning(f"pytesseract OCR 第 {page_idx + 1} 页失败: {e}")
                    continue

            if text:
                # 清理 OCR 结果中的常见噪点
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                ocr_texts.append(text)

        if max_pages < len(doc):
            ocr_texts.append(f"\n[... 仅 OCR 了前 {max_pages} 页，共 {len(doc)} 页 ...]")

        return '\n\n'.join(ocr_texts)
    except Exception as e:
        logging.error(f"OCR 失败: {e}")
        return ''

def extract_epub(path):
    """解析 EPUB 文件，提取正文和章节"""
    try:
        import zipfile
        from xml.etree import ElementTree as ET

        chapters = []
        all_text = []

        with zipfile.ZipFile(path, 'r') as zf:
            # 找到 content.opf
            container_xml = zf.read('META-INF/container.xml')
            container_tree = ET.fromstring(container_xml)
            ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
            rootfile = container_tree.find('.//c:rootfile', ns)
            if rootfile is None:
                # 尝试无命名空间
                rootfile = container_tree.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
            if rootfile is None:
                return {'text': 'EPUB 格式无法识别', 'chapters': [], 'page_count': 0, 'author': '', 'format': 'epub'}

            opf_path = rootfile.get('full-path')
            opf_dir = '/'.join(opf_path.split('/')[:-1])
            opf_data = zf.read(opf_path)
            opf_tree = ET.fromstring(opf_data)

            # 提取元数据
            author = ''
            dc_ns = 'http://purl.org/dc/elements/1.1/'
            opf_ns = 'http://www.idpf.org/2007/opf'
            creator_el = opf_tree.find(f'.//{{{dc_ns}}}creator')
            if creator_el is not None and creator_el.text:
                author = creator_el.text.strip()

            # 提取阅读顺序
            spine = opf_tree.find(f'{{{opf_ns}}}spine')
            if spine is None:
                spine = opf_tree.find('.//spine')
            manifest = opf_tree.find(f'{{{opf_ns}}}manifest')
            if manifest is None:
                manifest = opf_tree.find('.//manifest')

            # 构建 id -> href 映射
            id_to_href = {}
            if manifest is not None:
                for item in manifest:
                    item_id = item.get('id', '')
                    item_href = item.get('href', '')
                    media = item.get('media-type', '')
                    if 'html' in media or 'xhtml' in media:
                        id_to_href[item_id] = item_href

            # 按 spine 顺序读取章节
            if spine is not None:
                for itemref in spine:
                    idref = itemref.get('idref', '')
                    href = id_to_href.get(idref, '')
                    if not href:
                        continue

                    full_path = f"{opf_dir}/{href}" if opf_dir else href
                    try:
                        html_data = zf.read(full_path)
                    except KeyError:
                        # 尝试不带目录前缀
                        try:
                            html_data = zf.read(href)
                        except KeyError:
                            continue

                    html_str_raw = html_data.decode('utf-8', errors='ignore')
                    # 从 HTML/XHTML 提取纯文本
                    chapter_text = extract_text_from_html(html_str_raw)
                    if chapter_text.strip():
                        # 优先从 h1/h2/h3 标签提取章节标题
                        title = extract_heading_from_html(html_str_raw)
                        if not title:
                            lines = chapter_text.strip().split('\n')
                            title = lines[0][:80] if lines else f'Chapter {len(chapters) + 1}'
                        chapters.append({
                            'level': 1,
                            'title': title,
                            'page': len(chapters) + 1
                        })
                        all_text.append(chapter_text.strip())

        text = '\n\n'.join(all_text)
        return {
            'text': text,
            'chapters': chapters,
            'page_count': len(chapters),
            'author': author,
            'format': 'epub'
        }
    except Exception as e:
        logging.error(f"EPUB 解析失败: {e}")
        return {'text': f"EPUB 读取失败: {str(e)}", 'chapters': [], 'page_count': 0, 'author': '', 'format': 'epub'}

def extract_heading_from_html(html_str):
    """从 HTML 中提取 h1/h2/h3 标签作为章节标题"""
    for tag in ['h1', 'h2', 'h3']:
        match = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html_str, re.DOTALL | re.IGNORECASE)
        if match:
            # 去除内部 HTML 标签
            text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            if text and len(text) > 1:
                return text[:120]
    # 尝试匹配带 class 含 title/heading 的元素
    match = re.search(r'<[^>]+class="[^"]*(?:title|heading|chapter)[^"]*"[^>]*>(.*?)</\w+>', html_str, re.DOTALL | re.IGNORECASE)
    if match:
        text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        if text and len(text) > 1:
            return text[:120]
    return ''

def extract_text_from_html(html_str):
    """从 HTML/XHTML 中提取纯文本，过滤标签"""
    # 移除 script 和 style
    html_str = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
    # 段落和换行转换
    html_str = re.sub(r'<br\s*/?>', '\n', html_str, flags=re.IGNORECASE)
    html_str = re.sub(r'</(p|div|h[1-6]|li|tr)>', '\n', html_str, flags=re.IGNORECASE)
    # 移除所有 HTML 标签
    text = re.sub(r'<[^>]+>', '', html_str)
    # 清理 HTML 实体
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#\d+;', '', text)
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_book_smart(path):
    """统一入口：根据文件格式选择提取方式"""
    ext = path.lower()
    if ext.endswith('.epub'):
        return extract_epub(path)
    elif ext.endswith('.pdf'):
        return extract_pdf_smart(path)
    elif ext.endswith('.txt') or ext.endswith('.md'):
        try:
            for enc in ['utf-8', 'gb18030', 'latin-1']:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        text = f.read()
                    return {'text': text, 'chapters': [], 'page_count': 0, 'author': '', 'format': 'txt'}
                except (UnicodeDecodeError, UnicodeError):
                    continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return {'text': text, 'chapters': [], 'page_count': 0, 'author': '', 'format': 'txt'}
        except Exception as e:
            return {'text': f"文件读取失败: {str(e)}", 'chapters': [], 'page_count': 0, 'author': '', 'format': 'txt'}
    else:
        # 尝试作为文本读取
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return {'text': text, 'chapters': [], 'page_count': 0, 'author': '', 'format': os.path.splitext(path)[1]}
        except Exception as e:
            return {'text': f"不支持的格式: {str(e)}", 'chapters': [], 'page_count': 0, 'author': '', 'format': 'unknown'}

def register_book(name, user_id, source_type='upload', author='', chapters=None, file_format='', page_count=0):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    chapters_json = json_module.dumps(chapters, ensure_ascii=False) if chapters else None
    try:
        c.execute("""INSERT INTO books (user_id, title, source_type, author, chapters, file_format, page_count)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (user_id, name, source_type, author, chapters_json, file_format, page_count))
        bid = c.lastrowid
    except:
        c.execute("SELECT id FROM books WHERE user_id = ? AND title = ?", (user_id, name))
        bid = c.fetchone()[0]
        # 更新元数据
        c.execute("UPDATE books SET source_type=?, author=?, chapters=?, file_format=?, page_count=? WHERE id=?",
                  (source_type, author, chapters_json, file_format, page_count, bid))
    conn.commit()
    conn.close()
    return bid

# ============================================
# 🔐 登录/注册路由
# ============================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        # 传递 OAuth 配置到模板
        google_client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
        apple_client_id = os.environ.get('APPLE_CLIENT_ID', '')
        return render_template('login.html',
                               google_enabled=bool(google_client_id),
                               google_client_id=google_client_id,
                               apple_enabled=bool(apple_client_id),
                               apple_client_id=apple_client_id)

    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'status': 'error', 'message': '请输入用户名和密码'})

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id, password, is_guest FROM users WHERE username = ?", (username,))
    user = c.fetchone()

    if user and verify_password(user[1], password):
        session['user_id'] = user[0]
        session['username'] = username
        session['is_guest'] = bool(user[2]) if user[2] else False
        session.permanent = True
        # 自动升级旧版 SHA256 哈希到安全哈希
        if not user[1].startswith(('pbkdf2:', 'scrypt:')):
            c.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(password), user[0]))
        c.execute("UPDATE users SET last_login = ? WHERE id = ?",
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user[0]))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': '登录成功'})

    conn.close()
    return jsonify({'status': 'error', 'message': '用户名或密码错误'})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()
    
    if not username or not password:
        return jsonify({'status': 'error', 'message': '请输入用户名和密码'})
    
    if len(username) < 2:
        return jsonify({'status': 'error', 'message': '用户名至少2个字符'})
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return jsonify({'status': 'error', 'message': '用户名已存在'})
    
    try:
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  (username, hash_password(password), email))
        user_id = c.lastrowid
        c.execute("INSERT INTO user_settings (user_id, key, value) VALUES (?, 'vocab_size', '0')", (user_id,))
        conn.commit()
        conn.close()
        
        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        
        return jsonify({'status': 'success', 'message': '注册成功'})
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': f'注册失败: {str(e)}'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/user_info')
@login_required
def user_info():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT username, email, is_guest, oauth_provider, avatar_url, created_at FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return jsonify({
            'user_id': user_id,
            'username': row[0],
            'email': row[1] or '',
            'is_guest': bool(row[2]),
            'oauth_provider': row[3] or '',
            'avatar_url': row[4] or '',
            'created_at': row[5] or ''
        })
    return jsonify({'user_id': user_id, 'username': session.get('username')})

# ============================================
# 👤 游客模式
# ============================================

@app.route('/guest_login', methods=['POST'])
def guest_login():
    """创建游客账户，免注册即可试用"""
    guest_id = secrets.token_hex(6)
    guest_username = f"guest_{guest_id}"
    guest_password = secrets.token_hex(16)

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, is_guest) VALUES (?, ?, 1)",
                  (guest_username, hash_password(guest_password)))
        user_id = c.lastrowid
        c.execute("INSERT INTO user_settings (user_id, key, value) VALUES (?, 'vocab_size', '0')", (user_id,))
        conn.commit()
        conn.close()

        session['user_id'] = user_id
        session['username'] = guest_username
        session['is_guest'] = True
        session.permanent = True

        return jsonify({'status': 'success', 'message': '游客模式已开启'})
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': f'创建游客失败: {str(e)}'})

@app.route('/convert_guest', methods=['POST'])
@login_required
def convert_guest():
    """游客账户转正式账户"""
    user_id = get_current_user_id()
    if not session.get('is_guest'):
        return jsonify({'status': 'error', 'message': '当前不是游客账户'})

    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()

    if not username or not password:
        return jsonify({'status': 'error', 'message': '请输入用户名和密码'})
    if len(username) < 2:
        return jsonify({'status': 'error', 'message': '用户名至少2个字符'})

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return jsonify({'status': 'error', 'message': '用户名已存在'})

    try:
        c.execute("UPDATE users SET username = ?, password = ?, email = ?, is_guest = 0 WHERE id = ?",
                  (username, hash_password(password), email, user_id))
        conn.commit()
        conn.close()

        session['username'] = username
        session['is_guest'] = False

        return jsonify({'status': 'success', 'message': '账户升级成功，所有学习数据已保留'})
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': f'升级失败: {str(e)}'})

# ============================================
# 🔗 Google OAuth 登录
# ============================================

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')

@app.route('/auth/google', methods=['POST'])
def google_auth():
    """使用 Google ID Token 登录（前端通过 Google Sign-In 获取 token）"""
    if not GOOGLE_CLIENT_ID:
        return jsonify({'status': 'error', 'message': 'Google 登录未配置'})
    if not HAS_REQUESTS:
        return jsonify({'status': 'error', 'message': '缺少 requests 库'})

    id_token = request.json.get('id_token', '')
    if not id_token:
        return jsonify({'status': 'error', 'message': '缺少 id_token'})

    # 验证 Google ID Token
    try:
        verify_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
        resp = requests.get(verify_url, timeout=10)
        if resp.status_code != 200:
            return jsonify({'status': 'error', 'message': 'Google 验证失败'})
        token_info = resp.json()

        # 确认 token 是给我们的应用签发的
        if token_info.get('aud') != GOOGLE_CLIENT_ID:
            return jsonify({'status': 'error', 'message': 'Token 不匹配'})

        google_id = token_info.get('sub')
        email = token_info.get('email', '')
        name = token_info.get('name', '') or email.split('@')[0]
        avatar = token_info.get('picture', '')

    except Exception as e:
        logging.error(f"Google OAuth 验证失败: {e}")
        return jsonify({'status': 'error', 'message': f'验证出错: {str(e)}'})

    # 查找或创建用户
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("SELECT id, username FROM users WHERE oauth_provider = 'google' AND oauth_id = ?", (google_id,))
    user = c.fetchone()

    if user:
        # 已有用户，直接登录
        user_id, username = user
        c.execute("UPDATE users SET last_login = ?, avatar_url = ? WHERE id = ?",
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), avatar, user_id))
    else:
        # 新用户，自动注册
        # 确保用户名唯一
        base_name = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '', name) or 'user'
        username = base_name
        counter = 1
        while True:
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            if not c.fetchone():
                break
            username = f"{base_name}_{counter}"
            counter += 1

        random_pw = secrets.token_hex(16)
        c.execute("""INSERT INTO users (username, password, email, oauth_provider, oauth_id, avatar_url)
                     VALUES (?, ?, ?, 'google', ?, ?)""",
                  (username, hash_password(random_pw), email, google_id, avatar))
        user_id = c.lastrowid
        c.execute("INSERT INTO user_settings (user_id, key, value) VALUES (?, 'vocab_size', '0')", (user_id,))

    conn.commit()
    conn.close()

    session['user_id'] = user_id
    session['username'] = username
    session['is_guest'] = False
    session.permanent = True

    return jsonify({'status': 'success', 'message': '登录成功', 'username': username})

# ============================================
# 🍎 Apple Sign-In
# ============================================

APPLE_CLIENT_ID = os.environ.get('APPLE_CLIENT_ID', '')  # 即 Services ID
APPLE_TEAM_ID = os.environ.get('APPLE_TEAM_ID', '')
APPLE_KEY_ID = os.environ.get('APPLE_KEY_ID', '')
# Apple 私钥文件路径或内容（.p8 文件）
APPLE_PRIVATE_KEY = os.environ.get('APPLE_PRIVATE_KEY', '')

@app.route('/auth/apple', methods=['POST'])
def apple_auth():
    """Apple Sign-In：前端通过 Apple JS SDK 获取 authorization code + id_token"""
    if not APPLE_CLIENT_ID:
        return jsonify({'status': 'error', 'message': 'Apple 登录未配置'})

    data = request.json
    id_token = data.get('id_token', '')
    auth_code = data.get('code', '')
    user_info = data.get('user', {})  # Apple 首次登录时会传用户信息

    if not id_token:
        return jsonify({'status': 'error', 'message': '缺少 id_token'})

    # 解码 Apple ID Token（JWT）
    try:
        import base64
        # Apple ID Token 是标准 JWT，解码 payload 获取用户信息
        parts = id_token.split('.')
        if len(parts) != 3:
            return jsonify({'status': 'error', 'message': 'Token 格式错误'})

        # 解码 payload（第二部分）
        payload = parts[1]
        # 补齐 base64 填充
        payload += '=' * (4 - len(payload) % 4)
        decoded = json_module.loads(base64.urlsafe_b64decode(payload))

        # 验证 audience
        if decoded.get('aud') != APPLE_CLIENT_ID:
            return jsonify({'status': 'error', 'message': 'Token audience 不匹配'})

        # 验证 issuer
        if decoded.get('iss') != 'https://appleid.apple.com':
            return jsonify({'status': 'error', 'message': 'Token issuer 不合法'})

        # 验证过期时间
        if decoded.get('exp', 0) < datetime.now().timestamp():
            return jsonify({'status': 'error', 'message': 'Token 已过期'})

        apple_id = decoded.get('sub', '')
        email = decoded.get('email', '')
        # Apple 首次登录才返回姓名
        name = ''
        if user_info:
            first = user_info.get('name', {}).get('firstName', '')
            last = user_info.get('name', {}).get('lastName', '')
            name = f"{first} {last}".strip()

    except Exception as e:
        logging.error(f"Apple Sign-In 验证失败: {e}")
        return jsonify({'status': 'error', 'message': f'验证出错: {str(e)}'})

    if not apple_id:
        return jsonify({'status': 'error', 'message': '无法获取 Apple 用户ID'})

    # 查找或创建用户
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("SELECT id, username FROM users WHERE oauth_provider = 'apple' AND oauth_id = ?", (apple_id,))
    user = c.fetchone()

    if user:
        user_id, username = user
        c.execute("UPDATE users SET last_login = ? WHERE id = ?",
                  (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_id))
    else:
        # 新用户
        base_name = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '', name) if name else ''
        if not base_name:
            base_name = email.split('@')[0] if email else 'apple_user'
        username = base_name
        counter = 1
        while True:
            c.execute("SELECT id FROM users WHERE username = ?", (username,))
            if not c.fetchone():
                break
            username = f"{base_name}_{counter}"
            counter += 1

        random_pw = secrets.token_hex(16)
        c.execute("""INSERT INTO users (username, password, email, oauth_provider, oauth_id)
                     VALUES (?, ?, ?, 'apple', ?)""",
                  (username, hash_password(random_pw), email, apple_id))
        user_id = c.lastrowid
        c.execute("INSERT INTO user_settings (user_id, key, value) VALUES (?, 'vocab_size', '0')", (user_id,))

    conn.commit()
    conn.close()

    session['user_id'] = user_id
    session['username'] = username
    session['is_guest'] = False
    session.permanent = True

    return jsonify({'status': 'success', 'message': '登录成功', 'username': username})

# ============================================
# 🔑 密码找回 / 重置
# ============================================

@app.route('/forgot_password', methods=['GET'])
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route('/api/forgot_password', methods=['POST'])
def forgot_password():
    """通过用户名+邮箱验证身份，生成重置令牌"""
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()

    if not username or not email:
        return jsonify({'status': 'error', 'message': '请输入用户名和注册邮箱'})

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id, email FROM users WHERE username = ?", (username,))
    user = c.fetchone()

    if not user or not user[1]:
        conn.close()
        # 模糊回复，防止用户名枚举
        return jsonify({'status': 'success', 'message': '如果信息匹配，重置链接已生成'})

    if user[1].lower() != email.lower():
        conn.close()
        return jsonify({'status': 'success', 'message': '如果信息匹配，重置链接已生成'})

    # 生成重置令牌（有效期 30 分钟）
    token = secrets.token_urlsafe(32)
    c.execute("INSERT INTO password_reset_tokens (user_id, token) VALUES (?, ?)", (user[0], token))
    conn.commit()
    conn.close()

    # 返回令牌给前端（实际生产环境应通过邮件发送）
    logging.info(f"密码重置令牌已生成: 用户={username}, token={token}")
    return jsonify({
        'status': 'success',
        'message': '验证成功，请设置新密码',
        'reset_token': token
    })

@app.route('/reset_password', methods=['GET'])
def reset_password_page():
    token = request.args.get('token', '')
    return render_template('forgot_password.html', reset_token=token)

@app.route('/api/reset_password', methods=['POST'])
def reset_password():
    """使用重置令牌设置新密码"""
    data = request.json
    token = data.get('token', '').strip()
    new_password = data.get('new_password', '')

    if not token or not new_password:
        return jsonify({'status': 'error', 'message': '缺少令牌或新密码'})
    if len(new_password) < 4:
        return jsonify({'status': 'error', 'message': '新密码至少4个字符'})

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    # 查找有效令牌（30 分钟内未使用）
    c.execute("""SELECT user_id FROM password_reset_tokens
                 WHERE token = ? AND used = 0
                 AND created_at >= datetime('now', '-30 minutes')""", (token,))
    row = c.fetchone()

    if not row:
        conn.close()
        return jsonify({'status': 'error', 'message': '重置链接无效或已过期，请重新申请'})

    user_id = row[0]

    # 更新密码
    c.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(new_password), user_id))
    # 标记令牌已使用
    c.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,))
    # 清理该用户所有旧令牌
    c.execute("DELETE FROM password_reset_tokens WHERE user_id = ? AND (used = 1 OR created_at < datetime('now', '-1 hour'))", (user_id,))
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': '密码重置成功，请使用新密码登录'})

# ============================================
# 📖 章节导航
# ============================================

@app.route('/api/book_chapters/<int:book_id>')
@login_required
def get_book_chapters(book_id):
    """获取指定书籍的章节列表"""
    user_id = session['user_id']
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT chapters, title, author, page_count, file_format FROM books WHERE id = ? AND user_id = ?",
              (book_id, user_id))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'status': 'error', 'message': '书籍不存在'}), 404

    chapters_json, title, author, page_count, file_format = row
    chapters = []
    if chapters_json:
        try:
            chapters = json_module.loads(chapters_json)
        except Exception:
            pass

    return jsonify({
        'status': 'success',
        'book': {
            'id': book_id,
            'title': title,
            'author': author or '',
            'page_count': page_count or 0,
            'format': file_format or ''
        },
        'chapters': chapters
    })

# ============================================
# 📖 阅读进度
# ============================================

@app.route('/api/save_reading_progress', methods=['POST'])
@login_required
def save_reading_progress():
    """保存阅读进度（前端定期调用）"""
    user_id = session['user_id']
    data = request.json
    book_id = data.get('book_id')
    position = data.get('position', 0)  # 滚动位置（像素或 word index）
    percent = data.get('percent', 0)    # 阅读百分比
    chapter = data.get('chapter', '')   # 当前章节标题

    if not book_id:
        return jsonify({'status': 'error', 'message': '缺少 book_id'}), 400

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("""UPDATE books SET read_position = ?, read_percent = ?,
                 current_chapter = ?, last_read_at = CURRENT_TIMESTAMP
                 WHERE id = ? AND user_id = ?""",
              (position, percent, chapter, book_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/get_reading_progress/<int:book_id>')
@login_required
def get_reading_progress(book_id):
    """获取某本书的阅读进度"""
    user_id = session['user_id']
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("""SELECT read_position, read_percent, current_chapter, last_read_at,
                 chapters, page_count, file_format
                 FROM books WHERE id = ? AND user_id = ?""",
              (book_id, user_id))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'status': 'error', 'message': '书籍不存在'}), 404

    chapters = []
    if row[4]:
        try:
            chapters = json_module.loads(row[4])
        except Exception:
            pass

    return jsonify({
        'status': 'success',
        'progress': {
            'position': row[0] or 0,
            'percent': row[1] or 0,
            'current_chapter': row[2] or '',
            'last_read_at': row[3] or '',
        },
        'chapters': chapters,
        'page_count': row[5] or 0,
        'format': row[6] or ''
    })

@app.route('/api/start_reading_session', methods=['POST'])
@login_required
def start_reading_session():
    """开始一个阅读会话"""
    user_id = session['user_id']
    data = request.json
    book_id = data.get('book_id')
    position = data.get('position', 0)

    if not book_id:
        return jsonify({'status': 'error', 'message': '缺少 book_id'}), 400

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("""INSERT INTO reading_sessions (user_id, book_id, start_position)
                 VALUES (?, ?, ?)""", (user_id, book_id, position))
    session_id = c.lastrowid
    # 同时更新 last_read_at
    c.execute("UPDATE books SET last_read_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
              (book_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'session_id': session_id})

@app.route('/api/end_reading_session', methods=['POST'])
@login_required
def end_reading_session():
    """结束一个阅读会话"""
    user_id = session['user_id']
    data = request.json
    session_id = data.get('session_id')
    position = data.get('position', 0)

    if not session_id:
        return jsonify({'status': 'error', 'message': '缺少 session_id'}), 400

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("""UPDATE reading_sessions
                 SET end_time = CURRENT_TIMESTAMP,
                     end_position = ?,
                     duration_seconds = CAST((julianday(CURRENT_TIMESTAMP) - julianday(start_time)) * 86400 AS INTEGER)
                 WHERE id = ? AND user_id = ?""",
              (position, session_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/reading_stats')
@login_required
def reading_stats():
    """获取用户阅读统计（总时长、今日时长、最近阅读书籍）"""
    user_id = session['user_id']
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    # 总阅读时长
    c.execute("SELECT COALESCE(SUM(duration_seconds), 0) FROM reading_sessions WHERE user_id = ?", (user_id,))
    total_seconds = c.fetchone()[0]

    # 今日阅读时长
    c.execute("""SELECT COALESCE(SUM(duration_seconds), 0) FROM reading_sessions
                 WHERE user_id = ? AND date(start_time) = date('now')""", (user_id,))
    today_seconds = c.fetchone()[0]

    # 最近阅读的书（按 last_read_at 排序）
    c.execute("""SELECT id, title, read_percent, current_chapter, last_read_at, file_format
                 FROM books WHERE user_id = ? AND last_read_at IS NOT NULL
                 ORDER BY last_read_at DESC LIMIT 10""", (user_id,))
    recent_books = []
    for row in c.fetchall():
        recent_books.append({
            'id': row[0], 'title': row[1],
            'percent': row[2] or 0, 'chapter': row[3] or '',
            'last_read': row[4] or '', 'format': row[5] or ''
        })

    conn.close()
    return jsonify({
        'status': 'success',
        'total_minutes': round(total_seconds / 60),
        'today_minutes': round(today_seconds / 60),
        'recent_books': recent_books
    })

# ============================================
# 📤 数据导出
# ============================================

@app.route('/api/export_data', methods=['GET'])
@login_required
def export_data():
    """导出用户全部学习数据（JSON 格式）"""
    user_id = get_current_user_id()
    export_type = request.args.get('type', 'all')  # all / notebook / vocab / report

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    result = {'exported_at': datetime.now().isoformat(), 'user': session.get('username')}

    if export_type in ('all', 'notebook'):
        c.execute("SELECT text, explanation, type, created_at, next_review, quiz_correct_count FROM notebook WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        result['notebook'] = [dict(row) for row in c.fetchall()]

    if export_type in ('all', 'vocab'):
        c.execute("SELECT id, content FROM books WHERE user_id = ? ORDER BY id ASC", (user_id,))
        books = c.fetchall()
        ignore_set = get_user_ignore_set(user_id)
        all_vocab = set()
        book_vocab_list = []
        for b in books:
            words = extract_words(b['content'] or '', user_id, ignore_set)
            new_words = words - all_vocab
            all_vocab.update(words)
            book_vocab_list.append({'book_id': b['id'], 'new_words': sorted(list(new_words)), 'total_words': len(words)})
        result['vocabulary'] = {'total_unique_words': len(all_vocab), 'all_words': sorted(list(all_vocab)), 'by_book': book_vocab_list}

    if export_type in ('all', 'report'):
        c.execute("SELECT COUNT(*) FROM notebook WHERE user_id = ?", (user_id,))
        total_words_learned = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM books WHERE user_id = ?", (user_id,))
        total_books = c.fetchone()[0]
        c.execute("SELECT value FROM user_settings WHERE user_id = ? AND key = 'vocab_size'", (user_id,))
        urow = c.fetchone()
        vocab_size = int(urow[0]) if urow else 0
        c.execute("""SELECT DATE(created_at) as date, COUNT(*) as count
                     FROM notebook WHERE user_id = ? GROUP BY DATE(created_at) ORDER BY date""", (user_id,))
        daily_stats = [{'date': row['date'], 'count': row['count']} for row in c.fetchall()]
        result['report'] = {
            'total_words_learned': total_words_learned,
            'total_books': total_books,
            'vocab_size': vocab_size,
            'daily_activity': daily_stats
        }

    conn.close()

    json_str = json_module.dumps(result, ensure_ascii=False, indent=2)
    return Response(
        json_str,
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename=ai_reader_export_{datetime.now().strftime("%Y%m%d")}.json'}
    )

@app.route('/api/export_csv', methods=['GET'])
@login_required
def export_csv():
    """导出生词本为 CSV 格式"""
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT text, explanation, type, created_at, quiz_correct_count FROM notebook WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()

    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['单词/短语', '释义', '类型', '添加时间', '正确次数'])
    for r in rows:
        # 去除 HTML 标签
        explanation = re.sub(r'<[^>]+>', '', r['explanation'] or '')
        writer.writerow([r['text'], explanation, r['type'], r['created_at'], r['quiz_correct_count'] or 0])

    # 添加 BOM 头，确保 Excel 正确识别中文编码
    csv_content = '\ufeff' + output.getvalue()
    return Response(
        csv_content,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename=vocabulary_{datetime.now().strftime("%Y%m%d")}.csv'}
    )

# ============================================
# ⚙️ 账户管理
# ============================================

@app.route('/settings')
@login_required
def settings_page():
    return render_template('settings.html', username=session.get('username'))

@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    user_id = get_current_user_id()
    data = request.json
    old_password = data.get('old_password', '')
    new_password = data.get('new_password', '')

    if not old_password or not new_password:
        return jsonify({'status': 'error', 'message': '请输入旧密码和新密码'})
    if len(new_password) < 4:
        return jsonify({'status': 'error', 'message': '新密码至少4个字符'})

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()

    if not row or not verify_password(row[0], old_password):
        conn.close()
        return jsonify({'status': 'error', 'message': '旧密码错误'})

    c.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': '密码修改成功'})

@app.route('/api/update_email', methods=['POST'])
@login_required
def update_email():
    user_id = get_current_user_id()
    email = request.json.get('email', '').strip()

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': '邮箱已更新'})

@app.route('/api/delete_account', methods=['POST'])
@login_required
def delete_account():
    """删除账户及全部学习数据（不可恢复）"""
    user_id = get_current_user_id()
    data = request.json
    confirm = data.get('confirm', '')

    if confirm != 'DELETE':
        return jsonify({'status': 'error', 'message': '请输入 DELETE 确认删除'})

    # 非 OAuth 用户需要验证密码
    password = data.get('password', '')
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT password, oauth_provider FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()

    if row and not row[1] and not session.get('is_guest'):
        # 普通用户需要验证密码
        if not password or not verify_password(row[0], password):
            conn.close()
            return jsonify({'status': 'error', 'message': '密码验证失败'})

    try:
        # 删除所有用户数据
        c.execute("DELETE FROM reading_sessions WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM notebook WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM ignored_words WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM books WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM folders WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

        # 删除用户上传文件
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        if os.path.exists(user_upload_dir):
            import shutil
            shutil.rmtree(user_upload_dir, ignore_errors=True)

        session.clear()
        return jsonify({'status': 'success', 'message': '账户已删除，所有数据已清除'})
    except Exception as e:
        conn.close()
        logging.error(f"删除账户失败: {e}")
        return jsonify({'status': 'error', 'message': f'删除失败: {str(e)}'})

# ============================================
# 📚 主要路由
# ============================================

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    user_id = get_current_user_id()
    content, bid, title, stats = "", 0, "", {'new': 0, 'cumulative': 0, 'user_vocab': 0, 'word_count': 0}
    
    if request.method == 'POST' and 'file' in request.files:
        f = request.files['file']
        if f.filename:
            safe_name = secure_filename(f.filename)
            if not safe_name:
                safe_name = 'uploaded_file'
            user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if not os.path.exists(user_upload_folder):
                os.makedirs(user_upload_folder)

            path = os.path.join(user_upload_folder, safe_name)
            f.save(path)
            display_name = f.filename
            title = display_name

            # 统一使用 extract_book_smart，返回 dict
            result = extract_book_smart(path)
            content = result['text']
            bid = register_book(
                display_name, user_id,
                source_type='upload',
                author=result.get('author', ''),
                chapters=result.get('chapters'),
                file_format=result.get('format', ''),
                page_count=result.get('page_count', 0)
            )

            threading.Thread(target=analyze_vocabulary_task, args=(bid, content, user_id), daemon=True).start()
            
    elif request.args.get('book_id'):
        try:
            bid = int(request.args.get('book_id'))
        except (ValueError, TypeError):
            bid = 0
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("SELECT title, content FROM books WHERE id=? AND user_id=?", (bid, user_id))
        row = c.fetchone()
        conn.close()
        if row:
            title, content = row
    
    chapters = []
    read_position = 0
    read_percent = 0

    if bid:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("""SELECT new_vocab, cumulative_vocab, word_count, chapters,
                     read_position, read_percent FROM books WHERE id=? AND user_id=?""",
                  (bid, user_id))
        row = c.fetchone()
        c.execute("SELECT value FROM user_settings WHERE user_id=? AND key='vocab_size'", (user_id,))
        urow = c.fetchone()
        # 更新 last_read_at
        c.execute("UPDATE books SET last_read_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                  (bid, user_id))
        conn.commit()
        conn.close()
        if row:
            stats['new'], stats['cumulative'], stats['word_count'] = row[0], row[1], row[2]
            if row[3]:
                try:
                    chapters = json_module.loads(row[3])
                except Exception:
                    pass
            read_position = row[4] or 0
            read_percent = row[5] or 0
        if urow:
            stats['user_vocab'] = int(urow[0])

    return render_template('index.html',
                           content=content,
                           book_id=bid,
                           book_title=title,
                           stats=stats,
                           chapters=chapters,
                           read_position=read_position,
                           read_percent=read_percent,
                           username=session.get('username'))

# 🟢 智能判断答案
@app.route('/api/submit_quiz', methods=['POST'])
@login_required
def submit_quiz():
    user_id = get_current_user_id()
    card_id = request.json.get('card_id')
    user_input = request.json.get('user_input', '').strip().lower()
    correct_word = request.json.get('correct_word', '').strip().lower()
    
    is_correct = False
    
    if user_input == correct_word:
        is_correct = True
    elif lemmatizer:
        try:
            u_v = lemmatizer.lemmatize(user_input, 'v')
            c_v = lemmatizer.lemmatize(correct_word, 'v')
            u_n = lemmatizer.lemmatize(user_input, 'n')
            c_n = lemmatizer.lemmatize(correct_word, 'n')
            
            if u_v == c_v or u_n == c_n:
                is_correct = True
        except:
            pass

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM notebook WHERE id = ? AND user_id = ?", (card_id, user_id))
    card = c.fetchone()
    if not card:
        conn.close()
        return jsonify({'status': 'error', 'message': '卡片不存在'})
    
    card = dict(card)
    current_correct = card['quiz_correct_count'] or 0
    now = datetime.now()
    
    if is_correct:
        current_correct += 1
        next_review_dt = now + timedelta(minutes=90)
    else:
        current_correct = 0
        next_review_dt = now + timedelta(minutes=30)
        
    next_review_str = next_review_dt.strftime('%Y-%m-%d %H:%M:%S')
    c.execute('UPDATE notebook SET next_review = ?, quiz_correct_count = ? WHERE id = ? AND user_id = ?',
              (next_review_str, current_correct, card_id, user_id))
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'success', 'is_correct': is_correct, 'next_time': next_review_str})

# 🟢 获取待复习卡片
@app.route('/api/due_cards', methods=['GET'])
@login_required
def due_cards():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    force_all = request.args.get('force') == 'true'
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if force_all:
        query = "SELECT * FROM notebook WHERE user_id = ? ORDER BY created_at DESC"
        params = (user_id,)
    else:
        query = "SELECT * FROM notebook WHERE user_id = ? AND (next_review IS NULL OR next_review <= ?) ORDER BY created_at DESC"
        params = (user_id, now_str)
        
    try:
        c.execute(query, params)
        items = [dict(row) for row in c.fetchall()]
        c.execute("SELECT COUNT(*) FROM notebook WHERE user_id = ?", (user_id,))
        total_count = c.fetchone()[0]
    except Exception as e:
        items = []
        total_count = 0
        
    conn.close()
    return jsonify({'cards': items, 'count': len(items), 'total_exists': total_count})

# ============================================
# 💬 AI 对话式阅读助手
# ============================================

@app.route('/api/ai_chat', methods=['POST'])
@login_required
def ai_chat():
    """上下文感知的 AI 阅读助手对话"""
    user_id = get_current_user_id()
    data = request.json
    user_message = data.get('message', '').strip()
    context_text = data.get('context', '')  # 当前阅读的段落/页面文本
    book_title = data.get('book_title', '')
    history = data.get('history', [])  # 前端维护的对话历史

    if not user_message:
        return jsonify({'status': 'error', 'message': '请输入问题'})

    # 构建 system prompt
    system_prompt = "你是一个英语阅读助手，正在帮助用户阅读和理解英语文章。请用中文回答用户的问题。"
    if book_title:
        system_prompt += f"\n用户正在阅读的书籍：《{book_title}》"
    if context_text:
        # 限制上下文长度，避免 token 超限
        context_trimmed = context_text[:2000]
        system_prompt += f"\n\n以下是用户当前正在阅读的段落内容：\n---\n{context_trimmed}\n---\n请基于以上内容回答用户的问题。如果问题与上文无关，也可以正常回答。回答要简洁清晰。"

    # 构建 messages
    messages = [{"role": "system", "content": system_prompt}]

    # 加入对话历史（最近 10 轮，避免超长）
    for h in history[-10:]:
        role = h.get('role', 'user')
        content = h.get('content', '')
        if role in ('user', 'assistant') and content:
            messages.append({"role": role, "content": content})

    # 加入当前用户消息
    messages.append({"role": "user", "content": user_message})

    # 调用 AI
    reply = call_openai_chat(messages, max_tokens=800)

    if not reply or reply.startswith(('API Error', 'Net Error', '请设置')):
        return jsonify({'status': 'error', 'message': reply or 'AI 服务暂不可用'})

    return jsonify({
        'status': 'success',
        'reply': reply
    })

@app.route('/ask_ai', methods=['POST'])
@login_required
def ask_ai():
    user_id = get_current_user_id()
    d = request.json
    text, mode, bid, wid = d.get('text'), d.get('mode'), d.get('book_id'), d.get('word_index')
    
    if not text:
        return jsonify({'result': '请输入要查询的内容'})
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    res = ""
    if mode == 'word':
        # 直接用 OpenAI ChatGPT 查询中英文释义
        ai_result = call_openai(
            f"请为英语单词 '{text}' 提供简洁释义，严格按以下格式输出，不要任何多余内容：\n中文：一个简短的中文翻译\n英文：A brief English definition in one sentence.",
            150
        )
        logging.info(f"OpenAI 查词结果 [{text}]: {ai_result}")

        cn = ""
        en = ""
        if ai_result and not ai_result.startswith(('API Error', 'Net Error', '请设置')):
            for line in ai_result.strip().split('\n'):
                line = line.strip()
                if '中文' in line and ('：' in line or ':' in line):
                    cn = re.split(r'[：:]', line, 1)[-1].strip()
                elif '英文' in line and ('：' in line or ':' in line):
                    en = re.split(r'[：:]', line, 1)[-1].strip()
                elif 'Chinese' in line and ':' in line:
                    cn = line.split(':', 1)[-1].strip()
                elif 'English' in line and ':' in line:
                    en = line.split(':', 1)[-1].strip()
            if not cn and not en:
                cn = ai_result.strip()
        else:
            cn = ai_result or "查询失败"

        cn = str(html_escape(cn))
        en = str(html_escape(en))

        def wrap(m):
            return f'<span class="word nested-word">{m.group(0)}</span>'
        if en:
            en = re.sub(r'\b[a-zA-Z]{3,}\b', wrap, en)

        res = f"<div class='cn-def' style='font-size:18px;font-weight:bold;color:#2c3e50'>{cn}</div><div class='en-def' style='font-size:14px;color:#555;margin-top:4px'>{en}</div>"
    else:
        res = call_openai(f"请翻译并分析以下内容：{text}", 500)
        if not res:
            res = "无法获取解释"
        res = str(html_escape(res))

    # 保存到生词本（已有则更新释义，没有则插入）
    try:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        existing_id = None
        if mode == 'word' and wid is not None:
            c.execute("SELECT id FROM notebook WHERE user_id=? AND book_id=? AND word_index=?", (user_id, bid, wid))
            row = c.fetchone()
            if row:
                existing_id = row[0]
        if existing_id:
            c.execute("UPDATE notebook SET explanation=?, created_at=? WHERE id=?", (res, now_str, existing_id))
        else:
            c.execute("INSERT INTO notebook (user_id, text, explanation, type, book_id, word_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (user_id, text, res, mode, bid, wid, now_str))
        conn.commit()
    except Exception as e:
        logging.error(f"保存生词失败: {e}")
    
    conn.close()
    return jsonify({'result': res})

@app.route('/delete_book', methods=['POST'])
@login_required
def delete_book():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        c.execute("DELETE FROM notebook WHERE book_id = ? AND user_id = ?", (book_id, user_id))
        c.execute("DELETE FROM books WHERE id = ? AND user_id = ?", (book_id, user_id))
        conn.commit()
        recalculate_chain(user_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"删除书籍失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/reset_book_progress', methods=['POST'])
@login_required
def reset_book_progress():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    if not book_id:
        return jsonify({'status': 'error', 'message': '缺少 book_id'}), 400
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        # 彻底清除：删除该书所有查词记录、高亮、复习进度
        c.execute("DELETE FROM notebook WHERE book_id = ? AND user_id = ?", (book_id, user_id))
        # 重置用户词汇量统计
        c.execute("UPDATE user_settings SET value = '0' WHERE user_id = ? AND key = 'vocab_size'", (user_id,))
        # 重置该书的词汇统计数据
        c.execute("UPDATE books SET total_vocab = 0, new_vocab = 0, cumulative_vocab = 0 WHERE id = ? AND user_id = ?", (book_id, user_id))
        conn.commit()
        conn.close()
        # 重新计算词汇统计链
        recalculate_chain(user_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_stats', methods=['POST'])
@login_required
def get_stats_api():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT new_vocab, cumulative_vocab, word_count FROM books WHERE id = ? AND user_id = ?", (book_id, user_id))
    book_row = c.fetchone()
    c.execute("SELECT value FROM user_settings WHERE user_id = ? AND key='vocab_size'", (user_id,))
    user_row = c.fetchone()
    conn.close()
    res = {'new': 0, 'cumulative': 0, 'user_vocab': 0, 'word_count': 0}
    if book_row:
        res['new'], res['cumulative'], res['word_count'] = book_row
    if user_row:
        res['user_vocab'] = int(user_row[0])
        remaining = res['cumulative'] - res['user_vocab']
        res['new'] = remaining if remaining > 0 else 0
    return jsonify(res)

@app.route('/reset_book_review', methods=['POST'])
@login_required
def reset_book_review():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT cumulative_vocab FROM books WHERE id = ? AND user_id = ?", (book_id, user_id))
    row = c.fetchone()
    new_level = row[0] if row else 0
    c.execute("UPDATE user_settings SET value = ? WHERE user_id = ? AND key = 'vocab_size'", (str(new_level), user_id))
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("UPDATE notebook SET next_review = ?, quiz_correct_count = 0 WHERE book_id = ? AND user_id = ?", (now_str, book_id, user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'new_level': new_level})

@app.route('/api/notebook_data', methods=['POST'])
@login_required
def nb_data():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM folders WHERE user_id = ? ORDER BY created_at ASC", (user_id,))
    folders = [dict(row) for row in c.fetchall()]
    c.execute("SELECT id, title, total_vocab, new_vocab, word_count, folder_id FROM books WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    books = [dict(row) for row in c.fetchall()]
    
    if book_id and str(book_id) != "0":
        c.execute("SELECT * FROM notebook WHERE user_id = ? AND book_id = ? ORDER BY created_at DESC", (user_id, book_id))
    else:
        c.execute("SELECT * FROM notebook WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    items = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify({'folders': folders, 'books': books, 'items': items})

@app.route('/api/heatmap_data', methods=['GET'])
@login_required
def hm_data():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count 
        FROM notebook 
        WHERE user_id = ? AND created_at >= DATE('now', '-30 days')
        GROUP BY DATE(created_at)
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    result = {row[0]: row[1] for row in rows}
    return jsonify(result)

@app.route('/api/review_card', methods=['POST'])
@login_required
def review_card_api():
    user_id = get_current_user_id()
    d = request.json
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        now = datetime.now()
        if d['quality'] == 1:
            next_t = now + timedelta(minutes=1)
        elif d['quality'] == 3:
            next_t = now + timedelta(minutes=15)
        else:
            next_t = now + timedelta(hours=1)
        c.execute('UPDATE notebook SET next_review = ?, interval = 0 WHERE id = ? AND user_id = ?',
                  (next_t.strftime('%Y-%m-%d %H:%M:%S'), d['card_id'], user_id))
        conn.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"复习卡片更新失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/get_book_highlights', methods=['POST'])
@login_required
def get_highlights():
    user_id = get_current_user_id()
    conn = None
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("SELECT word_index FROM notebook WHERE user_id = ? AND book_id = ? AND word_index IS NOT NULL",
                  (user_id, request.json.get('book_id')))
        indices = [row[0] for row in c.fetchall()]
        return jsonify({'indices': indices})
    except Exception:
        return jsonify({'indices': []})
    finally:
        if conn:
            conn.close()

@app.route('/create_folder', methods=['POST'])
@login_required
def create_folder_api():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        c.execute("INSERT INTO folders (user_id, name) VALUES (?, ?)", (user_id, request.json.get('name')))
        conn.commit()
    except:
        pass
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/move_book', methods=['POST'])
@login_required
def move_book_api():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE books SET folder_id = ? WHERE id = ? AND user_id = ?",
              (request.json.get('folder_id'), request.json.get('book_id'), user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/clear_notebook', methods=['POST'])
@login_required
def clear_notebook_api():
    user_id = get_current_user_id()
    bid = request.json.get('book_id')
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    if bid and str(bid) != '0':
        c.execute("DELETE FROM notebook WHERE book_id = ? AND user_id = ?", (bid, user_id))
    else:
        c.execute("DELETE FROM notebook WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/delete_single_word', methods=['POST'])
@login_required
def del_word():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("DELETE FROM notebook WHERE id = ? AND user_id = ?", (request.json.get('id'), user_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/add_to_ignore', methods=['POST'])
@login_required
def ignore_word():
    user_id = get_current_user_id()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO ignored_words (user_id, word) VALUES (?, ?)", (user_id, request.json.get('word')))
        conn.commit()
    except:
        pass
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/get_vocab_details', methods=['POST'])
@login_required
def vocab_det():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    type_ = request.json.get('type')
    ignore_set = get_user_ignore_set(user_id)
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id, content FROM books WHERE user_id = ? ORDER BY id ASC", (user_id,))
    books = c.fetchall()
    conn.close()
    g_set = set()
    t_set = set()
    found = False
    for bid, ct in books:
        if str(bid) == str(book_id):
            t_set = extract_words(ct, user_id, ignore_set)
            found = True
            break
        else:
            g_set.update(extract_words(ct, user_id, ignore_set))
    res = sorted(list(t_set - g_set)) if type_ == 'new' and found else sorted(list(t_set.union(g_set))) if type_ == 'cumulative' else []
    return jsonify({'words': res, 'count': len(res)})

# ============================================
# 🏥 健康检查
# ============================================

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

@app.route('/_health')
def health_check():
    """Railway 健康检查端点"""
    return 'OK', 200

# ============================================
# 🚀 启动
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    print(f"📁 Data directory: {get_data_dir()}")
    app.run(host='0.0.0.0', port=port, debug=False)