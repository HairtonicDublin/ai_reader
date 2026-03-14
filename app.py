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
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
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

# 配置
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
logging.info(f"OPENAI_API_KEY 状态: {'已设置 (' + OPENAI_API_KEY[:8] + '...)' if OPENAI_API_KEY else '未设置'}")
if not OPENAI_API_KEY:
    logging.warning("⚠️ OPENAI_API_KEY 未设置，AI 翻译功能将不可用。请设置环境变量 OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
# 支持的模型: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, code-davinci-002（Codex）
AI_MODEL = os.environ.get('AI_MODEL', 'gpt-3.5-turbo')
logging.info(f"使用模型: {AI_MODEL}")

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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )''')
    
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, title)
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
    
    # 创建索引
    try:
        c.execute("CREATE INDEX IF NOT EXISTS idx_books_user ON books(user_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_notebook_user ON notebook(user_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_notebook_book ON notebook(book_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_notebook_created ON notebook(created_at);")
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

def call_openai(prompt, max_tokens=200):
    """调用 OpenAI API（支持 ChatGPT、Codex 等模型）"""
    if not OPENAI_API_KEY:
        return "请设置 OPENAI_API_KEY 环境变量以使用 AI 功能"
    if not HAS_REQUESTS:
        return "Error: requests library missing."
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 针对不同模型的不同请求格式
        if "code-davinci" in AI_MODEL or "codex" in AI_MODEL.lower():
            # Codex 模型使用 completions 端点（已弃用，这里用 chat completions 兼容）
            data = {
                "model": AI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3
            }
        else:
            # ChatGPT 模型
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

# 保持向后兼容性
def call_deepseek(prompt, max_tokens=200):
    """向后兼容：重定向到 call_openai"""
    return call_openai(prompt, max_tokens)

def clean_block_text(text):
    text = re.sub(r'(\w)-\n\s*(\w)', r'\1\2', text)
    text = text.replace('\n', ' ').strip()
    return re.sub(r'\s+', ' ', text)

def extract_book_smart(path):
    try:
        import fitz
        doc = fitz.open(path)
        text = []
        for p in doc:
            blocks = p.get_text("blocks")
            for b in blocks:
                if b[6] == 1:
                    continue
                raw = clean_block_text(b[4])
                if len(raw) > 3:
                    text.append(raw)
        return "\n\n".join(text)
    except Exception as e:
        return f"文件读取失败: {str(e)}"

def register_book(name, user_id):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        c.execute("INSERT INTO books (user_id, title) VALUES (?, ?)", (user_id, name))
        bid = c.lastrowid
    except:
        c.execute("SELECT id FROM books WHERE user_id = ? AND title = ?", (user_id, name))
        bid = c.fetchone()[0]
    conn.commit()
    conn.close()
    return bid

# ============================================
# 🔐 登录/注册路由
# ============================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'status': 'error', 'message': '请输入用户名和密码'})
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    
    if user and verify_password(user[1], password):
        session['user_id'] = user[0]
        session['username'] = username
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
    return jsonify({
        'user_id': session.get('user_id'),
        'username': session.get('username')
    })

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
            # 用原始文件名作为显示标题，安全文件名用于存储
            display_name = f.filename
            bid = register_book(display_name, user_id)
            title = display_name

            ext = safe_name.lower()
            if ext.endswith(('.pdf', '.epub', '.mobi')):
                content = extract_book_smart(path)
            else:
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        content = fh.read()
                except Exception:
                    try:
                        with open(path, 'r', encoding='gb18030') as fh:
                            content = fh.read()
                    except Exception:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                            content = fh.read()

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
    
    if bid:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("SELECT new_vocab, cumulative_vocab, word_count FROM books WHERE id=? AND user_id=?", (bid, user_id))
        row = c.fetchone()
        c.execute("SELECT value FROM user_settings WHERE user_id=? AND key='vocab_size'", (user_id,))
        urow = c.fetchone()
        conn.close()
        if row:
            stats['new'], stats['cumulative'], stats['word_count'] = row
        if urow:
            stats['user_vocab'] = int(urow[0])
    
    return render_template('index.html',
                           content=content,
                           book_id=bid,
                           book_title=title,
                           stats=stats,
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