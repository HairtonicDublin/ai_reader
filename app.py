import os
import sys
import sqlite3
import threading
import re
import time
import json
import logging
import traceback
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

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

# 从环境变量获取 secret key，或生成随机的
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(traceback.format_exc())
    return jsonify({'error': str(e)}), 500

# 配置
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-cae2748e8598422babdd661c334a70f0')
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

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
    """密码加密"""
    return hashlib.sha256(password.encode()).hexdigest()

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
def is_valid_word(word, user_id):
    word = word.lower()
    ignore_set = get_user_ignore_set(user_id)
    if word in ignore_set:
        return False
    if len(word) < 2 and word not in ['a', 'i']:
        return False
    if not re.search(r'[aeiouy]', word):
        return False
    return True

def extract_words_list(text, user_id):
    if not text:
        return []
    return [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if is_valid_word(w, user_id)]

def extract_words(text, user_id):
    return set(extract_words_list(text, user_id))

def recalculate_chain(user_id):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id, content FROM books WHERE user_id = ? ORDER BY id ASC", (user_id,))
    books = c.fetchall()
    global_vocab = set()
    for bid, content in books:
        words = set(extract_words_list(content, user_id))
        new_count = len(words - global_vocab)
        global_vocab.update(words)
        c.execute("UPDATE books SET total_vocab=?, new_vocab=?, cumulative_vocab=?, word_count=? WHERE id=?",
                  (len(words), new_count, len(global_vocab), len(extract_words_list(content, user_id)), bid))
    conn.commit()
    conn.close()

def analyze_vocabulary_task(bid, text, user_id):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE books SET content = ? WHERE id = ?", (text, bid))
    conn.commit()
    conn.close()
    recalculate_chain(user_id)

def call_deepseek(prompt, max_tokens=200):
    if not HAS_REQUESTS:
        return "Error: requests library missing."
    try:
        resp = requests.post(
            DEEPSEEK_URL,
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": max_tokens
            },
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return f"API Error: {resp.status_code}"
    except Exception as e:
        return f"Net Error: {e}"

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
    
    if user and user[1] == hash_password(password):
        session['user_id'] = user[0]
        session['username'] = username
        session.permanent = True
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
            user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if not os.path.exists(user_upload_folder):
                os.makedirs(user_upload_folder)
            
            path = os.path.join(user_upload_folder, f.filename)
            f.save(path)
            bid = register_book(f.filename, user_id)
            title = f.filename
            
            ext = f.filename.lower()
            if ext.endswith(('.pdf', '.epub', '.mobi')):
                content = extract_book_smart(path)
            else:
                try:
                    content = open(path, 'r', encoding='utf-8').read()
                except:
                    try:
                        content = open(path, 'r', encoding='gb18030').read()
                    except:
                        content = open(path, 'r', encoding='utf-8', errors='ignore').read()

            threading.Thread(target=analyze_vocabulary_task, args=(bid, content, user_id), daemon=True).start()
            
    elif request.args.get('book_id'):
        bid = int(request.args.get('book_id'))
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
    
    # 先查找已有的解释
    if mode == 'word':
        if wid:
            c.execute("SELECT explanation FROM notebook WHERE user_id=? AND book_id=? AND word_index=?", (user_id, bid, wid))
        else:
            c.execute("SELECT explanation FROM notebook WHERE user_id=? AND text=? ORDER BY created_at DESC LIMIT 1", (user_id, text))
        row = c.fetchone()
        if row and row[0]:
            conn.close()
            return jsonify({'result': row[0]})

    res = ""
    if mode == 'word':
        cn = ""
        en = ""
        
        # 尝试翻译
        try:
            if HAS_TRANSLATOR:
                cn = GoogleTranslator(source='auto', target='zh-CN').translate(text)
                if not cn:
                    cn = text  # 如果翻译返回空，使用原词
            else:
                cn = text
        except Exception as e:
            logging.error(f"翻译失败: {e}")
            cn = text
        
        # 尝试获取英文定义
        try:
            syns = wordnet.synsets(text)
            if syns:
                en = syns[0].definition().capitalize() + "."
            else:
                en = "No definition available."
        except Exception as e:
            logging.error(f"词典查询失败: {e}")
            en = "Definition not found."
        
        # 包装英文定义中的单词
        def wrap(m):
            return f'<span class="word nested-word">{m.group(0)}</span>'
        if en:
            en = re.sub(r'\b[a-zA-Z]{3,}\b', wrap, en)
        
        res = f"<div class='cn-def' style='font-size:18px;font-weight:bold;color:#2c3e50'>{cn}</div><div class='en-def' style='font-size:14px;color:#555;margin-top:4px'>{en}</div>"
    else:
        res = call_deepseek(f"Translate and Analyze: {text}", 500)
        if not res:
            res = "无法获取解释"

    # 保存到生词本
    try:
        c.execute("INSERT INTO notebook (user_id, text, explanation, type, book_id, word_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (user_id, text, res, mode, bid, wid, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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
        conn.close()
        recalculate_chain(user_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reset_book_progress', methods=['POST'])
@login_required
def reset_book_progress():
    user_id = get_current_user_id()
    book_id = request.json.get('book_id')
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    try:
        c.execute("DELETE FROM notebook WHERE book_id = ? AND user_id = ?", (book_id, user_id))
        c.execute("UPDATE user_settings SET value = '0' WHERE user_id = ? AND key = 'vocab_size'", (user_id,))
        conn.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

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
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/get_book_highlights', methods=['POST'])
@login_required
def get_highlights():
    user_id = get_current_user_id()
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("SELECT word_index FROM notebook WHERE user_id = ? AND book_id = ? AND word_index IS NOT NULL",
                  (user_id, request.json.get('book_id')))
        indices = [row[0] for row in c.fetchall()]
        conn.close()
        return jsonify({'indices': indices})
    except:
        return jsonify({'indices': []})

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
            t_set = extract_words(ct, user_id)
            found = True
            break
        else:
            g_set.update(extract_words(ct, user_id))
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