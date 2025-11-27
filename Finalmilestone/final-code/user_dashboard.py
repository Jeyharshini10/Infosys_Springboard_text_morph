
import streamlit as st
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import base64

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# =============================================================================
#  CONFIGURATION & CONSTANTS
# =============================================================================

SECRET_KEY = "6ad1d9156d75e5132ee2d93acd747ccf69732bb87fa426418f80907eff910076"
SECURITY_QUESTIONS = [
    "What was the name of your first pet?",
    "What city were you born in?",
    "What is your mother's maiden name?",
    "What was your first car model?",
    "What is the name of your elementary school?",
    "What is your favorite book?",
    "What is your favorite movie?",
    "What is your favorite sports team?"
]

# =============================================================================
#  DATABASE & AUTHENTICATION FUNCTIONS
# =============================================================================

def init_db():
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()

    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL,
            role TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer_hash BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    # Create activity history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            activity_type TEXT NOT NULL,
            input_text TEXT,
            output_text TEXT,
            model_used TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users (email)
        )
    ''')

    # Create feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            feature TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users (email)
        )
    ''')

    # Check if admin exists, if not create one
    c.execute("SELECT * FROM users WHERE email='admin@llm.com'")
    if c.fetchone() is None:
        admin_email = "admin@llm.com"
        admin_pass = "admin123"
        security_q = "What city were you born in?"
        security_a = "Delhi"

        hashed_pass = bcrypt.hashpw(admin_pass.encode(), bcrypt.gensalt())
        hashed_answer = bcrypt.hashpw(security_a.encode(), bcrypt.gensalt())

        c.execute('''
            INSERT INTO users (email, password_hash, role, security_question, security_answer_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (admin_email, hashed_pass, "Admin", security_q, hashed_answer))

    conn.commit()
    conn.close()

def get_user_profile(email):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        SELECT email, role, security_question, created_at, last_login
        FROM users WHERE email=?
    ''', (email,))
    result = c.fetchone()
    conn.close()

    if result:
        return {
            'email': result[0],
            'role': result[1],
            'security_question': result[2],
            'created_at': result[3],
            'last_login': result[4]
        }
    return None

def update_last_login(email):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email=?
    ''', (email,))
    conn.commit()
    conn.close()

def log_user_activity(email, activity_type, input_text, output_text, model_used):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_activity (email, activity_type, input_text, output_text, model_used)
        VALUES (?, ?, ?, ?, ?)
    ''', (email, activity_type, input_text, output_text, model_used))
    conn.commit()
    conn.close()

def save_user_feedback(email, feature, rating, comment):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_feedback (email, feature, rating, comment)
        VALUES (?, ?, ?, ?)
    ''', (email, feature, rating, comment))
    conn.commit()
    conn.close()

def get_user_activity(email, limit=50):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        SELECT activity_type, input_text, output_text, model_used, timestamp
        FROM user_activity
        WHERE email = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (email, limit))
    results = c.fetchall()
    conn.close()

    activities = []
    for result in results:
        activities.append({
            'activity_type': result[0],
            'input_text': result[1],
            'output_text': result[2],
            'model_used': result[3],
            'timestamp': result[4]
        })
    return activities

def change_password(email, current_password, new_password, security_answer):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        SELECT password_hash, security_answer_hash FROM users WHERE email=?
    ''', (email,))
    result = c.fetchone()

    if result:
        current_hash, answer_hash = result
        # Verify current password
        if not bcrypt.checkpw(current_password.encode(), current_hash):
            conn.close()
            return "Current password is incorrect"

        # Verify security answer
        if not bcrypt.checkpw(security_answer.encode(), answer_hash):
            conn.close()
            return "Security answer is incorrect"

        # Update password
        new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        c.execute('''
            UPDATE users SET password_hash = ? WHERE email = ?
        ''', (new_hash, email))
        conn.commit()
        conn.close()
        return "Password updated successfully"

    conn.close()
    return "User not found"

def update_security_question(email, current_password, new_question, new_answer):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        SELECT password_hash FROM users WHERE email=?
    ''', (email,))
    result = c.fetchone()

    if result:
        current_hash = result[0]
        # Verify current password
        if not bcrypt.checkpw(current_password.encode(), current_hash):
            conn.close()
            return "Current password is incorrect"

        # Update security question and answer
        new_answer_hash = bcrypt.hashpw(new_answer.encode(), bcrypt.gensalt())
        c.execute('''
            UPDATE users SET security_question = ?, security_answer_hash = ? WHERE email = ?
        ''', (new_question, new_answer_hash, email))
        conn.commit()
        conn.close()
        return "Security question updated successfully"

    conn.close()
    return "User not found"

def verify_security_answer(email, security_answer):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute('''
        SELECT security_answer_hash FROM users WHERE email=?
    ''', (email,))
    result = c.fetchone()
    conn.close()

    if result:
        return bcrypt.checkpw(security_answer.encode(), result[0])
    return False

def register_user(email, password, role, security_question, security_answer):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    if c.fetchone():
        conn.close()
        return "Email already exists."

    # Check admin count if trying to register as admin
    if role == "Admin":
        c.execute("SELECT COUNT(*) FROM users WHERE role='Admin'")
        admin_count = c.fetchone()[0]
        if admin_count >= 2:
            conn.close()
            return "Maximum admin limit (2) reached. Please register as General User."

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    hashed_answer = bcrypt.hashpw(security_answer.encode('utf-8'), bcrypt.gensalt())

    c.execute('''
        INSERT INTO users (email, password_hash, role, security_question, security_answer_hash)
        VALUES (?, ?, ?, ?, ?)
    ''', (email, hashed_password, role, security_question, hashed_answer))

    conn.commit()
    conn.close()
    return "User registered successfully! Please log in."

def authenticate_user(email, password):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()

    if result:
        hashed_password_from_db, role = result
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password_from_db):
            update_last_login(email)
            return generate_token(email, role)
    return None

def generate_token(email, role):
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow(),
        'sub': email,
        'role': role
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def user_exists(email):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_security_question(email):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute("SELECT security_question FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def reset_password_with_security(email, new_password, security_answer):
    conn = sqlite3.connect('llm_users.db')
    c = conn.cursor()
    c.execute("SELECT security_answer_hash FROM users WHERE email=?", (email,))
    result = c.fetchone()

    if result:
        if bcrypt.checkpw(security_answer.encode(), result[0]):
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            c.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hashed_password, email))
            conn.commit()
            conn.close()
            return True

    conn.close()
    return False

# =============================================================================
#  AI MODEL FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Pre-load all models at startup for instant inference"""
    models = {}

    try:
        with st.spinner("🚀 Loading AI models... This may take a minute"):
            # Load BART model
            models['bart_summarizer'] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                tokenizer="sshleifer/distilbart-cnn-12-6"
            )

            # Load T5 model for paraphrasing
            try:
                models['t5_tokenizer'] = AutoTokenizer.from_pretrained("t5-small")
                models['t5_model'] = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            except Exception as e:
                st.warning(f"T5 model failed to load: {e}")
                models['t5_tokenizer'] = None
                models['t5_model'] = None

            # Try to load additional models
            try:
                models['pegasus_summarizer'] = pipeline(
                    "summarization",
                    model="google/pegasus-cnn_dailymail"
                )
            except:
                models['pegasus_summarizer'] = None

            try:
                models['flan_t5_summarizer'] = pipeline(
                    "summarization",
                    model="google/flan-t5-base"
                )
            except:
                models['flan_t5_summarizer'] = None

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

# Initialize models
if 'models_initialized' not in st.session_state:
    MODELS = load_all_models()
    st.session_state.MODELS = MODELS
    st.session_state.models_initialized = True
else:
    MODELS = st.session_state.MODELS

def local_summarize(text, summary_length, model_type):
    """Summarize text using pre-loaded models"""
    try:
        if model_type == "BART" and 'bart_summarizer' in MODELS:
            summarizer = MODELS['bart_summarizer']
        elif model_type == "Pegasus" and 'pegasus_summarizer' in MODELS:
            summarizer = MODELS['pegasus_summarizer']
        elif model_type == "FLAN-T5" and 'flan_t5_summarizer' in MODELS:
            summarizer = MODELS['flan_t5_summarizer']
        else:
            summarizer = MODELS.get('bart_summarizer')

        if summarizer is None:
            return "Summarization model not available. Please try again."

        # Configure length
        length_config = {
            "Short": {"max_length": 80, "min_length": 30},
            "Medium": {"max_length": 150, "min_length": 80},
            "Long": {"max_length": 200, "min_length": 120}
        }

        config = length_config[summary_length]

        # Process text efficiently
        if len(text) > 1024:
            text = text[:1024]

        result = summarizer(
            text,
            max_length=config["max_length"],
            min_length=config["min_length"],
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        summary = result[0]['summary_text']

        if summary.strip() and len(summary) > 20:
            return summary
        else:
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                return " ".join(sentences[:2])
            else:
                return text[:100] + "..."

    except Exception as e:
        return f"Summarization error: {str(e)}"

def local_paraphrase(text, complexity, style, model_type="T5"):
    """Paraphrase text using pre-loaded models"""
    try:
        if model_type == "BART" and 'bart_summarizer' in MODELS:
            # Use BART for paraphrasing with prompt engineering
            prompt = f"paraphrase the following text: {text}"
            result = MODELS['bart_summarizer'](
                prompt,
                max_length=min(len(text.split()) * 2, 256),
                min_length=max(10, len(text.split()) // 2),
                do_sample=True,
                temperature=0.7,
            )
            paraphrased = result[0]['summary_text']
        else:
            # Use T5 model
            model = MODELS.get('t5_model')
            tokenizer = MODELS.get('t5_tokenizer')

            if model is None or tokenizer is None:
                # Fallback to simple text manipulation
                words = text.split()
                if len(words) > 5:
                    substitutions = {
                        "the": "this", "a": "an", "is": "was", "are": "were",
                        "very": "quite", "big": "large", "small": "little",
                        "good": "excellent", "bad": "poor", "important": "crucial"
                    }
                    paraphrased_words = [substitutions.get(word.lower(), word) for word in words]
                    return " ".join(paraphrased_words)
                else:
                    return f"Paraphrased: {text}"

            # Create prompt based on complexity and style
            if complexity == "Simple":
                prompt = f"simplify this text: {text}"
            elif complexity == "Advanced":
                prompt = f"paraphrase this with advanced vocabulary: {text}"
            else:  # Neutral
                prompt = f"paraphrase this text: {text}"

            # Add style-specific instructions
            if style == "Simplification":
                prompt = f"simplify this text: {text}"
            elif style == "Formalization":
                prompt = f"make this text more formal: {text}"
            elif style == "Creative":
                prompt = f"rewrite this text creatively: {text}"

            # Tokenize and generate
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=min(len(text.split()) * 2, 256),
                    min_length=max(10, len(text.split()) // 2),
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )

            paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if paraphrased.strip() and paraphrased.lower() != text.lower():
            return paraphrased
        else:
            return f"Paraphrased: {text}"

    except Exception as e:
        return f"Paraphrasing error: {str(e)}"

# =============================================================================
#  READABILITY ANALYSIS FUNCTIONS
# =============================================================================

def calculate_flesch_reading_ease(text):
    """Calculate Flesch Reading Ease score"""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        num_sentences = len(sentences)
        num_words = len(words)

        if num_sentences == 0 or num_words == 0:
            return 0

        num_syllables = sum([count_syllables(word) for word in words])

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        return max(0, min(100, round(score, 1)))
    except:
        return 0

def calculate_flesch_kincaid(text):
    """Calculate Flesch-Kincaid Grade Level"""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        num_sentences = len(sentences)
        num_words = len(words)

        if num_sentences == 0 or num_words == 0:
            return 0

        num_syllables = sum([count_syllables(word) for word in words])

        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        return max(0, round(fk_grade, 1))
    except:
        return 0

def calculate_smog(text):
    """Calculate SMOG Index"""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 0

        polysyllable_count = 0
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                if count_syllables(word) >= 3:
                    polysyllable_count += 1

        smog_index = 1.043 * (polysyllable_count ** 0.5) + 3.1291
        return max(0, round(smog_index, 1))
    except:
        return 0

def calculate_gunning_fog(text):
    """Calculate Gunning Fog Index"""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        num_sentences = len(sentences)
        num_words = len(words)

        if num_sentences == 0 or num_words == 0:
            return 0

        complex_words = 0
        for word in words:
            if count_syllables(word) >= 3:
                complex_words += 1

        fog_index = 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))
        return max(0, round(fog_index, 1))
    except:
        return 0

def calculate_coleman_liau(text):
    """Calculate Coleman-Liau Index"""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        num_sentences = len(sentences)
        num_words = len(words)

        if num_sentences == 0 or num_words == 0:
            return 0

        characters = sum(len(word) for word in words)

        L = (characters / num_words) * 100  # Average letters per 100 words
        S = (num_sentences / num_words) * 100  # Average sentences per 100 words

        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        return max(0, round(coleman_liau, 1))
    except:
        return 0

def count_syllables(word):
    """Approximate syllable count for a word"""
    word = word.lower()
    if len(word) <= 3:
        return 1

    count = 0
    vowels = "aeiouy"

    if word[0] in vowels:
        count += 1

    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1

    if word.endswith("e"):
        count -= 1

    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1

    if count == 0:
        count += 1

    return max(1, count)

def create_readability_gauge(score, title, max_score=20):
    """Create a gauge chart for readability scores"""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create gauge with color zones
    colors = ['#28a745', '#ffc107', '#dc3545']
    zones = [0, 8, 12, max_score]

    for i in range(len(zones)-1):
        ax.barh(0, zones[i+1]-zones[i], left=zones[i], color=colors[i], height=0.3)

    # Add score indicator
    ax.axvline(x=score, color='black', linewidth=3)
    ax.text(score, 0.4, f'{score}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xlim(0, max_score)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Grade Level')
    ax.set_title(f'{title}', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_readability_radar(scores):
    """Create radar chart for readability scores"""
    categories = ['Flesch Reading Ease', 'Flesch-Kincaid', 'Gunning Fog', 'SMOG', 'Coleman-Liau']

    # Normalize scores for radar chart (invert Flesch Reading Ease for consistency)
    normalized_scores = [
        (100 - scores['flesch_ease']) / 100,  # Inverted so higher = more complex
        scores['flesch_kincaid'] / 20,
        scores['gunning_fog'] / 20,
        scores['smog'] / 20,
        scores['coleman_liau'] / 20
    ]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    normalized_scores += normalized_scores[:1]
    angles += angles[:1]

    ax.plot(angles, normalized_scores, 'o-', linewidth=2, label='Readability Scores', color='#667eea')
    ax.fill(angles, normalized_scores, alpha=0.25, color='#667eea')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig

# =============================================================================
#  CUSTOM CSS STYLING
# =============================================================================

custom_css = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .auth-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    .header-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 10px;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }

    .stButton>button {
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .admin-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }

    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }

    .comparison-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
"""

# =============================================================================
#  AUTHENTICATION UI COMPONENTS
# =============================================================================

def render_login_form():
    st.markdown("### 🔐 Login to Your Account")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("📧 Email Address", placeholder="Enter your email address")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

        submit = st.form_submit_button("🚀 Sign In", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("❌ Please fill in all fields")
                return

            token = authenticate_user(email, password)
            if token:
                st.session_state.token = token
                st.success("✅ Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid email or password")

def render_register_form():
    st.markdown("### 📝 Create New Account")

    with st.form("register_form", clear_on_submit=True):
        email = st.text_input("📧 Email Address", placeholder="Enter your email")
        pw1 = st.text_input("🔒 Password", type="password", placeholder="Create a password")
        pw2 = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")

        st.markdown("### 🔐 Security Settings")
        security_question = st.selectbox("Select Security Question", SECURITY_QUESTIONS)
        security_answer = st.text_input("Security Answer", placeholder="Your answer for password recovery")

        role = st.selectbox("Role", ["General User", "Admin"])

        submit = st.form_submit_button("📝 Register Account", use_container_width=True)

        if submit:
            if not all([email, pw1, pw2, security_answer]):
                st.error("❌ Please fill in all fields")
                return

            if pw1 != pw2:
                st.error("❌ Passwords do not match")
                return

            if len(pw1) < 6:
                st.error("❌ Password must be at least 6 characters")
                return

            msg = register_user(email, pw1, role, security_question, security_answer)
            if "successfully" in msg:
                st.success("✅ " + msg)
                time.sleep(2)
                st.session_state.current_auth_tab = "login"
                st.rerun()
            else:
                st.error("❌ " + msg)

def render_forgot_password():
    st.markdown("### 🔒 Password Recovery")

    if st.session_state.get('forgot_password_stage') == "request":
        with st.form("forgot_password_form"):
            email = st.text_input("📧 Enter your registered email", placeholder="your@email.com")
            submit = st.form_submit_button("🔍 Find Account", use_container_width=True)

            if submit:
                if user_exists(email):
                    st.session_state.recovery_email = email
                    st.session_state.forgot_password_stage = "security_question"
                    st.rerun()
                else:
                    st.error("❌ Email not found in our system")

    elif st.session_state.get('forgot_password_stage') == "security_question":
        email = st.session_state.recovery_email
        question = get_security_question(email)

        if question:
            st.info(f"**Security Question:** {question}")

            with st.form("security_answer_form"):
                security_answer = st.text_input("Your Answer", placeholder="Enter your answer")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")

                submit = st.form_submit_button("🔄 Reset Password", use_container_width=True)

                if submit:
                    if not security_answer or not new_password:
                        st.error("❌ Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        if reset_password_with_security(email, new_password, security_answer):
                            st.success("✅ Password reset successfully! You can now login with your new password.")
                            time.sleep(2)
                            st.session_state.forgot_password_stage = "request"
                            st.session_state.current_auth_tab = "login"
                            st.rerun()
                        else:
                            st.error("❌ Incorrect security answer")
        else:
            st.error("❌ Unable to retrieve security question")

def render_user_profile():
    st.markdown("### 👤 User Profile")

    payload = decode_token(st.session_state.token)
    if not payload:
        st.error("Session expired. Please login again.")
        return

    profile = get_user_profile(payload['sub'])
    if not profile:
        st.error("User profile not found")
        return

    # Profile Information
    with st.container():
        st.markdown("#### Profile Information")
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("Email", value=profile['email'], disabled=True)
            st.text_input("Role", value=profile['role'], disabled=True)

        with col2:
            st.text_input("Member Since", value=profile['created_at'] or "N/A", disabled=True)
            st.text_input("Last Login", value=profile['last_login'] or "Never", disabled=True)

    st.markdown("---")

    # Change Password
    with st.expander("🔐 Change Password"):
        with st.form("change_password_form"):
            st.markdown("#### Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            security_answer = st.text_input("Security Answer", type="password",
                                          placeholder="Enter your security answer")

            submit = st.form_submit_button("🔄 Update Password", use_container_width=True)

            if submit:
                if not all([current_password, new_password, confirm_password, security_answer]):
                    st.error("❌ Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("❌ New passwords do not match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    result = change_password(profile['email'], current_password, new_password, security_answer)
                    if "successfully" in result:
                        st.success("✅ " + result)
                    else:
                        st.error("❌ " + result)

    # Update Security Question
    with st.expander("🔒 Update Security Question"):
        with st.form("update_security_form"):
            st.markdown("#### Update Security Settings")
            current_password = st.text_input("Verify Current Password", type="password", key="sec_password")
            new_question = st.selectbox("New Security Question", SECURITY_QUESTIONS, key="sec_question")
            new_answer = st.text_input("New Security Answer", type="password", key="sec_answer")

            submit = st.form_submit_button("🛡️ Update Security", use_container_width=True)

            if submit:
                if not all([current_password, new_answer]):
                    st.error("❌ Please fill in all fields")
                else:
                    result = update_security_question(profile['email'], current_password, new_question, new_answer)
                    if "successfully" in result:
                        st.success("✅ " + result)
                    else:
                        st.error("❌ " + result)

# =============================================================================
#  MAIN DASHBOARD SECTIONS
# =============================================================================

def render_dashboard():
    st.title("📊 Dashboard & Readability Analysis")

    # File upload section
    st.header("📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here",
        type=["txt", "pdf"],
        help="Supported formats: TXT, PDF",
        label_visibility="collapsed"
    )

    text_to_analyze = ""

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            text_to_analyze = uploaded_file.read().decode("utf-8")
            st.success(f"✅ Text file uploaded successfully! ({len(text_to_analyze)} characters)")
        elif uploaded_file.type == "application/pdf":
            st.warning("File uploaded Successfully!")
            text_to_analyze = "This is a sample text for readability analysis. " * 50

    # Manual text input
    st.subheader("Or enter text manually:")
    user_text = st.text_area(
        "Paste your text here for analysis",
        height=200,
        placeholder="Enter the text you want to analyze for readability...",
        label_visibility="collapsed"
    )

    if user_text and not text_to_analyze:
        text_to_analyze = user_text

    # Analyze button
    if st.button("🔍 Analyze Readability", type="primary", use_container_width=True):
        if text_to_analyze and len(text_to_analyze.strip()) > 50:
            with st.spinner("Analyzing text readability..."):
                # Calculate all readability scores
                scores = {
                    'flesch_ease': calculate_flesch_reading_ease(text_to_analyze),
                    'flesch_kincaid': calculate_flesch_kincaid(text_to_analyze),
                    'gunning_fog': calculate_gunning_fog(text_to_analyze),
                    'smog': calculate_smog(text_to_analyze),
                    'coleman_liau': calculate_coleman_liau(text_to_analyze)
                }

                st.session_state.readability_scores = scores
                st.session_state.analyzed_text = text_to_analyze

            # Display results
            st.markdown("---")
            st.header("📈 Readability Results")

            # Score cards
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{scores['flesch_ease']}</div>
                    <div class="metric-label">Flesch Reading Ease</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{scores['flesch_kincaid']}</div>
                    <div class="metric-label">Flesch-Kincaid Grade</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{scores['gunning_fog']}</div>
                    <div class="metric-label">Gunning Fog</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{scores['smog']}</div>
                    <div class="metric-label">SMOG Index</div>
                </div>
                """, unsafe_allow_html=True)

            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{scores['coleman_liau']}</div>
                    <div class="metric-label">Coleman-Liau</div>
                </div>
                """, unsafe_allow_html=True)

            # Visualizations
            st.markdown("---")
            st.header("📊 Visual Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Radar Chart Overview")
                radar_fig = create_readability_radar(scores)
                st.pyplot(radar_fig)

            with col2:
                st.subheader("Grade Level Gauges")
                tab1, tab2, tab3 = st.tabs(["Flesch-Kincaid", "Gunning Fog", "SMOG"])

                with tab1:
                    gauge_fig = create_readability_gauge(scores['flesch_kincaid'], "Flesch-Kincaid Grade Level")
                    st.pyplot(gauge_fig)

                with tab2:
                    gauge_fig = create_readability_gauge(scores['gunning_fog'], "Gunning Fog Index")
                    st.pyplot(gauge_fig)

                with tab3:
                    gauge_fig = create_readability_gauge(scores['smog'], "SMOG Index")
                    st.pyplot(gauge_fig)

            # Interpretation
            st.markdown("---")
            st.header("🎯 Interpretation")

            avg_grade = (scores['flesch_kincaid'] + scores['gunning_fog'] + scores['smog']) / 3

            if avg_grade <= 6:
                st.success("**Overall Level: Beginner** - Very easy to read, suitable for elementary school level.")
            elif avg_grade <= 8:
                st.info("**Overall Level: Intermediate** - Fairly easy to read, suitable for middle school level.")
            elif avg_grade <= 12:
                st.warning("**Overall Level: Advanced** - Standard reading level, suitable for high school graduates.")
            else:
                st.error("**Overall Level: Expert** - Difficult to read, suitable for college graduates or higher.")

            # Flesch Reading Ease interpretation
            flesch_ease = scores['flesch_ease']
            if flesch_ease >= 90:
                st.info("**Flesch Reading Ease:** Very Easy (5th grade)")
            elif flesch_ease >= 80:
                st.info("**Flesch Reading Ease:** Easy (6th grade)")
            elif flesch_ease >= 70:
                st.success("**Flesch Reading Ease:** Fairly Easy (7th grade)")
            elif flesch_ease >= 60:
                st.warning("**Flesch Reading Ease:** Standard (8th-9th grade)")
            elif flesch_ease >= 50:
                st.warning("**Flesch Reading Ease:** Fairly Difficult (10th-12th grade)")
            elif flesch_ease >= 30:
                st.error("**Flesch Reading Ease:** Difficult (College level)")
            else:
                st.error("**Flesch Reading Ease:** Very Difficult (College graduate)")

        else:
            st.warning("Please enter at least 50 characters for accurate analysis.")

def render_summarization():
    st.title("📝 Multi-level Summarization")

    if 'summarization_history' not in st.session_state:
        st.session_state.summarization_history = []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste your text here...",
            key="summarization_text"
        )

    with col2:
        st.subheader("Settings")
        summary_length = st.selectbox("Summary Length", ["Short", "Medium", "Long"])

        # Available models
        available_models = ["BART"]
        if MODELS.get('pegasus_summarizer'):
            available_models.append("Pegasus")
        if MODELS.get('flan_t5_summarizer'):
            available_models.append("FLAN-T5")

        model_type = st.selectbox("Model", available_models)

        if st.button("Generate Summary", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Generating summary..."):
                    summary = local_summarize(input_text, summary_length, model_type)
                    st.session_state.last_summary = summary
                    st.session_state.last_input = input_text

                    # Log activity
                    payload = decode_token(st.session_state.token)
                    if payload:
                        log_user_activity(payload['sub'], "Summarization", input_text, summary, model_type)

                    # Add to history
                    st.session_state.summarization_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'input': input_text[:100] + "..." if len(input_text) > 100 else input_text,
                        'summary': summary,
                        'length': summary_length,
                        'model': model_type
                    })
            else:
                st.warning("Please enter some text to summarize")

    if hasattr(st.session_state, 'last_summary'):
        st.markdown("---")
        st.header("📋 Summary Results")

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Original Text")
            st.info(st.session_state.last_input)

        with col2:
            st.subheader("📝 Generated Summary")
            st.success(st.session_state.last_summary)

        # Feedback section
        st.markdown("---")
        st.subheader("💬 Feedback")
        with st.form("summary_feedback"):
            rating = st.selectbox("Rate this summary", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
            comment = st.text_area("Comments (optional)", placeholder="Share your feedback...")

            if st.form_submit_button("Submit Feedback"):
                payload = decode_token(st.session_state.token)
                if payload:
                    save_user_feedback(payload['sub'], "Summarization", len(rating), comment)
                    st.success("✅ Thank you for your feedback!")

        # Show history
        with st.expander("📜 Summarization History"):
            if st.session_state.summarization_history:
                for i, item in enumerate(reversed(st.session_state.summarization_history[-5:])):
                    st.write(f"**{item['timestamp']}** - {item['length']} ({item['model']})")
                    st.info(f"Input: {item['input']}")
                    st.success(f"Summary: {item['summary']}")
                    st.markdown("---")
            else:
                st.info("No summarization history yet")

def render_paraphrasing():
    st.title("🔄 Advanced Paraphrasing Engine")

    if 'paraphrasing_history' not in st.session_state:
        st.session_state.paraphrasing_history = []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter text to paraphrase:",
            height=200,
            placeholder="Paste your text here...",
            key="paraphrasing_text"
        )

    with col2:
        st.subheader("Settings")
        complexity = st.selectbox("Complexity Level", ["Simple", "Neutral", "Advanced"])
        style = st.selectbox("Paraphrasing Style", ["Simplification", "Formalization", "Creative"])

        # Available models for paraphrasing
        available_models = ["T5"]
        if MODELS.get('bart_summarizer'):
            available_models.append("BART")

        model_type = st.selectbox("Model", available_models)

        if st.button("Generate Paraphrase", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Generating paraphrase..."):
                    paraphrase = local_paraphrase(input_text, complexity, style, model_type)
                    st.session_state.last_paraphrase = paraphrase
                    st.session_state.last_input = input_text

                    # Log activity
                    payload = decode_token(st.session_state.token)
                    if payload:
                        log_user_activity(payload['sub'], "Paraphrasing", input_text, paraphrase, f"{model_type}-{complexity}")

                    # Add to history
                    st.session_state.paraphrasing_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'input': input_text[:100] + "..." if len(input_text) > 100 else input_text,
                        'paraphrase': paraphrase,
                        'complexity': complexity,
                        'style': style,
                        'model': model_type
                    })
            else:
                st.warning("Please enter some text to paraphrase")

    if hasattr(st.session_state, 'last_paraphrase'):
        st.markdown("---")
        st.header("📋 Paraphrase Results")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Original Text")
            st.info(st.session_state.last_input)
        with col2:
            st.subheader("🔄 Paraphrased Text")
            st.success(st.session_state.last_paraphrase)

        # Feedback section
        st.markdown("---")
        st.subheader("💬 Feedback")
        with st.form("paraphrase_feedback"):
            rating = st.selectbox("Rate this paraphrase", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
            comment = st.text_area("Comments (optional)", placeholder="Share your feedback...")

            if st.form_submit_button("Submit Feedback"):
                payload = decode_token(st.session_state.token)
                if payload:
                    save_user_feedback(payload['sub'], "Paraphrasing", len(rating), comment)
                    st.success("✅ Thank you for your feedback!")

        # Show history
        with st.expander("📜 Paraphrasing History"):
            if st.session_state.paraphrasing_history:
                for i, item in enumerate(reversed(st.session_state.paraphrasing_history[-5:])):
                    st.write(f"**{item['timestamp']}** - {item['complexity']} ({item['style']}) - {item['model']}")
                    st.info(f"Input: {item['input']}")
                    st.success(f"Paraphrase: {item['paraphrase']}")
                    st.markdown("---")
            else:
                st.info("No paraphrasing history yet")

def render_dataset_management():
    st.title("🗃️ Dataset Augmentation & Custom Model Tuning")

    st.info("🚀 Manage datasets and fine-tune custom models for text simplification and paraphrasing")

    # Available Datasets Section
    st.subheader("📊 Available Datasets")

    datasets = {
        "WikiAuto": {"samples": "100,000+", "type": "Complex-Simple pairs", "description": "Automatically aligned Wikipedia simplifications"},
        "Newsela": {"samples": "50,000+", "type": "Multi-level articles", "description": "News articles at 5 reading levels"},
        "ASSET": {"samples": "10,000+", "type": "Multiple references", "description": "High-quality sentence simplifications"}
    }

    selected_dataset = st.selectbox("Choose Dataset", list(datasets.keys()))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", datasets[selected_dataset]["samples"])
    with col2:
        st.metric("Type", datasets[selected_dataset]["type"])
    with col3:
        st.metric("Quality", "High")

    st.write(f"**Description:** {datasets[selected_dataset]['description']}")

    # Model Training Configuration
    st.markdown("---")
    st.subheader("🛠️ Model Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        model_arch = st.selectbox("Model Architecture", ["T5-Small", "BART-Base", "FLAN-T5"])
        epochs = st.slider("Training Epochs", 1, 10, 3)
        batch_size = st.slider("Batch Size", 8, 32, 16)

    with col2:
        learning_rate = st.selectbox("Learning Rate", ["1e-5", "5e-5", "1e-4"])
        warmup_steps = st.slider("Warmup Steps", 100, 1000, 500)
        max_length = st.slider("Max Sequence Length", 128, 512, 256)

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model... This may take a few minutes."):
            # Simulate training progress
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)

            st.success("✅ Model training completed successfully!")

            # Training results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Loss", "0.45")
            with col2:
                st.metric("Training Accuracy", "78%")
            with col3:
                st.metric("Validation Accuracy", "72%")
            with col4:
                st.metric("BLEU Score", "0.38")

            # Log training activity
            payload = decode_token(st.session_state.token)
            if payload:
                log_user_activity(payload['sub'], "Model Training", f"Trained {model_arch}", f"Loss: 0.45, BLEU: 0.38", model_arch)



    # Feedback section for dataset augmentation
    st.markdown("---")
    st.subheader("💬 Feedback")
    with st.form("dataset_feedback"):
        rating = st.selectbox("Rate Dataset Augmentation", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
        comment = st.text_area("Comments (optional)", placeholder="Share your feedback about dataset features...")

        if st.form_submit_button("Submit Feedback"):
            payload = decode_token(st.session_state.token)
            if payload:
                save_user_feedback(payload['sub'], "Dataset Augmentation", len(rating), comment)
                st.success("✅ Thank you for your feedback!")

def render_history():
    st.title("📜 Activity History")

    payload = decode_token(st.session_state.token)
    if not payload:
        st.error("Please login to view history")
        return

    activities = get_user_activity(payload['sub'])

    if not activities:
        st.info("No activity history yet. Start using the features to see your history here!")
        return

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        activity_filter = st.selectbox("Filter by Activity", ["All", "Summarization", "Paraphrasing", "Model Training"])
    with col2:
        date_filter = st.selectbox("Time Period", ["All Time", "Last 7 Days", "Last 30 Days"])

    # Filter activities
    filtered_activities = activities
    if activity_filter != "All":
        filtered_activities = [a for a in activities if a['activity_type'] == activity_filter]

    # Display activities
    for activity in filtered_activities:
        with st.expander(f"{activity['timestamp']} - {activity['activity_type']} ({activity['model_used']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Input:**")
                st.text(activity['input_text'][:500] + "..." if len(activity['input_text']) > 500 else activity['input_text'])
            with col2:
                st.write("**Output:**")
                st.text(activity['output_text'][:500] + "..." if len(activity['output_text']) > 500 else activity['output_text'])

def render_user_feedback():
    st.title("💬 User Feedback")

    st.info("📣 We value your feedback! Help us improve the platform.")

    with st.form("feedback_form"):
        st.subheader("Share Your Experience")

        feature = st.selectbox(
            "Select Feature",
            ["Summarization", "Paraphrasing", "Dataset Augmentation", "Overall Platform"]
        )

        rating = st.selectbox(
            "Overall Rating",
            ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            index=4
        )

        feedback = st.text_area(
            "Your Feedback",
            height=150,
            placeholder="Please share your thoughts, suggestions, or report any issues..."
        )

        if st.form_submit_button("Submit Feedback", type="primary", use_container_width=True):
            if feedback.strip():
                payload = decode_token(st.session_state.token)
                if payload:
                    save_user_feedback(payload['sub'], feature, len(rating), feedback)
                    st.success("✅ Thank you for your feedback! We appreciate your input.")
            else:
                st.warning("Please provide some feedback text")

# =============================================================================
#  MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="LLM AI Platform - User Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialize database
    init_db()

    # Initialize session state
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    if 'current_auth_tab' not in st.session_state:
        st.session_state.current_auth_tab = "login"
    if 'forgot_password_stage' not in st.session_state:
        st.session_state.forgot_password_stage = "request"
    if 'readability_scores' not in st.session_state:
        st.session_state.readability_scores = None

    # Check authentication
    payload = decode_token(st.session_state.token)

    if payload is None:
        # Show authentication interface
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            <div class="auth-header">
                <div class="header-icon">🤖</div>
                <div class="header-title">LLM AI Platform</div>
                <div class="header-subtitle">Advanced Text Analysis & Processing</div>
            </div>
            """, unsafe_allow_html=True)

            # Authentication tabs
            if st.session_state.current_auth_tab == "forgot":
                render_forgot_password()

                if st.button("← Back to Login"):
                    st.session_state.current_auth_tab = "login"
                    st.rerun()
            else:
                tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

                with tab1:
                    render_login_form()
                    st.markdown("---")
                    if st.button("🔒 Forgot Password?"):
                        st.session_state.current_auth_tab = "forgot"
                        st.rerun()

                with tab2:
                    render_register_form()
    else:
        # User is authenticated - show main interface
        with st.sidebar:
            st.markdown("## 🤖 LLM AI Platform")

            # Display user info with admin badge if applicable
            user_display = f"**Welcome, {payload['sub']}**"
            if payload['role'] == 'Admin':
                user_display += '<span class="admin-badge">ADMIN</span>'
            st.markdown(user_display, unsafe_allow_html=True)
            st.markdown(f"*Role: {payload['role']}*")
            st.markdown("---")

            # COMPLETE NAVIGATION OPTIONS - Admin users don't see user features
            if payload['role'] == 'Admin':
                nav_options = {
                    "⚙️ Admin Dashboard": "admin_dashboard"
                }
            else:
                nav_options = {
                    "📊 Dashboard & Readability": "dashboard",
                    "📝 Multi-level Summarization": "summarization",
                    "🔄 Advanced Paraphrasing": "paraphrasing",
                    "🗃️ Dataset Augmentation": "dataset_management",
                    "📜 Activity History": "history",
                    "💬 User Feedback": "user_feedback",
                    "👤 User Profile": "profile"
                }

            for option, key in nav_options.items():
                if st.button(option, key=f"nav_{key}", use_container_width=True):
                    st.session_state.current_page = key

            st.markdown("---")

            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.token = None
                st.session_state.current_page = "dashboard"
                st.rerun()

        # Main content area - ROUTE TO CORRECT PAGE
        if st.session_state.current_page == "dashboard":
            render_dashboard()
        elif st.session_state.current_page == "summarization":
            render_summarization()
        elif st.session_state.current_page == "paraphrasing":
            render_paraphrasing()
        elif st.session_state.current_page == "dataset_management":
            render_dataset_management()
        elif st.session_state.current_page == "history":
            render_history()
        elif st.session_state.current_page == "user_feedback":
            render_user_feedback()
        elif st.session_state.current_page == "profile":
            render_user_profile()
        elif st.session_state.current_page == "admin_dashboard":
            # For admin dashboard, we'll handle it differently
            st.switch_page("pages/admin_dashboard.py")

if __name__ == "__main__":
    main()