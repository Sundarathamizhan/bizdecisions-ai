import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline as hf_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
import collections
import datetime
import random
import time
import io
import matplotlib
import bcrypt
from dotenv import load_dotenv
load_dotenv()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
import hashlib
import plotly.graph_objects as go
from groq import Groq
# reportlab — PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

try:
    import tweepy
    TWEEPY_OK = True
except ImportError:
    TWEEPY_OK = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

try:
    from google_play_scraper import reviews as gplay_reviews, Sort
    GPLAY_OK = True
except ImportError:
    GPLAY_OK = False

# Enforce consistent language detection
DetectorFactory.seed = 0

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="BizDecisions AI",
    page_icon="🏪",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS STYLE
# --------------------------------------------------
st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }
    
    /* Input Fields */
    .stTextArea>div>div>textarea, .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #38bdf8;
        color: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #38bdf8;
        font-size: 2.2rem !important;
        font-weight: 700;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 1.5 DATABASE & AUTHENTICATION
# --------------------------------------------------
def _bcrypt_hash(password: str) -> str:
    """Hash a password with bcrypt (salted). Returns the stored string."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _bcrypt_check(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except Exception:
        return False

def init_db():
    conn = sqlite3.connect("business_intelligence.db", check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS reviews_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  review TEXT,
                  aspect TEXT,
                  sentiment TEXT,
                  confidence REAL,
                  username TEXT,
                  source TEXT DEFAULT 'upload')''')

    # Ensure reviews_history has username/source columns (migration guard)
    try:
        c.execute("ALTER TABLE reviews_history ADD COLUMN username TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE reviews_history ADD COLUMN source TEXT DEFAULT 'upload'")
    except Exception:
        pass

    # Create default admin with bcrypt — re-create if still using old sha256 hash
    c.execute("SELECT password FROM users WHERE username='admin'")
    row = c.fetchone()
    if not row:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  ("admin", _bcrypt_hash("password")))
    elif not row[0].startswith("$2"):  # old sha256 hash — upgrade it
        c.execute("UPDATE users SET password=? WHERE username='admin'",
                  (_bcrypt_hash("password"),))

    conn.commit()
    return conn

conn = init_db()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>BizDecisions AI SaaS Platform</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username=?", (username,))
                row = c.fetchone()
                if row and _bcrypt_check(password, row[0]):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
                    
    with col2:
        st.info("Demo Login: \n\n**User:** `admin`\n\n**Pass:** `password`")
    st.stop()


# --------------------------------------------------
# 2. BUSINESS ADVICE DB
# --------------------------------------------------
ADVICE_DB = {
    "product": {
        "Negative": "Check product quality, inventory, and address common defect reports.",
        "Positive": "Promote top-performing products or services on social media."
    },
    "service": {
        "Negative": "Conduct staff training on politeness, responsiveness, and speed.",
        "Positive": "Reward staff for excellent customer service."
    },
    "price": {
        "Negative": "Review pricing strategy or offer discounts.",
        "Positive": "Pricing strategy is effective."
    },
    "ambience": {
        "Negative": "Improve layout, cleanliness, lighting, and noise control.",
        "Positive": "Ambience and environment are appreciated by customers."
    }
}

# --------------------------------------------------
# 3. LOAD MODEL
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model_path = os.getenv("MODEL_PATH", r"C:\new\my_bert_model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_zeroshot_model(model_name: str):
    """Load (and permanently cache) a zero-shot classification pipeline.
    Uses the top-level hf_pipeline alias so there is never a redundant import.
    Streamlit's @st.cache_resource ensures this runs exactly ONCE per model name.
    """
    return hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

# ── SESSION STATE MODEL GUARD ─────────────────────────────────
# Only calls load_model() ONCE. Subsequent tab switches skip this entirely.
if "_tokenizer" not in st.session_state:
    _tok, _mdl = load_model()
    st.session_state["_tokenizer"] = _tok
    st.session_state["_bert_model"] = _mdl

tokenizer = st.session_state["_tokenizer"]
model     = st.session_state["_bert_model"]

# --------------------------------------------------
# 4. SENTIMENT + CONFIDENCE ENGINE
# --------------------------------------------------
def analyze_aspect(text, aspect):
    text = str(text)
    if aspect.lower() == "food":
        aspect = "product"

    negative_keywords = {
        "product": [
            "bad", "cold", "stale", "broken", "defective", "poor", "cheap",
            "tasteless", "raw", "burnt", "damaged", "late", "delay",
            "faulty", "not working", "stopped", "malfunction", "issue"
        ],
        "service": ["rude", "slow", "ignored", "late", "waiting", "unhelpful"],
        "price": ["expensive", "costly", "high", "overpriced", "rip-off"],
        "ambience": ["dirty", "noisy", "smell", "loud", "crowded", "messy"]
    }

    neutral_words = ["okay", "average", "fine", "decent", "normal", "acceptable"]

    # -------- 1️⃣ MODEL PREDICTION FIRST --------
    inputs = tokenizer(
        text,
        text_pair=aspect,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    idx = int(torch.argmax(probs))
    confidence = float(probs[0][idx])

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = label_map[idx]
    
    # -------- EXPLAINABLE AI (WORD IMPORTANCE) --------
    # Extract attention weights from the last layer, average across heads
    attentions = outputs.attentions[-1]  # (batch, heads, seq, seq)
    avg_attention = torch.mean(attentions, dim=1)  # (batch, seq, seq)
    cls_attention = avg_attention[0, 0, :].cpu().numpy()  # CLS row

    # Full attention matrix for heatmap (seq×seq), trimmed to real tokens
    full_attn_matrix = avg_attention[0].cpu().numpy()

    tokens_raw = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    word_importances = []
    clean_tokens_list = []  # parallel list for heatmap axis labels
    token_indices = []      # indices in the full token list that survive filtering

    for idx_t, (token, weight) in enumerate(zip(tokens_raw, cls_attention)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        clean_token = token.replace("##", "")
        if clean_token.strip() == "":
            continue
        word_importances.append((clean_token, float(weight)))
        clean_tokens_list.append(clean_token)
        token_indices.append(idx_t)

    # Trim the attention matrix to only real-token rows/cols
    if token_indices:
        attn_trimmed = full_attn_matrix[np.ix_(token_indices, token_indices)]
    else:
        attn_trimmed = full_attn_matrix

    # -------- 2️⃣ PREDICTION CONFIDENCE OVERRIDE --------
    # Let the ML model's prediction speak for itself. Only flag as uncertain if confidence is low.
    if confidence < 0.5:
        sentiment = "Uncertain"

    action = ADVICE_DB.get(aspect, {}).get(sentiment, "No action needed")

    return sentiment, confidence, action, word_importances, attn_trimmed, clean_tokens_list


def batch_analyze_aspects(texts, aspects, batch_size=32):
    """
    Run BERT sentiment in bulk for a list of (text, aspect) pairs.
    Returns lists of (sentiment, confidence, action, _, _, _) matching analyze_aspect format.
    Does not compute XAI word_importances/heatmaps to save memory on large datasets.
    """
    if not texts:
        return []

    mapped_aspects = ["product" if a.lower() == "food" else a for a in aspects]
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_aspects = mapped_aspects[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            text_pair=batch_aspects,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = probs[torch.arange(len(preds)), preds].cpu().numpy()

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        for j in range(len(batch_texts)):
            sentiment = label_map[preds[j]]
            conf = float(confidences[j])
            
            # Confidence override
            if conf < 0.5:
                sentiment = "Uncertain"
                
            action = ADVICE_DB.get(batch_aspects[j], {}).get(sentiment, "No action needed")
            # Bulk inference skips XAI for speed
            results.append((sentiment, conf, action, [], None, []))

    return results


def evaluate_absa(test_df):
    reviews = test_df["review"].astype(str).tolist()
    aspects = test_df["aspect"].astype(str).str.lower().str.strip().tolist()
    true_labels = test_df["true_sentiment"].tolist()

    y_true = []
    y_pred = []
    misclassified = []

    batch_results = batch_analyze_aspects(reviews, aspects)

    for i in range(len(test_df)):
        review = reviews[i]
        aspect = aspects[i]
        true_label = true_labels[i]
        pred_label, confidence, _, _, _, _ = batch_results[i]

        y_true.append(true_label)
        y_pred.append(pred_label)
        
        if pred_label != true_label:
            misclassified.append({
                "Review": review,
                "Aspect": aspect,
                "True Sentiment": true_label,
                "Predicted Sentiment": pred_label,
                "Confidence": round(confidence, 2)
            })

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=["Positive", "Neutral", "Negative"]
    )

    misclassified_df = pd.DataFrame(misclassified)

    return accuracy, precision, recall, f1, cm, misclassified_df

# --------------------------------------------------
# 5. GROQ AI RESPONDER
# --------------------------------------------------
def draft_ai_reply(review_text, sentiment, aspect, api_key):
    if not api_key:
        return "⚠️ Please enter a valid Groq API Key in the sidebar."
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are the friendly and professional manager of a small business.
        A customer just left the following review focusing on the '{aspect}':
        "{review_text}"
        
        The automated sentiment analysis detected this as: {sentiment}.
        
        Please draft a concise, empathetic, and professional response to this review. 
        If it's negative, apologize and offer to make things right. 
        If it's positive, thank them and invite them back.
        Keep the response under 3 sentences.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant", 
        )
        
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating reply: {str(e)}"

# --------------------------------------------------
# 6. OVERALL AI STRATEGY GENERATOR
# --------------------------------------------------
def generate_ai_strategy(result_df, api_key):
    if not api_key:
        return "⚠️ Please enter a valid Groq API Key to generate an overall business strategy."
    
    product_neg = (result_df.get("Product Sentiment", pd.Series()) == "Negative").sum()
    service_neg = (result_df.get("Service Sentiment", pd.Series()) == "Negative").sum()
    price_neg = (result_df.get("Price Sentiment", pd.Series()) == "Negative").sum()
    ambience_neg = (result_df.get("Ambience Sentiment", pd.Series()) == "Negative").sum()
    
    stats_summary = f"""
    - Product: {product_neg} negative reviews
    - Service: {service_neg} negative reviews
    - Price: {price_neg} negative reviews
    - Ambience: {ambience_neg} negative reviews
    """
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are an expert business consultant. Based on the following negative feedback statistics from recent customer reviews, 
        provide a 3-point actionable strategy to improve the business. Be specific, encouraging, and direct.
        
        Negative Review Statistics:
        {stats_summary}
        """
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", 
        )
        
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating strategy: {str(e)}"

# --------------------------------------------------
# 7. FILE READER (CSV / EXCEL SAFE)
# --------------------------------------------------
def read_file_safe(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    try:
        return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, engine="python", encoding="latin-1", on_bad_lines="skip")

# --------------------------------------------------
# 8. CSV PREPROCESSING
# --------------------------------------------------
def preprocess_csv(df):
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    for col in ["review", "text", "content", "comment", "feedback"]:
        if col in df.columns:
            target_col = col
            break
    else:
        return None, None, 0, None
        
    date_col = None
    for col in ["date", "timestamp", "created_at", "time"]:
        if col in df.columns:
            date_col = col
            break

    before = len(df)
    df = df.dropna(subset=[target_col])
    df = df[df[target_col].astype(str).str.strip() != ""]
    removed = before - len(df)

    return df, target_col, removed, date_col

# --------------------------------------------------
# 9. RECOMMENDATION ENGINE (FALLBACK)
# --------------------------------------------------
def generate_recommendations(result_df):
    product_neg = (result_df["Product Sentiment"] == "Negative").sum() if "Product Sentiment" in result_df else 0
    service_neg = (result_df["Service Sentiment"] == "Negative").sum() if "Service Sentiment" in result_df else 0
    price_neg = (result_df["Price Sentiment"] == "Negative").sum() if "Price Sentiment" in result_df else 0
    ambience_neg = (result_df["Ambience Sentiment"] == "Negative").sum() if "Ambience Sentiment" in result_df else 0

    negatives = {
        "Product": product_neg,
        "Service": service_neg,
        "Price": price_neg,
        "Ambience": ambience_neg
    }

    dominant = max(negatives, key=negatives.get)

    if negatives[dominant] == 0:
        return ["✅ Overall sentiment is positive. Maintain current strategy."]

    if dominant == "Product":
        return [
            "📦 Improve product/service quality and delivery",
            "� Review supplier and inventory issues",
            "📋 Remove low-rated or defective items"
        ]
    elif dominant == "Service":
        return [
            "🧑‍🤝‍🧑 Train staff on customer handling",
            "⏱️ Increase staff during peak hours",
            "⭐ Reward high-performing employees"
        ]
    elif dominant == "Price":
        return [
            "💰 Re-evaluate pricing strategy",
            "🎯 Introduce bundled offers",
            "📊 Align quality with price expectations"
        ]
    else:
        return [
            "🏪 Improve store environment and cleanliness",
            "🎵 Adjust lighting and background noise"
        ]

# --------------------------------------------------
# 10. DYNAMIC ASPECT DETECTION
# --------------------------------------------------
def get_relevant_aspects(text):
    text = str(text).strip()
    if not text:
        return ["product"]
        
    labels = ["product", "service", "price", "ambience"]
    try:
        # ML-based zero-shot aspect classification
        backbone = st.session_state.get("zeroshot_backbone", "cross-encoder/nli-distilroberta-base")
        zeroshot_classifier = load_zeroshot_model(backbone)
        res = zeroshot_classifier(text, candidate_labels=labels, multi_label=True)
        # Take aspects with > 0.4 probability
        aspects = [label for label, score in zip(res['labels'], res['scores']) if score > 0.4]
        if not aspects:
            aspects.append(res['labels'][0]) # Backup top label
        return aspects
    except Exception:
        # Fallback if pipeline fails
        return ["product"]

# --------------------------------------------------
# 11. TRANSLATION HELPER
# --------------------------------------------------
@st.cache_data
def translate_if_needed(text):
    text_str = str(text)
    if not text_str.strip():
        return text_str, "en"
        
    try:
        lang = detect(text_str)
        if lang != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text_str)
            return translated, lang
        return text_str, "en"
    except Exception:
        return text_str, "unknown"

# --------------------------------------------------
# XAI — HELPER FUNCTIONS
# --------------------------------------------------

def plot_attention_heatmap(tokens: list, attn_matrix: np.ndarray, sentiment: str) -> go.Figure:
    """
    Renders a full token×token attention heatmap using Plotly.
    Each cell shows how much token-i attends to token-j in the last BERT layer.
    """
    color_scale = (
        "Reds"   if sentiment == "Negative" else
        "Greens" if sentiment == "Positive" else
        "YlOrBr"
    )
    # Trim to first 20 tokens max for readability
    n = min(len(tokens), 20)
    toks = tokens[:n]
    mat  = attn_matrix[:n, :n]

    fig = px.imshow(
        mat,
        x=toks, y=toks,
        color_continuous_scale=color_scale,
        labels={"color": "Attention"},
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#f8fafc", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        coloraxis_showscale=False,
        title=dict(text="🔥 Token Attention Heatmap (Last Layer)", font=dict(size=13, color="#38bdf8")),
        xaxis=dict(tickangle=-40, gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
    )
    return fig


def plot_word_importance_bar(word_importances: list, sentiment: str) -> go.Figure:
    """
    Horizontal bar chart — top 15 tokens ranked by CLS attention weight.
    """
    if not word_importances:
        return None

    bar_color = (
        "#ef4444" if sentiment == "Negative" else
        "#22c55e" if sentiment == "Positive" else
        "#eab308"
    )

    top = sorted(word_importances, key=lambda x: x[1], reverse=True)[:15]
    tokens_bar = [t for t, _ in top][::-1]
    weights    = [w for _, w in top][::-1]

    fig = go.Figure(go.Bar(
        x=weights, y=tokens_bar,
        orientation="h",
        marker=dict(
            color=weights,
            colorscale=[[0, "#1e293b"], [1, bar_color]],
            showscale=False,
        ),
        text=[f"{w:.4f}" for w in weights],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=10),
    ))
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#f8fafc", size=11),
        margin=dict(l=10, r=30, t=30, b=10),
        height=340,
        xaxis=dict(showgrid=True, gridcolor="#334155", color="#94a3b8", title="Attention Weight"),
        yaxis=dict(showgrid=False, color="#f8fafc"),
        title=dict(text="📊 Word Importance (CLS Attention)", font=dict(size=13, color="#38bdf8")),
    )
    return fig


def compute_loo_attribution(text: str, aspect: str, baseline_confidence: float,
                             baseline_sentiment: str) -> list:
    """
    Leave-One-Out (LOO) attribution — SHAP conceptual equivalent for BERT.
    Removes each meaningful word one at a time and measures the confidence delta.
    Returns list of (word, delta) sorted by |delta| descending.

    Viva explanation:
    'We compute word-level attribution by measuring how much each word's removal
     changes the model prediction confidence — this is the core idea behind SHAP
     (SHapley Additive exPlanations): the marginal contribution of each feature.'
    """
    words = [w for w in text.split() if len(w) > 2]
    if not words:
        return []

    attributions = []
    label_idx = {"Negative": 0, "Neutral": 1, "Positive": 2}
    base_idx   = label_idx.get(baseline_sentiment, 1)

    for i, word in enumerate(words):
        masked_words = words[:i] + words[i+1:]
        masked_text  = " ".join(masked_words)
        if not masked_text.strip():
            continue
        try:
            inputs_m = tokenizer(
                masked_text, text_pair=aspect,
                return_tensors="pt", max_length=128,
                padding="max_length", truncation=True
            ).to(device)
            with torch.no_grad():
                out_m = model(**inputs_m)
            probs_m  = F.softmax(out_m.logits, dim=1)[0].cpu().numpy()
            conf_m   = float(probs_m[base_idx])
            # Positive delta = this word was supporting the prediction
            delta = baseline_confidence - conf_m
            attributions.append((word, round(delta, 4)))
        except Exception:
            attributions.append((word, 0.0))

    return sorted(attributions, key=lambda x: abs(x[1]), reverse=True)


def explain_why(word_importances: list, loo_scores: list,
                sentiment: str, aspect: str, text: str) -> str:
    """
    Generates a natural-language explanation of why the model predicted this sentiment.
    Works entirely without an API key.
    """
    top_attn = [t for t, _ in sorted(word_importances, key=lambda x: x[1], reverse=True)[:3]]
    top_loo  = [w for w, d in loo_scores[:3] if abs(d) > 0.005] if loo_scores else []

    sentiment_desc = {
        "Negative": "negative issues",
        "Positive": "positive signals",
        "Neutral":  "neutral/mixed language",
    }.get(sentiment, "mixed signals")

    attn_part = (f"Words **{', '.join(f'*{t}*' for t in top_attn)}** received highest attention weights"
                 if top_attn else "No dominant tokens identified")

    if top_loo:
        delta_val = abs(loo_scores[0][1]) * 100 if loo_scores else 0
        loo_part  = (f"Removing **{top_loo[0]}** shifts model confidence by "
                     f"**{delta_val:.1f}%** — confirming it drives the prediction")
    else:
        loo_part = "Run Deep Attribution to see word-level causal scores"

    return (
        f"**Why {sentiment}?** The model detected {sentiment_desc} in the "
        f"**{aspect.capitalize()}** aspect. {attn_part}. {loo_part}."
    )

# --------------------------------------------------
# SOCIAL MONITOR — CONNECTOR FUNCTIONS

# --------------------------------------------------

def fetch_tweets(keyword: str, bearer_token: str, max_results: int = 10):
    """Fetch recent tweets matching keyword using Twitter v2 API (tweepy)."""
    if not TWEEPY_OK:
        return [], "tweepy not installed"
    if not bearer_token.strip():
        return [], "No Bearer Token provided"
    try:
        client = tweepy.Client(bearer_token=bearer_token.strip(), wait_on_rate_limit=False)
        query = f"{keyword} -is:retweet lang:en"
        resp = client.search_recent_tweets(
            query=query,
            max_results=max(10, min(max_results, 100)),
            tweet_fields=["text", "created_at", "author_id"]
        )
        if not resp.data:
            return [], "No tweets found for this keyword"
        texts = [t.text for t in resp.data]
        return texts, None
    except Exception as e:
        return [], str(e)


def fetch_google_play_reviews(app_id: str, count: int = 10):
    """Fetch latest reviews for a Google Play app (no auth required)."""
    if not GPLAY_OK:
        return [], "google-play-scraper not installed"
    if not app_id.strip():
        return [], "No App ID provided"
    try:
        result, _ = gplay_reviews(
            app_id.strip(),
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=count
        )
        texts = [r["content"] for r in result if r.get("content")]
        return texts, None
    except Exception as e:
        return [], str(e)


def fetch_paste_reviews(raw_text: str):
    """Parse manually pasted reviews (one per line or comma-separated)."""
    if not raw_text.strip():
        return [], "No text pasted"
    lines = [l.strip() for l in raw_text.replace(",\n", "\n").splitlines() if l.strip()]
    return lines, None


def simulate_live_reviews(session_reviews: list, batch_size: int = 5):
    """
    Replay reviews from the uploaded CSV as a simulated live feed.
    Adds random slight delay and shuffles each call.
    """
    if not session_reviews:
        return [], "No reviews available — upload a CSV in Tab 2 first"
    pool = session_reviews.copy()
    random.shuffle(pool)
    batch = pool[:batch_size]
    return batch, None


def run_monitor_analysis(texts: list, brand_keyword: str = ""):
    """
    Run BERT aspect sentiment on a list of review/post texts using batch processing.
    Returns a list of result dicts and aggregate sentiment counts.
    """
    now = datetime.datetime.now().strftime("%H:%M:%S")
    positive_n, neutral_n, negative_n = 0, 0, 0
    items = []

    if not texts:
        return items, 0, 0, 0

    batch_inputs = [] # (idx, text_orig, translated, lang, dominant_aspect)
    for i, text in enumerate(texts):
        translated, lang = translate_if_needed(text)
        aspects = get_relevant_aspects(translated)
        dominant_aspect = aspects[0] if aspects else "product"
        batch_inputs.append((i, text, translated, lang, dominant_aspect))

    if batch_inputs:
        b_texts = [x[2] for x in batch_inputs]
        b_aspects = [x[4] for x in batch_inputs]
        b_results = batch_analyze_aspects(b_texts, b_aspects)

        for (idx, text_orig, translated, lang, dom_asp), res in zip(batch_inputs, b_results):
            sentiment, confidence, action, _, _, _ = res

            if sentiment == "Positive":
                positive_n += 1
            elif sentiment == "Negative":
                negative_n += 1
            elif sentiment == "Neutral":
                neutral_n += 1

            items.append({
                "time":       now,
                "text":       text_orig[:180] + ("…" if len(text_orig) > 180 else ""),
                "aspect":     dom_asp.capitalize(),
                "sentiment":  sentiment,
                "confidence": round(confidence, 2),
                "lang":       lang
            })

    return items, positive_n, neutral_n, negative_n


# --------------------------------------------------
# PDF REPORT GENERATOR
# --------------------------------------------------

def _make_chart_image(result_df: pd.DataFrame, aspects: list) -> io.BytesIO:
    """Render a 2×2 grid of sentiment bar charts and return as PNG bytes."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.patch.set_facecolor("#1e293b")
    sent_colors = {"Positive": "#22c55e", "Neutral": "#eab308", "Negative": "#ef4444"}

    for ax, asp in zip(axes.flatten(), aspects):
        col = f"{asp} Sentiment"
        ax.set_facecolor("#0f172a")
        if col in result_df.columns and not result_df[col].dropna().empty:
            counts = result_df[col].dropna().value_counts()
            bar_colors = [sent_colors.get(k, "#64748b") for k in counts.index]
            ax.bar(counts.index, counts.values, color=bar_colors, width=0.5)
        ax.set_title(asp, color="#f8fafc", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.spines[:].set_color("#334155")
        ax.yaxis.label.set_color("#94a3b8")
    for ax in axes.flatten()[len(aspects):]:
        ax.set_visible(False)

    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_trend_chart(result_df: pd.DataFrame, date_col: str) -> io.BytesIO | None:
    """Render sentiment trend over time and return PNG bytes, or None."""
    if date_col not in result_df.columns:
        return None
    try:
        sent_cols = [c for c in result_df.columns if "Sentiment" in c]
        rdf = result_df.dropna(subset=[date_col]).copy()
        rdf[date_col] = pd.to_datetime(rdf[date_col], errors="coerce")
        rdf = rdf.dropna(subset=[date_col])
        if rdf.empty:
            return None

        trend_rows = []
        for dt, grp in rdf.groupby(date_col):
            all_sents = pd.concat([grp[c].dropna() for c in sent_cols if c in grp])
            trend_rows.append({
                "Date": dt,
                "Positive": (all_sents == "Positive").sum(),
                "Neutral":  (all_sents == "Neutral").sum(),
                "Negative": (all_sents == "Negative").sum(),
            })
        if not trend_rows:
            return None
        trend_df = pd.DataFrame(trend_rows).set_index("Date").sort_index()

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#0f172a")
        ax.plot(trend_df.index, trend_df["Positive"], color="#22c55e", marker="o", label="Positive")
        ax.plot(trend_df.index, trend_df["Neutral"],  color="#eab308", marker="s", label="Neutral")
        ax.plot(trend_df.index, trend_df["Negative"], color="#ef4444", marker="^", label="Negative")
        ax.legend(facecolor="#1e293b", labelcolor="#f8fafc", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.spines[:].set_color("#334155")
        ax.set_title("Sentiment Trend Over Time", color="#f8fafc", fontsize=11)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def generate_pdf_report(
    result_df: pd.DataFrame,
    health_score: float | None,
    leaderboard: list,
    ai_strategy: str,
    business_name: str = "Your Business",
    date_col: str | None = None,
) -> bytes:
    """
    Build a professional A4 PDF report using reportlab.
    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    # ── Styles ──────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    DARK  = colors.HexColor("#0f172a")
    MID   = colors.HexColor("#1e293b")
    BLUE  = colors.HexColor("#38bdf8")
    LIGHT = colors.HexColor("#f8fafc")
    MUTED = colors.HexColor("#94a3b8")
    RED   = colors.HexColor("#ef4444")
    GRN   = colors.HexColor("#22c55e")
    YEL   = colors.HexColor("#eab308")

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        textColor=BLUE, fontSize=26, spaceAfter=6, alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        textColor=MUTED, fontSize=10, spaceAfter=4, alignment=TA_CENTER
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=BLUE, fontSize=14, spaceBefore=10, spaceAfter=6,
        fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        textColor=LIGHT, fontSize=9, leading=14,
        fontName="Helvetica"
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        textColor=MUTED, fontSize=8, alignment=TA_CENTER
    )

    week_str = datetime.datetime.now().strftime("%B %d, %Y")
    story = []

    # ══════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("📊 Business Intelligence Report", title_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(business_name, ParagraphStyle(
        "BizName", parent=styles["Normal"],
        textColor=LIGHT, fontSize=18, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Generated: {week_str}", sub_style))
    story.append(Spacer(1, 1.5*cm))

    # Health score badge
    if health_score is not None:
        hs_color = GRN if health_score >= 65 else (YEL if health_score >= 40 else RED)
        hs_label = "Healthy 🟢" if health_score >= 65 else ("Needs Work 🟡" if health_score >= 40 else "Critical 🔴")
        hs_data = [
            [Paragraph("Business Health Score", ParagraphStyle("HS_lab", parent=styles["Normal"],
                        textColor=MUTED, fontSize=9, alignment=TA_CENTER))],
            [Paragraph(f"{health_score}", ParagraphStyle("HS_val", parent=styles["Normal"],
                        textColor=hs_color, fontSize=48, alignment=TA_CENTER,
                        fontName="Helvetica-Bold"))],
            [Paragraph(f"/ 100 — {hs_label}", ParagraphStyle("HS_sub", parent=styles["Normal"],
                        textColor=MUTED, fontSize=9, alignment=TA_CENTER))],
        ]
        hs_table = Table(hs_data, colWidths=[10*cm])
        hs_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), MID),
            ("ROWPADDING",  (0,0), (-1,-1), 8),
            ("BOX",         (0,0), (-1,-1), 1.5, BLUE),
            ("ROUNDEDCORNERS", [8]),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ]))
        story.append(hs_table)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 2 — EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("📋 Executive Summary", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    total_reviews = len(result_df)
    sent_cols_all = [c for c in result_df.columns if "Sentiment" in c]
    all_sents = pd.concat([result_df[c].dropna() for c in sent_cols_all]) if sent_cols_all else pd.Series([], dtype=str)
    total_sents = len(all_sents)
    pos_pct = round((all_sents == "Positive").sum() / total_sents * 100, 1) if total_sents else 0
    neu_pct = round((all_sents == "Neutral").sum()  / total_sents * 100, 1) if total_sents else 0
    neg_pct = round((all_sents == "Negative").sum() / total_sents * 100, 1) if total_sents else 0

    summary_data = [
        ["Metric", "Value"],
        ["Total Reviews Analysed", str(total_reviews)],
        ["Positive Sentiment %",  f"{pos_pct}%"],
        ["Neutral Sentiment %",   f"{neu_pct}%"],
        ["Negative Sentiment %",  f"{neg_pct}%"],
        ["Business Health Score", f"{health_score}/100" if health_score else "N/A"],
        ["Report Date", week_str],
    ]
    summary_tbl = Table(summary_data, colWidths=[9*cm, 8*cm])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0), DARK),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [MID, DARK]),
        ("TEXTCOLOR",    (0,1), (-1,-1), LIGHT),
        ("ALIGN",        (1,0), (1,-1), "CENTER"),
        ("ROWPADDING",   (0,0), (-1,-1), 8),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
    ]))
    story.append(summary_tbl)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 3 — KEY PROBLEMS LEADERBOARD
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("🔥 Key Problems Detected", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=10))

    if leaderboard:
        prob_data = [["Aspect", "Negatives", "Total", "Severity %", "Status"]]
        for p in leaderboard:
            prob_data.append([
                p.get("Aspect", ""),
                str(p.get("Negatives", "")),
                str(p.get("Total Mentions", "")),
                f"{p.get('Severity %', '')}%",
                p.get("Status", ""),
            ])
        prob_tbl = Table(prob_data, colWidths=[4*cm, 3*cm, 3*cm, 3.5*cm, 3.5*cm])
        row_bgs = []
        for i, p in enumerate(leaderboard, 1):
            pct = p.get("Severity %", 0)
            bg = colors.HexColor("#7f1d1d") if pct >= 60 else (
                 colors.HexColor("#713f12") if pct >= 30 else colors.HexColor("#14532d"))
            row_bgs.append(("BACKGROUND", (0,i), (-1,i), bg))
        prob_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), RED),
            ("TEXTCOLOR",  (0,0), (-1,0), DARK),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("TEXTCOLOR",  (0,1), (-1,-1), LIGHT),
            ("ROWPADDING", (0,0), (-1,-1), 7),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
            ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ] + row_bgs))
        story.append(prob_tbl)
    else:
        story.append(Paragraph("✅ No critical problems detected this period.", body_style))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 4 — SENTIMENT CHARTS
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("📊 Sentiment Distribution by Aspect", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))
    aspects_present = [a for a in ["Product","Service","Price","Ambience"]
                       if f"{a} Sentiment" in result_df.columns]
    if aspects_present:
        chart_buf = _make_chart_image(result_df, aspects_present)
        chart_img = RLImage(chart_buf, width=16*cm, height=9.6*cm)
        story.append(chart_img)
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Green = Positive · Yellow = Neutral · Red = Negative",
                                caption_style))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 5 — AI STRATEGY
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("🧠 AI Strategy Recommendations", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#818cf8"),
                             spaceAfter=10))
    strat_text = ai_strategy if ai_strategy.strip() else (
        "No Groq AI strategy generated yet. Upload data and click 'Generate Deep AI Strategy' in Tab 2."
    )
    for line in strat_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))
            story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 6 — WEEKLY TREND (if date data available)
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("📅 Weekly Sentiment Trend", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=GRN, spaceAfter=10))
    trend_buf = _make_trend_chart(result_df, date_col or "Date")
    if trend_buf:
        trend_img = RLImage(trend_buf, width=16*cm, height=5*cm)
        story.append(trend_img)
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Green = Positive · Yellow = Neutral · Red = Negative",
                                caption_style))
    else:
        story.append(Paragraph(
            "No date column detected. Add a 'date' column to your CSV to enable trend analysis.",
            body_style
        ))

    # ── Build PDF ────────────────────────────────────────
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# 12. SMART COPILOT — MODULE 1: TOP PROBLEM DETECTOR

# --------------------------------------------------
def detect_top_problems(result_df, df_reviews=None):
    """
    Ranks business problem areas by severity.
    Returns a list of dicts: {aspect, negative_count, total, severity_pct, badge}.
    Also scans raw text for recurring complaint keywords.
    """
    aspects = ["Product", "Service", "Price", "Ambience"]
    leaderboard = []

    for asp in aspects:
        col_s = f"{asp} Sentiment"
        if col_s not in result_df.columns:
            continue
        total = result_df[col_s].notna().sum()
        if total == 0:
            continue
        neg = (result_df[col_s] == "Negative").sum()
        pct = round((neg / total) * 100, 1)
        if pct >= 60:
            badge = "🔴 Critical"
        elif pct >= 30:
            badge = "🟡 Moderate"
        else:
            badge = "🟢 Minor"
        leaderboard.append({
            "Aspect": asp, "Negatives": int(neg),
            "Total Mentions": int(total), "Severity %": pct, "Status": badge
        })

    leaderboard.sort(key=lambda x: x["Severity %"], reverse=True)

    # Keyword cluster scan on raw review text
    keyword_clusters = {}
    if df_reviews is not None and "Original Review" in result_df.columns:
        complaint_words = [
            "wait","slow","cold","rude","dirty","broken","expensive",
            "bad","late","wrong","missing","poor","damaged","never","worst"
        ]
        all_text = " ".join(result_df["Original Review"].astype(str).str.lower().tolist())
        words = all_text.split()
        freq = collections.Counter(w for w in words if w in complaint_words)
        keyword_clusters = dict(freq.most_common(5))

    return leaderboard, keyword_clusters


# --------------------------------------------------
# 13. SMART COPILOT — MODULE 2: CHURN RISK PREDICTOR
# --------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def get_churn_model(result_df_json: str):
    """
    Builds and trains a RandomForest churn predictor using the real dataset.
    We pass result_df_json (string) instead of the dataframe so Streamlit can hash it for @st.cache_data.
    """
    df = pd.read_json(io.StringIO(result_df_json)) if result_df_json else pd.DataFrame()
    X_train, y_train = [], []
    aspects = ["Product", "Service", "Price", "Ambience"]

    # If we have real data, build training set from it using logical heuristics as ground truth
    if not df.empty and len(df) > 5:
        for _, row in df.iterrows():
            neg = sum(1 for asp in aspects if row.get(f"{asp} Sentiment") == "Negative")
            pos = sum(1 for asp in aspects if row.get(f"{asp} Sentiment") == "Positive")
            neu = sum(1 for asp in aspects if row.get(f"{asp} Sentiment") == "Neutral")
            confs = [row.get(f"{asp} Confidence") for asp in aspects if pd.notna(row.get(f"{asp} Confidence"))]
            conf = sum(confs)/len(confs) if confs else 0.5
            
            # Ground truth heuristic for training based on real rows:
            if neg >= 2: churn = 1
            elif neg == 1 and conf < 0.55: churn = 1
            elif pos >= 2 and neg == 0: churn = 0
            else: churn = 1 if random.random() < 0.15 else 0
            
            X_train.append([pos, neu, neg, conf])
            y_train.append(churn)
    else:
        # Fallback to simulated data if no real data uploaded yet
        for _ in range(500):
            pos = random.randint(0, 5)
            neu = random.randint(0, 5)
            neg = random.randint(0, 4)
            conf = random.uniform(0.3, 0.99)
            if neg > pos: churn = 1
            elif neg == 0 and pos > 0: churn = 0
            elif neg > 0 and conf < 0.5: churn = 1
            else: churn = 1 if random.random() < 0.2 else 0
            X_train.append([pos, neu, neg, conf])
            y_train.append(churn)
        
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict_churn_risk(result_df):
    """
    Scores each review row with a 0–100 churn risk score using a RandomForest ML model.
    Returns result_df with 'Churn Risk Score' and 'Churn Risk Level' columns,
    plus aggregate counts.
    """
    aspects = ["Product", "Service", "Price", "Ambience"]
    risk_scores = []
    
    # Train/load model using the current dataset's json signature
    df_json = result_df.to_json(orient="records") if not result_df.empty else ""
    clf = get_churn_model(df_json)

    for _, row in result_df.iterrows():
        neg_count = sum(1 for asp in aspects if row.get(f"{asp} Sentiment") == "Negative")
        pos_count = sum(1 for asp in aspects if row.get(f"{asp} Sentiment") == "Positive")
        neu_count = sum(1 for asp in aspects if row.get(f"{asp} Sentiment") == "Neutral")
        
        confs = [row.get(f"{asp} Confidence") for asp in aspects if not pd.isna(row.get(f"{asp} Confidence"))]
        avg_conf = sum(confs) / len(confs) if confs else 0.5
        
        prob = clf.predict_proba([[pos_count, neu_count, neg_count, avg_conf]])[0][1]
        score = int(prob * 100)
        risk_scores.append(score)

    result_df = result_df.copy()
    result_df["Churn Risk Score"] = risk_scores
    result_df["Churn Risk Level"] = result_df["Churn Risk Score"].apply(
        lambda s: "🔴 High Risk" if s >= 70 else ("🟡 Medium Risk" if s >= 40 else "🟢 Low Risk")
    )

    high   = (result_df["Churn Risk Level"] == "🔴 High Risk").sum()
    medium = (result_df["Churn Risk Level"] == "🟡 Medium Risk").sum()
    low    = (result_df["Churn Risk Level"] == "🟢 Low Risk").sum()

    return result_df, int(high), int(medium), int(low)


# --------------------------------------------------
# 14. SMART COPILOT — MODULE 3: REVENUE ACTIONS ENGINE
# --------------------------------------------------
REVENUE_ACTIONS = {
    "Product": {
        "threshold": 30,
        "icon": "📦",
        "action": "Launch a product quality audit. Defective/cold items destroy repeat business — fixing this can recover 20–30% lost return visits.",
        "quick_win": "Pull your 3 most-complained-about items and audit this week."
    },
    "Service": {
        "threshold": 30,
        "icon": "🧑‍🤝‍🧑",
        "action": "Faster service = higher table turnover = +15% revenue potential. Schedule a 1-hour staff roleplay training session.",
        "quick_win": "Assign a 'speed champion' per shift for the next 2 weeks."
    },
    "Price": {
        "threshold": 30,
        "icon": "💰",
        "action": "Introduce value bundles or combo deals. Perceived value raises spend-per-visit even without cutting prices.",
        "quick_win": "Create one 'best value' combo this week and promote it on social media."
    },
    "Ambience": {
        "threshold": 30,
        "icon": "🏪",
        "action": "Environment drives customer dwell time. Longer stays = larger basket size. Small fixes (lighting, cleanliness) have outsized ROI.",
        "quick_win": "Do a cleanliness walk-through before opening for the next 5 days."
    }
}

def generate_revenue_actions(result_df, high_churn_pct=0):
    """Returns a list of revenue action dicts for aspects with >threshold% negative reviews."""
    triggered = []
    for asp, cfg in REVENUE_ACTIONS.items():
        col = f"{asp} Sentiment"
        if col not in result_df.columns:
            continue
        total = result_df[col].notna().sum()
        if total == 0:
            continue
        neg_pct = (result_df[col] == "Negative").sum() / total * 100
        if neg_pct >= cfg["threshold"]:
            triggered.append({
                "aspect": asp, "icon": cfg["icon"],
                "neg_pct": round(neg_pct, 1),
                "action": cfg["action"], "quick_win": cfg["quick_win"]
            })

    if high_churn_pct >= 20:
        triggered.append({
            "aspect": "Retention", "icon": "🔄",
            "neg_pct": round(high_churn_pct, 1),
            "action": "High churn risk detected. Deploy a loyalty program — retaining 5% more customers can increase profit by up to 25%.",
            "quick_win": "Start a simple punch-card or digital loyalty scheme this week."
        })

    return triggered


# --------------------------------------------------
# 15. SMART COPILOT — MODULE 4: WEEKLY REPORT (GROQ)
# --------------------------------------------------
def compute_health_score(result_df):
    """Weighted business health score 0–100."""
    aspects = ["Product", "Service", "Price", "Ambience"]
    all_sentiments = []
    for asp in aspects:
        col = f"{asp} Sentiment"
        if col in result_df.columns:
            all_sentiments.extend(result_df[col].dropna().tolist())
    if not all_sentiments:
        return None
    pos = all_sentiments.count("Positive")
    neu = all_sentiments.count("Neutral")
    neg = all_sentiments.count("Negative")
    total = len(all_sentiments)
    score = ((pos * 1.0) + (neu * 0.5) + (neg * 0.0)) / total * 100
    return round(score, 1)


def generate_weekly_report_groq(result_df, health_score, leaderboard, revenue_actions, api_key):
    """Sends full stats to Groq and returns a concise weekly executive brief."""
    if not api_key:
        return None

    top_problem = leaderboard[0]["Aspect"] if leaderboard else "N/A"
    top_pct     = leaderboard[0]["Severity %"] if leaderboard else 0
    actions_txt = "\n".join(
        [f"- {r['aspect']}: {r['action']}" for r in revenue_actions]
    ) if revenue_actions else "No critical actions triggered."

    prompt = f"""
You are an expert business intelligence analyst. A small business owner is reviewing
their customer feedback for the week. Here is the summary:

- Business Health Score: {health_score}/100
- Top Problem Area: {top_problem} ({top_pct}% negative reviews)
- Problem Leaderboard: {', '.join([f"{r['Aspect']} ({r['Severity %']}%)" for r in leaderboard])}
- Revenue Actions Triggered:
{actions_txt}

Generate a concise "What to Fix This Week" executive report in exactly 5 bullet points.
Each bullet should be one specific, actionable instruction. Be direct, encouraging, and data-driven.
Do not use vague language. Start each bullet with an emoji.
"""
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"


def auto_save_to_db(df: pd.DataFrame):
    """Silently saves bulk analysis results to the database to build historical data."""
    if df.empty or not st.session_state.get("authenticated"):
        return
    
    current_user = st.session_state.get("username", "admin")
    c = conn.cursor()
    # Create simple unique identifier (hash of review text) to avoid duplicates
    try:
        c.execute("ALTER TABLE reviews_history ADD COLUMN text_hash TEXT")
    except Exception:
        pass
    
    saved_count = 0
    for _, r in df.iterrows():
        dt = str(r.get("Date", datetime.datetime.now().strftime("%Y-%m-%d")))
        rev = str(r.get("Original Review", ""))
        if not rev.strip(): continue
        
        text_hash = hashlib.md5(rev.encode()).hexdigest()
        
        # Check if already saved
        c.execute("SELECT id FROM reviews_history WHERE text_hash=?", (text_hash,))
        if c.fetchone():
            continue
            
        for asp in ["Product", "Service", "Price", "Ambience"]:
            sent_col = f"{asp} Sentiment"
            conf_col = f"{asp} Confidence"
            if sent_col in r and not pd.isna(r[sent_col]):
                c.execute(
                    "INSERT INTO reviews_history (date, review, aspect, sentiment, confidence, username, source, text_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (dt, rev, asp, str(r[sent_col]), float(r[conf_col]), current_user, "csv_upload", text_hash)
                )
        saved_count += 1
    
    if saved_count > 0:
        conn.commit()


# --------------------------------------------------
# 16. UI
# --------------------------------------------------

st.sidebar.title("👤 User Protocol")
st.sidebar.markdown(f"Logged in as: **{st.session_state.get('username', 'Admin')}**")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("⚙️ AI Settings")

backbone_choice = st.sidebar.selectbox(
    "🧠 Zero-Shot Backbone",
    [
        "cross-encoder/nli-distilroberta-base",
        "facebook/bart-large-mnli",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    ],
    index=0,
    help="Select the architecture for dynamic aspect extraction."
)
st.session_state["zeroshot_backbone"] = backbone_choice

groq_api_key = st.sidebar.text_input("Groq API Key (for replies)", value=os.getenv("GROQ_API_KEY", ""), type="password")
st.sidebar.markdown("[Get a free Groq key here](https://console.groq.com/keys)")

st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Predictions below this threshold are labeled 'Uncertain'"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #94a3b8; padding-top: 20px;">
    Powered by <b>BERT</b> & <b>Groq</b><br>
    <i>v2.0 Premium Edition</i>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>💼 Small Business Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Turn customer reviews into data-driven <b>Business Decisions</b>.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Single Review Analysis",
    "📂 Bulk Upload (CSV)",
    "📊 Model Evaluation",
    "🎯 Weekly AI Report",
    "📡 Social Media Monitor",
    "⚔️ Competitor Comparison"
])

# Session state for cross-tab data sharing
if "copilot_result_df" not in st.session_state:
    st.session_state["copilot_result_df"] = None
if "copilot_leaderboard" not in st.session_state:
    st.session_state["copilot_leaderboard"] = []
if "copilot_revenue_actions" not in st.session_state:
    st.session_state["copilot_revenue_actions"] = []
if "copilot_health_score" not in st.session_state:
    st.session_state["copilot_health_score"] = None

# Monitor session state
if "monitor_feed" not in st.session_state:
    st.session_state["monitor_feed"] = []          # rolling list of result dicts
if "monitor_trend" not in st.session_state:
    st.session_state["monitor_trend"] = []          # list of {time, pos, neu, neg}
if "monitor_alerts" not in st.session_state:
    st.session_state["monitor_alerts"] = []          # alert history
if "weekly_brief" not in st.session_state:
    st.session_state["weekly_brief"] = ""
if "ai_strategy_output" not in st.session_state:
    st.session_state["ai_strategy_output"] = ""

# ================= TAB 1 =================
with tab1:
    st.markdown("### 📝 Analyze a Single Review")
    st.markdown("Instantly break down a customer's review into actionable aspects and sentiments.")
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        review_text = st.text_area(
            "Paste Customer Review Here:",
            "The product arrived on time, but the packaging was damaged and customer service was unhelpful.",
            height=150
        )
        analyze_btn = st.button("🚀 Analyze Review Aspects", use_container_width=True)

    with col2:
        if analyze_btn:
            with st.spinner("Processing with BERT Model..."):
                translated_text, lang = translate_if_needed(review_text)
                
                if lang != "en" and lang != "unknown":
                    st.info(f"🌐 **Translated from detected language ({lang}):**\n*{translated_text}*")
                
                aspects_to_analyze = get_relevant_aspects(translated_text)
                st.markdown(f"**Detected Relevant Aspects:** {', '.join([f'`{a.capitalize()}`' for a in aspects_to_analyze])}")
                st.markdown("---")
                
                for aspect in aspects_to_analyze:
                    sentiment, confidence, action, importances, attn_matrix, clean_toks = \
                        analyze_aspect(translated_text, aspect)

                    # Confidence threshold override
                    if confidence < confidence_threshold:
                        sentiment = "Uncertain"

                    # ── Sentiment Display Card ──────────────────────
                    if sentiment == "Negative":
                        sc_color = "#ef4444"
                        st.error(f"**{aspect.capitalize()}** → 📉 Negative (Confidence: {confidence:.2f})\n\n💡 **Action:** {action}")
                    elif sentiment == "Positive":
                        sc_color = "#22c55e"
                        st.success(f"**{aspect.capitalize()}** → 📈 Positive (Confidence: {confidence:.2f})\n\n💡 **Action:** {action}")
                    elif sentiment == "Uncertain":
                        sc_color = "#64748b"
                        st.info(f"**{aspect.capitalize()}** → ❓ Uncertain (Confidence: {confidence:.2f}) — below threshold")
                    else:
                        sc_color = "#eab308"
                        st.warning(f"**{aspect.capitalize()}** → ➖ Neutral (Confidence: {confidence:.2f})\n\n💡 **Action:** {action}")

                    # ── XAI DASHBOARD ───────────────────────────────
                    with st.expander(f"🔬 Explainable AI — {aspect.capitalize()} Analysis", expanded=True):

                        # LAYER 1: Coloured token spans (keep the existing feel)
                        if importances:
                            max_w = max(w for _, w in importances) or 1.0
                            html_str = "<div style='background:#0f172a;padding:12px;border-radius:8px;margin-bottom:12px;'>"
                            html_str += "<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:6px;'>🔍 <b>Attention Heatmap (token level)</b> — darker = higher attention</div>"
                            html_str += "<div style='line-height:2.4;'>"
                            for token, weight in importances:
                                intensity = min(weight / max_w, 1.0)
                                if sentiment == "Negative":
                                    bg = f"rgba(239,68,68,{intensity*0.85})"
                                elif sentiment == "Positive":
                                    bg = f"rgba(34,197,94,{intensity*0.85})"
                                else:
                                    bg = f"rgba(234,179,8,{intensity*0.85})"
                                size_rem = 0.82 + intensity * 0.22
                                html_str += (
                                    f"<span style='background:{bg};padding:3px 6px;margin:2px;"
                                    f"border-radius:4px;font-family:monospace;color:white;"
                                    f"font-size:{size_rem:.2f}rem;' "
                                    f"title='Attention: {weight:.4f}'>{token}</span>"
                                )
                            html_str += "</div></div>"
                            st.markdown(html_str, unsafe_allow_html=True)

                        # LAYER 2: Plotly heatmap + bar chart side-by-side
                        xai_c1, xai_c2 = st.columns(2)
                        with xai_c1:
                            if attn_matrix is not None and len(clean_toks) >= 2:
                                fig_hm = plot_attention_heatmap(clean_toks, attn_matrix, sentiment)
                                st.plotly_chart(fig_hm, use_container_width=True)
                        with xai_c2:
                            if importances:
                                fig_bar = plot_word_importance_bar(importances, sentiment)
                                if fig_bar:
                                    st.plotly_chart(fig_bar, use_container_width=True)

                        # LAYER 3: Why panel (no API key needed)
                        why_key = f"loo_{aspect}_{hash(translated_text)}"
                        loo_cache = st.session_state.get(why_key, [])
                        why_text = explain_why(importances, loo_cache, sentiment, aspect, translated_text)
                        st.markdown(f"""
                        <div style='background:#1e293b;border-left:4px solid {sc_color};border-radius:8px;
                                    padding:14px 18px;margin-top:6px;'>
                            <div style='font-size:0.82rem;color:#94a3b8;margin-bottom:4px;'>💬 Why did the model predict this?</div>
                            <div style='color:#f8fafc;font-size:0.92rem;line-height:1.6;'>{why_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # LAYER 4: LOO Attribution (SHAP-style) — on-demand button
                        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                        loo_btn = st.button(
                            f"🎯 Run Deep Attribution (SHAP-style) — {aspect.capitalize()}",
                            key=f"loo_btn_{aspect}_{hash(translated_text)}",
                            help="Removes each word one-at-a-time and measures how much the model confidence changes. This is the Leave-One-Out (LOO) SHAP-equivalent attribution."
                        )
                        if loo_btn:
                            with st.spinner(f"Computing LOO attribution for {len(translated_text.split())} words..."):
                                loo_scores = compute_loo_attribution(
                                    translated_text, aspect, confidence, sentiment
                                )
                                st.session_state[why_key] = loo_scores

                        if loo_cache:
                            top_loo = loo_cache[:12]
                            words_loo   = [w for w, _ in top_loo]
                            deltas_loo  = [d for _, d in top_loo]
                            colors_loo  = [
                                "#ef4444" if d > 0 else "#22c55e" for d in deltas_loo
                            ]
                            fig_loo = go.Figure(go.Bar(
                                x=deltas_loo,
                                y=words_loo,
                                orientation="h",
                                marker_color=colors_loo,
                                text=[f"{d:+.3f}" for d in deltas_loo],
                                textposition="outside",
                                textfont=dict(color="#94a3b8", size=10),
                            ))
                            fig_loo.update_layout(
                                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                                font=dict(color="#f8fafc", size=11),
                                margin=dict(l=10, r=40, t=35, b=10),
                                height=320,
                                xaxis=dict(title="Confidence Δ (positive = supports prediction)",
                                           gridcolor="#334155", color="#94a3b8",
                                           zeroline=True, zerolinecolor="#475569"),
                                yaxis=dict(showgrid=False, color="#f8fafc"),
                                title=dict(
                                    text="🎯 SHAP-Style Word Attribution (LOO)",
                                    font=dict(size=13, color="#38bdf8")
                                ),
                            )
                            st.plotly_chart(fig_loo, use_container_width=True)
                            st.caption(
                                "🔴 Red = removing this word **drops** confidence (word supports prediction) · "
                                "🟢 Green = removing it **raises** confidence (word contradicts prediction)"
                            )

                    if groq_api_key:
                        with st.expander(f"✨ Draft AI Reply for {aspect.capitalize()}"):
                            with st.spinner("Generating reply with Groq..."):
                                reply = draft_ai_reply(review_text, sentiment, aspect, groq_api_key)
                                st.write(reply)



# ================= TAB 2 =================
with tab2:
    uploaded_file = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # ── PERFORMANCE FIX: Only re-run heavy analysis when a NEW file is uploaded ──
        # On tab switches, Streamlit re-runs the whole script. Without this guard,
        # the entire BERT batch analysis re-executes every time you click a tab.
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        already_analyzed = st.session_state.get("_analyzed_file_id") == current_file_id

        if already_analyzed:
            # Load from cache — nearly instant!
            raw_df   = st.session_state["_raw_df"]
            df_clean = st.session_state["_df_clean"]
            col_name = st.session_state["_col_name"]
            date_col = st.session_state["_date_col"]
            results  = st.session_state["_results"]
            result_df = pd.DataFrame(results)
            st.success(f"✅ Loaded **{len(result_df)}** cached results (tab switch — no re-analysis needed)")
        else:
            raw_df = read_file_safe(uploaded_file)

            if raw_df.empty:
                st.error("Uploaded file is empty.")
                st.stop()

            df_clean, col_name, removed, date_col = preprocess_csv(raw_df)

            if df_clean is None:
                st.error("No review column found.")
                st.stop()

            if date_col:
                st.info(f"📅 Detected date column: `{date_col}`. Trend analysis enabled.")

            st.success("File loaded successfully.")

            results = []
            limit = min(500, len(df_clean))
            st.write(f"Analyzing {limit} reviews...")
            progress = st.progress(0, text="Starting BERT batch analysis...")

            # 1. Gather all inputs
            batch_inputs = []
            row_datas = []

            for i in range(limit):
                original_text = str(df_clean.iloc[i][col_name])
                text, lang = translate_if_needed(original_text)
                aspects_to_analyze = get_relevant_aspects(text)

                row_data = {"Original Review": original_text, "Translated Review": text if lang != 'en' else "-"}
                if date_col:
                    row_data["Date"] = df_clean.iloc[i][date_col]

                for aspect_name in ["product", "service", "price", "ambience"]:
                    row_data[f"{aspect_name.capitalize()} Sentiment"] = None
                    row_data[f"{aspect_name.capitalize()} Confidence"] = None

                    if aspect_name in aspects_to_analyze:
                        batch_inputs.append((len(row_datas), text, aspect_name))

                row_datas.append(row_data)
                if i % 20 == 0:
                    progress.progress(int(i / limit * 70), text=f"Preparing review {i+1}/{limit}...")

            # 2. Run batch inference in one go
            if batch_inputs:
                progress.progress(75, text="Running BERT batch inference...")
                b_texts   = [x[1] for x in batch_inputs]
                b_aspects = [x[2] for x in batch_inputs]
                b_results = batch_analyze_aspects(b_texts, b_aspects)

                for (idx, _, aspect_name), res in zip(batch_inputs, b_results):
                    s, c, _, _, _, _ = res
                    row_datas[idx][f"{aspect_name.capitalize()} Sentiment"] = s
                    row_datas[idx][f"{aspect_name.capitalize()} Confidence"] = round(c, 2)

            results = row_datas
            progress.progress(100, text="✅ Analysis complete!")
            progress.empty()

            # Cache everything so tab switches are instant
            st.session_state["_analyzed_file_id"] = current_file_id
            st.session_state["_raw_df"]   = raw_df
            st.session_state["_df_clean"] = df_clean
            st.session_state["_col_name"] = col_name
            st.session_state["_date_col"] = date_col
            st.session_state["_results"]  = results
        # 2. Run blazing-fast batch inference
        if batch_inputs:
            b_texts = [x[1] for x in batch_inputs]
            b_aspects = [x[2] for x in batch_inputs]
            b_results = batch_analyze_aspects(b_texts, b_aspects)
            
            # 3. Map back to rows
            for (idx, _, aspect_name), res in zip(batch_inputs, b_results):
                s, c, _, _, _, _ = res
                row_datas[idx][f"{aspect_name.capitalize()} Sentiment"] = s
                row_datas[idx][f"{aspect_name.capitalize()} Confidence"] = round(c, 2)
                
        results = row_datas

        st.markdown("### 📊 Enterprise Dashboard")
        st.markdown("Get a bird's eye view of customer satisfaction and operations.")
        
        result_df = pd.DataFrame(results)
        
        # Auto-save to DB silently (like a real SaaS)
        auto_save_to_db(result_df)
        
        # Override Streamlit dataframe styling for a cleaner look
        st.markdown("#### 📄 Analyzed Data")
        with st.expander("View Data Table & AI Replies"):
            st.dataframe(result_df, use_container_width=True)
            
            generate_replies = st.checkbox("✨ Generate AI Replies for all reviews (requires Groq key)")
            
            if generate_replies and not groq_api_key:
                st.warning("Please enter your Groq API key in the sidebar to use this feature.")
                
            elif generate_replies and st.button("Start Generating Replies"):
                with st.spinner("Drafting replies via Groq..."):
                    replies = []
                    for idx, row in result_df.iterrows():
                        reply_aspect = "service"
                        worst_sentiment = "Positive"
                        for a in ["Product", "Service", "Price", "Ambience"]:
                            if row.get(f"{a} Sentiment") == "Negative":
                                reply_aspect = a.lower()
                                worst_sentiment = "Negative"
                                break
                            
                        reply = draft_ai_reply(row.get("Translated Review", row["Original Review"]) if row.get("Translated Review", "-") != "-" else row["Original Review"], worst_sentiment, reply_aspect, groq_api_key)
                        replies.append(reply)
                    
                    result_df["Suggested AI Reply"] = replies
                    st.success("Replies generated successfully!")
                    st.rerun() # Refresh to show new column
                    
            col_csv, col_xlsx, col_db = st.columns(3)
            with col_csv:
                st.download_button(
                    "⬇️ Download CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "analyzed_reviews.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_xlsx:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Analyzed Data')
                st.download_button(
                    "📊 Download Excel",
                    output.getvalue(),
                    "analyzed_reviews.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_db:
                if st.button("💾 Save to Database", use_container_width=True):
                    with st.spinner("Saving to SQLite..."):
                        c = conn.cursor()
                        for _, r in result_df.iterrows():
                            dt = str(r.get("Date", datetime.datetime.now().strftime("%Y-%m-%d")))
                            rev = str(r.get("Original Review", ""))
                            for asp in ["Product", "Service", "Price", "Ambience"]:
                                sent_col = f"{asp} Sentiment"
                                conf_col = f"{asp} Confidence"
                                if sent_col in r and not pd.isna(r[sent_col]):
                                    c.execute(
                                        "INSERT INTO reviews_history (date, review, aspect, sentiment, confidence) VALUES (?, ?, ?, ?, ?)",
                                        (dt, rev, asp, str(r[sent_col]), float(r[conf_col]))
                                    )
                        conn.commit()
                    st.success("Saved to DB manually!")
        
        st.markdown("#### 🗄️ Database Viewer")
        with st.expander("View Saved Historical Data (reviews_history)"):
            c = conn.cursor()
            try:
                db_df = pd.read_sql_query("SELECT * FROM reviews_history ORDER BY id DESC LIMIT 100", conn)
                if not db_df.empty:
                    st.dataframe(db_df, use_container_width=True)
                else:
                    st.info("No data saved in the database yet.")
            except Exception as e:
                st.error(f"Could not load database view: {e}")

        st.markdown("---")
        
        score_map = {"Positive": 1, "Neutral": 0, "Negative": -1, "Uncertain": 0}
        temp_df = result_df[[c for c in result_df.columns if "Sentiment" in c]].replace(score_map)
        overall_score = temp_df.apply(pd.to_numeric, errors='coerce').mean(axis=1).mean()
        
        score_col, strat_col = st.columns([1, 2.5])
        with score_col:
            st.markdown("<div style='background: #1e293b; padding: 20px; border-radius: 12px; text-align: center;'>", unsafe_allow_html=True)
            if not pd.isna(overall_score):
                st.metric("Aggregate Satisfaction Score", round(overall_score, 2))
            else:
                st.metric("Aggregate Satisfaction Score", "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

        with strat_col:
            st.markdown("<div style='background: #1e293b; padding: 15px; border-radius: 12px;'>", unsafe_allow_html=True)
            st.markdown("#### 💡 Quick Recommendations")
            for rec in generate_recommendations(result_df):
                st.success(rec)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Sentiment Distribution by Aspect")
        c1, c2, c3, c4 = st.columns(4)
        if "Product Sentiment" in result_df.columns and not result_df["Product Sentiment"].dropna().empty:
            c1.bar_chart(result_df["Product Sentiment"].dropna().value_counts())
        if "Service Sentiment" in result_df.columns and not result_df["Service Sentiment"].dropna().empty:
            c2.bar_chart(result_df["Service Sentiment"].dropna().value_counts())
        if "Price Sentiment" in result_df.columns and not result_df["Price Sentiment"].dropna().empty:
            c3.bar_chart(result_df["Price Sentiment"].dropna().value_counts())
        if "Ambience Sentiment" in result_df.columns and not result_df["Ambience Sentiment"].dropna().empty:
            c4.bar_chart(result_df["Ambience Sentiment"].dropna().value_counts())

        # =========================================================
        # 🔍 ADVANCED TOPIC MODELING (LDA)
        # =========================================================
        st.markdown("---")
        st.markdown("#### 🔍 Deep Topic Extraction (Top Complaints)")
        st.write("Automatically groups negative reviews to find exact underlying root issues without manual reading.")
        
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        neg_df = result_df.copy()
        sent_cols = [c for c in neg_df.columns if "Sentiment" in c]
        if sent_cols:
            is_neg = neg_df[sent_cols].eq("Negative").any(axis=1)
            trans_series = neg_df.loc[is_neg, "Translated Review"]
            orig_series = neg_df.loc[is_neg, "Original Review"]
            neg_reviews = trans_series.mask(trans_series == "-", orig_series).astype(str).tolist()
            
            if len(neg_reviews) >= 5:
                with st.spinner("Running LDA Topic Extraction..."):
                    try:
                        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
                        dtm = vectorizer.fit_transform(neg_reviews)
                        lda = LatentDirichletAllocation(n_components=3, random_state=42)
                        lda.fit(dtm)
                        
                        feature_names = vectorizer.get_feature_names_out()
                        
                        tcol1, tcol2, tcol3 = st.columns(3)
                        cols = [tcol1, tcol2, tcol3]
                        
                        for topic_idx, topic in enumerate(lda.components_):
                            top_features_ind = topic.argsort()[:-6:-1]
                            top_features = [feature_names[i] for i in top_features_ind]
                            
                            with cols[topic_idx]:
                                st.error(f"**Topic Motif {topic_idx + 1}:**\n\n`{', '.join(top_features)}`")
                                
                    except Exception as e:
                        st.warning(f"Not enough linguistic diversity for deep extraction. Need more diverse reviews.")
            else:
                st.info("Not enough negative reviews to run Deep Topic Extraction (requires at least 5).")

        # =========================================================
        # 🧠 PREMIUM AI STRATEGY GENERATOR
        # =========================================================
        st.markdown("---")
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;
                    border-radius:14px;padding:24px 28px;margin-bottom:20px;'>
            <div style='font-size:1.5rem;font-weight:800;color:#38bdf8;margin-bottom:6px;'>
                🧠 AI Strategy Generator
            </div>
            <div style='color:#94a3b8;font-size:0.9rem;'>
                Transforms your BERT sentiment analysis into a structured, actionable business strategy
                — with root causes, priority actions, and expected revenue impact.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-aspect strategy cards (always shown, no API key needed) ──
        STRATEGY_DB = {
            "Product": {
                "icon": "📦",
                "root_cause": "Defective/cold/damaged items reaching customers",
                "action": "Conduct a 3-item quality audit this week. Pull top complained items from inventory.",
                "outcome": "20–30% reduction in product complaints within 2 weeks",
                "priority": "HIGH"
            },
            "Service": {
                "icon": "🧑‍🤝‍🧑",
                "root_cause": "Staff response times or attitude issues at customer touchpoints",
                "action": "Run a 1-hour staff roleplay session. Introduce a customer satisfaction check-in.",
                "outcome": "+15% faster service = higher table turnover = direct revenue gain",
                "priority": "HIGH"
            },
            "Price": {
                "icon": "💰",
                "root_cause": "Perceived value gap — customers feel price doesn't match quality",
                "action": "Launch a 'Best Value Combo' bundle this week. Promote on social channels.",
                "outcome": "Increased spend-per-visit without cutting individual item prices",
                "priority": "MEDIUM"
            },
            "Ambience": {
                "icon": "🏠",
                "root_cause": "Environmental friction — noise, cleanliness, lighting reducing dwell time",
                "action": "Do daily opening walk-through checklist. Fix top 2 complaints this week.",
                "outcome": "Longer dwell time = larger basket size per visit",
                "priority": "MEDIUM"
            }
        }

        priority_colors = {"HIGH": ("#ef4444", "🔴"), "MEDIUM": ("#eab308", "🟡"), "LOW": ("#22c55e", "🟢")}

        aspects_with_negatives = []
        for asp_name, cfg in STRATEGY_DB.items():
            col_s = f"{asp_name} Sentiment"
            if col_s in result_df.columns:
                neg_c = (result_df[col_s] == "Negative").sum()
                total_c = result_df[col_s].notna().sum()
                if neg_c > 0:
                    aspects_with_negatives.append((asp_name, neg_c, total_c, cfg))

        if aspects_with_negatives:
            aspects_with_negatives.sort(key=lambda x: x[1], reverse=True)
            st.markdown("##### 📍 Problem-by-Problem Strategy Cards")
            for i, (asp_name, neg_c, total_c, cfg) in enumerate(aspects_with_negatives):
                pct = round(neg_c / total_c * 100, 1) if total_c > 0 else 0
                p_color, p_icon = priority_colors.get(cfg["priority"], ("#94a3b8", "⚪"))
                card_html = f"""
                <div style='background:#1e293b;border-left:4px solid {p_color};border-radius:10px;
                            padding:16px 20px;margin-bottom:14px;'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;'>
                        <div style='font-size:1.1rem;font-weight:700;color:#f8fafc;'>
                            {cfg['icon']} {asp_name} &nbsp;
                            <span style='font-size:0.78rem;background:#0f172a;padding:2px 8px;
                                        border-radius:8px;color:{p_color};'>{neg_c} negative ({pct}%)</span>
                        </div>
                        <span style='font-size:0.82rem;color:{p_color};font-weight:600;'>{p_icon} {cfg['priority']} PRIORITY</span>
                    </div>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;'>
                        <div style='background:#0f172a;border-radius:8px;padding:10px 12px;'>
                            <div style='color:#94a3b8;font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                        letter-spacing:0.04em;margin-bottom:4px;'>🔍 Root Cause</div>
                            <div style='color:#e2e8f0;font-size:0.85rem;'>{cfg['root_cause']}</div>
                        </div>
                        <div style='background:#0f172a;border-radius:8px;padding:10px 12px;'>
                            <div style='color:#94a3b8;font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                        letter-spacing:0.04em;margin-bottom:4px;'>⚡ Action This Week</div>
                            <div style='color:#38bdf8;font-size:0.85rem;'>{cfg['action']}</div>
                        </div>
                        <div style='background:#0f172a;border-radius:8px;padding:10px 12px;'>
                            <div style='color:#94a3b8;font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                        letter-spacing:0.04em;margin-bottom:4px;'>📈 Expected Outcome</div>
                            <div style='color:#22c55e;font-size:0.85rem;'>{cfg['outcome']}</div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.success("✅ No negative issues detected across all aspects. Business is performing well!")

        # ── Groq Deep Strategy ────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 🧠 Groq AI Deep Strategy Report")

        if not groq_api_key:
            st.markdown("""
            <div style='background:#1e293b;border:1px dashed #334155;border-radius:8px;padding:18px 22px;color:#94a3b8;'>
                🔑 <b>Enter your Groq API Key in the sidebar</b> to unlock a personalised AI strategy report.
                The AI will act as a business consultant and generate a structured 5-point action plan
                based on your exact data, complete with percentage impact estimates.
            </div>
            """, unsafe_allow_html=True)
        else:
            strat_prompt_extra = ""
            if aspects_with_negatives:
                strat_prompt_extra = " | ".join(
                    [f"{a}: {n} neg/{t} total ({round(n/t*100,1) if t>0 else 0}%)" for a, n, t, _ in aspects_with_negatives]
                )

            if st.button("✨ Generate Deep AI Strategy", use_container_width=True, key="deep_strategy_btn"):
                with st.spinner("🧠 AI consultant analysing your business data..."):
                    try:
                        client_s = Groq(api_key=groq_api_key)
                        deep_prompt = f"""
You are a senior business strategy consultant. A small business owner has just received
their customer review analysis. Here are the negative issue counts per aspect:
{strat_prompt_extra if strat_prompt_extra else 'No data yet'}
Overall Satisfaction Score: {round(overall_score, 2) if not pd.isna(overall_score) else 'N/A'}

Generate a structured 5-point business improvement strategy. For each point:
1. Name the Problem clearly
2. Explain the Root Cause in one sentence
3. Give one specific Action to take THIS WEEK
4. State the Expected Outcome with a % improvement estimate

Format each point as: Problem | Root Cause | Action | Expected Outcome
Be direct, data-driven, and encouraging. Start each point with an emoji.
"""
                        resp_s = client_s.chat.completions.create(
                            messages=[{"role": "user", "content": deep_prompt}],
                            model="llama-3.1-8b-instant"
                        )
                        strategy_output = resp_s.choices[0].message.content.strip()
                        st.session_state["ai_strategy_output"] = strategy_output
                    except Exception as e:
                        st.error(f"❌ Strategy error: {e}")

            if st.session_state.get("ai_strategy_output"):
                st.markdown(f"""
                <div style='background:#1e293b;border-left:4px solid #818cf8;border-radius:10px;
                            padding:20px 24px;margin-top:10px;color:#f8fafc;line-height:1.85;font-size:0.93rem;'>
                    <div style='color:#818cf8;font-weight:700;font-size:0.8rem;letter-spacing:0.05em;
                                text-transform:uppercase;margin-bottom:10px;'>🧠 AI Consultant Report</div>
                    {st.session_state['ai_strategy_output'].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

                # Download strategy
                strat_csv = pd.DataFrame([{"AI Strategy": st.session_state['ai_strategy_output']}])
                st.download_button(
                    "⬇️ Download Strategy as CSV",
                    strat_csv.to_csv(index=False).encode("utf-8"),
                    f"ai_strategy_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown("#### Model Confidence Analysis")
        conf_cols = [c for c in result_df.columns if "Confidence" in c]
        if conf_cols:
            all_conf = result_df[conf_cols].melt(value_name="Confidence").dropna()
            if not all_conf.empty:
                st.bar_chart(all_conf["Confidence"].value_counts(bins=5))
                
        if date_col and "Date" in result_df.columns:
            st.markdown("---")
            st.markdown("#### 📈 Negative Sentiment Trend")
            
            # Format date column and extract negatives over time
            trend_df = result_df.copy()
            trend_df["Date"] = pd.to_datetime(trend_df["Date"], errors='coerce')
            trend_df = trend_df.dropna(subset=["Date"])
            
            if not trend_df.empty:
                # Count negative sentiments across all tracked aspects per Date
                trend_df["Total Negatives"] = trend_df.apply(lambda r: sum(1 for c in trend_df.columns if "Sentiment" in c and r[c] == "Negative"), axis=1)
                trend_grouped = trend_df.groupby(trend_df["Date"].dt.date)["Total Negatives"].sum()
                
                if not trend_grouped.empty:
                    st.line_chart(trend_grouped)
                else:
                    st.write("No negative trends found over time.")
            else:
                 st.write("Could not parse dates for trend analysis.")

        # --------------------------------------------------
        # 🔍 ERROR / LOW-CONFIDENCE ANALYSIS SECTION
        # --------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔍 Prediction Confidence Analysis")

        # Rating-Aware Calibration
        if "rating" in df_clean.columns:
            st.markdown("#### ⭐ Rating-Sentiment Mismatch")
            mismatch_count = 0
            for i in range(limit):
                try:
                    rating_val = float(df_clean.iloc[i].get("rating", 3))
                    row_r = results[i]
                    # Check if any aspect is Positive but rating is ≤ 2
                    for asp in ["Product", "Service", "Price", "Ambience"]:
                        if row_r.get(f"{asp} Sentiment") == "Positive" and rating_val <= 2:
                            mismatch_count += 1
                            break
                except (ValueError, TypeError):
                    pass
            if mismatch_count > 0:
                st.error(f"⚠️ Sentiment mismatch with rating detected in {mismatch_count} review(s). Low rating + Positive prediction may indicate model overconfidence.")
            else:
                st.success("✅ No rating-sentiment mismatches detected.")

        # Confidence distribution across all aspects
        conf_all = result_df[[c for c in result_df.columns if "Confidence" in c]].melt(value_name="Confidence").dropna()
        if not conf_all.empty:
            low_conf_count = (conf_all["Confidence"] < 0.6).sum()
            if low_conf_count > 0:
                st.warning(f"⚠️ {low_conf_count} predictions have low confidence (< 0.60). Review these carefully.")
            else:
                st.info("All predictions have confidence ≥ 0.60.")

        # ══════════════════════════════════════════════════
        # 🤖 SMART BUSINESS COPILOT — MODULES 1, 2, 3
        # ══════════════════════════════════════════════════

        # ── MODULE 1: TOP PROBLEM DETECTOR ──────────────
        st.markdown("---")
        st.markdown("### 📊 Auto Top-Problem Detector")
        leaderboard, keyword_clusters = detect_top_problems(result_df, df_reviews=df_clean)
        if leaderboard:
            lb_df = pd.DataFrame(leaderboard)
            for i, row_pb in enumerate(leaderboard):
                col_l, col_r = st.columns([3, 1])
                with col_l:
                    bar_pct = int(row_pb["Severity %"])
                    bar_html = f"""
                    <div style='background:#1e293b;border-radius:8px;padding:10px 14px;margin-bottom:8px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                            <span style='font-weight:700;color:#f8fafc;font-size:1rem;'>#{i+1} {row_pb['Aspect']}</span>
                            <span style='font-size:1.1rem;'>{row_pb['Status']}</span>
                        </div>
                        <div style='background:#0f172a;border-radius:4px;height:10px;'>
                            <div style='background:{'#ef4444' if bar_pct>=60 else '#eab308' if bar_pct>=30 else '#22c55e'};width:{bar_pct}%;height:10px;border-radius:4px;'></div>
                        </div>
                        <div style='margin-top:4px;color:#94a3b8;font-size:0.82rem;'>{row_pb['Negatives']} negative / {row_pb['Total Mentions']} total mentions &nbsp;|&nbsp; <b>{bar_pct}%</b> severity</div>
                    </div>
                    """
                    st.markdown(bar_html, unsafe_allow_html=True)
            if keyword_clusters:
                st.markdown("**🔑 Top Recurring Complaint Words:**")
                kw_html = " ".join(
                    [f"<span style='background:#334155;color:#f8fafc;padding:3px 10px;border-radius:12px;margin:2px;font-size:0.85rem;'>{w} ({n}×)</span>"
                     for w, n in keyword_clusters.items()]
                )
                st.markdown(f"<div style='line-height:2.2;'>{kw_html}</div>", unsafe_allow_html=True)
        else:
            st.info("Not enough aspect data to build problem leaderboard.")

        # ── MODULE 2: CHURN RISK PREDICTOR ──────────────
        st.markdown("---")
        st.markdown("### 🧠 Customer Churn Risk Predictor")
        result_df_churn, high_c, medium_c, low_c = predict_churn_risk(result_df)
        total_c = high_c + medium_c + low_c
        if total_c > 0:
            high_pct   = round(high_c / total_c * 100, 1)
            medium_pct = round(medium_c / total_c * 100, 1)
            low_pct    = round(low_c / total_c * 100, 1)

            cr1, cr2, cr3 = st.columns(3)
            cr1.metric("🔴 High Risk",   f"{high_c} reviews",   f"{high_pct}% of total")
            cr2.metric("🟡 Medium Risk", f"{medium_c} reviews", f"{medium_pct}% of total")
            cr3.metric("🟢 Low Risk",    f"{low_c} reviews",    f"{low_pct}% of total")

            if high_pct >= 30:
                st.error(f"🚨 **{high_pct}%** of customers are HIGH churn risk — immediate intervention recommended.")
            elif high_pct >= 15:
                st.warning(f"⚠️ {high_pct}% high churn risk detected. Monitor closely and act on top problems.")
            else:
                st.success(f"✅ Churn risk is manageable. {low_pct}% of customers show positive signals.")

            with st.expander("🔍 View per-review Churn Risk Scores"):
                churn_display = result_df_churn[["Original Review", "Churn Risk Score", "Churn Risk Level"]]
                st.dataframe(churn_display, use_container_width=True)
        else:
            st.info("Run bulk analysis above to see churn predictions.")
            high_pct = 0

        # ── MODULE 3: REVENUE ACTIONS ENGINE ──────────────
        st.markdown("---")
        st.markdown("### 📈 Revenue Improvement Actions")
        high_pct_for_rev = round(high_c / total_c * 100, 1) if total_c > 0 else 0
        revenue_acts = generate_revenue_actions(result_df, high_churn_pct=high_pct_for_rev)

        if revenue_acts:
            for act in revenue_acts:
                act_html = f"""
                <div style='background:#1e293b;border-left:4px solid {'#ef4444' if act['neg_pct']>=60 else '#eab308'};border-radius:8px;padding:14px 18px;margin-bottom:12px;'>
                    <div style='font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:6px;'>{act['icon']} {act['aspect']} &nbsp;<span style='font-size:0.8rem;background:#0f172a;padding:2px 8px;border-radius:8px;color:#ef4444;'>{act['neg_pct']}% negative</span></div>
                    <div style='color:#cbd5e1;margin-bottom:8px;'>{act['action']}</div>
                    <div style='color:#38bdf8;font-size:0.85rem;'>⚡ Quick Win: {act['quick_win']}</div>
                </div>
                """
                st.markdown(act_html, unsafe_allow_html=True)
        else:
            st.success("✅ No critical revenue risks detected. All major areas are performing well.")

        # Save data to session state for Tab 4
        health = compute_health_score(result_df)
        st.session_state["copilot_result_df"]       = result_df
        st.session_state["copilot_leaderboard"]     = leaderboard
        st.session_state["copilot_revenue_actions"] = revenue_acts
        st.session_state["copilot_health_score"]    = health

# ================= TAB 3 =================
with tab3:
    st.subheader("📊 Model Evaluation (ABSA Test Dataset)")
    st.markdown("Upload labeled ground truth data to evaluate the BERT model performance.")

    c1, c2 = st.columns([1, 2])
    with c1:
        eval_file = st.file_uploader(
            "Upload ABSA Test CSV",
            type=["csv"],
            key="eval_upload"
        )

    if eval_file is not None:
        test_df = pd.read_csv(eval_file)

        required_cols = {"review", "aspect", "true_sentiment"}
        if not required_cols.issubset(test_df.columns):
            st.error("CSV must contain: review, aspect, true_sentiment")
            st.stop()

        st.info(f"Loaded **{len(test_df)}** labeled samples ready for evaluation.")

        if st.button("🚀 Run Large-Scale Evaluation", use_container_width=True):
            with st.spinner("Evaluating model..."):
                accuracy, precision, recall, f1, cm, misclassified_df = evaluate_absa(test_df)

            st.success("✅ Evaluation Complete")

            st.markdown("---")
            st.markdown("### 🎯 Model Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy:.2f}")
            m2.metric("Precision", f"{precision:.2f}")
            m3.metric("Recall", f"{recall:.2f}")
            m4.metric("F1 Score", f"{f1:.2f}")

            st.markdown("---")
            col_matrix, col_misc = st.columns([1, 1.3])
            
            with col_matrix:
                st.markdown("### 🧮 Confusion Matrix")
                cm_df = pd.DataFrame(
                    cm,
                    index=["True Pos", "True Neu", "True Neg"],
                    columns=["Pred Pos", "Pred Neu", "Pred Neg"]
                )
                st.dataframe(cm_df, use_container_width=True)

            with col_misc:
                st.markdown("### 🚨 Misclassified Samples")
                if not misclassified_df.empty:
                    st.warning(f"Total Incorrect Predictions: {len(misclassified_df)}")
                    st.dataframe(misclassified_df, use_container_width=True)

                    # Error Analysis Dashboard
                    st.markdown("### 🔍 Error Analysis")
                    st.write("Common Mistakes by Aspect:")
                    st.bar_chart(misclassified_df["Aspect"].value_counts())

                    # Low Confidence Detection
                    low_conf = misclassified_df[misclassified_df["Confidence"] < 0.6]
                    if not low_conf.empty:
                        st.warning(f"⚠️ {len(low_conf)} misclassified predictions also had low confidence (< 0.60). Model was uncertain AND wrong on these.")
                        st.dataframe(low_conf, use_container_width=True)
                else:
                    st.success("Wow! 100% Accuracy on this batch. No misclassified samples.")

# NOTE (Speed Optimization — Future Work):
# Currently reviews are processed one at a time (sequential inference).
# For large datasets, batch processing would significantly improve speed:
#   inputs = tokenizer(list_of_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
# This can reduce total inference time by up to 5-10× on GPU.

# ================= TAB 4: WEEKLY AI REPORT =================
with tab4:
    st.markdown("### 🎯 Weekly AI Business Report")
    st.markdown("Your automated *'What to Fix This Week'* executive brief — powered by BERT analysis + Groq AI.")

    rdf    = st.session_state.get("copilot_result_df")
    lb     = st.session_state.get("copilot_leaderboard", [])
    r_acts = st.session_state.get("copilot_revenue_actions", [])
    health = st.session_state.get("copilot_health_score")

    if rdf is None:
        st.info("📂 **No data yet.** Please upload a CSV in the **Bulk Upload** tab first, then return here.")
        st.markdown("""
        <div style='background:#1e293b;border-radius:12px;padding:30px;text-align:center;margin-top:20px;'>
            <div style='font-size:3rem;'>📋</div>
            <div style='color:#94a3b8;margin-top:10px;'>
                Upload reviews in <b>Tab 2 → Bulk Upload</b> to unlock your Weekly AI Report
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── BUSINESS HEALTH SCORE ─────────────────────────────
        st.markdown("---")
        st.markdown("#### 💚 Business Health Score")

        if health is not None:
            score_color = "#22c55e" if health >= 65 else ("#eab308" if health >= 40 else "#ef4444")
            score_label = "Healthy 🟢" if health >= 65 else ("Needs Work 🟡" if health >= 40 else "Critical 🔴")

            h1, h2 = st.columns([1, 2])
            with h1:
                st.markdown(f"""
                <div style='background:#1e293b;border-radius:12px;padding:24px;text-align:center;'>
                    <div style='font-size:3.5rem;font-weight:800;color:{score_color};'>{health}</div>
                    <div style='color:#94a3b8;font-size:0.9rem;'>/100</div>
                    <div style='margin-top:8px;font-size:1rem;color:#f8fafc;'>{score_label}</div>
                </div>
                """, unsafe_allow_html=True)

            with h2:
                # Score breakdown bar
                pos_pct = health
                rem_pct  = 100 - health
                st.markdown(f"""
                <div style='background:#1e293b;border-radius:12px;padding:20px;height:100%;'>
                    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:8px;'>Score Breakdown (based on all aspect sentiments)</div>
                    <div style='background:#0f172a;border-radius:6px;height:18px;margin-bottom:12px;'>
                        <div style='background:{score_color};width:{health}%;height:18px;border-radius:6px;transition:width 0.5s;'></div>
                    </div>
                    <div style='color:#cbd5e1;font-size:0.85rem;'>
                        Formula: <code>(Positive×1 + Neutral×0.5 + Negative×0) / Total × 100</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Could not compute health score — no sentiment data found in results.")

        # ── WHAT TO FIX THIS WEEK (Priority List) ────────────
        st.markdown("---")
        st.markdown("#### 📌 What To Fix This Week")

        if lb:
            priority_items = []
            for i, p in enumerate(lb[:4]):
                if p["Severity %"] >= 30:
                    priority_items.append({
                        "rank": i + 1,
                        "text": f"**{p['Aspect']}** — {p['Severity %']}% of reviews are negative ({p['Negatives']}/{p['Total Mentions']} mentions)",
                        "badge": p["Status"]
                    })

            if r_acts:
                for act in r_acts[:3]:
                    priority_items.append({
                        "rank": len(priority_items) + 1,
                        "text": f"**{act['aspect']} Action** — {act['quick_win']}",
                        "badge": "⚡ Quick Win"
                    })

            if priority_items:
                for item in priority_items[:5]:
                    st.markdown(f"""
                    <div style='background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px;'>
                        <div style='background:#38bdf8;color:#0f172a;font-weight:800;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;flex-shrink:0;'>{item['rank']}</div>
                        <div style='flex:1;color:#f8fafc;'>{item['text']}</div>
                        <div style='font-size:0.8rem;color:#94a3b8;white-space:nowrap;'>{item['badge']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No urgent fixes needed this week. Your business is performing well!")
        else:
            st.info("Problem data not yet computed. Ensure Bulk Analysis has run.")

        # ── GROQ AI WEEKLY BRIEF ──────────────────────────────
        st.markdown("---")
        st.markdown("#### 🤖 AI-Generated Weekly Brief")

        if not groq_api_key:
            st.warning("⚠️ Enter your Groq API Key in the sidebar to unlock the AI brief.")
            st.markdown("""
            <div style='background:#1e293b;border-radius:8px;padding:16px;color:#94a3b8;font-size:0.9rem;'>
                The AI brief will give you a personalised 5-point action plan based on your
                actual review data — written like an expert business consultant.
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("✨ Generate This Week's AI Report", use_container_width=True):
                with st.spinner("Consulting AI analyst... analysing your business data..."):
                    brief = generate_weekly_report_groq(
                        rdf, health or 0, lb, r_acts, groq_api_key
                    )
                if brief:
                    st.session_state["weekly_brief"] = brief

            if st.session_state.get("weekly_brief"):
                st.markdown(f"""
                <div style='background:#1e293b;border-left:4px solid #38bdf8;border-radius:8px;padding:20px;margin-top:8px;color:#f8fafc;line-height:1.8;'>
                {st.session_state['weekly_brief'].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

        # ── DOWNLOAD WEEKLY REPORT ────────────────────────────
        st.markdown("---")
        st.markdown("#### ⬇️ Download Weekly Report Data")

        report_rows = []
        report_rows.append({"Metric": "Business Health Score", "Value": health or "N/A", "Notes": ""})
        for p in lb:
            report_rows.append({
                "Metric": f"Problem — {p['Aspect']}",
                "Value": f"{p['Severity %']}% negative",
                "Notes": p["Status"]
            })
        for act in r_acts:
            report_rows.append({
                "Metric": f"Revenue Action — {act['aspect']}",
                "Value": f"{act['neg_pct']}% neg",
                "Notes": act["quick_win"]
            })
        if st.session_state.get("weekly_brief"):
            report_rows.append({
                "Metric": "AI Brief",
                "Value": st.session_state["weekly_brief"][:200] + "...",
                "Notes": "Full brief available in app"
            })

        report_df = pd.DataFrame(report_rows)
        week_str  = datetime.datetime.now().strftime("%Y-W%V")
        st.download_button(
            f"⬇️ Download Report — Week {week_str}",
            report_df.to_csv(index=False).encode("utf-8"),
            f"weekly_report_{week_str}.csv",
            "text/csv",
            use_container_width=True
        )

# NOTE (Speed Optimization — Future Work):
# Currently reviews are processed one at a time (sequential inference).
# For large datasets, batch processing would significantly improve speed:
#   inputs = tokenizer(list_of_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
# This can reduce total inference time by up to 5-10× on GPU.

# ================= TAB 5: SOCIAL MEDIA MONITOR =================
with tab5:
    st.markdown("### 📡 Real-Time Social Media Monitor")
    st.markdown("Track brand reputation live — analyse tweets, app reviews, and social posts with BERT in real time.")

    # ── AUTO REFRESH (if package available) ────────────────────
    is_auto_trigger = False
    if AUTOREFRESH_OK:
        refresh_interval_ms = st.sidebar.select_slider(
            "⏱️ Auto-Refresh Interval",
            options=[0, 30_000, 60_000, 300_000],
            format_func=lambda x: "Off" if x == 0 else f"{x//1000}s",
            value=0,
            key="monitor_refresh_interval"
        )
        if refresh_interval_ms > 0:
            if "last_auto_counter" not in st.session_state:
                st.session_state["last_auto_counter"] = 0
            curr_counter = st_autorefresh(interval=refresh_interval_ms, key="monitor_autorefresh")
            if curr_counter > st.session_state["last_auto_counter"]:
                is_auto_trigger = True
                st.session_state["last_auto_counter"] = curr_counter

    # ── SOURCE SELECTOR ─────────────────────────────────────────
    mon_col1, mon_col2 = st.columns([1, 2])

    with mon_col1:
        st.markdown("#### ⚙️ Monitor Settings")

        source = st.selectbox(
            "📡 Data Source",
            ["🎭 Simulation Mode", "🐦 Twitter / X", "🏪 Google Play Reviews", "📋 Instagram / Manual Paste"],
            key="monitor_source"
        )

        brand_keyword = st.text_input(
            "🔍 Brand / Keyword",
            placeholder="e.g. @YourBrand or BurgerPalace",
            key="monitor_keyword"
        )

        batch_size = st.slider("📦 Batch size per refresh", 3, 20, 5, key="monitor_batch")
        alert_threshold = st.slider(
            "🚨 Alert threshold (% negative)",
            0, 100, 40,
            help="Fire an alert if negative% in latest batch exceeds this value",
            key="monitor_alert_thresh"
        )

        # Source-specific credentials / settings
        st.markdown("---")

        if "Twitter" in source:
            bearer_token = st.text_input(
                "🔑 Twitter Bearer Token",
                type="password",
                key="twitter_bearer",
                help="Get from developer.twitter.com → Your App → Keys & Tokens"
            )
            st.markdown("[Get Twitter API keys →](https://developer.twitter.com/en/portal/dashboard)",
                        unsafe_allow_html=False)
            if not TWEEPY_OK:
                st.warning("⚠️ `tweepy` not installed. Run: `pip install tweepy`")

        elif "Google Play" in source:
            gplay_app_id = st.text_input(
                "📱 Google Play App ID",
                placeholder="e.g. com.ubercab or com.spotify.music",
                key="gplay_app_id",
                help="The package name visible in the Play Store URL"
            )
            if not GPLAY_OK:
                st.warning("⚠️ `google-play-scraper` not installed. Run: `pip install google-play-scraper`")

        elif "Instagram" in source or "Manual" in source or "Paste" in source:
            paste_text = st.text_area(
                "📋 Paste reviews / comments (one per line)",
                placeholder="The food was amazing!\nService was very slow...\nGreat ambience but overpriced.",
                height=130,
                key="monitor_paste_text"
            )
            st.caption("✅ Works offline — no API keys needed")

        elif "Simulation" in source:
            st.info("Uses your Tab 2 uploaded CSV as a live replay source. No API keys required.")
            sim_reviews_pool = []
            if st.session_state.get("copilot_result_df") is not None:
                sim_reviews_pool = st.session_state["copilot_result_df"]["Original Review"].dropna().tolist()
                st.success(f"✅ {len(sim_reviews_pool)} reviews loaded from your uploaded CSV")
            else:
                st.warning("🔶 No CSV uploaded yet — go to Tab 2 and upload a CSV first")

        st.markdown("---")
        refresh_btn = st.button("🔄 Fetch & Analyse Now", use_container_width=True, key="monitor_refresh_btn")
        clear_btn   = st.button("🗑️ Clear Feed", use_container_width=True, key="monitor_clear_btn")

        if clear_btn:
            st.session_state["monitor_feed"]   = []
            st.session_state["monitor_trend"]  = []
            st.session_state["monitor_alerts"] = []
            st.rerun()

    # ── LIVE FEED & ANALYTICS ────────────────────────────────────
    with mon_col2:
        # ─ FETCH ─
        if refresh_btn or is_auto_trigger:
            texts = []
            fetch_error = None

            with st.spinner("🔍 Fetching & analysing with BERT..."):
                if "Twitter" in source:
                    bt = st.session_state.get("twitter_bearer", "")
                    texts, fetch_error = fetch_tweets(brand_keyword or "product review", bt, batch_size)

                elif "Google Play" in source:
                    app_id_val = st.session_state.get("gplay_app_id", "")
                    texts, fetch_error = fetch_google_play_reviews(app_id_val, count=batch_size)

                elif "Instagram" in source or "Paste" in source or "Manual" in source:
                    raw = st.session_state.get("monitor_paste_text", "")
                    texts, fetch_error = fetch_paste_reviews(raw)
                    texts = texts[:batch_size]

                else:  # Simulation
                    pool = []
                    if st.session_state.get("copilot_result_df") is not None:
                        pool = st.session_state["copilot_result_df"]["Original Review"].dropna().tolist()
                    texts, fetch_error = simulate_live_reviews(pool, batch_size)

            if fetch_error:
                st.error(f"❌ {fetch_error}")
            elif texts:
                new_items, pos_n, neu_n, neg_n = run_monitor_analysis(texts, brand_keyword)

                # Append to rolling feed (keep last 50)
                st.session_state["monitor_feed"] = (new_items + st.session_state["monitor_feed"])[:50]

                # Append to trend history
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state["monitor_trend"].append({
                    "Time": ts, "Positive": pos_n, "Neutral": neu_n, "Negative": neg_n
                })

                # Check alert threshold
                total_batch = pos_n + neu_n + neg_n
                neg_pct_batch = (neg_n / total_batch * 100) if total_batch > 0 else 0
                if neg_pct_batch > alert_threshold:
                    alert_msg = (
                        f"🚨 [{ts}] NEGATIVE SPIKE: {neg_pct_batch:.0f}% negative "
                        f"in last batch of {total_batch} items "
                        f"(threshold: {alert_threshold}%)"
                    )
                    st.session_state["monitor_alerts"].insert(0, alert_msg)

        # ─ ALERTS ─
        if st.session_state["monitor_alerts"]:
            for alert_msg in st.session_state["monitor_alerts"][:3]:
                st.error(alert_msg)

        # ─ FEED CARDS ─
        feed = st.session_state["monitor_feed"]
        if feed:
            st.markdown(f"#### 📬 Live Feed &nbsp; <span style='color:#94a3b8;font-size:0.85rem;'>({len(feed)} items in session)</span>", unsafe_allow_html=True)

            for item in feed[:15]:
                s = item["sentiment"]
                if s == "Negative":
                    border, icon, s_color = "#ef4444", "📉", "#ef4444"
                elif s == "Positive":
                    border, icon, s_color = "#22c55e", "📈", "#22c55e"
                else:
                    border, icon, s_color = "#eab308", "➖", "#eab308"

                lang_badge = f"<span style='background:#334155;padding:1px 7px;border-radius:8px;font-size:0.75rem;color:#94a3b8;'>{item['lang']}</span>" if item["lang"] != "en" else ""
                card_html = f"""
                <div style='background:#1e293b;border-left:4px solid {border};border-radius:8px;
                            padding:10px 14px;margin-bottom:8px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
                        <span style='color:#94a3b8;font-size:0.78rem;'>⏱ {item['time']} &nbsp;|&nbsp; {item['aspect']} &nbsp;{lang_badge}</span>
                        <span style='color:{s_color};font-weight:700;font-size:0.9rem;'>{icon} {s} &nbsp;<span style='color:#64748b;font-size:0.78rem;'>({item['confidence']:.0%})</span></span>
                    </div>
                    <div style='color:#e2e8f0;font-size:0.9rem;line-height:1.45;'>{item['text']}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

        # ─ TREND CHART ─
        trend = st.session_state["monitor_trend"]
        if len(trend) >= 2:
            st.markdown("---")
            st.markdown("#### 📉 Sentiment Trend (per refresh batch)")
            trend_df = pd.DataFrame(trend).set_index("Time")
            st.line_chart(trend_df, color=["#22c55e", "#eab308", "#ef4444"])

            # Business Health Score sparkline
            total_counts = trend_df.sum()
            total_all    = total_counts.sum()
            if total_all > 0:
                live_health = round(
                    (total_counts["Positive"] * 1.0 + total_counts["Neutral"] * 0.5) / total_all * 100, 1
                )
                h_color = "#22c55e" if live_health >= 65 else ("#eab308" if live_health >= 40 else "#ef4444")
                st.markdown(f"""
                <div style='background:#1e293b;border-radius:8px;padding:12px 18px;margin-top:6px;display:flex;align-items:center;gap:16px;'>
                    <div style='font-size:2rem;font-weight:800;color:{h_color};'>{live_health}</div>
                    <div>
                        <div style='color:#f8fafc;font-size:0.9rem;font-weight:600;'>Live Session Health Score</div>
                        <div style='color:#64748b;font-size:0.78rem;'>Across {int(total_all)} analysed items this session</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif not feed:
            st.markdown("""
            <div style='background:#1e293b;border-radius:12px;padding:40px;text-align:center;'>
                <div style='font-size:3rem;'>📡</div>
                <div style='color:#94a3b8;margin-top:10px;font-size:0.95rem;'>
                    Select a source, set your keyword, then click <b>Fetch &amp; Analyse Now</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ─ DOWNLOAD FEED ─
        if feed:
            st.markdown("---")
            feed_csv = pd.DataFrame(feed).to_csv(index=False).encode("utf-8")
            ts_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ Download Session Feed CSV",
                feed_csv,
                f"monitor_feed_{ts_str}.csv",
                "text/csv",
                use_container_width=True,
                key="monitor_download"
            )

# ================= TAB 6: COMPETITOR COMPARISON =================
with tab6:
    st.markdown("### ⚔️ Competitor Comparison Dashboard")
    st.markdown("Measure your brand perception precisely against your top competitor.")
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("#### 🏪 Your Business Baseline")
        if st.session_state.get("copilot_result_df") is None:
            st.warning("Please upload your data in Tab 2 (Bulk Upload) to establish your baseline.")
            my_df = None
        else:
            my_df = st.session_state["copilot_result_df"]
            my_score_map = {"Positive": 1, "Neutral": 0, "Negative": -1, "Uncertain": 0}
            temp_my_df = my_df[[c for c in my_df.columns if "Sentiment" in c]].replace(my_score_map)
            my_overall_score = temp_my_df.apply(pd.to_numeric, errors='coerce').mean(axis=1).mean()
            st.success(f"Loaded {len(my_df)} baseline reviews.")
            st.metric("Your Aggregated Score", round(my_overall_score, 2))
            
    with comp_col2:
        st.markdown("#### 🏙️ Competitor Feedback")
        competitor_name = st.text_input("Competitor Name", "Burger King")
        comp_reviews = st.text_area("Paste competitor review samples (1 per line)", "The food was okay but expensive.\nTerrible customer service.", height=150)
        
        if st.button("🚀 Compare Sentiment", use_container_width=True):
            if comp_reviews.strip() and my_df is not None:
                with st.spinner(f"Analysing {competitor_name} with BERT..."):
                    comp_texts = [r.strip() for r in comp_reviews.split('\n') if r.strip()]
                    b_texts = []
                    b_aspects = []
                    for t in comp_texts:
                        asps = get_relevant_aspects(t)
                        for asp in asps:
                            b_texts.append(t)
                            b_aspects.append(asp)
                    
                    if b_texts:
                        b_results = batch_analyze_aspects(b_texts, b_aspects)
                        
                        comp_scores = {"Product": [], "Service": [], "Price": [], "Ambience": []}
                        total_sent = 0
                        for (ctx, casp), (sent, _, _, _, _, _) in zip(zip(b_texts, b_aspects), b_results):
                            cap = casp.capitalize()
                            val = 1 if sent == "Positive" else (0 if sent == "Neutral" else -1)
                            if cap in comp_scores:
                                comp_scores[cap].append(val)
                            total_sent += val
                                
                        comp_overall = total_sent / len(b_results)
                        st.metric(f"{competitor_name} Aggregated Score", round(comp_overall, 2))
                        
                        # Radar Chart logic
                        st.markdown(f"#### 📊 You vs. {competitor_name}")
                        categories = ['Product', 'Service', 'Price', 'Ambience']
                        
                        my_cat_scores = []
                        for a in categories:
                            if f"{a} Sentiment" in my_df.columns:
                                val = pd.to_numeric(my_df[f"{a} Sentiment"].replace(my_score_map), errors='coerce').mean()
                                my_cat_scores.append(val if not pd.isna(val) else 0)
                            else:
                                my_cat_scores.append(0)
                        
                        comp_cat_scores = []
                        for a in categories:
                            if comp_scores[a]:
                                comp_cat_scores.append(sum(comp_scores[a])/len(comp_scores[a]))
                            else:
                                comp_cat_scores.append(0)
                            
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=my_cat_scores,
                            theta=categories,
                            fill='toself',
                            name='You',
                            line_color='#38bdf8'
                        ))
                        fig.add_trace(go.Scatterpolar(
                            r=comp_cat_scores,
                            theta=categories,
                            fill='toself',
                            name=competitor_name,
                            line_color='#ef4444'
                        ))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                            showlegend=True,
                            template="plotly_dark",
                            paper_bgcolor="#0f172a",
                            plot_bgcolor="#0f172a"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Price perception delta
                        my_price = my_cat_scores[2]
                        comp_price = comp_cat_scores[2]
                        st.info(f"**Pricing Insight**: Your price perception is **{'better' if my_price > comp_price else 'worse'}** than {competitor_name} by {round(abs(my_price - comp_price), 2)} points.")
