"""
Football Sentiment Analysis — Improved Flask App
Features:
  - Live prediction: type any text and get instant sentiment + confidence
  - Dashboard: sentiment breakdown charts per player
  - Player filter: view results for individual players
  - Uses the trained models saved by model_training.py
  - Graceful fallback if models aren't trained yet
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import re
import json
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

app = Flask(__name__)

# ── LOAD MODELS ──────────────────────────────────────────────
MODELS_DIR = "saved_models"

def load_models():
    models  = {}
    encoder = None
    tfidf_word = None
    tfidf_char = None
    ready   = False

    try:
        tfidf_word = joblib.load(os.path.join(MODELS_DIR, "tfidf_word.pkl"))
        tfidf_char = joblib.load(os.path.join(MODELS_DIR, "tfidf_char.pkl"))
        encoder    = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        for name in ["Logistic_Regression", "Linear_SVM",
                     "Random_Forest", "Gradient_Boosting"]:
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            if os.path.exists(path):
                models[name.replace("_", " ")] = joblib.load(path)
        ready = len(models) > 0
    except Exception as e:
        print(f"Models not loaded: {e}")

    return models, encoder, tfidf_word, tfidf_char, ready

models, encoder, tfidf_word, tfidf_char, MODELS_READY = load_models()

# ── TEXT CLEANING (mirrors cleaning.py) ──────────────────────
SLANG = {
    "goat": "greatest of all time", "lfg": "lets go", "w": "win",
    "l": "loss", "bottled": "failed under pressure", "baller": "excellent player",
    "worldie": "spectacular goal", "howler": "bad mistake", "sitter": "easy missed chance",
    "flop": "disappointing player", "fraud": "disappointing player", "elite": "top quality",
}

stop_words  = set(stopwords.words('english')) - {"no", "not", "never", "nor"}
lemmatizer  = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    tokens = [SLANG.get(t, t) for t in text.split()]
    text   = ' '.join(tokens)
    text   = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text   = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

def predict(text, model_name="Logistic Regression"):
    if not MODELS_READY:
        return None, None, None
    cleaned = clean_text(text)
    X_word  = tfidf_word.transform([cleaned])
    X_char  = tfidf_char.transform([cleaned])
    X       = hstack([X_word, X_char])
    model   = models.get(model_name)
    if model is None:
        return None, None, None
    pred    = model.predict(X)[0]
    label   = encoder.inverse_transform([pred])[0]
    # Confidence: use decision_function if available, else None
    conf    = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        conf  = round(float(np.max(proba)) * 100, 1)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)[0]
        exp    = np.exp(scores - np.max(scores))
        proba  = exp / exp.sum()
        conf   = round(float(np.max(proba)) * 100, 1)
    return label, conf, cleaned

# ── LOAD DATA ────────────────────────────────────────────────
def load_data():
    for fname in ["final_cleaned_sentiment_analysis.csv",
                  "final_sentiment_analysis.csv"]:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            df = df.fillna('')
            return df
    return pd.DataFrame(columns=['Text', 'cleaned_text', 'Sentiment', 'Player', 'Source'])

df_global = load_data()

def get_stats(df):
    if df.empty or 'Sentiment' not in df.columns:
        return {}, [], []
    counts  = {k.lower(): v for k, v in df['Sentiment'].value_counts().to_dict().items()}
    players = sorted(df['Player'].unique().tolist()) if 'Player' in df.columns else []
    sources = sorted(df['Source'].unique().tolist()) if 'Source' in df.columns else []
    return counts, players, sources

# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def index():
    player = request.args.get('player', 'All')
    source = request.args.get('source', 'All')
    df     = df_global.copy()

    if player != 'All' and 'Player' in df.columns:
        df = df[df['Player'] == player]
    if source != 'All' and 'Source' in df.columns:
        df = df[df['Source'] == source]

    counts, players, sources = get_stats(df_global)
    # Normalise keys to lowercase so lookups work regardless of source capitalisation
    filtered_counts = {k.lower(): v for k, v in
                       df['Sentiment'].value_counts().to_dict().items()} if not df.empty else {}

    total    = sum(filtered_counts.values()) if filtered_counts else 0
    pos_pct  = round(filtered_counts.get('positive', 0) / total * 100, 1) if total else 0
    neg_pct  = round(filtered_counts.get('negative', 0) / total * 100, 1) if total else 0
    neu_pct  = round(filtered_counts.get('neutral',  0) / total * 100, 1) if total else 0

    # Recent records for the table (last 50)
    records = df.tail(50).to_dict(orient='records')
    records.reverse()

    return render_template(
        "index.html",
        records        = records,
        counts         = filtered_counts,
        players        = players,
        sources        = sources,
        selected_player = player,
        selected_source = source,
        total          = total,
        pos_pct        = pos_pct,
        neg_pct        = neg_pct,
        neu_pct        = neu_pct,
        models_ready   = MODELS_READY,
        model_names    = list(models.keys()),
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    data       = request.get_json()
    text       = data.get("text", "")
    model_name = data.get("model", "Logistic Regression")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    label, conf, cleaned = predict(text, model_name)

    if label is None:
        return jsonify({"error": "Models not trained yet. Run model_training.py first."}), 503

    return jsonify({
        "sentiment":    label,
        "confidence":   conf,
        "cleaned_text": cleaned,
    })


@app.route("/stats")
def stats():
    player = request.args.get('player', 'All')
    df     = df_global.copy()
    if player != 'All' and 'Player' in df.columns:
        df = df[df['Player'] == player]

    if df.empty or 'Sentiment' not in df.columns:
        return jsonify({})

    result = {
        "counts": df['Sentiment'].value_counts().to_dict(),
        "total":  len(df),
    }
    if 'Player' in df.columns:
        per_player = {}
        for p, grp in df.groupby('Player'):
            per_player[p] = grp['Sentiment'].value_counts().to_dict()
        result["per_player"] = per_player

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
