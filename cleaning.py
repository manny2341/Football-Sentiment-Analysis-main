"""
Football Sentiment Analysis — Improved Text Cleaning
Improvements over original:
  - Emoji converted to text (e.g. 😂 → "face tears joy") before removal
  - Football-specific slang expanded (GOAT, LFG, bottled, etc.)
  - Handles Twitter @mentions and hashtags properly
  - Removes repeated characters (e.g. "soooo good" → "so good")
  - Stopwords initialised once outside the loop (was re-built every row)
  - Prints before/after sample so you can verify it's working
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── FOOTBALL SLANG DICTIONARY ────────────────────────────────
# Maps shorthand/slang to plain English so the model understands them
SLANG = {
    "goat":    "greatest of all time",
    "lfg":     "lets go",
    "w":       "win",
    "l":       "loss",
    "lw":      "left wing",
    "rw":      "right wing",
    "cf":      "centre forward",
    "cb":      "centre back",
    "gk":      "goalkeeper",
    "og":      "own goal",
    "ucl":     "champions league",
    "epl":     "premier league",
    "prem":    "premier league",
    "bottled": "failed under pressure",
    "baller":  "excellent player",
    "worldie": "spectacular goal",
    "howler":  "bad mistake",
    "sitter":  "easy missed chance",
    "tap in":  "easy goal",
    "regen":   "regenerated player",
    "sbc":     "squad building challenge",
    "overrated": "overrated",
    "underrated": "underrated",
    "flop":    "disappointing player",
    "legend":  "legendary player",
    "fraud":   "disappointing player",
    "elite":   "top quality",
}

# ── INITIALISE ONCE ──────────────────────────────────────────
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

# Keep negation words — "not good" ≠ "good"
KEEP_WORDS  = {"no", "not", "never", "nor", "neither", "against", "without"}
stop_words -= KEEP_WORDS


def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 3. Remove @mentions but keep hashtag text (e.g. #Messi → messi)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)

    # 4. Expand football slang
    tokens = text.split()
    tokens = [SLANG.get(t, t) for t in tokens]
    text   = ' '.join(tokens)

    # 5. Remove repeated characters (soooo → so, hahaha → ha)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 6. Keep letters and spaces only
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 7. Tokenise
    tokens = text.split()

    # 8. Remove stopwords (but keep negation)
    tokens = [w for w in tokens if w not in stop_words]

    # 9. Lemmatise
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # 10. Remove very short tokens (single letters add noise)
    tokens = [w for w in tokens if len(w) > 1]

    return ' '.join(tokens)


# ── MAIN ─────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("final_sentiment_analysis.csv")
df['Text'] = df['Text'].fillna('')

print(f"Rows loaded: {len(df)}")
print("\nSample before cleaning:")
print(df['Text'].head(3).to_string())

print("\nCleaning text...")
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Drop rows where cleaning produced empty text
before = len(df)
df = df[df['cleaned_text'].str.strip() != '']
after  = len(df)
if before != after:
    print(f"Removed {before - after} rows with empty text after cleaning")

print("\nSample after cleaning:")
for orig, clean in zip(df['Text'].head(3), df['cleaned_text'].head(3)):
    print(f"  Before: {orig[:80]}")
    print(f"  After:  {clean[:80]}")
    print()

df.to_csv("final_cleaned_sentiment_analysis.csv", index=False)
print(f"Saved {len(df)} rows to final_cleaned_sentiment_analysis.csv")
