"""
Football Sentiment Analysis — Improved Model Training
Improvements over original:
  - SVM added (best for text classification)
  - 5-fold cross-validation for reliable accuracy
  - class_weight='balanced' to handle imbalanced labels
  - TF-IDF features increased to 5,000 + char n-grams
  - All models saved to disk with joblib
  - Confusion matrix + top feature importance charts
  - Full comparison table at the end
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────
DATA_FILE   = "final_cleaned_sentiment_analysis.csv"
MODELS_DIR  = "saved_models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("=" * 60)
print("  Football Sentiment Analysis — Model Training")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df['cleaned_text'] = df['cleaned_text'].fillna('')

print(f"\nDataset shape: {df.shape}")
print("\nSentiment distribution:")
dist = df['Sentiment'].value_counts()
for label, count in dist.items():
    pct = count / len(df) * 100
    print(f"  {label:<12} {count:>5}  ({pct:.1f}%)")

# ── 2. LABEL ENCODING ────────────────────────────────────────
encoder = LabelEncoder()
y = encoder.fit_transform(df['Sentiment'])
print(f"\nClasses: {encoder.classes_}")

# ── 3. IMPROVED TF-IDF ───────────────────────────────────────
# Word-level: 5,000 features, unigrams + bigrams
# Original had 1,000 — 5x more vocabulary coverage
tfidf_word = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,           # ignore terms in fewer than 2 documents
    sublinear_tf=True,  # log normalisation dampens very frequent words
    strip_accents='unicode',
)

# Character-level: catches misspellings, slang, abbreviations
# e.g. "goat", "LFG", "bottled" — things VADER misses
tfidf_char = TfidfVectorizer(
    max_features=3000,
    analyzer='char_wb',
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
)

X_word = tfidf_word.fit_transform(df['cleaned_text'])
X_char = tfidf_char.fit_transform(df['cleaned_text'])
X      = hstack([X_word, X_char])   # combine both

print(f"\nFeature matrix: {X.shape}")
print(f"  Word features: {X_word.shape[1]}")
print(f"  Char features: {X_char.shape[1]}")

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────
# stratify=y ensures class ratios are the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── 5. DEFINE MODELS ─────────────────────────────────────────
# class_weight='balanced' automatically compensates for unequal class sizes
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
    ),
    "Linear SVM": LinearSVC(
        max_iter=2000,
        class_weight='balanced',
        C=1.0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    ),
}

# ── 6. CROSS-VALIDATION + TRAINING ───────────────────────────
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
test_results = {}

print("\n" + "=" * 60)
print("  Training & Evaluating Models")
print("=" * 60)

for name, model in MODELS.items():
    print(f"\n[{name}]")

    # 5-fold CV — gives a more reliable estimate than a single split
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring='f1_weighted', n_jobs=-1)
    cv_scores[name] = scores
    print(f"  CV F1 (5-fold):  {scores.mean():.3f} ± {scores.std():.3f}")

    # Final train on full training set, evaluate on held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred,
                                   target_names=encoder.classes_,
                                   output_dict=True)
    test_results[name] = report
    print(f"  Test Accuracy:   {report['accuracy']:.3f}")
    print(f"  Test F1 (macro): {report['macro avg']['f1-score']:.3f}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save each model
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
    print(f"  Saved to saved_models/{name.replace(' ', '_')}.pkl")

# Save vectorisers and encoder — Flask app needs these to make predictions
joblib.dump(tfidf_word, os.path.join(MODELS_DIR, "tfidf_word.pkl"))
joblib.dump(tfidf_char, os.path.join(MODELS_DIR, "tfidf_char.pkl"))
joblib.dump(encoder,    os.path.join(MODELS_DIR, "label_encoder.pkl"))
print("\nVectorisers and encoder saved to saved_models/")

# ── 7. MODEL COMPARISON CHART ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — Football Sentiment Analysis",
             fontsize=14, fontweight='bold')

# Cross-validation box plot
cv_data  = [cv_scores[n] for n in MODELS]
axes[0].boxplot(cv_data, labels=list(MODELS.keys()), patch_artist=True,
                boxprops=dict(facecolor='#4488ff', alpha=0.7))
axes[0].set_title("5-Fold Cross-Validation F1")
axes[0].set_ylabel("Weighted F1 Score")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(axis='y', alpha=0.3)

# Test set bar chart
test_accs = [test_results[n]['accuracy'] for n in MODELS]
test_f1s  = [test_results[n]['macro avg']['f1-score'] for n in MODELS]
x         = np.arange(len(MODELS))
w         = 0.35
b1 = axes[1].bar(x - w/2, test_accs, w, label='Accuracy',  color='#ff6644', alpha=0.8)
b2 = axes[1].bar(x + w/2, test_f1s,  w, label='Macro F1', color='#44bb66', alpha=0.8)
axes[1].set_title("Test Set Performance")
axes[1].set_xticks(x)
axes[1].set_xticklabels(list(MODELS.keys()), rotation=15)
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/model_comparison.png")

# ── 8. CONFUSION MATRICES ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')

for i, (name, model) in enumerate(MODELS.items()):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
    disp.plot(ax=axes.flatten()[i], colorbar=False, cmap='Blues')
    axes.flatten()[i].set_title(name, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/confusion_matrices.png")

# ── 9. TOP FEATURES PER CLASS ────────────────────────────────
logreg        = MODELS["Logistic Regression"]
feature_names = (tfidf_word.get_feature_names_out().tolist() +
                 tfidf_char.get_feature_names_out().tolist())

fig, axes = plt.subplots(1, len(encoder.classes_), figsize=(16, 5))
fig.suptitle("Top 15 Words Per Sentiment Class (Logistic Regression)",
             fontsize=13, fontweight='bold')

palette = {'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#27ae60'}

for i, cls in enumerate(encoder.classes_):
    coefs     = logreg.coef_[i]
    top_idx   = np.argsort(coefs)[-15:]
    top_words = [feature_names[j] for j in top_idx]
    top_vals  = coefs[top_idx]
    color     = palette.get(cls.lower(), '#3498db')
    axes[i].barh(top_words, top_vals, color=color, alpha=0.8)
    axes[i].set_title(cls.title(), fontweight='bold')
    axes[i].set_xlabel("Coefficient weight")
    axes[i].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/top_features.png")

# ── 10. FINAL SUMMARY TABLE ──────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"\n{'Model':<22} {'CV F1':>8} {'±':>6} {'Test Acc':>10} {'Test F1':>9}")
print("-" * 60)
for name in MODELS:
    cv_mean = cv_scores[name].mean()
    cv_std  = cv_scores[name].std()
    acc     = test_results[name]['accuracy']
    f1      = test_results[name]['macro avg']['f1-score']
    print(f"{name:<22} {cv_mean:>8.3f} {cv_std:>6.3f} {acc:>10.3f} {f1:>9.3f}")

best = max(MODELS, key=lambda n: test_results[n]['macro avg']['f1-score'])
print(f"\nBest model: {best}")
print("\nDone. All models saved to saved_models/")
