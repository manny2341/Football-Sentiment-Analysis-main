# Beyond The Score — Football Sentiment Analysis

A machine learning project that analyses public opinion about football players using Reddit, Twitter, and ESPN data. Classifies sentiment as Positive, Negative, or Neutral using NLP and 4 trained models, served through a live Flask web application.

---

## What This Project Does

Scrapes public comments and news about football players, cleans and processes the text, trains ML models to classify sentiment, and displays everything in a dashboard with live prediction.

> **Can you type any text about a player and instantly know if public opinion is positive, negative, or neutral?**
> Yes — that's exactly what the live prediction feature does.

---

## Players Covered

| Player | Club |
|---|---|
| Erling Haaland | Manchester City |
| Lamine Yamal | Barcelona |
| Kylian Mbappé | Real Madrid |
| Vinicius Junior | Real Madrid |
| Jude Bellingham | Real Madrid |
| Raphinha | Barcelona |
| Mohamed Salah | Liverpool |
| Cole Palmer | Chelsea |
| Alexander Isak | Newcastle |
| Bryan Mbeumo | Brentford |
| Bukayo Saka | Arsenal |
| Chris Wood | Nottingham Forest |
| Jarrod Bowen | West Ham |
| Ollie Watkins | Aston Villa |
| Nicolas Jackson | Chelsea |

---

## Dataset

| Detail | Value |
|---|---|
| Total records | 214,738 |
| Sources | Reddit, ESPN |
| Sentiment labels | Positive (44.1%) · Neutral (31.9%) · Negative (24.0%) |
| Players | 15 |
| Categories | Match Performance, Transfers, Injuries, Off-field Behaviour |

---

## Model Results

| Model | CV F1 (5-fold) | Test Accuracy | Test F1 |
|---|---|---|---|
| **Random Forest** | **0.888 ± 0.002** | **89.9%** | **0.892** |
| Linear SVM | 0.852 ± 0.002 | 85.6% | 0.847 |
| Logistic Regression | 0.846 ± 0.004 | 85.0% | 0.842 |
| Gradient Boosting | 0.768 ± 0.005 | 77.6% | 0.752 |

**Best model: Random Forest — 89.9% accuracy on 42,948 unseen records**

---

## Installation

```bash
git clone https://github.com/manny2341/Football-Sentiment-Analysis-main.git
cd Football-Sentiment-Analysis-main
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Clean the data
```bash
python3 cleaning.py
```

### Step 2 — Train the models
```bash
python3 model_training.py
```
Trains 4 models with 5-fold cross-validation. All models saved to `saved_models/`.

### Step 3 — Start the web app
```bash
python3 app.py
```
Open **http://localhost:5001**

---

## Web App Features

| Feature | Description |
|---|---|
| Live Prediction | Type any text → instant sentiment + confidence score |
| Model Selector | Choose between all 4 trained models |
| Dashboard Charts | Doughnut + bar chart showing sentiment breakdown |
| Player Filter | View results for individual players |
| Source Filter | Filter by Reddit or ESPN |
| Data Table | Browse the last 50 records |

---

## Project Structure

```
Football-Sentiment-Analysis-main/
├── app.py                          # Flask web application
├── cleaning.py                     # Text preprocessing pipeline
├── model_training.py               # Train + evaluate all 4 models
├── redditscrape.py                 # Scrape Reddit r/soccer
├── add_new_players.py              # Add new players without re-scraping
├── analyze_reddit.py               # VADER sentiment labelling
├── combine_datasets.py             # Merge Reddit + ESPN data
├── templates/
│   └── index.html                  # Dashboard UI
├── saved_models/                   # Trained models (local only)
│   ├── Logistic_Regression.pkl
│   ├── Linear_SVM.pkl
│   ├── Random_Forest.pkl
│   ├── Gradient_Boosting.pkl
│   ├── tfidf_word.pkl
│   ├── tfidf_char.pkl
│   └── label_encoder.pkl
└── results/                        # Charts (local only)
    ├── model_comparison.png
    ├── confusion_matrices.png
    └── top_features.png
```

---

## Pipeline

```
Reddit / ESPN / Twitter
        ↓
redditscrape.py / fetchtweets.py
        ↓
analyze_reddit.py  (VADER sentiment labelling)
        ↓
combine_datasets.py  (merge all sources)
        ↓
cleaning.py  (slang expansion, lemmatisation, stopwords)
        ↓
model_training.py  (TF-IDF 8,000 features → 4 ML models)
        ↓
app.py  (Flask dashboard + live prediction)
```

---

## Key Improvements Over Original

| Area | Original | Improved |
|---|---|---|
| Models | 2 (LR, RF) | 4 (+ SVM, Gradient Boosting) |
| Evaluation | Single 80/20 split | 5-fold cross-validation |
| TF-IDF features | 1,000 | 8,000 (word + character n-grams) |
| Class imbalance | Not handled | `class_weight='balanced'` |
| Football slang | Stripped as noise | Expanded to plain English |
| Negation words | Removed | Kept ("not good" ≠ "good") |
| Web app | Static CSV dump | Live prediction + charts + filters |
| Model saving | Nothing saved | All models saved with joblib |

---

## Adding New Players

```bash
python3 add_new_players.py
```
Scrapes Reddit for new players only, merges with existing data, and re-cleans automatically.

---

## Future Improvements

- BERT / RoBERTa transformer model for higher accuracy
- Twitter data re-integration when API access available
- Real-time scraping and automatic model retraining
- Confidence calibration for better probability estimates

---

## Data Source

Reddit r/soccer — scraped using PRAW (Python Reddit API Wrapper)
ESPN Football News — scraped using BeautifulSoup

---

## Disclaimer

This project is for educational purposes only. Sentiment scores reflect public Reddit/ESPN commentary and do not represent the views of the project author.

---

## Licence

MIT Licence — free to use, modify, and distribute with attribution.
