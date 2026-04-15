"""
Scrape Reddit data for new PL Top 10 G+A players and merge into existing dataset.
New players: Cole Palmer, Alexander Isak, Bryan Mbeumo, Bukayo Saka,
             Chris Wood, Jarrod Bowen, Ollie Watkins, Nicolas Jackson
             + Mohamed Salah (if not already present)
"""

import praw
import pandas as pd
import time
from analyze_reddit import get_sentiment as classify_sentiment

# ── REDDIT CREDENTIALS ───────────────────────────────────────
reddit = praw.Reddit(
    client_id="Pr57J5TKyIxnIeb09AsU6g",
    client_secret="qL_Cxx2n_w6XtYBE2TcoI5tG7EYBGw",
    user_agent="SentimentFootballAnalysis"
)

# ── NEW PLAYERS TO ADD ───────────────────────────────────────
# PL Top 10 G+A 2024/25 (Haaland + existing players already in dataset)
NEW_PLAYERS = [
    "Mohamed Salah",
    "Cole Palmer",
    "Alexander Isak",
    "Bryan Mbeumo",
    "Bukayo Saka",
    "Chris Wood",
    "Jarrod Bowen",
    "Ollie Watkins",
    "Nicolas Jackson",
]

CATEGORIES = {
    "Match Performance":  ["goal", "assist", "penalty", "dribble", "hat-trick", "performance"],
    "Transfer Rumors":    ["transfer", "signing", "new club", "contract", "leaving"],
    "Injury Reports":     ["injury", "out for weeks", "rehab", "hamstring", "fitness"],
    "Off-field Behavior": ["controversy", "scandal", "charity", "foundation"],
}

subreddit = reddit.subreddit("soccer")

# ── CHECK WHICH PLAYERS ALREADY HAVE DATA ────────────────────
existing_df = pd.read_csv("final_sentiment_analysis.csv")
existing_players = existing_df['Player'].unique().tolist()
print("Already in dataset:", existing_players)

players_to_scrape = [p for p in NEW_PLAYERS if p not in existing_players]
print(f"\nScraping data for: {players_to_scrape}\n")

# ── SCRAPE ───────────────────────────────────────────────────
new_data = []

for player in players_to_scrape:
    print(f"\n{'='*50}")
    print(f"  Scraping: {player}")
    print(f"{'='*50}")
    count = 0

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            try:
                posts = subreddit.search(f"{player} {keyword}", limit=10)
                for post in posts:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        body = comment.body.strip()
                        if len(body) < 10:
                            continue
                        sentiment = classify_sentiment(body)
                        new_data.append({
                            "Player":    player,
                            "Text":      body,
                            "Sentiment": sentiment,
                            "Source":    "Reddit",
                        })
                        count += 1
                time.sleep(1.5)
            except Exception as e:
                print(f"  Error on {player}/{keyword}: {e}")
                time.sleep(5)

    print(f"  Collected {count} comments for {player}")

# ── MERGE & SAVE ─────────────────────────────────────────────
if new_data:
    new_df = pd.DataFrame(new_data)
    print(f"\nNew records collected: {len(new_df)}")
    print(new_df['Player'].value_counts().to_string())

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv("final_sentiment_analysis.csv", index=False)
    print(f"\nUpdated final_sentiment_analysis.csv — total rows: {len(combined)}")

    # Re-run cleaning on new data only
    print("\nCleaning new data...")
    import subprocess
    subprocess.run(["python3", "cleaning.py"])

    print("\nDone! Now retrain the model:")
    print("  python3 model_training.py")
else:
    print("No new data collected.")
