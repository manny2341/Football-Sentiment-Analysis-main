import praw
import pandas as pd
import time

# Reddit API credentials (Replace with your own credentials)
reddit = praw.Reddit(
    client_id="Pr57J5TKyIxnIeb09AsU6g",
    client_secret="qL_Cxx2n_w6XtYBE2TcoI5tG7EYBGw",
    user_agent="SentimentFootballAnalysis"
)

# List of players
players = [
    # Original
    "Mbappe", "Lamine Yamal", "Pedri", "Jude Bellingham",
    "Vinicius Junior", "Raphinha", "Erling Haaland",
    "Cristiano Ronaldo", "Neymar", "Messi", "Baleba",
    "Robert Lewandowski", "Mohamed Salah", "Reece James",
    # PL Top 10 G+A 2024/25 (new)
    "Cole Palmer", "Alexander Isak", "Bryan Mbeumo",
    "Bukayo Saka", "Chris Wood", "Jarrod Bowen",
    "Ollie Watkins", "Nicolas Jackson",
]

# Define relevant topics with keywords
categories = {
    "Match Performance": ["goal", "assist", "penalty", "dribble", "mistake", "hat-trick", "performance", "match"],
    "Transfer Rumors": ["transfer", "deal", "signing", "new club", "leaving", "contract", "buyout", "release clause"],
    "Injury Reports": ["injury", "out for weeks", "medical test", "rehab", "ankle", "hamstring", "muscle tear"],
    "Off-field Behavior": ["controversy", "scandal", "fight", "donation", "charity", "community work", "foundation"]
}

# Store data
all_reddit_data = []

# Choose subreddit (Soccer-related)
subreddit = reddit.subreddit("soccer")

# Fetch posts and comments for each player and category
for player in players:
    for category, keywords in categories.items():
        for keyword in keywords:
            print(f"Fetching Reddit posts for {player} - {category} ({keyword})...")
            
            # Search for Reddit posts
            posts = subreddit.search(f"{player} {keyword}", limit=10)  # Fetch top 10 posts
            
            for post in posts:
                post.comments.replace_more(limit=0)  # Expand comments
                for comment in post.comments.list():
                    all_reddit_data.append([player, category, keyword, post.title, comment.body, post.created_utc])

            time.sleep(2)  # Prevent hitting Reddit's rate limits

# Convert to DataFrame
df = pd.DataFrame(all_reddit_data, columns=["Player", "Category", "Keyword", "Post Title", "Comment", "Timestamp"])

# Save to CSV
df.to_csv("reddit_football_data.csv", index=False)

print("Reddit posts and comments collected successfully!")