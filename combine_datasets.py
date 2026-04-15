import pandas as pd

# Load Twitter Data
#twitter_df = pd.read_csv("football_players_tweets_v2.csv")
#twitter_df["Source"] = "Twitter"  # Add source column

# Load Reddit Data
reddit_df = pd.read_csv("reddit_sentiment_analysis.csv")
reddit_df = reddit_df.rename(columns={"Comment": "Text"})  # Rename for consistency
reddit_df["Source"] = "Reddit"

# Load ESPN Data
espn_df = pd.read_csv("espn_football_news.csv")
espn_df = espn_df.rename(columns={"Headline": "Text"})  # Rename for consistency
espn_df["Source"] = "ESPN"

# Standardizing Twitter column names
#twitter_df = twitter_df.rename(columns={"Tweet": "Text"})

# Select common columns for merging
columns_to_keep = ["Player", "Category", "Keyword", "Text", "Source"]

# Ensure all dataframes have the same structure
reddit_df = reddit_df[["Player", "Text", "Sentiment", "Source"]]
espn_df = espn_df[["Player", "Text", "Source"]]
#twitter_df = twitter_df[["Player", "Category", "Keyword", "Text", "Source"]]

# Merge all datasets
combined_df = pd.concat([reddit_df, espn_df], ignore_index=True)
# add this when twitter api set ok twitter_df
# Save the final dataset
combined_df.to_csv("final_sentiment_analysis.csv", index=False)

print("✅ Final dataset created successfully! Check 'final_sentiment_analysis.csv'")