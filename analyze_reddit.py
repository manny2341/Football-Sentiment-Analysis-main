import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Reddit data
df = pd.read_csv("reddit_football_data.csv")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.1:
        return "Positive"
    elif score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Comment"].apply(get_sentiment)

# Save results
df.to_csv("reddit_sentiment_analysis.csv", index=False)
print("Sentiment analysis on Reddit comments completed!")