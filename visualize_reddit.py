import pandas as pd
import matplotlib.pyplot as plt

# Load sentiment results
df = pd.read_csv("reddit_sentiment_analysis.csv")

# Count sentiment distribution
sentiment_counts = df["Sentiment"].value_counts()

# Plot bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "gray", "red"])
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.title("Sentiment Analysis of Reddit Comments on Football Players")
plt.show()