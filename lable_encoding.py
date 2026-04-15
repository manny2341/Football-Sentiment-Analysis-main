import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the cleaned CSV file clearly
df = pd.read_csv("final_cleaned_sentiment_analysis.csv")

# Encode sentiment labels clearly
encoder = LabelEncoder()
y = encoder.fit_transform(df['Sentiment'])  # converts "positive", "negative", "neutral" to numerical labels (0,1,2)

# clearly print results to verify
print("✅ Labels encoded successfully:")
print(pd.DataFrame({'Original Label': df['Sentiment'], 'Encoded Label': y}).head())