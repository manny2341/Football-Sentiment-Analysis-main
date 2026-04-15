import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your cleaned CSV
df = pd.read_csv("final_cleaned_sentiment_analysis.csv")

# Ensure no NaNs remain (fix)
df['cleaned_text'] = df['cleaned_text'].fillna('')

# Now perform TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text']).toarray()

print("✅ TF-IDF vectorization completed successfully!")