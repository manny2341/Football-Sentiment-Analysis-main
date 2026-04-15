import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your cleaned dataset
df = pd.read_csv("final_cleaned_sentiment_analysis.csv")
df['cleaned_text'] = df['cleaned_text'].fillna('')

# Step 2: TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text']).toarray()

# Step 3: Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(df['Sentiment'])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data split completed:")
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")