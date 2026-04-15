import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load cleaned data
df = pd.read_csv("final_cleaned_sentiment_analysis.csv")

# 1. Sentiment distribution plot
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
plt.show()

# 2. Generate Word Clouds per sentiment
sentiments = ['positive', 'negative', 'neutral']

for sentiment in sentiments:
    text = ' '.join(df[df['Sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.title(f'{sentiment.capitalize()} Sentiment Word Cloud')
    plt.axis('off')
    plt.savefig(f'{sentiment}_wordcloud.png')
    plt.show()

print("✅ Visualizations created successfully!")