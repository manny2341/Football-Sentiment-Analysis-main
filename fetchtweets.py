import tweepy
import pandas as pd

# Twitter API keys (replace with own api)
API_KEY = "OfCLJbGS8kz6hCqwsEPyg0GoB"
API_SECRET = "PQRVd2rM8ucPETUmO2fSlA8LTa9y8wshehqmI1kkWdJ4NhF52U"
#ACCESS_TOKEN = "1671097325421305857-htnajYjNbFzxgUUE5bxSsh2st8Af7j"
#ACCESS_SECRET = "Y8KaXvHpMSh6MXqcmB28Bt3KWqpLpJ9tm0ktw3kFIIon4"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAI%2FezgEAAAAAIvk6trqpQ2XlAc4wW%2FhpbLbmL2s%3DDz9X19TMMUQ6RkARPvucBvQuiQhzAif5ZeQA8euCDsoBcEKlmv"

# Authenticate
#auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
#auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
#api = tweepy.API(auth, wait_on_rate_limit=True)
# Authenticate using Twitter API v2
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# List of players
players = ["Mbappe", "Lamine Yamal", "Pedri", "Jude Bellingham", "Vinicius Junior", "Raphinha", "Erling Haaland"]

# Define relevant topics with keywords
categories = {
    "Match Performance": ["goal", "assist", "penalty", "hat-trick", "dribble", "save", "mistake", "poor game"],
    "Transfer Rumors": ["transfer", "deal", "signing", "new club", "leaving", "contract", "buyout", "release clause"],
    "Injury Reports": ["injury", "out for weeks", "medical test", "rehab", "ankle", "hamstring", "muscle tear"],
    "Off-field Behavior": ["controversy", "scandal", "fight", "donation", "charity", "community work", "foundation"]
}

# Store data
all_tweets = []

# Function to handle rate limits and retry automatically
def fetch_tweets(query, retry_count=0):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["created_at"])
        return tweets.data if tweets.data else []

    except tweepy.TooManyRequests:
        if retry_count < 3:  # Only retry up to 3 times
            wait_time = (retry_count + 1) * 600  # Increases wait time (10 mins, 20 mins, 30 mins)
            print(f"Rate limit reached. Waiting for {wait_time // 60} minutes before retrying...")
            time.sleep(wait_time)
            return fetch_tweets(query, retry_count + 1)  # Retry after waiting
        else:
            print("Max retries reached. Skipping this request.")
            return []

# Fetch tweets for each player and category
for player in players:
    for category, keywords in categories.items():
        for keyword in keywords:
            query = f'"{player}" "{keyword}" -is:retweet lang:en'
            print(f"Fetching tweets for {player} - {category} ({keyword})...")

            tweets = fetch_tweets(query)

            for tweet in tweets:
                all_tweets.append([player, category, keyword, tweet.text, tweet.created_at])

            time.sleep(30)  # Adds a **30-second** delay between requests

# Convert to DataFrame
df = pd.DataFrame(all_tweets, columns=["Player", "Category", "Keyword", "Tweet", "Date"])

# Save to CSV
df.to_csv("football_players_tweets_v2.csv", index=False)
print("Tweets collected successfully with rate limit handling!")