import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define player names and their ESPN URLs (Update IDs if needed)
players = {
    "Kylian Mbappe": "231254",
    "Lamine Yamal": "478413",
    "Pedri": "456923",
    "Jude Bellingham": "437430",
    "Vinicius Junior": "401526",
    "Raphinha": "220766",
    "Erling Haaland": "396253"
}

# Store news headlines
all_headlines = []

# Loop through each player and scrape ESPN news
for player, player_id in players.items():
    url = f"https://www.espn.com/soccer/player/_/id/{player_id}/{player.replace(' ', '-')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all news headlines
    articles = soup.find_all("h1", class_="headline")
    headlines = [article.text.strip() for article in articles]

    # Append to list
    for headline in headlines:
        all_headlines.append([player, headline])

# Convert to DataFrame
df = pd.DataFrame(all_headlines, columns=["Player", "Headline"])

# Save to CSV
df.to_csv("espn_football_news.csv", index=False)

print("News headlines collected for all players!")