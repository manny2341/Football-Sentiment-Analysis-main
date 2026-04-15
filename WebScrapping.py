import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of Wikipedia URLs to scrape
urls = [
    'https://en.wikipedia.org/wiki/Neymar',
    'https://en.wikipedia.org/wiki/Lionel_Messi',
    'https://en.wikipedia.org/wiki/Jude_Bellingham',
    'https://en.wikipedia.org/wiki/Raphinha',
    'https://en.wikipedia.org/wiki/Vin%C3%ADcius_J%C3%BAnior',
    'https://en.wikipedia.org/wiki/Cristiano_Ronaldo'
]

# Initialize a list to store player data
players_data = []

# Loop through each URL to fetch and parse content
for url in urls:
    # Fetch the content from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract player's name from URL
    player_name = url.split("/")[-1].replace("_", " ")

    # Extract paragraphs from the HTML content
    paragraphs = soup.find_all('p')

    # Join paragraphs with proper spacing
    text = "\n\n".join([para.text.strip() for para in paragraphs if para.text.strip()])

    # Add a separator for clear distinction between players
    text += "\n\n" + ("-" * 50) + "\n\n"

    # Create a dictionary with player name and content
    data = {
        'Player': player_name,
        'Content': text
    }

    # Append the player's data to the list
    players_data.append(data)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(players_data)

# Save the DataFrame to a CSV file
df.to_csv('football_players.csv', index=False)

# Print confirmation message
print("Data successfully scraped and saved to 'football_players.csv'")
