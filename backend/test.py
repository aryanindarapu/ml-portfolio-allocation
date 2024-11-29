import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

tickers = []
table = soup.find('table', {'class': 'wikitable'})
rows = table.find_all('tr')[1:]  # Skip header row

for row in rows:
    ticker = row.find_all('td')[0].text.strip()  # First column
    tickers.append(ticker)

tickers.sort()
print(tickers)  # List of all S&P 500 tickers
print(len(tickers))  # List of all S&P 500 tickers
