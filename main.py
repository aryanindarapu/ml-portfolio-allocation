import numpy as np
from utils import get_portfolio_weights

# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "CLF"]
start_date = "2010-01-01"
end_date = "2023-12-31"
weights = get_portfolio_weights(tickers, strategy="rpp", start_date=start_date, end_date=end_date, l=0.5)
print(weights)
