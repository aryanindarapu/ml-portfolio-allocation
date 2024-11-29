import numpy as np
from quant.utils import get_portfolio_weights, get_monthly_returns

# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]
tickers = ["AAPL", 'NVDA', "MSFT", "GOOGL", "AMZN", 'META', "TSLA"]
# tickers = ["AMD", "INTC", "QCOM", "IBM", "PINS", "AAPL", "GOOGL"]
# include 100 different small cap stocks
# tickers = 
initial_amount = 10000
start_date = "2012-01-10"
end_date = "2024-11-19"
strategy = "rpp"
model_type = "ff5" # Either capm, ff3, or ff5
portfolio = get_monthly_returns(tickers, start_date, end_date)
print(portfolio)
