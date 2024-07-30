import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import numpy as np

from utils import get_ticker_data, plot_returns
from strategies import ewp, ivp, gmvp, rpp
from returns import run_model, actual_returns, portfolio_return2

if __name__ == "__main__":
    # parse file path arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to data file")
    parser.add_argument("-t", "--tickers", help="List of tickers")
    parser.add_argument("-i", "--initial_amount", help="Initial amount to invest")
    args = parser.parse_args()

    if args.data:
        data = pd.read_csv(args.data)
    else:
        # tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'CLF', 'META', 'TSLA', 'NVDA']
        tickers = ['AAPL', 'MSFT']
        # data = get_ticker_data(tickers, start='2020-01-01', end='2023-12-31')
        data = get_ticker_data(tickers, start='2023-11-01', end='2023-12-31')
        
    if args.initial_amount:
        initial_amount = float(args.initial_amount)
    else:
        initial_amount = 10000.0
        
    # output, weights = ewp(data, initial_amount)
    output, weights = ivp(data, initial_amount)
    # output, weights = gmvp(data, initial_amount)
    # output, weights = rpp(data, initial_amount)
    print("Portfolio Allocation:\n", output.sort_values(ascending=False))  
    print("\nIVP Weights:\n", weights.sort_values(ascending=False))

    # historical_data = data.loc[:'2023-01-01']
    # test_data = data.loc['2023-01-01':]

    # Compute actual portfolio returns
    cumulative_portfolio_return, actual_portfolio_return = actual_returns(data, weights)
    print("\nActual Portfolio Return:", actual_portfolio_return)
    print("Total Cumulative Return:", cumulative_portfolio_return)
    
    # Compute expected returns using selected model
    # results, projected_returns, projected_returns_cumulative = capm_model(data)
    # results, projected_returns, projected_returns_cumulative = ff3_model(data)
    # results, projected_returns, projected_returns_cumulative = ff5_model(data)
    results, projected_returns, projected_returns_cumulative = run_model(data, model="CAPM")
    # expected_return, total_expected_return = portfolio_return(output, projected_returns_cumulative)
    predicted_return = portfolio_return2(weights, projected_returns)
    print("\nExpected Return:", predicted_return)
    
    
    plot_returns(tickers, projected_returns, data, predicted_return, actual_portfolio_return)