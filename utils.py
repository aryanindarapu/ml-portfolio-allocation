import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from strategies import ewp, mvp, ivp, rpp

""" Helper Functions"""

def select_factor_list(t):
    if t == "capm":
        return ["Mkt-RF"]
    elif t == "ff3":
        return ["Mkt-RF", "SMB", "HML"]
    elif t == "ff5":
        return ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    
    raise ValueError("Invalid factor model. Please choose from 'capm', 'ff3', or 'ff5'.")

def get_portfolio_weights(tickers, strategy, initial_amount, start_date, end_date, **kwargs):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.pct_change().dropna()
    print(data)
    if strategy == "ewp":
        return ewp(data, initial_amount)
    elif strategy == "mvp":
        return mvp(data, initial_amount, kwargs.get("l", 1))
    elif strategy == "ivp":
        return ivp(data, initial_amount)
    elif strategy == "rpp":
        return rpp(data, initial_amount)
    
    raise ValueError("Invalid strategy. Please choose from 'ewp', 'mvp', 'ivp', or 'rpp'.")

def get_monthly_returns(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        returns = yf.download(ticker, start=start_date, end=end_date)
        data = returns["Adj Close"].resample("ME").ffill().to_frame()
        data.index = data.index.to_period("M")
        data["Return"] = data["Adj Close"].pct_change() * 100
        data.dropna(inplace=True)
        all_data[ticker] = data["Return"]
    return pd.DataFrame(all_data)

def get_factors():
    factors_monthly = pd.read_csv("data/ff5_monthly_data.csv", index_col=0)
    factors_monthly.index.name = "Date"
    factors_monthly.index = pd.to_datetime(factors_monthly.index, format="%Y%m")
    factors_monthly.index = factors_monthly.index.to_period("M")
    
    return factors_monthly

def get_stock_details(tickers):
    stock_details = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        stock_details.append({
            "Ticker": ticker,
            "Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry")
        })
    return pd.DataFrame(stock_details)

def prepare_features(data, lag=12):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)
    return df

""" Visualization Functions """
def visualize_returns(historical_returns, forecasted_returns, initial_investment):
    plt.figure(figsize=(10, 5))
    
    # Plot historical returns (percentage change)
    historical_returns.plot(label='Historical Returns (Percentage Change)', color='blue')
    
    # Create a DataFrame for forecasted returns
    forecast_index = pd.period_range(start=historical_returns.index[-1] + 1, periods=12, freq='M')
    forecasted_returns_df = pd.Series(forecasted_returns, index=forecast_index, name='Forecasted Returns')
    
    # Plot forecasted returns (percentage change)
    forecasted_returns_df.plot(label='Forecasted Returns (Percentage Change)', color='red', linestyle='--')
    
    plt.title('Historical and Forecasted Portfolio Returns (Percentage Change)')
    plt.xlabel('Date')
    plt.ylabel('Monthly Return (%)')
    plt.legend()
    plt.show()

    # Calculate portfolio value over time
    historical_portfolio_value = (1 + historical_returns / 100).cumprod() * initial_investment
    forecasted_portfolio_value = (1 + forecasted_returns_df / 100).cumprod() * historical_portfolio_value.iloc[-1]
    
    plt.figure(figsize=(10, 5))
    
    # Plot historical portfolio value
    historical_portfolio_value.plot(label='Historical Portfolio Value', color='green')
    
    # Plot forecasted portfolio value
    forecasted_portfolio_value.plot(label='Forecasted Portfolio Value', color='orange', linestyle='--')
    
    plt.title('Historical and Forecasted Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.show()
