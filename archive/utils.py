import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

### Portfolio Optimization Functions ###

def get_risk_contribution(weights: pd.Series, data: pd.DataFrame) -> pd.Series:
    returns = data.pct_change()    
    cov_matrix = returns.cov()
    
    # Calculate portfolio variance
    portfolio_variance = (weights.T @ cov_matrix @ weights) #[0]
    
    # Calculate marginal risk contribution for each asset
    risk_contribution = (weights * (cov_matrix @ weights)) / portfolio_variance
    
    return risk_contribution

def get_sharpe_ratio(weights: pd.Series, data: pd.DataFrame) -> float:
    pass # TODO

def get_factors(model: str) -> pd.DataFrame:
    factors = []
    if model == "CAPM":
        factor_data = pd.read_csv("data/ff3_daily_data.csv")
        factor_data['Date'] = pd.to_datetime(factor_data['Date'], format='%Y%m%d')#.dt.strftime('%Y-%m-%d')
        factor_data.set_index('Date', inplace=True)
        rf = factor_data["RF"]
        factors = factor_data[['Mkt-RF']]
    elif model == "FF3":
        factor_data = pd.read_csv("data/ff3_daily_data.csv")
        factor_data['Date'] = pd.to_datetime(factor_data['Date'], format='%Y%m%d')#.dt.strftime('%Y-%m-%d')
        factor_data.set_index('Date', inplace=True)
        rf = factor_data["RF"]
        factors = factor_data[['Mkt-RF', 'SMB', 'HML']]   
    elif model == "FF5":
        factor_data = pd.read_csv("data/ff5_daily_data.csv")
        factor_data['Date'] = pd.to_datetime(factor_data['Date'], format='%Y%m%d')#.dt.strftime('%Y-%m-%d')
        factor_data.set_index('Date', inplace=True)
        rf = factor_data["RF"]
        factors = factor_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    else:
        raise ValueError("Invalid model type. Choose from CAPM, FF3, FF5")
    
    # rf = factor_data["RF"] # .loc[f"{date.year}{date.month:02d}"] # get the risk free rate for the current month of the data
    # rf = ff3_data.loc[ff3_data['Date'] == end_date, 'RF'].values[0] # get the risk free rate for the current month of the data
    return factors, rf

### Ticker Data Function ###

def get_sp500_data() -> pd.DataFrame:
    data = yf.download("^GSPC", start="2015-01-01", end="2023-12-31")['Adj Close']
    return data

def get_ticker_data_visual(tickers: list) -> pd.DataFrame:
    # data: pd.DataFrame = yf.download(tickers, group_by="Ticker", start="2015-01-01", end="2023-12-31")
    df_list = []
    for ticker in tickers:
        data = yf.download(ticker, group_by="Ticker", start="2015-01-01", end="2023-12-31")
        data["Ticker"] = ticker
        df_list.append(data)
        
    data = pd.concat(df_list)
    
    # save data to csv
    plot_ticker_data(data)
    return data

def get_ticker_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    # TODO: add date range as argument
    # get_ticker_data_visual(tickers)
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    data = data.pct_change().dropna()
    data.to_csv("data/data.csv")
    return data

def plot_ticker_data(data: pd.DataFrame):
    data["Adj Close"].resample('D').mean().plot()  # Daily plot
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.title("Daily Plot")
    plt.show()
    
    data["Adj Close"].resample('M').mean().plot()  # Monthly plot
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.title("Monthly Plot")
    plt.show()
    
    data["Adj Close"].resample('Y').mean().plot()  # Yearly plot
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.title("Yearly Plot")
    plt.show()

def plot_returns(tickers, predicted_returns, actual_returns, portfolio_predicted_returns, portfolio_actual_returns):
    """
    Plot predicted returns and actual returns on a graph for comparison.
    
    :param tickers: List of stock tickers
    :param predicted_returns: Dictionary of predicted returns for each stock
    :param actual_returns: Series of actual cumulative returns for each stock
    :param portfolio_predicted_returns: Series of predicted cumulative returns for the portfolio
    :param portfolio_actual_returns: Series of actual cumulative returns for the portfolio
    """
    plt.figure(figsize=(12, 8))
    
    # Plot individual stock returns
    for ticker in tickers:
        plt.figure(figsize=(12, 8))
        # plt.plot(predicted_returns[ticker].index, (predicted_returns[ticker] + 1).cumprod(), label=f'Predicted {ticker}')
        plt.plot(actual_returns[ticker].index, (actual_returns[ticker] + 1).cumprod(), label=f'Actual {ticker}', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title(f'Predicted vs Actual Cumulative Returns for {ticker}')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f"images/{ticker}_returns.png")
        plt.clf()
    
    # Plot portfolio returns
    # plt.figure(figsize=(12, 8))
    plt.plot(portfolio_predicted_returns.index, (portfolio_predicted_returns + 1).cumprod(), label='Predicted Portfolio')
    plt.plot(portfolio_actual_returns.index, (portfolio_actual_returns + 1).cumprod(), label='Actual Portfolio', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Predicted vs Actual Cumulative Returns for Portfolio')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("images/portfolio_returns.png")

def get_stock_tickers() -> list:
    # TODO: change to use yahoo finance API
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'CLF', 'META', 'TSLA', 'NVDA']