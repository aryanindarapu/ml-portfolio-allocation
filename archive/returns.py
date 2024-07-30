import pandas as pd
import numpy as np
from utils import get_sp500_data, get_factors

from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.api import OLS


# def portfolio_return(outputs: pd.Series, expected_returns: pd.Series):
#     return outputs * expected_returns, np.dot(outputs, expected_returns)

def actual_returns(returns: pd.DataFrame, weights: pd.Series):
    weighted_returns = returns.mul(weights, axis=1)
    portfolio_return = weighted_returns.sum(axis=1)
    
    cumulative_return = (portfolio_return + 1).cumprod().iloc[-1] - 1
    return cumulative_return, portfolio_return

def portfolio_return2(weights: pd.Series, predicted_returns: pd.DataFrame):
    weighted_predicted_returns = predicted_returns.mul(weights, axis=1)
    portfolio_predicted_returns = weighted_predicted_returns.sum(axis=1)
    portfolio_predicted_returns.index = pd.to_datetime(portfolio_predicted_returns.index)
    return portfolio_predicted_returns

# CAPM Model
# def capm_model(data: pd.DataFrame, market_data: pd.DataFrame):
#     market_returns = market_data.pct_change().dropna()
#     data_returns = data.pct_change().dropna()
    
#     betas, alphas = dict(), dict()
#     for ticker in data.columns:
#         b, a = np.polyfit(market_returns, data_returns[ticker], 1)
#         plt.plot(market_returns, b*market_returns + a, color='r')
#         betas[ticker] = b
#         alphas[ticker] = a
    
#     plt.show()
#     return betas, alphas

# def compute_capm_returns(historical_data: pd.DataFrame, test_data: pd.DataFrame, end_date: str):
    sp500_data = get_sp500_data()
    betas, alphas = capm_model(historical_data, sp500_data)
    _, rf = get_factors(end_date, "CAPM")
    
    rm = sp500_data.pct_change().mean()
    capm_returns = dict()
    for ticker in test_data.columns:
        capm_returns[ticker] = rf + betas[ticker] * (rm- rf)
        
    return pd.Series(capm_returns)

# def run_model(data: pd.DataFrame, model: str):
#     factors, rf = get_factors(model)
#     factors = factors.loc[data.index]
#     rf = rf.loc[data.index]
    
#     results = {}
#     predicted_returns = {}
#     cumulative_predicted_returns = {}
#     for ticker in data.columns:
#         returns = data[ticker]
#         X = sm.add_constant(factors)
#         model = OLS(returns, X).fit()
#         predicted_return = model.predict(X)
        
#         results[ticker] = model.summary()
#         predicted_returns[ticker] = predicted_return
#         cumulative_predicted_returns[ticker] = (predicted_return + 1).cumprod().iloc[-1] - 1
        
#     return results, pd.DataFrame(predicted_returns), pd.Series(cumulative_predicted_returns)

# def capm_model(data: pd.DataFrame):
#     factors, rf = get_factors("CAPM")
#     factors = factors.loc[data.index]
#     rf = rf.loc[data.index]
    
#     results = {}
#     predicted_returns = {}
#     cumulative_predicted_returns = {}
#     for ticker in data.columns:
#         # Calculate the daily returns for the stock
#         returns = data[ticker]
#         X = sm.add_constant(factors)

#         # Fit the CAPM model: (R_i - R_f) ~ alpha + beta * (R_Mkt - R_f)
#         model = OLS(returns - rf, X).fit()
#         predicted_return = model.predict(X) + rf

#         results[ticker] = model.summary()
#         predicted_returns[ticker] = predicted_return
#         cumulative_predicted_returns[ticker] = (predicted_return + 1).cumprod().iloc[-1] - 1
        
#     return results, pd.DataFrame(predicted_returns), None

# # French-Fama Three Factor Model
# def ff3_model(data: pd.DataFrame):
#     factors, rf = get_factors("FF3")
#     factors = factors.loc[data.index]
#     rf = rf.loc[data.index]
    
#     results = {}
#     predicted_returns = {}
#     cumulative_predicted_returns = {}
#     for ticker in data.columns:
#         returns = data[ticker]
#         X = sm.add_constant(factors)
#         model = OLS(returns - rf, X).fit()
#         predicted_return = model.predict(X) + rf
        
#         results[ticker] = model.summary()
#         predicted_returns[ticker] = predicted_return
#         cumulative_predicted_returns[ticker] = (predicted_return + 1).cumprod().iloc[-1] - 1
        
#     # Convert cumulative returns to a DataFrame with keys as index
#     # cumulative_predicted_returns = pd.DataFrame(cumulative_predicted_returns, index=["Ticker"])
            
#     return results, pd.DataFrame(predicted_returns), pd.Series(cumulative_predicted_returns)

# # French-Fama Five Factor Model
# def ff5_model(data: pd.DataFrame):
    factors, rf = get_factors("FF5")
    factors = factors.loc[data.index]
    rf = rf.loc[data.index]
    
    results = {}
    predicted_returns = {}
    cumulative_predicted_returns = {}
    for ticker in data.columns:
        returns = data[ticker]
        X = sm.add_constant(factors)
        model = OLS(returns - rf, X).fit()
        predicted_return = model.predict(X) + rf
        
        results[ticker] = model.summary()
        predicted_returns[ticker] = predicted_return
        cumulative_predicted_returns[ticker] = (predicted_return + 1).cumprod().iloc[-1] - 1
        
    return results, pd.DataFrame(predicted_returns), pd.Series(cumulative_predicted_returns)