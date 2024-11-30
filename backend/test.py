import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define stock tickers and the time period for analysis
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA']  # Example tickers
start_date = '2018-01-01'
end_date = '2023-01-01'

# Fetch adjusted close prices from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily logarithmic returns
returns = np.log(data / data.shift(1)).dropna()

# Calculate mean annual return and covariance matrix
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev

# Function to minimize negative Sharpe ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

# Constraints: sum of weights is 1, weights are non-negative
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Initial guess (equal distribution)
initial_weights = np.array(len(tickers) * [1. / len(tickers)])

# Perform optimization to maximize Sharpe ratio
# optimized = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix),
#                      method='SLSQP', bounds=bounds, constraints=constraints)

# Retrieve optimal weights and portfolio performance
# optimal_weights = optimized.x
# expected_return, expected_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
# sharpe_ratio = (expected_return - 0.01) / expected_volatility

# print("Optimal Weights:", optimal_weights)
# print("Expected Annual Return:", expected_return)
# print("Expected Volatility:", expected_volatility)
# print("Sharpe Ratio:", sharpe_ratio)

# Function to target a specific portfolio volatility
def target_volatility(weights, mean_returns, cov_matrix, target_vol):
    _, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return (p_std_dev - target_vol)**2

# Example target volatility (e.g., 15% annualized)
target_vol = 0.10

# Perform optimization to achieve target volatility
optimized_vol = minimize(target_volatility, initial_weights, args=(mean_returns, cov_matrix, target_vol),
                         method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights_vol = optimized_vol.x
expected_return_vol, expected_volatility_vol = portfolio_performance(optimal_weights_vol, mean_returns, cov_matrix)

print("\nOptimal Weights for Target Volatility:", optimal_weights_vol)
print("Expected Annual Return:", expected_return_vol)
print("Expected Volatility:", expected_volatility_vol)

# Function to simulate random portfolios and plot the efficient frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        p_return, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = p_std_dev
        results[1,i] = p_return
        results[2,i] = (p_return - risk_free_rate) / p_std_dev
    return results, weights_record

# Generate and plot the efficient frontier
results, weights_record = efficient_frontier(mean_returns, cov_matrix)

# print(weights_record)
# plot the portfolio weights as a bar graph
plt.bar(tickers, optimal_weights_vol)
plt.title('Portfolio Weights')
plt.xlabel('Portfolio')
plt.ylabel('Weight')
plt.show()

# plot the weights record
plt.plot(weights_record)
plt.title('Weights Record')
plt.xlabel('Portfolio')
plt.ylabel('Weight')
plt.show()

print(results, type(results), results.shape)
plt.figure(figsize=(10, 7))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.show()
