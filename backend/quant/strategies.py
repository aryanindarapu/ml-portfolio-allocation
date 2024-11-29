import pandas as pd
import numpy as np
from math import sqrt
from scipy.optimize import minimize

""" These methods are simple heuristics and don't consider correlation between assets.
"""
# Equal Weight Portfolio
def ewp(data: pd.DataFrame):
    """ The simplest method, where all assets are given equal weight in the portfolio."""
    weights = [1 / data.shape[1]] * data.shape[1]
    weights = pd.Series(weights, index=data.columns)
        
    return weights

# Inverse Volatility Portfolio
def ivp(data: pd.DataFrame):
    """ Returns the optimal portfolio allocation using the Inverse Volatility Portfolio method. This method aims to allocate
    the weights inversely proportional to the volatility of the assets."""
    
    std = data.std()

    weights = 1 / std
    weights /= weights.sum()
    weights = pd.Series(weights, index=data.columns)
    
    # return weights * initial_amount, weights
    return weights

# Global Minimum Variance - ignore expected returns and focus on minimizing the portfolio variance
def gmvp(data: pd.DataFrame):
    """ Returns the optimal portfolio allocation using the Global Minimum Variance Portfolio method. This method solely aims to minimize the 
    portfolio's variance, but completely ignores the expected returns."""
    
    inv_cov_matrix = np.linalg.inv(data.cov())
    ones = np.ones(len(data.columns))
    weights = (inv_cov_matrix @ ones) / (ones.T @ inv_cov_matrix @ ones)
    weights /= weights.sum()
    weights = pd.Series(weights, index=data.columns)
    
    return weights

# Mean-Variance (Markowitz Portfolio)
def mvp(data: pd.DataFrame, l: float):
    """Returns the optimal portfolio allocation using the Markowitz Portfolio Optimization method. This method aims to balance the 
    trade-off between risk and return by minimizing the negative of the risk-adjusted return (i.e. the variance) 
    while maximizing the expected return.

    Args:
        data (pd.DataFrame): The time series data of the assets
        l (float): The risk coefficient (controls the trade-off between risk and return). A high value of l will result in a more conservative portfolio.
    """
    # Calculate mean returns and inverse covariance matrix
    mean_returns = data.mean().values
    cov_matrix = data.cov().values

    # Objective function: Minimize the negative of the risk-adjusted return
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.dot(weights.T, np.dot(cov_matrix, weights))
        return - (portfolio_return - l * portfolio_risk)
    
    # Constraints: Sum of weights is 1, all weights are non-negative
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = [(0, 1) for _ in range(len(mean_returns))]

    initial_weights = np.ones(len(mean_returns)) / len(mean_returns)
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization did not converge")
    
    return pd.Series(result.x, index=data.columns)
    
    # ones = np.ones(len(data.columns))
    
    # mean_returns = data.mean()
    # inv_cov_matrix = np.linalg.inv(data.cov())
    
    # # Ensure alpha is non-negative
    # alpha_numerator = 2 * l - (ones.T @ inv_cov_matrix @ mean_returns)
    # alpha_denominator = ones.T @ inv_cov_matrix @ ones
    
    # alpha = alpha_numerator / alpha_denominator
    # print(alpha)
    # weights = inv_cov_matrix @ (mean_returns + alpha * ones)
    # weights /= 2 * l
    
    # weights /= weights.sum()
    # # print(weights)
    # return weights
    
def nrbp(data: pd.DataFrame):
    """Returns the optimal portfolio allocation using the Naive Risk Budgeting Portfolio method. Each asset is given a weight
    such that it contributes equally to the overall portfolio risk. Note that this method assumes that the returns are diagonal, i.e.
    there are no correlations between the returns of the assets (independence assumption).

    Args:
        data (pd.DataFrame): The time series data of the assets
    """
    N = len(data.columns)
    std = data.std()
    weights = [sqrt(1/N) / std[i] for i in range(N)]
    weights /= sum(weights)
    weights = pd.Series(weights, index=data.columns)
    return weights

def rpp(data: pd.DataFrame):
    """ Risk Parity Portfolio. Each asset contributes equally to the overall portfolio risk, with no assumptions on the data. """
    cov = data.cov().values
    w_0 = x_0 = [1 / data.shape[1]] * data.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
    result = minimize(_risk_budget_objective, w_0, args=[cov, x_0], method='SLSQP', constraints=constraints, options={'disp': True, 'ftol': 1e-20})
    weights = np.asmatrix(result.x)
    weights = pd.Series(np.array(weights)[0], index=data.columns)
    return weights

def _portfolio_variance(w, V):
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def _risk_contribution(w, V):
    w = np.matrix(w)
    sigma = np.sqrt(_portfolio_variance(w, V))
    MRC = V * w.T # Marginal Risk Contribution
    RC = np.multiply(MRC, w.T)/sigma # Risk Contribution
    return RC

# Objective function is to minimize risk contribution difference via sum of squared error
def _risk_budget_objective(x, args):
    cov, x_t = args
    sig_p = np.sqrt(_portfolio_variance(x, cov)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = _risk_contribution(x, cov)
    J = sum(np.square(asset_RC - risk_target.T))[0,0] # * 1000 # NOTE: to run more than 1 iteration # sum of squared error
    return J

def srp(data: pd.DataFrame, risk_free_rate: float):
    """Maximizes the Sharpe ratio of a portfolio using historical returns.

    Args:
        data (pd.DataFrame): The time series data of the assets
        risk_free_rate (float): The risk-free rate

    Returns:
        np.ndarray: The optimal portfolio weights that maximize the Sharpe ratio
    """
    # Calculate mean returns and covariance matrix
    mean_returns = data.mean().values
    cov_matrix = data.cov().values
    num_assets = len(mean_returns)

    # Objective function: Minimize the negative of the Sharpe ratio
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        return -sharpe_ratio  # Minimize the negative of the Sharpe ratio
    
    # Constraints: Sum of weights is 1, all weights are non-negative
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization did not converge")

    return pd.Series(result.x, index=data.columns)

# Portfolio Transformer - https://arxiv.org/pdf/2206.03246

# Max Diversification Portfolio
# Maximum Sharpe Ratio Portfolio
# Sharpe Ratio Maximization

# Mix of Markowitz and Risk Parity portfolio