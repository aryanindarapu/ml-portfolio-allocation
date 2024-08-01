import pandas as pd
import numpy as np
from math import sqrt
from scipy.optimize import minimize
# import pdb


""" These methods are simple heuristics and don't consider correlation between assets.
"""
# Equal Weight Portfolio
def ewp(data: pd.DataFrame, initial_amount: float):
    # pdb.set_trace()
    # weights = np.array([1 / data.shape[1]] * data.shape[1])
    weights = [1 / data.shape[1]] * data.shape[1]
    weights = pd.Series(weights, index=data.columns)
        
    return weights * initial_amount, weights

# Mean-Variance (Markowitz Portfolio)
def mvp(data: pd.DataFrame, initial_amount: float, l: float):
    """Returns the optimal portfolio allocation using the Markowitz Portfolio Optimization method.

    Args:
        data (pd.DataFrame): The time series data of the assets
        initial_amount (float): The initial amount to invest
        l (float): The risk coefficient (controls the trade-off between risk and return)
    """
    
    mean_returns = data.mean()
    std_inv = 1 / data.std()
    cov_matrix = data.cov()
    
    alpha = 2 * l - std_inv.dot(mean_returns)
    alpha /= std_inv.sum()
    
    weights = np.linalg.inv(cov_matrix).dot(mean_returns + alpha.sum())
    weights /= 2 * l
    
    return weights * initial_amount, weights
    
# Global Minimum Variance - ignore expected returns and focus on minimizing the portfolio variance
def gmvp(data: pd.DataFrame, initial_amount: float):
    std_inv = 1 / data.std()
    weights = std_inv / std_inv.sum()
    
    return weights * initial_amount, weights

""" These methods are focused on balancing risk between the assets in the portfolio. They consider the 
    volatility of the assets and the correlation between them.
"""

# Naive Risk Parity Portfolio 
def ivp(data: pd.DataFrame, initial_amount: float):
    std = data.std()

    weights = 1 / std
    weights /= weights.sum()
    
    return weights * initial_amount, weights

# Naive Risk Budgeting Portfolio - each asset contributes equally to the overall portfolio risk
def nrbp(data: pd.DataFrame, initial_amount: float):
    """Returns the optimal portfolio allocation using the Naive Risk Budgeting Portfolio method.

    Args:
        data (pd.DataFrame): The time series data of the assets
        initial_amount (float): The initial amount to invest
    """
    N = len(data.columns)
    std = data.std()
    weights = [sqrt(1/N) / std[i] for i in range(N)]
    weights /= sum(weights)
    
    return weights * initial_amount, weights

def rpo(w, cov):
    cov_w = np.dot(cov, w)
    # a = (w[:, None] * cov_w - w * cov_w[:, None]) ** 2
    term = np.outer(w, cov_w) - np.outer(cov_w, w)
    return np.sum(term ** 2)
    
# Risk Parity Portfolio - each asset contributes equally to the overall portfolio risk
def rpp(data: pd.DataFrame, initial_amount: float):
    cov_matrix = data.cov()
    num_assets = cov_matrix.shape[0]
    initial_weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    weights = minimize(rpo, initial_weights, args=(cov_matrix,), method='SLSQP',
                      bounds=bounds, constraints=constraints)
    
    print(weights.x)
    return pd.Series(weights.x) * initial_amount, pd.Series(weights.x)
    
    
# def rpp(data: pd.DataFrame, initial_amount: float):
#     # Calculate the returns
#     returns = data.pct_change().dropna()

#     # Compute the covariance matrix of the returns
#     cov_matrix = returns.cov().values

#     # Number of assets
#     N = cov_matrix.shape[0]

#     # Objective function for risk parity
#     def risk_parity_objective(w, grad):
#         Sigma_w = np.dot(cov_matrix, w)
#         term = np.outer(w, Sigma_w) - np.outer(Sigma_w, w)
#         if grad.size > 0:
#             grad[:] = 4 * (np.dot(cov_matrix, np.dot(cov_matrix, w)) - Sigma_w * np.sum(Sigma_w))
#         return np.sum(term**2)

#     # Constraint function: sum of weights is 1
#     def constraint_sum_weights(result, w, grad):
#         result[0] = np.sum(w) - 1
#         if grad.size > 0:
#             grad[:] = np.ones_like(w)
#         return result

#     # Initial guess for the weights
#     w0 = np.ones(N) / N

#     # Create optimizer
#     opt = nlopt.opt(nlopt.LD_SLSQP, N)

#     # Set objective function
#     opt.set_min_objective(risk_parity_objective)

#     # Set constraint
#     opt.add_equality_constraint(lambda w, grad: constraint_sum_weights(np.array([0.0]), w, grad), 1e-8)

#     # Set bounds for weights
#     opt.set_lower_bounds(0.0)
#     opt.set_upper_bounds(1.0)

#     # Set stopping criteria
#     opt.set_xtol_rel(1e-6)

#     # Optimize
#     optimal_weights = opt.optimize(w0)
#     minf = opt.last_optimum_value()

#     # Compute the allocation based on the optimal weights
#     allocation = optimal_weights * initial_amount

#     return optimal_weights, allocation, minf

# Max Diversification Portfolio - 

# Maximum Sharpe Ratio Portfolio
# Sharpe Ratio Maximization

