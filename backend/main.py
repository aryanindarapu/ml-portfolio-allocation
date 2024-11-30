from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
import numpy as np

from quant.strategies import mvp, rpp, portfolio_optimization
from quant.utils import get_factors, get_monthly_returns, get_stock_details, get_portfolio_weights, select_factor_list, visualize_returns
from quant.helpers import run_regression, compute_annual_return, get_insights, forecast_factors, compute_monthly_expected_returns

origins = [
    "http://localhost:3000",
    # Add other origins if needed
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods
    allow_headers=["*"],    # Allow all headers
)

class DataRequest(BaseModel):
    start_date: str
    end_date: str
    initial_amount: float
    tickers: List[str]
    strategy: str
    risk_level: str

@app.post("/process-data")
async def process_data(request: DataRequest):
    # Placeholder for data processing logic
    # For now, we'll just return the received data
    return {
        "message": "Data processed successfully",
        "data": request.model_dump()
    }
    
@app.get("/tickers")
async def get_tickers():
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
    return tickers

def risk_adjusted_portfolio(tickers, start_date, end_date, target_vol):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std_dev

    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
        p_returns, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_returns - risk_free_rate) / p_std_dev

    # Constraints: sum of weights is 1, weights are non-negative
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))

    # Initial guess (equal distribution)
    initial_weights = np.array(len(tickers) * [1. / len(tickers)])

    # Function to target a specific portfolio volatility
    def target_volatility(weights, mean_returns, cov_matrix, target_vol):
        _, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        return (p_std_dev - target_vol)**2

    # Perform optimization to achieve target volatility
    optimized_vol = minimize(target_volatility, initial_weights, args=(mean_returns, cov_matrix, target_vol),
                            method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights_vol = optimized_vol.x
    expected_return_vol, expected_volatility_vol = portfolio_performance(optimal_weights_vol, mean_returns, cov_matrix)

    print("\nOptimal Weights for Target Volatility:", optimal_weights_vol)
    print("Expected Annual Return:", expected_return_vol)
    print("Expected Volatility:", expected_volatility_vol)
    
    # weights_dict = {k: v for k, v in zip(tickers, optimal_weights_vol)}
    # return optimal_weights_vol

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
    
    return optimal_weights_vol, results
    
@app.post('/portfolio-analysis')
async def run_portfolio_analysis(request: DataRequest):
    tickers, start_date, end_date, initial_amount, strategy, risk_level = \
        request.tickers, request.start_date, request.end_date, request.initial_amount, request.strategy, request.risk_level
    
    # TODO: adjust the stock tickers based on risk level
    # TODO: keep track of portfolio over time and assess performance (server + db, etc.)
    l = None
    if risk_level == "very_low":
        l = 0.15
    elif risk_level == "low":
        l = 0.25
    elif risk_level == "medium":
        l = 0.31
    elif risk_level == "high":
        l = 0.375
    elif risk_level == "very_high":
        l = 0.45
    
    weights, frontier = risk_adjusted_portfolio(tickers, start_date, end_date, l)
    weights_dict = {k: v * initial_amount for k, v in zip(tickers, weights)}
    
    output = {
        "weights": weights_dict,
        "frontier": frontier.tolist()
    }
    
    return output
