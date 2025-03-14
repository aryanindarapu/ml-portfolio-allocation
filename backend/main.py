from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Import our agents module
from agents import Agent, Runner, trace

# Your existing quant imports (assumed to be in your project)
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
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    print(data)
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_performance(weights, mean_returns, cov_matrix):
        ret = np.sum(mean_returns * weights)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, std_dev

    def target_volatility(weights, mean_returns, cov_matrix, target_vol):
        _, std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        return (std_dev - target_vol) ** 2

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    initial_weights = np.array(len(tickers) * [1. / len(tickers)])

    optimized = minimize(target_volatility, initial_weights, args=(mean_returns, cov_matrix, target_vol),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = optimized.x
    exp_return, exp_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

    print("\nOptimal Weights for Target Volatility:", optimal_weights)
    print("Expected Annual Return:", exp_return)
    print("Expected Volatility:", exp_vol)
    
    # Generate a basic efficient frontier (simulation with fewer portfolios)
    def efficient_frontier(mean_returns, cov_matrix, num_portfolios=1000):
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            ret, std = portfolio_performance(weights, mean_returns, cov_matrix)
            results[0, i] = std
            results[1, i] = ret
            results[2, i] = (ret - 0.01) / std  # Sharpe ratio with risk-free rate of 0.01
        return results
    frontier = efficient_frontier(mean_returns, cov_matrix)
    
    return optimal_weights, frontier

@app.post('/portfolio-analysis')
async def run_portfolio_analysis(request: DataRequest):
    tickers = request.tickers
    start_date = request.start_date
    end_date = request.end_date
    initial_amount = request.initial_amount
    strategy = request.strategy
    risk_level = request.risk_level
    
    # Map risk level to target volatility
    risk_map = {"very_low": 0.15, "low": 0.25, "medium": 0.31, "high": 0.375, "very_high": 0.45}
    if risk_level not in risk_map:
        raise HTTPException(status_code=400, detail="Invalid risk level provided")
    l = risk_map[risk_level]

    weights, frontier = risk_adjusted_portfolio(tickers, start_date, end_date, l)
    weights_dict = {k: v * initial_amount for k, v in zip(tickers, weights)}
    
    output = {
        "weights": weights_dict,
        "frontier": frontier.tolist()
    }
    
    return output

@app.post('/run-explanation-agents')
async def run_explanation_agents(request: DataRequest):
    # Use the risk-adjusted portfolio function to get the allocation summary.
    risk_map = {"very_low": 0.15, "low": 0.25, "medium": 0.31, "high": 0.375, "very_high": 0.45}
    if request.risk_level not in risk_map:
        raise HTTPException(status_code=400, detail="Invalid risk level provided")
    l = risk_map[request.risk_level]

    weights, _ = risk_adjusted_portfolio(request.tickers, request.start_date, request.end_date, l)
    allocation_summary = f"Portfolio allocation weights: {dict(zip(request.tickers, weights))}"

    with trace("Ticker Suggestion Agent"):
        ticker_agent = Agent(
            name="ticker_suggestion_agent",
            instructions="Based on the provided portfolio allocation, suggest additional tickers to boost potential returns."
        )
        ticker_suggestions = await Runner.run(ticker_agent, allocation_summary)
    
    return {"ticker_suggestions": ticker_suggestions}

@app.post('/run-efficient-frontier-agent')
async def run_efficient_frontier_agent(request: DataRequest):
    risk_map = {"very_low": 0.15, "low": 0.25, "medium": 0.31, "high": 0.375, "very_high": 0.45}
    if request.risk_level not in risk_map:
        raise HTTPException(status_code=400, detail="Invalid risk level provided")
    l = risk_map[request.risk_level]

    weights, _ = risk_adjusted_portfolio(request.tickers, request.start_date, request.end_date, l)
    allocation_summary = f"Optimized portfolio allocation weights: {dict(zip(request.tickers, weights))}"

    with trace("Portfolio Optimizer Agent"):
        optimizer_agent = Agent(
            name="portfolio_optimizer_agent",
            instructions="Generate the best portfolio allocation based on the optimization numbers."
        )
        optimized_allocation = await Runner.run(optimizer_agent, allocation_summary)
    
    with trace("Efficient Frontier Explanation Agent"):
        explainer_agent = Agent(
            name="efficient_frontier_explainer_agent",
            instructions="Explain the efficient frontier concept and why this allocation lies on it."
        )
        explanation = await Runner.run(explainer_agent, optimized_allocation)
    
    return {
        "optimized_allocation": optimized_allocation,
        "efficient_frontier_explanation": explanation
    }

@app.post('/run-help-agent')
async def run_help_agent(request: DataRequest):
    # Here we assume the "strategy" field holds a help question.
    query = request.strategy or "How can I help you with portfolio allocation?"
    with trace("Help Agent"):
        help_agent = Agent(
            name="help_agent",
            instructions="Answer any questions regarding the portfolio allocation app clearly and helpfully."
        )
        help_response = await Runner.run(help_agent, query)
    
    return {"help_response": help_response}

@app.post('/run-complex-workflow')
async def run_complex_workflow(request: DataRequest):
    # --- Input Guardrails ---
    if request.initial_amount <= 0:
        raise HTTPException(status_code=400, detail="Initial amount must be positive")
    valid_risk_levels = {"very_low", "low", "medium", "high", "very_high"}
    if request.risk_level not in valid_risk_levels:
        raise HTTPException(status_code=400, detail="Invalid risk level provided")
    
    # Map risk level to target volatility.
    risk_map = {"very_low": 0.15, "low": 0.25, "medium": 0.31, "high": 0.375, "very_high": 0.45}
    l = risk_map[request.risk_level]
    
    # --- Step 1: Compute the Basic Portfolio Allocation ---
    weights, _ = risk_adjusted_portfolio(request.tickers, request.start_date, request.end_date, l)
    allocation_summary = f"Optimized portfolio allocation weights: {dict(zip(request.tickers, weights))}"
    
    # --- Step 2: Iterative Optimization & Evaluation ---
    optimizer_agent = Agent(
        name="portfolio_optimizer_agent",
        instructions="Generate the best portfolio allocation based on the optimization numbers."
    )
    evaluator_agent = Agent(
        name="evaluator_agent",
        instructions="Evaluate the given portfolio allocation. Return 'pass' if it meets quality standards, else 'needs_improvement: <feedback>'."
    )
    
    max_iterations = 3
    iteration = 0
    optimized_allocation = allocation_summary
    evaluator_response = ""
    
    while iteration < max_iterations:
        with trace(f"Portfolio Optimizer Agent Iteration {iteration+1}"):
            optimized_allocation = await Runner.run(optimizer_agent, optimized_allocation)
        
        with trace("Evaluator Agent"):
            evaluator_response = await Runner.run(evaluator_agent, optimized_allocation)
        
        # If evaluator returns "pass", we consider the output acceptable.
        if "pass" in evaluator_response.lower():
            break
        else:
            # Append evaluator feedback to improve the allocation.
            optimized_allocation += f" | Feedback: {evaluator_response}"
        iteration += 1
    
    # --- Output Guardrail ---
    if "pass" not in evaluator_response.lower():
        raise HTTPException(status_code=500, detail="Optimized allocation did not meet quality standards after multiple iterations.")
    
    # --- Step 3: Run Additional Agents ---
    with trace("Ticker Suggestion Agent"):
        ticker_agent = Agent(
            name="ticker_suggestion_agent",
            instructions="Based on the provided portfolio allocation, suggest additional tickers to boost potential returns."
        )
        ticker_suggestions = await Runner.run(ticker_agent, optimized_allocation)
    
    with trace("Efficient Frontier Explanation Agent"):
        explainer_agent = Agent(
            name="efficient_frontier_explainer_agent",
            instructions="Explain the efficient frontier concept and its significance for this portfolio allocation."
        )
        explanation = await Runner.run(explainer_agent, optimized_allocation)
    
    # --- Step 4: Synthesize Final Output & Output Guardrail ---
    final_output = (
        f"Final Optimized Allocation:\n{optimized_allocation}\n\n"
        f"Ticker Suggestions:\n{ticker_suggestions}\n\n"
        f"Efficient Frontier Explanation:\n{explanation}"
    )
    
    if not final_output.strip():
        raise HTTPException(status_code=500, detail="Final output guardrail triggered: Empty output.")
    
    return {"final_output": final_output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
