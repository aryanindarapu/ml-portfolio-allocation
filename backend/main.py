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
    model_type: str
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

# def portfolio_allocation(tickers, start_date, end_date, strategy):
    
#     return weights, factors_subset

# def factor_model(factors_subset, model_type):
    
#     return 

    
@app.post('/portfolio-analysis')
async def run_portfolio_analysis(request: DataRequest):
    tickers, start_date, end_date, initial_amount, strategy, model_type, risk_level = \
        request.tickers, request.start_date, request.end_date, request.initial_amount, request.strategy, request.model_type, request.risk_level
    
    portfolio = get_monthly_returns(tickers, start_date, end_date)
    factors_monthly = get_factors()
    factors_subset = factors_monthly[factors_monthly.index.isin(portfolio.index)].copy()
    weights = get_portfolio_weights(tickers, strategy=strategy, start_date=start_date, 
                                    end_date=end_date, l=5, rf=factors_subset["RF"].mean())
    portfolio_return = (portfolio * weights).sum(axis=1)
    factors_subset["Excess Returns"] = portfolio_return - factors_subset["RF"]
    
    # =======================================================================================
    
    factors_monthly = get_factors()
    factor_list = select_factor_list(model_type)
    model = run_regression(factors_subset, factor_list)

    forecasted_factors = forecast_factors(factors_monthly, model_type=model_type, steps=12)
    monthly_expected_returns = compute_monthly_expected_returns(model, forecasted_factors)
    annual_expected_return = compute_annual_return(monthly_expected_returns)

    # print(monthly_expected_returns)
    print("Expected Annual Return for the Next 12 Months: {:.2f}%".format(annual_expected_return * 100))
    portfolio_return = (portfolio * weights).sum(axis=1)
    
    # =======================================================================================
    
    historical_returns = portfolio_return
    forecasted_returns = monthly_expected_returns
    
    # add last point of historical returns to forecasted returns
    forecasted_returns = [historical_returns.iloc[-1]] + forecasted_returns

    # Create a DataFrame for forecasted returns
    forecast_index = pd.period_range(start=historical_returns.index[-1] + 1, periods=13, freq='M')
    forecasted_returns_df = pd.Series(forecasted_returns, index=forecast_index, name='Forecasted Returns')
        
    # Calculate portfolio value over time
    historical_portfolio_value = (1 + historical_returns / 100).cumprod() * initial_amount
    forecasted_portfolio_value = (1 + forecasted_returns_df / 100).cumprod() * historical_portfolio_value.iloc[-1]
    
     # Directory to save plots
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Historical and Forecasted Portfolio Returns
    plt.figure(figsize=(10, 5))
    historical_returns.plot(label='Historical Returns (Percentage Change)', color='blue')
    forecasted_returns_df.plot(label='Forecasted Returns (Percentage Change)', color='red', linestyle='--')
    plt.title('Historical and Forecasted Portfolio Returns (Percentage Change)')
    plt.xlabel('Date')
    plt.ylabel('Monthly Return (%)')
    plt.legend()
    plot1_path = os.path.join(plot_dir, 'returns_plot.png')
    plt.savefig(plot1_path)
    plt.close()

    # Plot 2: Historical and Forecasted Portfolio Value
    plt.figure(figsize=(10, 5))
    historical_portfolio_value.plot(label='Historical Portfolio Value', color='green')
    forecasted_portfolio_value.plot(label='Forecasted Portfolio Value', color='orange', linestyle='--')
    plt.title('Historical and Forecasted Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plot2_path = os.path.join(plot_dir, 'value_plot.png')
    plt.savefig(plot2_path)
    plt.close()

    # Return the paths to the saved plots
    return {"portfolio": weights * initial_amount, "plot1": plot1_path, "plot2": plot2_path}

@app.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    plot_path = os.path.join('plots', plot_name)
    print(plot_name)
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Plot not found")