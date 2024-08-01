import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

from utils import prepare_features

""" Main Functions """

def run_regression(data, factor_list):
    X = sm.add_constant(data[factor_list])
    y = data["Excess Returns"]
    model = sm.OLS(y, X).fit()
    # model.summary()
    
    return model

def get_insights(client, dataframe, portfolio, stock_details, advanced=False):
    data_string = dataframe.to_string()
    portfolio_string = portfolio.to_string()
    stock_details_string = stock_details.to_string()

    prompt = f"""
    Analyze the following regression results data and provide insights:

    {data_string}

    Additionally, here is the portfolio composition with associated weights:

    {portfolio_string}

    Here are the details of the selected stocks:

    {stock_details_string}

    Determine which factor correlations affect which tickers in the portfolio the most based on the regression coefficients and provide a detailed analysis of how these factors relate to each stock versus the entire portfolio.
    """
    
    if advanced:
        messages = [
            {"role": "system", "content": "You are a financial analyst who provides insights on regression analysis for either a single stock or a portfolio of stocks. You are expected to provide more detailed and technical insights."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a financial analyst who provides insights on regression analysis for either a single stock or a portfolio of stock. You are speaking to a general audience and should provide high-level insights without getting too technical."},
            {"role": "user", "content": prompt}
        ]
    
    completion = client.chat.completions.create(model='gpt-4o-mini', messages=messages)
    
    return completion.choices[0].message.content.strip()

def forecast_factors(factors, model_type, steps=12):
    forecasted_factors = {}
    
    if model_type == "capm":
        factor_headers = ["Mkt-RF"]
    elif model_type == "ff3":
        factor_headers = ["Mkt-RF", "SMB", "HML"]
    elif model_type == "ff5":
        factor_headers = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    else:
        raise ValueError("Invalid model type. Choose from 'capm', 'ff3', or 'ff5'.")
        
    lag = 12  # Number of lagged values to use as features
    
    for factor in factor_headers:
        df = prepare_features(factors[factor], lag)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
        
        last_values = X[-1].reshape(1, -1)
        forecast = []
        
        for _ in range(steps):
            next_value = model.predict(last_values)[0]
            forecast.append(next_value)
            last_values = np.roll(last_values, -1)
            last_values[0, -1] = next_value
        
        forecasted_factors[factor] = pd.Series(forecast, index=pd.period_range(start=factors.index[-1] + 1, periods=steps, freq='M'))
    
    return forecasted_factors

def compute_monthly_expected_returns(model, forecasted_factors):
    monthly_expected_returns = []
    for i in range(12):
        monthly_return = model.params['const']
        for factor in forecasted_factors:
            monthly_return += model.params[factor] * forecasted_factors[factor].iloc[i]
        monthly_expected_returns.append(monthly_return)
    return monthly_expected_returns

def compute_annual_return(monthly_expected_returns):
    cumulative_return = 1
    for monthly_return in monthly_expected_returns:
        cumulative_return *= (1 + monthly_return / 100)
    annual_return = cumulative_return - 1
    return annual_return