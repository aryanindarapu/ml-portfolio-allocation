import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf

class FinancialDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
        """
        Args:
            csv_file (str): Path to the CSV file with daily returns data.
            sequence_length (int): Number of days in each sequence.
        """
        self.sequence_length = sequence_length
        self.data = pd.read_csv(csv_file, index_col='Date')
        
        # Ensure data is a PyTorch tensor
        self.returns = torch.tensor(self.data.values, dtype=torch.float32)

    def __len__(self):
        # Number of possible sequences in the dataset
        return len(self.returns) - self.sequence_length

    def __getitem__(self, idx):
        # Sequence of returns for the model input
        x = self.returns[idx:idx + self.sequence_length]
        # The next day's returns, used as the target
        y = self.returns[idx + 1: idx + self.sequence_length + 1]
        
        return x, y

if __name__ == "__main__":
    # Define the ETFs
    etfs = ['AGG', 'DBC', '^VIX', 'VTI']
    start_date = "2006-01-01"
    end_date = "2023-12-31"

    # Download data
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Save to CSV
    returns.to_csv("etf_returns_2006_2023.csv")
    print("ETF returns saved.")