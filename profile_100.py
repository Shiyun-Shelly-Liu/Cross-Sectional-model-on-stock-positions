# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:57:30 2024

@author: Shiyun Liu
"""

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
# %% Data collection
# Define the list of stock tickers
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'NVDA', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'DIS', 'ADBE', 'PYPL', 'NFLX', 'XOM', 'CMCSA', 'KO',
    'NKE', 'PFE', 'PEP', 'MRK', 'INTC', 'CSCO', 'T', 'ABT', 'CVX', 'VZ',
    'ABBV', 'LLY', 'MCD', 'DHR', 'WMT', 'BMY', 'MDT', 'HON', 'BA', 'AMGN',
    'UNP', 'IBM', 'LIN', 'LOW', 'PM', 'MS', 'NEE', 'RTX', 'TXN', 'CAT',
    'GS', 'BLK', 'GE', 'SPGI', 'LMT', 'PLD', 'ADP', 'MMM', 'ZTS', 'ISRG',
    'CHTR', 'AMT', 'MDLZ', 'TMO', 'GILD', 'USB', 'CI', 'SBUX', 'SCHW', 'FIS',
    'BKNG', 'SYK', 'TGT', 'CSX', 'CB', 'BDX', 'EL', 'REGN', 'NSC', 'MO',
    'WM', 'SO', 'APD', 'COP', 'NOW', 'CL', 'AXP', 'ICE', 'BSX', 'VRTX',
    'MMC', 'COST', 'SPG', 'ADI', 'KLAC', 'EW', 'D', 'HUM', 'TJX', 'LRCX'
]
# Download historical data
data = yf.download(tickers, start='2015-01-01', end='2023-12-31')

# Save the DataFrame to an Excel file
output_file = 'raw_data.xlsx'
data.to_excel(output_file, index=True)

# %% simpler model, only using 6-month return as the position

# Create a DataFrame to store features
feature_data = pd.DataFrame(index=data.index)
price_data = pd.DataFrame(index=data.index)
# Calculate momentum and technical features for each ticker
for ticker in tickers:
    # Extract the adjusted close prices for the current ticker
    adj_close = data['Adj Close'][ticker]

    # # Ensure enough data points to calculate returns and moving averages
    # if len(adj_close) > 105:  # Check if there are more than 200 data points

    # Momentum features
    feature_data[ticker] = adj_close.pct_change(
        231)  # Calculate 5-month return

for ticker in tickers:
    price_data[(ticker)] = adj_close.shift(-21).dropna()


# output_file = 'feature_data.xlsx'
# feature_data.to_excel(output_file, index=True)

# Drop rows with NaN values
feature_data = feature_data.dropna()

# Calculate the mean of each feature across all stocks
feature_means = feature_data.mean(axis=0)

feature_std = feature_data.std(axis=0)

standardized_feature_data = (feature_data - feature_means) / feature_std
# Print the standardized feature data for verification
print('Standardized Feature Data:')
print(standardized_feature_data)

# %%
positions = standardized_feature_data
aligned_positions = positions.shift(1).fillna(0)

# aligned_positions = aligned_positions.div(
#     aligned_positions.abs().sum(axis=1), axis=0)

returns = price_data[[
    f'{ticker}' for ticker in tickers]].pct_change().dropna()

portfolio_returns = (returns[230:] * aligned_positions[1:]).sum(axis=1)
print(portfolio_returns)

# Calculate performance metrics
sharpe_ratio = (portfolio_returns.mean() /
                portfolio_returns.std()) * np.sqrt(252)
gmv = np.sum(np.abs(aligned_positions))

turnover = np.sum(np.sum(np.abs(aligned_positions.diff().dropna())
                         )) / np.sum(gmv)

bias = np.sum(portfolio_returns)/np.sum(gmv)

print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Turnover Ratio: {turnover}')
print(f'Bias: {bias}')
# %%
# plotting the portfolio returns
plt.plot(portfolio_returns.index, portfolio_returns.cumsum())
title = '100_3months'
plt.title(title)
plt.savefig(f'plots/{title}.pdf', dpi=600)
