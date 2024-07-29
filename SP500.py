# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:45:31 2024

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
# Fetch the S&P 500 tickers from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
df = tables[0]

# Clean up ticker symbols (remove extra spaces, replace periods with dashes)
df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
tickers = df['Symbol'].tolist()

# Remove the invalid tickers
invalid_tickers = ['SOLV', 'SW', 'GEV', 'ABNB', 'CARR', 'CEG', 'CRWD', 'CTVA', 'DAY', 'DOW', 'ETSY', 'FOX', 'FOXA',
                   'FTV', 'GDDY', 'GEHC', 'HPE', 'HWM', 'INVH', 'IR', 'KHC', 'KVUE', 'LW',
                   'MRNA', 'OTIS', 'PYPL', 'UBER', 'VICI', 'VLTO', 'VST']
tickers = [ticker for ticker in tickers if ticker not in invalid_tickers]

print(tickers)
# %%
# Download the data
data = yf.download(tickers, start='2015-01-01', end='2023-12-31')

# %%
# Calculate the number of NaN values for each ticker
nan_counts = data.isna().sum()

# Identify tickers with non-zero NaN values
non_zero_nan_tickers = nan_counts[nan_counts >
                                  0].index.get_level_values(1).unique()

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
        105)  # Calculate 5-month return

# Assuming 'data' is your existing DataFrame containing 'Adj Close' prices
shifted_data = {}

for ticker in tickers:

    adj_close = data['Adj Close'][ticker]

    # Shift the adjusted close prices by -21 and drop NA values
    shifted_data[ticker] = adj_close.shift(-21)

# Concatenate all shifted data into a single DataFrame and drop rows with NA values
price_data = pd.concat(shifted_data, axis=1).dropna()

# Print the resulting DataFrame
print(price_data)


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

portfolio_returns = (returns[104:] * aligned_positions[1:]).sum(axis=1)
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
title = 'SP500_3months'
plt.title(title)
plt.savefig(f'plots/{title}.pdf', dpi=600)
