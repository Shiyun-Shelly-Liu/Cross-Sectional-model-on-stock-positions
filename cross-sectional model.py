# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:08:18 2024

@author: Shiyun Liu
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import yfinance as yf

# %% Data collection
# Define the list of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Download historical data
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')

# Save the DataFrame to an Excel file
output_file = 'raw_data.xlsx'
data.to_excel(output_file, index=True)

# %% Feature engineering
# Calculate RSI (Relative Strength Index)


def compute_rsi(series, window=14):
    delta = series.diff()  # Calculate the difference between consecutive prices
    gain = delta.where(delta > 0, 0).rolling(
        window=window).mean()  # Calculate the rolling mean of gains
    # Calculate the rolling mean of losses
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss  # Calculate the relative strength
    return 100 - (100 / (1 + rs))  # Calculate the RSI


# Create a DataFrame to store features
feature_data = pd.DataFrame(index=data.index)

# Calculate momentum and technical features for each ticker
for ticker in tickers:
    # Extract the adjusted close prices for the current ticker
    adj_close = data['Adj Close'][ticker]

    # Ensure enough data points to calculate returns and moving averages
    if len(adj_close) > 200:  # Check if there are more than 200 data points

        # Momentum features
        feature_data[(ticker, '1M_return')] = adj_close.pct_change(
            21)  # Calculate 1-month return
        feature_data[(ticker, '3M_return')] = adj_close.pct_change(
            63)  # Calculate 3-month return

        # Moving averages
        feature_data[(ticker, '50_MA')] = adj_close.rolling(
            window=50).mean()  # Calculate 50-day moving average
        feature_data[(ticker, '200_MA')] = adj_close.rolling(
            window=200).mean()  # Calculate 200-day moving average

        # RSI
        feature_data[(ticker, 'RSI')] = compute_rsi(adj_close)  # Calculate RSI

        # Add adjusted close prices to the feature_data DataFrame for target variable alignment
        feature_data[(ticker, 'Adj_Close')] = adj_close

# Drop rows with NaN values
feature_data = feature_data.dropna()

# Flatten the MultiIndex columns
feature_data.columns = ['_'.join(col).strip()
                        for col in feature_data.columns.values]

output_file = 'feature_data.xlsx'
feature_data.to_excel(output_file, index=True)
# %%

# Split the data into training and testing sets
train_data = feature_data[:'2022-01-01']
test_data = feature_data['2022-01-02':]

# Prepare features and target
features = ['1M_return', '3M_return', '50_MA', '200_MA', 'RSI']
target = 'Adj_Close'

# Initialize empty lists to store training and testing data for all tickers
X_train_list, y_train_list, X_test_list = [], [], []
# Track the number of rows for each ticker in X_test
test_lengths = []

for ticker in tickers:
    # Prepare the training data
    X_train = train_data[[
        f'{ticker}_{feature}' for feature in features]].dropna()
    y_train = train_data[(f'{ticker}_{target}')].shift(-21).dropna()

    # Align X and y
    X_train = X_train.loc[y_train.index]

    # Prepare the testing data
    X_test = test_data[[
        f'{ticker}_{feature}' for feature in features]].dropna()

    test_lengths.append(len(X_test))

    # Append the data for this ticker to the lists
    X_train_list.append(X_train)
    y_train_list.append(y_train)
    X_test_list.append(X_test)

# Concatenate the training and testing data for all tickers
X_train = pd.concat(X_train_list, axis=1)
y_train = pd.concat(y_train_list, axis=1)
X_test = pd.concat(X_test_list, axis=1)

# Train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


# %%
# Make predictions
predictions = -1*model.predict(X_test)
percentage_threshold = 0.01

# Initialize a DataFrame for positions
positions = pd.DataFrame(0, index=X_test.index, columns=tickers)

start_idx = 0
# Assuming all tickers have the same length in test_lengths
end_idx = max(test_lengths)

# Loop over tickers
for ticker in tickers:
    print(ticker, start_idx, end_idx)

    # Extract the relevant predictions for the ticker
    ticker_predictions = predictions[:, tickers.index(ticker)]
    print('ticker_predictions', ticker_predictions)

    # Get the current price for comparison
    current_prices = test_data[f'{ticker}_{target}']
    print('current_prices', current_prices)

    # Determine positions based on the percentage threshold
    long_positions = (ticker_predictions > current_prices *
                      (1 + percentage_threshold)).astype(int)
    short_positions = (ticker_predictions < current_prices *
                       (1 - percentage_threshold)).astype(int)
    print('long_positions', long_positions)
    print('short_positions', short_positions)

    # Combine positions
    positions[ticker] = long_positions - short_positions
    print('positions', positions)

# %%

# Calculate returns for the testing period
returns = test_data[[
    f'{ticker}_{target}' for ticker in tickers]].pct_change().dropna()

# Align positions with returns index
aligned_positions = positions.shift(1).reindex(returns.index).fillna(0)

# Check and align column names
if not returns.columns.equals(aligned_positions.columns):
    returns.columns = aligned_positions.columns  # Align column names

# Calculate portfolio returns
portfolio_returns = (returns * aligned_positions).mean(axis=1)


# Calculate performance metrics
sharpe_ratio = (portfolio_returns.mean() /
                portfolio_returns.std()) * np.sqrt(252)
turnover = np.sum(np.abs(aligned_positions.diff().dropna())
                  ) / len(aligned_positions)
bias = ((predictions -
        test_data[[f'{ticker}_{target}' for ticker in tickers]]) * 10000).mean()

print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Turnover Ratio: {turnover}')
print(f'Bias (in basis points): {bias}')
