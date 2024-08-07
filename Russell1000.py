# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:13:30 2024

@author: Shiyun Liu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
# %% Data collection
# Fetch the Russell 1000 Index tickers from Wikipedia
url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
tables = pd.read_html(url)
df = tables[2]

# Clean up ticker symbols (remove extra spaces, replace periods with dashes)
df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
tickers = df['Symbol'].tolist()

# Download the data
data = yf.download(tickers, start='2015-01-01', end='2023-12-31')

# Calculate the number of NaN values for each ticker
nan_counts = data.isna().sum()

# Identify tickers with non-zero NaN values
non_zero_nan_tickers = nan_counts[nan_counts >
                                  0].index.get_level_values(1).unique()

# Remove the invalid tickers
tickers = [ticker for ticker in tickers if ticker not in non_zero_nan_tickers]

data = yf.download(tickers, start='2015-01-01', end='2023-12-31')
# %% model composition

# Creating a dataFrame containing adjusted closing prices for all tickers
closes = data.loc[:, pd.IndexSlice['Adj Close', :]]

# Calculates the percentage change of the adjusted closing prices to get the returns
returns = closes.pct_change().dropna(axis=1, how='all')

# Calculates the rolling mean of the returns over a window of 11 months
signal = returns.rolling(window=252-21, min_periods=21).mean().shift(21)

# Calculate the z-score of signal
signal = signal.sub(signal.mean(axis=1), axis=0).div(
    signal.std(axis=1), axis=0)

# Scaling the signal to unit gmv
signal = signal.div(signal.abs().sum(axis=1), axis=0)

# portfolio returns per ticker
pnl = signal.mul(returns).dropna(axis=0, how='all')

# total portfolio returns across all tickers
pnl_total = pnl.sum(axis=1)

# %% Metrics calculation

# Function for calculating the Sharpe_ratio


def sharpe(pnl: pd.Series) -> float:
    return pnl.mean() / pnl.std() * np.sqrt(252)

# Function for calculating the turnover rate


def turnover(pos: pd.DataFrame) -> float:
    return pos.diff().abs().sum(axis=1).div(
        pos.abs().sum(axis=1)
    ).mean()

# Function for calculating the bias


def bias(pnl: pd.Series, pos: pd.DataFrame) -> float:
    return 1e4 * pnl.sum() / pos.diff().abs().sum(axis=1).sum()


# Compute the metrics
sharpe_ratio = sharpe(pnl_total)
turnover = turnover(signal)
bias = bias(pnl_total, signal)
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Turnover Ratio: {turnover}')
print(f'Bias: {bias}')

# %% Plotting total pnl over time

# plotting the portfolio returns
plt.plot(pnl_total.index, pnl_total.cumsum())
title = 'R1000_12months'
plt.title(title)
plt.savefig(f'plots/{title}.pdf', dpi=600)
