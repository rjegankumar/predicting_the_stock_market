import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# reading the historical S&P 500 price data
sp500 = pd.read_csv('sphist.csv')

# converting date column to datetime format
sp500['Date'] = pd.to_datetime(sp500['Date'], format='%Y-%m-%d')

# sorting the dataframe in ascending order of date
sp500.sort_values('Date', ascending=True, inplace=True)
sp500.reset_index(inplace=True, drop=True)

# computing rolling/ moving weekly and monthly averages of the closing price
sp500['roll_wk_close'] = sp500['Close'].rolling(window = 5).mean()
sp500['roll_mon_close'] = sp500['Close'].rolling(window = 21).mean()

# computing the rolling ratio between the highest closing price in the previous year to the current closing price
sp500['high_curr_close'] = sp500['Close'].rolling(window = 252).max()/ sp500['Close']

sp500[['roll_wk_close','roll_mon_close','high_curr_close']] = \
sp500[['roll_wk_close','roll_mon_close','high_curr_close']].shift()

# assingning day of week to a new column to be used as a feature
sp500['dow'] = sp500['Date'].dt.dayofweek

# dropping rows with NaN values
sp500.dropna(axis=0, inplace=True)

# selecting train and test data
train = sp500[sp500['Date']<datetime(year=2013, month=1, day=1)]
test = sp500[sp500['Date']>=datetime(year=2013, month=1, day=1)]

# training and predicting daily closing S&P 500 prices using a linear regression model
lr = LinearRegression()
lr.fit(train[['roll_wk_close','roll_mon_close','high_curr_close','dow']], train['Close'])
test['close_prediction'] = lr.predict(test[['roll_wk_close','roll_mon_close','high_curr_close','dow']])

# calculating mean absolute error of the model
mae = mean_absolute_error(test['Close'], test['close_prediction'])
print(mae)
print(test.tail())
