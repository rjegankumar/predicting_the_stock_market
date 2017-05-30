import pandas as pd
import numpy as np

# reading the historical S&P 500 price data
sp = pd.read_csv('sphist.csv')

# converting date column to datetime format
sp['Date'] = pd.to_datetime(sp['Date'], format='%Y-%m-%d')

# sorting the dataframe in ascending order of date
sp.sort_values('Date', ascending=True, inplace=True)
sp.reset_index(inplace=True, drop=True)

# filtering dataframe to be in the range from oldest year + 1 to newest year - 1
max_year = max(sp['Date'].dt.year)
min_year = min(sp['Date'].dt.year)
sp = sp[(sp['Date'].dt.year > min_year) & (sp['Date'].dt.year < max_year)]

# Computing stats of days in year, month and week
yr_mon_wk = pd.DataFrame()
yr_mon_wk['year'] = sp['Date'].dt.year
yr_mon_wk['month'] = sp['Date'].dt.month
yr_mon_wk['week'] = sp['Date'].dt.week

print('Days in year stats','\n')
print(yr_mon_wk['year'].value_counts().describe(),'\n')
print(yr_mon_wk['year'].value_counts().mode()[0],'\n')
print('Days in month stats','\n')
print(pd.crosstab(index= yr_mon_wk['year'], columns= yr_mon_wk['month']).stack().describe(),'\n')
print(pd.crosstab(index= yr_mon_wk['year'], columns= yr_mon_wk['month']).stack().mode()[0],'\n')
print('Days in week stats','\n')
print(pd.crosstab(index= yr_mon_wk['year'], columns= yr_mon_wk['week']).stack().describe(),'\n')
print(pd.crosstab(index= yr_mon_wk['year'], columns= yr_mon_wk['week']).stack().mode()[0],'\n')