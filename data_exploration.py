import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_processing import preprocess_data
from economic_collector import fetch_economic_data
from stock_collector import fetch_stock_data
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Setting up constant names, change to your filenames if you wish them to be different.
STOCK_FILE = 'stock_historical_data.csv'
ECONOMIC_FILE = 'economic_data.csv'
DATA_ = 'combined_data.csv'
LAG_COLUMNS = ['Interest Rate','Inflation Rate','GDP Growth','Unemployment Rate','PPI']
LAG_DAYS = [60]  # Days to lag by, can be multiple lags, default is 60

def add_lag_columns(data, columns, LAG_DAYS):
    '''
    Create lagged columns for the specified columns.

    Parameters:
    data (pd.DataFrame): The data to add lagged columns to.
    columns (list): The columns to add lagged columns for.
    LAG_DAYS (list): The number of days to lag by.

    Returns:
    pd.DataFrame: The data with lagged columns added.
    '''
    for col in columns:
        for lag in LAG_DAYS:
            data[f'{col} {lag} day lag'] = data[col].shift(lag)
    return data

def feature_engineering(data, LAG_COLUMNS, LAG_DAYS):
    '''
    Add new features to the data.

    Parameters:
    data (pd.DataFrame): The data to add features to.
    LAG_COLUMNS (list): The columns to add lagged columns for.
    LAG_DAYS (list): The number of days to lag by.

    Returns:
    pd.DataFrame: The data with new features.
    '''
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['quarter'] = np.where(data['month']%3==0,1,0)
    data = add_lag_columns(data, LAG_COLUMNS, LAG_DAYS)
    return data

def prepare_data(data, LAG_DAYS):
    '''
    Preparing data for modeling.

    Parameters:
    data (pd.DataFrame): The data to prepare.
    LAG_DAYS (list): The number of days to lag by.

    Returns:
    pd.DataFrame: The prepared data.
    '''
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.dropna()

    # Net change between open and close, positive means daily increase, negative means daily decrease
    numeric_data['open-close']  = numeric_data['Open'] - numeric_data['Close']
    
    # Stock price range for the day, grants an idea for volatility. 
    numeric_data['low-high']  = numeric_data['Low'] - numeric_data['High']

    # If the price goes up in the next 20 days, we label it as 1, else 0
    # Target for models to predict on, better to predict whether will be an increase
    # rather than the price itself
    numeric_data['target'] = np.where(numeric_data['Close'].shift(-max(LAG_DAYS)) > numeric_data['Close'], 1, 0)
    return numeric_data


# Getting the stock symbol from the user and fetching the data
stock_symbol = input("Please enter the stock symbol (e.g. AAPL for Apple): ")
fetch_stock_data(stock_symbol)

# Fetching the economic data, promted if user wants to update or not
# Economic data is updated less often, and API calls may be limited
# but make sure economic_data.csv exists in directory
fetch_economic_data()
logging.info(f'Processing the data for {stock_symbol} and economic indicators... ')
data = preprocess_data(STOCK_FILE, ECONOMIC_FILE)
logging.info('Preprocessing is complete.')


# Plotting the price of the stock

plt.figure(figsize=(15,5))

plt.plot(data['Close'])

plt.title(f'{stock_symbol} Close price.', fontsize=15)

plt.ylabel('Price in USD')
plt.show()



# Distribution plot for stock prices

stock = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))
 

for i, col in enumerate(stock):

  plt.subplot(2,3,i+1)

  sns.distplot(data[col])
plt.show()



# Distribution plot for economic indicators

indicators = ['Interest Rate', 'GDP Growth', 'PPI', 'Inflation Rate', 'Unemployment Rate']

plt.subplots(figsize=(20,10))
 

for i, col in enumerate(indicators):

  plt.subplot(2,3,i+1)

  sns.distplot(data[col])
plt.show()

# Figuring out outliers

# Boxplot for stock prices
plt.subplots(figsize=(20,10))

for i, col in enumerate(stock):

  plt.subplot(2,3,i+1)

  sns.boxplot(data[col])
plt.show()



# Boxplot for economic indicators
plt.subplots(figsize=(20,10))

for i, col in enumerate(indicators):

  plt.subplot(2,3,i+1)

  sns.boxplot(data[col])
plt.show()


data = feature_engineering(data, LAG_COLUMNS, LAG_DAYS)

numeric_data = prepare_data(data, LAG_DAYS)


# Bar Graph for stock prices

data_grouped = numeric_data.groupby('year').mean()

plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):

  plt.subplot(2,2,i+1)

  data_grouped[col].plot.bar()
plt.show()



# Bar Graph for economic indicators

plt.subplots(figsize=(20,10))

for i, col in enumerate(['Interest Rate', 'GDP Growth', 'PPI', 'Inflation Rate', 'Unemployment Rate']):

  plt.subplot(3,2,i+1)

  data_grouped[col].plot.bar()
plt.show()



# Grouped to check differences bewteen quarter end and not quarter end

# grouped = numeric_data.groupby('quarter').mean()
# grouped.to_csv('grouped_data.csv')


# numeric_data.to_csv('numeric_data.csv')


# Correlation plot set to only check for 0.9 and above correlation
plt.figure(figsize=(10, 10))
 
# As concern is with the highly
# correlated features only I will visualize
# the heatmap as per that criteria only.

sns.heatmap(numeric_data.corr() > 0.9, annot=True, cbar=False)
plt.show()
