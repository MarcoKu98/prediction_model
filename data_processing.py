import pandas as pd
import numpy as np
import logging
from data_loading import load_data

# Setting up constants
STOCK_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
ECONOMIC_COLS = ['Interest Rate', 'GDP Growth', 'PPI', 'Inflation Rate', 'Unemployment Rate']
EXCLUDE_COLS = ['Dividends', 'Stock Splits']

# Function to handle duplicate dates in the data
def handle_duplicates(data):
    # Check if there's duplication
    if data.index.duplicated().any():
        logging.info("Found Duplicate dates. Aggregating data...")
        # Aggregate data by date with index level 0
        data = data.groupby(level=0).mean()
    return data

# Function to align timeframes of two datasets
def align_timeframes(data1, data2):
    # Find the latest start date and earliest end date
    start_date = max(data1.index.min(), data2.index.min())
    end_date = min(data1.index.max(), data2.index.max())
    # Return data for the common timeframe
    return data1[start_date:end_date], data2[start_date:end_date]

# Function to interpolate missing values in specified columns
def interpolate_columns(data, columns, method):
    for column in columns:
        # Interpolate missing values in the column
        data[column].interpolate(method=method, inplace=True)
    return data

# Function to fill missing values in specified columns
def fillna_columns(data, columns, method):
    for column in columns:
        # Fill missing values in the column
        data[column].fillna(method=method, inplace=True)
    return data

def preprocess_data(stock_file, economic_file):

    """
    Clean the data and return a combined ready-to-use dataset.

    Parameters:
    stock_file (str): The name of the stock data CSV file.
    economic_file (str): The name of the economic data CSV file.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    logging.info('Loading data...')
    stock_data = load_data(stock_file)
    economic_data = load_data(economic_file)

    # Drop columns that are not needed
    stock_data.drop(EXCLUDE_COLS, axis=1, inplace=True)

    # Handle duplicates
    stock_data = handle_duplicates(stock_data)
    economic_data = handle_duplicates(economic_data)

    for col in STOCK_COLS:
        '''
        # Removal of outliers
        Q1 = stock_data[col].quantile(0.25)
        Q3 = stock_data[col].quantile(0.75)
        IQR = Q3 - Q1
        stock_data = stock_data[~((stock_data[col] < (Q1 - 1.5 * IQR)) | (stock_data[col] > (Q3 + 1.5 * IQR)))]


        # Median replacement
        Q1 = stock_data[col].quantile(0.25)
        Q3 = stock_data[col].quantile(0.75)
        IQR = Q3 - Q1
        stock_data.loc[((stock_data[col] < (Q1 - 1.5 * IQR)) | (stock_data[col] > (Q3 + 1.5 * IQR))), col] = stock_data[col].median()
        '''
        #Capped outliers
        Q1 = stock_data[col].quantile(0.25)
        Q3 = stock_data[col].quantile(0.75)
        IQR = Q3 - Q1
        stock_data.loc[stock_data[col] < (Q1 - 1.5 * IQR), col] = Q1 - 1.5 * IQR
        stock_data.loc[stock_data[col] > (Q3 + 1.5 * IQR), col] = Q3 + 1.5 * IQR

    for col in ECONOMIC_COLS:
        '''
        # Removal of outliers
        Q1 = economic_data[col].quantile(0.25)
        Q3 = economic_data[col].quantile(0.75)
        IQR = Q3 - Q1
        economic_data = economic_data[~((stock_data[col] < (Q1 - 1.5 * IQR)) | (economic_data[col] > (Q3 + 1.5 * IQR)))]


        # Median replacement
        Q1 = economic_data[col].quantile(0.25)
        Q3 = economic_data[col].quantile(0.75)
        IQR = Q3 - Q1
        economic_data.loc[((economic_data[col] < (Q1 - 1.5 * IQR)) | (economic_data[col] > (Q3 + 1.5 * IQR))), col] = economic_data[col].median()
        '''
        
        #Capped outliers
        Q1 = economic_data[col].quantile(0.25)
        Q3 = economic_data[col].quantile(0.75)
        IQR = Q3 - Q1
        economic_data.loc[economic_data[col] < (Q1 - 1.5 * IQR), col] = Q1 - 1.5 * IQR
        economic_data.loc[economic_data[col] > (Q3 + 1.5 * IQR), col] = Q3 + 1.5 * IQR
    


    stock_data, economic_data = align_timeframes(stock_data, economic_data)

    combined_data = pd.concat([stock_data, economic_data], axis=1)

    '''
        checking if needed to preemtively fill the first empty cell
        since interpolate can't possibly do that
    '''
    for column in combined_data.columns:
        if pd.isnull(combined_data[column].iloc[0]):
            combined_data[column].iloc[0] = combined_data[column].iloc[1]

    '''
        ~Reasoning: 
        Interpolation columns benefit most from linear interpolation
        since the data is most commonly continuous.
        Forward fill columns benefit most from forward filling since the data is
        most commonly discrete and the data is most likely to be the same as the previous observation
    '''
    cols_to_interpolate = STOCK_COLS + ['Interest Rate']
    cols_to_forward_fill = ['GDP Growth', 'PPI', 'Inflation Rate', 'Unemployment Rate']

    combined_data = interpolate_columns(combined_data, cols_to_interpolate, 'linear')
    combined_data = fillna_columns(combined_data, cols_to_forward_fill, 'ffill')

    combined_data = combined_data.fillna(0)

    # Iterate over the columns
    for column in combined_data.columns:
        # Find the index of the first non-zero value in the column
        non_zero_data = combined_data.loc[combined_data[column] != 0]
        if not non_zero_data.empty:
            first_non_zero_index = non_zero_data.index[0]
            # Get the first non-zero value in the column
            first_non_zero_value = combined_data.loc[first_non_zero_index, column]

            # Replace the initial sequence of zeros or NaNs in the column with the first non-zero value
            combined_data.loc[:first_non_zero_index, column] = combined_data.loc[:first_non_zero_index, column].replace([0, np.nan], first_non_zero_value)

    # Resetting index to utilize Date for feature engineering
    combined_data.reset_index(inplace=True)
    return combined_data
