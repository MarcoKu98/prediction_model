import pandas as pd
import numpy as np
import logging
from sklearn import metrics
from data_processing import preprocess_data
from economic_collector import fetch_economic_data
from stock_collector import fetch_stock_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setting up constant names, change to your filenames if you wish them to be different.
STOCK_FILE = 'stock_historical_data.csv'
ECONOMIC_FILE = 'economic_data.csv'
DATA_ = 'combined_data.csv'
LAG_COLUMNS = ['Interest Rate','Inflation Rate','GDP Growth','Unemployment Rate','PPI']
LAG_DAYS = [60]  # Days to lag by, can be multiple lags, default is 60
TEST_ = 0.2  # Percentage of data to use for testing
TRAIN_ = 0.25  # Percentage of data to use for training

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

def get_lag_columns(df):
    '''
    Get all column names in a DataFrame that contain "day lag" in their name.

    Parameters:
    df (pd.DataFrame): The data to search.

    Returns:
    list: A list of column names that contain "day lag".
    '''
    return [col for col in df.columns if 'day lag' in col]

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
    data['month'] = data['Date'].dt.month
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

    # Net change between open and close, positive means daily increase, negative means daily decrease
    numeric_data['open-close']  = numeric_data['Open'] - numeric_data['Close']
    
    # Stock price range for the day, grants an idea for volatility. 
    numeric_data['low-high']  = numeric_data['Low'] - numeric_data['High']

    # If the price goes up in the next amount of days, we label it as 1, else 0
    # Target for models to predict on, better to predict whether will be an increase
    # rather than the price itself
    numeric_data['target'] = np.where(numeric_data['Close'].shift(-max(LAG_DAYS)) > numeric_data['Close'], 1, 0)
    
    # Newly added columns will have NaN values, need to drop them
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.dropna()
    
    return numeric_data

def split_data(features, target):
    '''
    Split the data into training, validation, and test sets.

    Parameters:
    features (pd.DataFrame): The features to split.
    target (pd.Series): The target variable to split.

    Returns:
    tuple: The training, validation, and test sets for the features and target variable.
    '''
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_temp, X_test, Y_temp, Y_test = train_test_split(features, target, test_size=TEST_, random_state=62)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_temp, Y_temp, test_size=TRAIN_, random_state=62)
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

# Getting the stock symbol from the user and fetching the data
stock_symbol = input("Please enter the stock symbol (e.g. AAPL for Apple): ")
fetch_stock_data(stock_symbol)

# Fetching the economic data, promted if user wants to update or not
# Economic data is updated less often, and API calls may be limited
# but make sure economic_data.csv exists in directory
fetch_economic_data()
logging.info(f'Processing the data for {stock_symbol} and economic indicators... ')
data = preprocess_data(STOCK_FILE, ECONOMIC_FILE)
data.to_csv(DATA_)

logging.info('Preprocessing is complete. Combined file created as combined_data.csv')

data = feature_engineering(data, LAG_COLUMNS, LAG_DAYS)

numeric_data = prepare_data(data, LAG_DAYS)

lagged_column_names = get_lag_columns(numeric_data)

# Remove anything that has inflation rate in it from lagged columns
# Inflation rate is highly correlated with prices, and we want to avoid data leakage
lagged_column_names = [col for col in lagged_column_names if 'Inflation Rate' not in col]

# Selecting important features to avoid noise
features = numeric_data[['open-close','low-high', 'quarter'] + lagged_column_names]
target = numeric_data['target']

# Split your data into training, validation, and test sets
X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_data(features, target)


'''
# Testing three different models to determine which is best
# models = [LogisticRegression(), SVC(

#   kernel='poly', probability=True), XGBClassifier()]
 

# for i in range(3):

#   models[i].fit(X_train, Y_train)
 

#   print(f'{models[i]} : ')

#   print('Training Accuracy : ', metrics.roc_auc_score(

#     Y_train, models[i].predict_proba(X_train)[:,1]))

#   print('Validation Accuracy : ', metrics.roc_auc_score(

#     Y_valid, models[i].predict_proba(X_valid)[:,1]))

#   print()

'''

'''
# XGBoost model had the best accuracy, tuning the hyperparameters

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.005, 0.01]
}

xgb_model = XGBClassifier(random_state=42)

# Estimated general range of parameters to start with
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    cv=3,  # Number of cross-validation folds
    scoring='accuracy',  # Scoring metric
    verbose=2,  # Prints out updates
    n_jobs=-1  # Uses all CPU cores
)

# Using GridSearchCV to find the best parameters
grid_search.fit(X_train, Y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
'''

# Best Parameters: {'colsample_bytree': 0.6, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'subsample': 0.6}
# Best XGBoost model
best_xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0,
    random_state=42
)

# Fitting into training data
best_xgb_model.fit(
    X_train, 
    Y_train, 
    eval_set=[(X_valid, Y_valid)], 
    verbose=False
)

print('Best XGB Model: ')

# Using ROC and AUC for training and validation accuracy
train_accuracy = metrics.roc_auc_score(Y_train, best_xgb_model.predict_proba(X_train)[:,1])
print('Training Accuracy : ', train_accuracy)

valid_accuracy = metrics.roc_auc_score(Y_valid, best_xgb_model.predict_proba(X_valid)[:,1])
print('Validation Accuracy : ', valid_accuracy)

# Predictions on the validation data
Y_pred = best_xgb_model.predict(X_valid)

# Confusion matrix
cm = metrics.confusion_matrix(Y_valid, Y_pred)

print(cm)

# Predictions on the test data
Y_test_pred = best_xgb_model.predict(X_test)

# Compute and print the test accuracy
test_accuracy = metrics.accuracy_score(Y_test, Y_test_pred)
print('Test Accuracy : ', test_accuracy)

# Confusion Matrix
cm_test = metrics.confusion_matrix(Y_test, Y_test_pred)

print(cm_test)

# Further evaluation with precision, recall, and F1 score
# for both validation and test data
precision = precision_score(Y_valid, Y_pred)
recall = recall_score(Y_valid, Y_pred)
f1 = f1_score(Y_valid, Y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

test_precision = precision_score(Y_test, Y_test_pred)
test_recall = recall_score(Y_test, Y_test_pred)
test_f1 = f1_score(Y_test, Y_test_pred)

print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
print(f'Test F1 Score: {test_f1}')

# Evaluating importance for features
importances = best_xgb_model.feature_importances_
feature_names = features.columns

# Sorting by highest importance
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances= feature_importances.sort_values('importance', ascending=False)

print(feature_importances)