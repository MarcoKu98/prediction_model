# DISCLAIMER

This project is not the final product. As stock prediction is very complex and requires multiple analyses, what this project displays is only a small portion of the analysis that needs to be done to get a somewhat accurate stock prediction.

# Stock Prediction Model

This project focuses on building a predictive model that forecasts whether the price will increase or not, on a chosen stock's historical data, as well as economic indicators for the same time frame.
It initially extracts data from yfinance and Nasdaq Data Link APIs then combines them into a single dataframe using the date as the index.
Given that there are several missing data, especially due to the different publishing ratios for the data, interpolation, and forward-filling
is used appropriately to handle the empty rows.
After the data is processed, cleaned, and free of missing values, it runs through a series of data exploration steps where key features 
are decided upon, and new features are engineered with the help of visuals from the Matplot library.
Then the data is split for training and testing, and three models are tested initially to gauge accuracy through a ROC-AUC curve. 
As the model is decided, a grid search is run to estimate the best parameters, and then that best model is evaluated again through various methods.

## Run it yourself

Download the python files, the model.py is all you need to run. If you would like to explore the data yourself, data_exploration.py can be run and displays a few different plots, bar graphs, etc. 

### API key
To set up the API key for the Nasdaq Data Link API, you would need to sign up and then be able to get your key. More information here: https://docs.data.nasdaq.com/docs/getting-started
This code only accesses free publishers from Nasdaq so you won't need a premium account, but if you wish to switch publishers, you should be able to do so on the economic_collector.py file

### Prerequisites
Make sure to install yfinance and Nasdaq Data Link. You can simply use PyPI to get the right packages.
Make sure to check the websites for these packages to get the right version for your operating system.
APIs: 
 - pip install yfinance
 - pip install nasdaq-data-link
Other packages:
 - https://scikit-learn.org/stable/install.html
 - https://xgboost.readthedocs.io/en/stable/install.html


## Author

Marco Kushta 
