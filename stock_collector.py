import yfinance as yf
import os
import datetime
import sys

# API key setup
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Setting up current date as the last date to retrieve
# and start date as 10 years ago
endDate = datetime.date.today()
startDate = endDate - datetime.timedelta(days=365*10)

# Converting to right format (YYYY-MM-DD)
endDateStr = endDate.isoformat()
startDateStr = startDate.isoformat()


def fetch_stock_data(stock_symbol):
    # Validate user input
    if not stock_symbol:
        print("You must enter a stock symbol.")
        return

    try:
        # Ticker object for the stock
        ticker = yf.Ticker(stock_symbol)

        # Check if the stock symbol is valid
        if not ticker.info:
            raise ValueError(f"{stock_symbol} is not a valid stock symbol.")

        # Fetch data using yfinance
        data = ticker.history(start=startDateStr, end=endDateStr)

        # Check if any data was returned
        if data.empty:
            raise ValueError(f"No data found for {stock_symbol}.")

        # Save the data to a CSV file
        data.to_csv('stock_historical_data.csv')

        print(f"Historical data for {stock_symbol} has been saved to stock_historical_data.csv")

    except Exception as e:
        print(f"Error fetching data for this stock: {e}")
        sys.exit(1)