import nasdaqdatalink as ndl
import os
import pandas as pd
import datetime
import sys

'''
    API key setup
    Read instructions on how to set up your own key on your machine
'''
api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")

# Check if API key is set
if api_key is None:
    raise ValueError("NASDAQ_DATA_LINK_API_KEY not found, refer to setup instructions")

# Setting up current date as the last date to retrieve
# and start date as 10 years ago
endDate = datetime.date.today()
startDate = endDate - datetime.timedelta(days=365*20)

# Converting to right format (YYYY-MM-DD)
endDateStr = endDate.isoformat()
startDateStr = startDate.isoformat()

def fetch_economic_data():
    """
    Fetch economic data from the Nasdaq Data Link API and save it to a CSV file.
    
    """

    decision = input("Do you want to fetch economic data? (y/n): ")
    if decision.lower() != 'y':
        return

    try:
        # Fetching 10 years of data
        interest_rates = ndl.get("FRED/DTB3", start_date = startDateStr, end_date = endDateStr)  # 3-Month Treasury Bill: Secondary Market Rate
        inflation_rates = ndl.get("FRED/CPIAUCSL", start_date = startDateStr, end_date = endDateStr)  # USA Consumer Price Index
        gdp_growth = ndl.get("FRED/GDP", start_date = startDateStr, end_date = endDateStr)  # Gross Domestic Product
        unemployment_rates = ndl.get("FRED/UNRATE", start_date = startDateStr, end_date = endDateStr)  # Unemployment Rate
        ppi = ndl.get("FRED/PPIACO", start_date = startDateStr, end_date = endDateStr)  # Producer Price Index by Commodity: All Commodities
    
    except Exception as e:
        print(f"Error fetching economic indicator data: {e}")
        sys.exit(1)

    # Renaming for clearer file
    interest_rates.rename(columns={'Value': 'Interest Rate'}, inplace=True)
    inflation_rates.rename(columns={'Value': 'Inflation Rate'}, inplace=True)
    gdp_growth.rename(columns={'Value': 'GDP Growth'}, inplace=True)
    unemployment_rates.rename(columns={'Value': 'Unemployment Rate'}, inplace=True)
    ppi.rename(columns={'Value': 'PPI'}, inplace=True)

    # Combine all dataframes into one
    economic_data = pd.concat([interest_rates, inflation_rates, gdp_growth, unemployment_rates, ppi], axis=1)

    # Save the combined economic data to a CSV file
    economic_data.to_csv('economic_data.csv')

    print("Economic data has been saved to economic_data.csv")

