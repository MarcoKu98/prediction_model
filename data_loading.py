import pandas as pd

def load_data(file_name):
    """
    Load data from a CSV file and set 'Date' as index.

    Parameters:
    file_name (str): The name of the CSV file.

    Returns:
    pd.DataFrame: The loaded data.

    Raises:
    FileNotFoundError: If the file does not exist.
    """
    try:
        # Load data
        data = pd.read_csv(file_name)
        # Convert 'Date' column to datetime and remove time part
        data['Date'] = pd.to_datetime(data['Date'].str.split(" ").str[0])
        # Setting 'Date' as index
        data.set_index('Date', inplace=True)
        return data
    except Exception as e:
        print(f"Error loading data from {file_name}: {e}")
        return FileNotFoundError