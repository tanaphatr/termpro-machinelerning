# Datafile/load_data.py
import requests
import pandas as pd


# ดึงค่า API_URL
API_URL = 'https://termpro-api-production.up.railway.app/'

if not API_URL:
    raise EnvironmentError("API_URL environment variable is not set.")

def load_data():
    """
    Fetch data from the API endpoint for sales data.
    """
    endpoint = f"{API_URL}/salesdata"  # Adjust the endpoint as needed
    response = requests.get(endpoint)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        df = pd.DataFrame(data)  # Convert to DataFrame
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")

def load_dataps():
    """
    Fetch data from the API endpoint for product sales data.
    """
    endpoint = f"{API_URL}/product_sales"  # Adjust the endpoint as needed
    response = requests.get(endpoint)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        dfps = pd.DataFrame(data)  # Convert to DataFrame
        return dfps
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")