import pandas as pd
from calendar import monthrange

import pandas as pd

def preprocess_data(df):
    # Ensure sale_date is in datetime format
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')

    # Log missing sale_date entries
    print(f"Missing sale_date entries: {df['sale_date'].isnull().sum()}")

    # Drop rows with invalid sale_date
    df = df.dropna(subset=['sale_date'])

    # Ensure sales_amount is numeric
    df['sales_amount'] = pd.to_numeric(df['sales_amount'], errors='coerce')

    # Replace missing sales_amount with the mean
    df['sales_amount'].fillna(df['sales_amount'].mean(), inplace=True)

    # Drop unnecessary columns
    df.drop(columns=['profit_amount'], inplace=True)
    
    # Add day_of_year column
    df['day_of_year'] = df['sale_date'].dt.dayofyear

    # Extract Day, Month, Year
    df['Day'] = df['sale_date'].dt.day
    df['Month'] = df['sale_date'].dt.month
    df['Year'] = df['sale_date'].dt.year
    
    return df

def preprocess_dataps(dfps):
    # Ensure Date is in datetime format
    dfps['Date'] = pd.to_datetime(dfps['Date'], errors='coerce')

    # Drop rows with invalid or missing Date
    dfps = dfps.dropna(subset=['Date'])

    # Group by Date and Product_code, summing Quantity and Total_Sale
    dfps = dfps.groupby(['Date', 'Product_code'], as_index=False).agg({
        'Quantity': 'sum',
        'Total_Sale': 'sum'
    })

    # Generate all possible combinations of Date and Product_code
    all_dates = pd.date_range(dfps['Date'].min(), dfps['Date'].max(), freq='D')
    all_product_codes = dfps['Product_code'].unique()
    expanded_df = pd.DataFrame([(date, code) for date in all_dates for code in all_product_codes], 
                               columns=['Date', 'Product_code'])

    # Merge with original data and fill missing values
    dfps_full = pd.merge(expanded_df, dfps, on=['Date', 'Product_code'], how='left').fillna({'Quantity': 0, 'Total_Sale': 0})

    # Ensure Quantity and Total_Sale are numeric
    dfps_full['Quantity'] = pd.to_numeric(dfps_full['Quantity'], errors='coerce').fillna(0)
    dfps_full['Total_Sale'] = pd.to_numeric(dfps_full['Total_Sale'], errors='coerce').fillna(0)

    # Replace 0 values with the average for each Product_code
    avg_values = dfps_full.groupby('Product_code')[['Quantity', 'Total_Sale']].mean().round(0).astype(int)
    dfps_full.loc[dfps_full['Quantity'] == 0, 'Quantity'] = dfps_full['Product_code'].map(avg_values['Quantity'])
    dfps_full.loc[dfps_full['Total_Sale'] == 0, 'Total_Sale'] = dfps_full['Product_code'].map(avg_values['Total_Sale'])

    # Filter specific Product_codes
    product_codes_to_show = ['A1034', 'A1004', 'A1001', 'B1003', 'B1002', 'D1003', 'A1002']
    dfps_full = dfps_full[dfps_full['Product_code'].isin(product_codes_to_show)]

    # Sort by Date and Product_code
    dfps_full = dfps_full.sort_values(by=['Date', 'Product_code']).reset_index(drop=True)

    return dfps_full
