import pandas as pd
from calendar import monthrange

import pandas as pd

def preprocess_data(df):
    # แปลงปีพุทธศักราชเป็นปีคริสต์ศักราช
    df['sale_date'] = pd.to_datetime(df['sale_date'].astype(str).str.replace(r'(\d{4})', lambda x: str(int(x.group(0)) - 543), regex=True), errors='coerce')

    # Log จำนวนวันที่ขาดหาย
    print(f"Missing sale_date entries: {df['sale_date'].isnull().sum()}")

    # Ensure sales_amount is numeric
    df['sales_amount'] = pd.to_numeric(df['sales_amount'], errors='coerce')

    # แทนที่ค่าที่ขาดหายใน sales_amount ด้วยค่าเฉลี่ย
    df['sales_amount'].fillna(df['sales_amount'].mean(), inplace=True)

    # ลบคอลัมน์ profit_amount
    df.drop(columns=['profit_amount'], inplace=True)
    
    # เพิ่มคอลัมน์ day_of_year
    df['day_of_year'] = df['sale_date'].dt.dayofyear

    # แยกวันที่เป็น Date, Month, Year
    df['Day'] = df['sale_date'].dt.day
    df['Month'] = df['sale_date'].dt.month
    df['Year'] = df['sale_date'].dt.year
    
    # สร้างไฟล์ CSV หลังจากประมวลผลข้อมูล
    # df.to_csv("processed_data111.csv", index=False)
    return df

def preprocess_dataps(dfps):
    # ตรวจสอบและแปลงปี พ.ศ. เป็น ค.ศ.
    dfps['Date'] = dfps['Date'].astype(str)  
    dfps['Date'] = dfps['Date'].apply(lambda x: str(int(x[:4]) - 543) + x[4:] if x[:4].isdigit() else x)

    # แปลงคอลัมน์ 'Date' เป็น datetime
    dfps['Date'] = pd.to_datetime(dfps['Date'], errors='coerce')

    # กรองข้อมูลที่ Date เป็น NaT ออกไป
    dfps = dfps.dropna(subset=['Date'])

    # รวมค่าจำนวน Quantity และ Total_Sale ถ้ามี Product_code และ Date ซ้ำกัน
    dfps = dfps.groupby(['Date', 'Product_code'], as_index=False).agg({
        'Quantity': 'sum',
        'Total_Sale': 'sum'
    })

    # สร้างลิสต์ของทุกๆ Date และ Product_code ที่เป็นไปได้
    all_dates = pd.date_range(dfps['Date'].min(), dfps['Date'].max(), freq='D')
    all_product_codes = dfps['Product_code'].unique()

    # สร้าง DataFrame เปล่าที่จะเติมข้อมูล
    expanded_df = pd.DataFrame([(date, code) for date in all_dates for code in all_product_codes], columns=['Date', 'Product_code'])

    # ผสมข้อมูลเดิมกับ DataFrame ที่เติมค่าด้วยค่า 0
    dfps_full = pd.merge(expanded_df, dfps, on=['Date', 'Product_code'], how='left').fillna({'Quantity': 0, 'Total_Sale': 0})

    # Ensure Quantity and Total_Sale are numeric
    dfps_full['Quantity'] = pd.to_numeric(dfps_full['Quantity'], errors='coerce').fillna(0)
    dfps_full['Total_Sale'] = pd.to_numeric(dfps_full['Total_Sale'], errors='coerce').fillna(0)

    # Calculate the average values for Quantity and Total_Sale
    avg_values = dfps_full.groupby('Product_code')[['Quantity', 'Total_Sale']].mean().round(0).astype(int)

    # Replace 0 values with the average for each Product_code
    dfps_full.loc[dfps_full['Quantity'] == 0, 'Quantity'] = dfps_full['Product_code'].map(avg_values['Quantity'])
    dfps_full.loc[dfps_full['Total_Sale'] == 0, 'Total_Sale'] = dfps_full['Product_code'].map(avg_values['Total_Sale'])

    # กรองแสดงเฉพาะ Product_code ที่ต้องการ
    product_codes_to_show = ['A1034', 'A1004', 'A1001', 'B1003', 'B1002', 'D1003', 'A1002']
    dfps_full = dfps_full[dfps_full['Product_code'].isin(product_codes_to_show)]

    # เรียงลำดับข้อมูลตาม Date และ Product_code
    dfps_full = dfps_full.sort_values(by=['Date', 'Product_code']).reset_index(drop=True)

    # บันทึกไฟล์ CSV
    # dfps_full.to_csv("product.csv", index=False)

    return dfps_full
