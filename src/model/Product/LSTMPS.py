import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, app, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from tensorflow.keras.regularizers import l2
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PJML')))


product_codes = ["A1001", "A1002", "A1004", "A1034", "B1002", "B1003", "D1003"]
# product_codes = ["A1001", "A1002", "A1004", "A1034", "B1002", "B1003", "D1003"]

def add_time_features(df):
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Add cyclical encoding for seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    return df

def add_new_features(df):
    df['prev_day_diff'] = df['Quantity'] - df['Quantity'].shift(1)
    df['rolling_avg_60'] = df['Quantity'].rolling(window=60).mean()
    return df

def add_lag_features(df, lags=[1, 2, 3, 7, 14, 30]):
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['Quantity'].shift(lag)
    return df

def add_rolling_features(df):
    windows = [7, 14, 30]
    for window in windows:
        df[f'sales_ma_{window}'] = df['Quantity'].rolling(window=window).mean()
        df[f'sales_std_{window}'] = df['Quantity'].rolling(window=window).std()
        df[f'sales_min_{window}'] = df['Quantity'].rolling(window=window).min()
        df[f'sales_max_{window}'] = df['Quantity'].rolling(window=window).max()
    return df

def augment_time_series(df, random_seed=42):
    np.random.seed(random_seed)  # กำหนด seed ให้ noise คงที่ทุกครั้ง
    augmented_data = pd.DataFrame()
    
    # Time shift augmentation
    for shift in [-2, -1, 1, 2]:
        shifted = df.copy()
        shifted['Quantity'] = shifted['Quantity'].shift(shift).bfill()  # แทนค่าหาย
        augmented_data = pd.concat([augmented_data, shifted.dropna()])
    
    for _ in range(5):  # ทำซ้ำ 5 รอบ
        scale = np.random.uniform(0.02, 0.04)  # ปรับ Noise Scale ให้อยู่ในช่วงแคบลง
        noisy = df.copy()
        noise = np.random.normal(0, df['Quantity'].std() * scale, len(df))
        noisy['Quantity'] = np.clip(noisy['Quantity'] + noise, 0, None)
        augmented_data = pd.concat([augmented_data, noisy])

    for _ in range(5):
        scale = np.random.uniform(0.95, 1.05)  # ใช้ Scaling แบบสุ่ม
        scaled = df.copy()
        scaled['Quantity'] = scaled['Quantity'] * scale
        augmented_data = pd.concat([augmented_data, scaled])

    
    # Trend augmentation
    trend = df.copy()
    trend['Quantity'] = trend['Quantity'] * (1 + np.linspace(0, 0.1, len(df)))
    augmented_data = pd.concat([augmented_data, trend])
    
    return pd.concat([df, augmented_data]).sort_values('Date').reset_index(drop=True)

# def save_to_csv(df, filename):
#     df.to_csv(filename, index=False)
#     print(f"✅ Data saved to {filename}")

def fetch_temperature_data():
    """Fetch temperature data from the API."""
    url = "https://termpro-api-production.up.railway.app/Temp"
    response = requests.get(url)
    if response.status_code == 200:
        temp_data = pd.DataFrame(response.json())
        temp_data['Date'] = pd.to_datetime(temp_data['date']).dt.tz_localize(None)  # Remove timezone
        temp_data = temp_data[['Date', 'Temp']]
        return temp_data
    else:
        raise Exception(f"Failed to fetch temperature data. Status code: {response.status_code}")

def prepare_data(df):

    print("🔄 เริ่มการเตรียมข้อมูล...")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)  # Remove timezone
    df = df.dropna(subset=['Date'])
    
    print("🌡️ กำลังดึงข้อมูลอุณหภูมิ...")
    temp_data = fetch_temperature_data()
    df = pd.merge(df, temp_data, on='Date', how='left')

    print("➕ กำลังเพิ่ม features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_new_features(df)
    df = add_rolling_features(df)
    df = df.dropna()
    
    print("🔄 กำลังทำ Data Augmentation...")
    df_augmented = augment_time_series(df)

    # Scale features
    scaler = StandardScaler()
    features = ['prev_day_diff', 'day_of_week', 'month','day_of_year','rolling_avg_60', 'Temp'] + \
              [col for col in df.columns if 'sales_lag_' in col or 
                                          'sales_ma_' in col or 
                                          'sales_std_' in col or 
                                          'sales_min_' in col or 
                                          'sales_max_' in col]
    
    df_augmented[features] = scaler.fit_transform(df_augmented[features])
    
    sequence_length = 60
    X, y = [], []
    
    print("🎯 กำลังสร้าง sequences...")
    for i in range(sequence_length, len(df_augmented)):
        X.append(df_augmented.iloc[i-sequence_length:i][features].values)
        y.append(df_augmented['Quantity'].iloc[i])
    
    return np.array(X), np.array(y), df_augmented, scaler

def train_lstm_model(X_train, y_train, X_val, y_val, model_dir, product_code):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm2')
    
    # สร้าง path ของไฟล์โมเดลและพารามิเตอร์ที่ดีที่สุด
    model_path2 = os.path.join(model_dir, f'lstm_model_{product_code}.pkl')
    best_params_path = os.path.join(model_dir, f'best_params_{product_code}.pkl')
    os.makedirs(model_dir, exist_ok=True)

    # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
    if os.path.exists(model_path2):
        print(f"📥 โหลดโมเดล {model_path2} ที่เก็บไว้แล้ว")
        model = joblib.load(model_path2)
        return model

    print(f"🛠️ กำลังสร้างโมเดลใหม่สำหรับ {model_dir}...")

    # ตรวจสอบว่ามีพารามิเตอร์ที่ดีที่สุดที่บันทึกไว้หรือไม่
    if os.path.exists(best_params_path):
        print("📥 โหลดพารามิเตอร์ที่ดีที่สุดที่บันทึกไว้")
        best_params = joblib.load(best_params_path)
    else:
        print("🔍 กำลังทำ Grid Search เพื่อหาพารามิเตอร์ที่เหมาะสม...")
        param_grid = {
            'lstm_units': [64, 128, 256],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005]
        }
        best_val_loss = float('inf')
        best_params = {}

        for lstm_units in param_grid['lstm_units']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    print(f"ทดสอบ: lstm_units={lstm_units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
                    model = Sequential([
                        Input(shape=(X_train.shape[1], X_train.shape[2])),
                        Bidirectional(LSTM(lstm_units, return_sequences=True)),
                        BatchNormalization(),
                        Dropout(dropout_rate),
                        Bidirectional(LSTM(lstm_units // 2, return_sequences=True)),
                        BatchNormalization(),
                        Dropout(dropout_rate - 0.1),
                        Bidirectional(LSTM(lstm_units // 4)),
                        BatchNormalization(),
                        Dropout(dropout_rate - 0.2),
                        Dense(32, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber(), metrics=['mae', 'mape'])
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=30,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            'lstm_units': lstm_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate
                        }
        print(f"🏆 พารามิเตอร์ที่ดีที่สุด: {best_params}")
        print(f"🏆 ค่า validation loss ที่ดีที่สุด: {best_val_loss}")
        joblib.dump(best_params, best_params_path)

    # สร้างโมเดลสุดท้ายด้วยพารามิเตอร์ที่ดีที่สุด
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(best_params['lstm_units'], return_sequences=True)),
        BatchNormalization(),
        Dropout(best_params['dropout_rate']),
        Bidirectional(LSTM(best_params['lstm_units'] // 2, return_sequences=True)),
        BatchNormalization(),
        Dropout(best_params['dropout_rate'] - 0.1),
        Bidirectional(LSTM(best_params['lstm_units'] // 4)),
        BatchNormalization(),
        Dropout(best_params['dropout_rate'] - 0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss=Huber(), metrics=['mae', 'mape'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )

    # บันทึกโมเดลและพารามิเตอร์
    joblib.dump(model, model_path2)
    print(f"✅ บันทึกโมเดลของ {model_dir} เรียบร้อยแล้ว")
    print(f"✅ บันทึกพารามิเตอร์ที่ดีที่สุดไว้ที่ {best_params_path}")

    return model

def predict_next_sales(model, X, df):
    last_sequence = X[-1].reshape(1, -1, X.shape[2])
    prediction = model.predict(last_sequence)[0][0]
    predicted_date = df['Date'].iloc[-1] + pd.DateOffset(days=1)
    return prediction, predicted_date

def get_product_name(product_code):
    if product_code == "A1001":
        return " Osida shoes"
    elif product_code == "A1002":
        return " Adda shoes"
    elif product_code == "A1004":
        return " Fashion shoes"
    elif product_code == "A1034":
        return " Court Shoes"
    elif product_code == "B1002":
        return " Long socks"
    elif product_code == "B1003":
        return " Short socks"
    elif product_code == "D1003":
        return " Mask pack"
    
def get_price(product_code, next_day_prediction):
    prices = {
        "A1001": 150,
        "A1002": 100,
        "A1004": 139,
        "A1034": 350,
        "B1002": 20,
        "B1003": 15,
        "D1003": 10
    }
    return prices.get(product_code, 0) * int(next_day_prediction)
