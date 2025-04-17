import os
import sys
from unicodedata import bidirectional

from sklearn.discriminant_analysis import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error
from tensorflow.keras.regularizers import l2
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PJML')))

def add_time_features(df):
    df['day_of_week'] = df['sale_date'].dt.dayofweek
    df['month'] = df['sale_date'].dt.month
    df['quarter'] = df['sale_date'].dt.quarter
    df['year'] = df['sale_date'].dt.year
    df['day_of_year'] = df['sale_date'].dt.dayofyear
    
    # Add cyclical encoding for seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    return df

def add_new_features(df):
    df['prev_day_diff'] = df['sales_amount'] - df['sales_amount'].shift(1)
    df['rolling_avg_60'] = df['sales_amount'].rolling(window=60).mean()
    return df

def add_lag_features(df, lags=[1, 2, 3, 7, 14, 30]):
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['sales_amount'].shift(lag)
    return df

def add_rolling_features(df):
    windows = [7, 14, 30]
    for window in windows:
        df[f'sales_ma_{window}'] = df['sales_amount'].rolling(window=window).mean()
        df[f'sales_std_{window}'] = df['sales_amount'].rolling(window=window).std()
        df[f'sales_min_{window}'] = df['sales_amount'].rolling(window=window).min()
        df[f'sales_max_{window}'] = df['sales_amount'].rolling(window=window).max()
    return df

def augment_time_series(df, random_seed=42):
    np.random.seed(random_seed)  # กำหนด seed ให้ noise คงที่ทุกครั้ง
    augmented_data = pd.DataFrame()
    
    # Time shift augmentation
    for shift in [-3, -2, -1, 1, 2, 3]:
        shifted = df.copy()
        shifted['sales_amount'] = shifted['sales_amount'].shift(shift)
        augmented_data = pd.concat([augmented_data, shifted.dropna()])
    
    # Random noise augmentation with different intensities
    noise_scales = [0.03, 0.05, 0.07]
    for scale in noise_scales:
        noisy = df.copy()
        noise = np.random.normal(0, df['sales_amount'].std() * scale, len(df))
        noisy['sales_amount'] += noise
        augmented_data = pd.concat([augmented_data, noisy])
    
    # Scaling augmentation
    for scale in [0.9, 0.95, 1.05, 1.1]:
        scaled = df.copy()
        scaled['sales_amount'] = scaled['sales_amount'] * scale
        augmented_data = pd.concat([augmented_data, scaled])
    
    # Trend augmentation
    trend = df.copy()
    trend['sales_amount'] = trend['sales_amount'] * (1 + np.linspace(0, 0.1, len(df)))
    augmented_data = pd.concat([augmented_data, trend])
    
    return pd.concat([df, augmented_data]).sort_values('sale_date').reset_index(drop=True)

# def save_to_csv(df, filename):
#     df.to_csv(filename, index=False)
#     print(f"✅ Data saved to {filename}")

def prepare_data(df):
    print("🔄 เริ่มการเตรียมข้อมูล...")
    
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.dropna(subset=['sale_date'])
    
    print("➕ กำลังเพิ่ม features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_new_features(df)
    df = add_rolling_features(df)
    df = df.dropna()
    
    print("🔄 กำลังทำ Data Augmentation...")
    df_augmented = augment_time_series(df)

    # Save augmented data
    # save_to_csv(df_augmented, 'augmented_data.csv')

    # # Scale features
    # scaler = StandardScaler()
    # features = ['Temperature', 'day_of_week', 'month', 'quarter', 'year', 
    #             'day_of_year', 'month_sin', 'month_cos', 'day_of_week_sin', 
    #             'day_of_week_cos'] + \
    #           [col for col in df.columns if 'sales_lag_' in col or 
    #                                       'sales_ma_' in col or 
    #                                       'sales_std_' in col or 
    #                                       'sales_min_' in col or 
    #                                       'sales_max_' in col]
    
    # Scale features
    scaler = StandardScaler()
    features = ['Temperature', 'prev_day_diff', 'day_of_week', 'month','day_of_year','rolling_avg_60'] + \
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
        y.append(df_augmented['sales_amount'].iloc[i])
    
    return np.array(X), np.array(y), df_augmented, scaler

def train_lstm_model(X_train, y_train, X_val, y_val):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm1')
    model_path = os.path.join(model_dir, 'lstm_model1.pkl')
    grid_search_results_path = os.path.join(model_dir, 'grid_search_results.pkl')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if the model already exists
    if os.path.exists(model_path):
        print("📥 โหลดโมเดลจากไฟล์ที่เก็บไว้แล้ว")
        model = joblib.load(model_path)
        return model
    
    print("🛠️ กำลังสร้างโมเดลใหม่...")
    
    # Define model-building function
    def create_lstm_model(lstm_units, dropout_rate, learning_rate):
        model = Sequential([
            Bidirectional(LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001))),
            BatchNormalization(),
            Dropout(dropout_rate),

            Bidirectional(LSTM(lstm_units//2, return_sequences=True, kernel_regularizer=l2(0.001))),
            BatchNormalization(),
            Dropout(dropout_rate),

            Bidirectional(LSTM(lstm_units//4, kernel_regularizer=l2(0.001))),
            BatchNormalization(),
            Dropout(dropout_rate),

            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=Huber(),
            metrics=['mae', 'mape']
        )
        
        return model
    
    # Define parameter grid for Grid Search
    param_grid = {
        'lstm_units': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005]
    }
    
    # Manual Grid Search implementation
    print("🔍 เริ่มทำ Grid Search เพื่อหาพารามิเตอร์ที่ดีที่สุด...")
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        param_grid['lstm_units'],
        param_grid['dropout_rate'],
        param_grid['learning_rate']
    ))
    
    best_mae = float('inf')
    best_params = None
    grid_results = []
    
    # Split training data for validation during grid search
    X_grid_train, X_grid_val, y_grid_train, y_grid_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    if os.path.exists(grid_search_results_path):
        print("📥 โหลดผลลัพธ์ Grid Search ที่มีอยู่แล้ว...")
        grid_search_results = joblib.load(grid_search_results_path)
        best_params = grid_search_results['best_params']
        best_mae = grid_search_results['best_mae']
        print(f"🏆 พารามิเตอร์ที่ดีที่สุดจาก Grid Search เดิม: {best_params}")
        print(f"🎯 ค่า MAE ที่ดีที่สุดจาก Grid Search เดิม: {best_mae:.4f}")
    else:
        print("🔍 เริ่มทำ Grid Search ใหม่...")
        # Loop through all parameter combinations
        for i, (lstm_units, dropout_rate, learning_rate) in enumerate(param_combinations):
            print(f"\n🔄 ทดสอบพารามิเตอร์ชุดที่ {i+1}/{len(param_combinations)}")
            print(f"   lstm_units: {lstm_units}, dropout_rate: {dropout_rate}, learning_rate: {learning_rate}")
            
            # Create and train model with current parameters
            model = create_lstm_model(lstm_units, dropout_rate, learning_rate)
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            )
            
            history = model.fit(
                X_grid_train, y_grid_train,
                validation_data=(X_grid_val, y_grid_val),
                epochs=30,  # ลดจำนวน epoch ลงเพื่อให้ grid search เร็วขึ้น
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate model on validation set
            val_loss, val_mae, val_mape = model.evaluate(X_val, y_val, verbose=0)
            
            # Save results
            result = {
                'params': {
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate
                },
                'val_mae': val_mae,
                'val_loss': val_loss,
                'val_mape': val_mape
            }
            grid_results.append(result)
            
            # Check if this is the best model so far
            if val_mae < best_mae:
                best_mae = val_mae
                best_params = result['params']
                print(f"✅ พบพารามิเตอร์ที่ดีกว่า! MAE: {val_mae:.4f}")
            
            # Clear session to free memory
            from tensorflow.keras import backend as K
            K.clear_session()
        
        # Save grid search results
        grid_search_results = {
            'results': grid_results,
            'best_params': best_params,
            'best_mae': best_mae
        }
        joblib.dump(grid_search_results, grid_search_results_path)
        print(f"\n✅ Grid Search เสร็จสิ้น")
        print(f"🏆 พารามิเตอร์ที่ดีที่สุด: {best_params}")
        print(f"🎯 ค่า MAE ที่ดีที่สุด: {best_mae:.4f}")

    # Train final model with best parameters
    print("🔄 กำลังเทรนโมเดลสุดท้ายด้วยพารามิเตอร์ที่ดีที่สุด...")
    
    final_model = create_lstm_model(
        best_params['lstm_units'],
        best_params['dropout_rate'],
        best_params['learning_rate']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        mode='min'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    history = final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Save best model
    joblib.dump(final_model, model_path)
    print("✅ บันทึกโมเดลเรียบร้อยแล้ว")
    
    return final_model

def predict_next_sales(model, X, df):
    last_sequence = X[-1].reshape(1, -1, X.shape[2])
    prediction = model.predict(last_sequence)[0][0]
    predicted_date = df['sale_date'].iloc[-1] + pd.DateOffset(days=1)
    return prediction, predicted_date

