import os
import sys
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
import numpy as np

# เพิ่มเส้นทางไปยังโฟลเดอร์รากของโปรเจกต์
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data.load_data import load_data, load_dataps
from src.data.preprocess_data import preprocess_data, preprocess_dataps
from src.model.Daily.LSTM import prepare_data as prepare_daily_data, train_lstm_model as train_daily_model, predict_next_sales as predict_daily_sales
from src.model.Product.LSTMPS import prepare_data as prepare_monthly_data, train_lstm_model as train_monthly_model, predict_next_sales as predict_monthly_sales, get_product_name, get_price

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales_api():
    predictions = {}

    # Daily Mode
    try:
        print("🔄 กำลังโหลดข้อมูลรายวัน...")
        df = load_data()
        df_preprocessed = preprocess_data(df)

        X, y, df_prepared, scaler = prepare_daily_data(df_preprocessed)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.45, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)

        model = train_daily_model(X_train, y_train, X_val, y_val)

        print("🔮 กำลังทำนายผลรายวัน...")
        predicted_sales = model.predict(X_test)
        mae = mean_absolute_error(predicted_sales, y_test)
        mape = mean_absolute_percentage_error(predicted_sales, y_test)
        r2 = r2_score(y_test, predicted_sales)
        rmse = mean_squared_error(y_test, predicted_sales, squared=False)

        next_day_prediction, predicted_date = predict_daily_sales(model, X, df_prepared)

        predictions['daily'] = {
            'predicted_sales': float(next_day_prediction),
            'predicted_date': str(predicted_date),
            'model_name': "LSTM Daily Model",
            'metrics': {
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2),
                'rmse': float(rmse)
            }
        }
    except Exception as e:
        print(f"⚠️ Error in Daily Mode: {e}")

    # Monthly Mode
    try:
        print("🔄 กำลังโหลดข้อมูลรายเดือน...")
        df = load_dataps()
        df_preprocessed = preprocess_dataps(df)
    except Exception as e:
        print(f"⚠️ Error in Monthly Mode: {e}")
        return jsonify({'error': 'An error occurred in Monthly Mode.'})

    product_codes = ["A1001", "A1002", "A1004", "A1034", "B1002", "B1003", "D1003"]

    for product_code in product_codes:
        print(f"🔄 กำลังเทรนโมเดลสำหรับ {product_code}...")

        df_product = df_preprocessed[df_preprocessed['Product_code'] == product_code]

        if df_product.empty:
            print(f"⚠️ ไม่มีข้อมูลสำหรับ {product_code} ข้ามไป...")
            continue

        X, y, df_prepared, scaler = prepare_monthly_data(df_product)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)

        model_dir = os.path.join('ModelLstm2', product_code)
        model = train_monthly_model(X_train, y_train, X_val, y_val, model_dir, product_code)

        print(f"🔮 กำลังทำนายผลสำหรับ {product_code}...")
        predicted_sales = model.predict(X_test)
        mae = mean_absolute_error(predicted_sales, y_test)
        mape = mean_absolute_percentage_error(predicted_sales, y_test)
        rmse = mean_squared_error(y_test, predicted_sales, squared=False)
        r2 = r2_score(y_test, predicted_sales)

        next_day_prediction, predicted_date = predict_monthly_sales(model, X, df_prepared)
        product_name = get_product_name(product_code)
        price = get_price(product_code, next_day_prediction)

        predictions[product_code + product_name] = {
            'predicted_sales': int(next_day_prediction),
            'predicted_date': str(predicted_date),
            'price': price,
            'metrics': {
                'mae': round(float(mae), 2),
                'mape': round(float(mape), 2),
                'r2': round(float(r2), 2),
                'rmse': round(float(rmse), 2)
            }
        }

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='localhost', port=8877, debug=True)