import os
import sys
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error, root_mean_squared_error
import numpy as np
import requests  # เพิ่มไลบรารีสำหรับการส่ง HTTP requests

# เพิ่มเส้นทางไปยังโฟลเดอร์รากของโปรเจกต์
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data.load_data import load_data, load_dataps
from src.data.preprocess_data import preprocess_data, preprocess_dataps
from src.model.Daily.LSTM import prepare_data as prepare_daily_data, train_lstm_model as train_daily_model, predict_next_sales as predict_daily_sales
from src.model.Product.LSTMPS import prepare_data as prepare_monthly_data, train_lstm_model as train_monthly_model, predict_next_sales as predict_monthly_sales, get_product_name, get_price

app = Flask(__name__)

def send_prediction_to_api(date, type, result):
    url = "https://termpro-api-production.up.railway.app/History_predic/"
    payload = {
        "date": date,
        "type": type,
        "result": result
    }
    try:
        # ตรวจสอบว่ามีข้อมูลวันที่และประเภทนั้นอยู่ใน API หรือไม่
        response = requests.get(url, params={"type": type})
        if response.status_code == 200:
            data = response.json()
            # กรองข้อมูลให้ตรงกับวันที่และประเภทที่ต้องการ
            existing_data = [
                item for item in data 
                if item['date'][:10] == date and item['type'] == type
            ]
            if existing_data:  # ตรวจสอบว่ามีข้อมูลจริง
                print(f"⚠️ ข้อมูลวันที่ {date} และประเภท {type} มีอยู่แล้วในระบบ: {existing_data}")
                return {"status": "duplicate", "message": "Data already exists in the system."}

        # ถ้าไม่มีข้อมูล ให้เพิ่มข้อมูลใหม่
        response = requests.post(url, json=payload)
        if response.status_code == 201:
            print(f"✅ เพิ่มข้อมูลสำเร็จ: {payload}")
            return {"status": "success", "message": "Data added successfully."}
        else:
            print(f"⚠️ ไม่สามารถเพิ่มข้อมูลได้: {response.status_code}, {response.text}")
            return {"status": "error", "message": f"Failed to add data: {response.status_code}"}
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการส่งข้อมูลไปยัง API: {e}")
        return {"status": "error", "message": f"Exception occurred: {e}"}

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
        rmse = root_mean_squared_error(y_test, predicted_sales)

        next_day_prediction, predicted_date = predict_daily_sales(model, X, df_prepared)

        predictions['daily'] = {
            'predicted_sales': round(float(next_day_prediction), 2),
            'predicted_date': str(predicted_date),
            'model_name': "LSTM Daily Model",
            'metrics': {
                'mae': round(float(mae), 2),
                'mape': round(float(mape), 2),
                'r2': round(float(r2), 2),
                'rmse': round(float(rmse), 2)
            }
        }
        # เรียกใช้ฟังก์ชันส่งค่าทำนาย
        send_prediction_to_api(predicted_date.strftime('%Y-%m-%d'), "Daily", float(next_day_prediction))  # แปลงเป็น float
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
        try:
            print(f"🔄 กำลังเทรนโมเดลสำหรับ {product_code}...")

            df_product = df_preprocessed[df_preprocessed['Product_code'] == product_code]

            if df_product.empty:
                print(f"⚠️ ไม่มีข้อมูลสำหรับ {product_code} ข้ามไป...")
                continue

            X, y, df_prepared, scaler = prepare_monthly_data(df_product)

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)

            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model', 'Product', 'ModelLstm2', product_code))
            os.makedirs(model_dir, exist_ok=True)
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
                'predicted_sales': int(next_day_prediction),  # แปลงเป็น int
                'predicted_date': str(predicted_date),
                'price': price,
                'metrics': {
                    'mae': round(float(mae), 2),
                    'mape': round(float(mape), 2),
                    'r2': round(float(r2), 2),
                    'rmse': round(float(rmse), 2)
                }
            }
            # เรียกใช้ฟังก์ชันส่งค่าทำนาย
            send_prediction_to_api(predicted_date.strftime('%Y-%m-%d'), product_code, int(next_day_prediction))  # แปลงเป็น int
        except Exception as e:
            print(f"⚠️ Error for {product_code}: {e}")

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='localhost', port=8877, debug=True)

