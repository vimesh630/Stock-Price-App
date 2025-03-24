# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model with error handling
try:
    model = joblib.load("stock_model_realtime.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the preprocessed dataset with error handling
try:
    # Load the CSV without parsing dates initially
    df = pd.read_csv("multi_stock_data_preprocessed.csv")
    # Convert the 'Date' column to datetime, handling timezone-aware dates
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    # Remove timezone information to make the index timezone-naive
    df.index = df.index.tz_convert(None)
    print("Dataset loaded successfully!")
    print("Index type:", type(df.index))
    print("Sample dates:", df.index[:5])
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None

# Load the model metrics with error handling
try:
    metrics_df = pd.read_csv("linear_regression_metrics.csv")
    print("Metrics loaded successfully!")
except Exception as e:
    print(f"Error loading metrics: {e}")
    metrics_df = None

# List of stocks
stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]

@app.route('/')
def index():
    if model is None or df is None or metrics_df is None:
        return "Error: Could not load model, dataset, or metrics. Check the server logs for details.", 500

    # Prepare historical data for Chart.js
    historical_data = {}
    for stock in stocks:
        # Create a list of {x, y} objects for each stock
        data_points = [
            {"x": date.strftime("%Y-%m-%d"), "y": price}
            for date, price in zip(df.index, df[stock])
        ]
        historical_data[stock] = data_points
    
    # Debug: Print the first few data points for AAPL
    print("Sample historical data for AAPL:", historical_data['AAPL'][:5])

    # Prepare latest Prev and MA7 values for each stock
    latest_features = {}
    for stock in stocks:
        # Get the last row for the stock
        last_row = df[[f"{stock}_Prev", f"{stock}_MA7"]].iloc[-1]
        latest_features[stock] = {
            "prev": round(last_row[f"{stock}_Prev"], 2),
            "ma7": round(last_row[f"{stock}_MA7"], 2)
        }
    print("Latest features:", latest_features)
    
    # Prepare metrics for display
    metrics = metrics_df.to_dict(orient="records")
    
    return render_template("index.html", stocks=stocks, historical_data=historical_data, latest_features=latest_features, metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check the server logs for details."}), 500

    try:
        # Get the selected stock and input features from the form
        stock = request.form['stock']
        prev_price = float(request.form['prev_price'])
        ma7 = float(request.form['ma7'])
        
        # Prepare the input features for prediction asap
        # The model expects features in the order: [AAPL_Prev, AAPL_MA7, TSLA_Prev, TSLA_MA7, ..., NVDA_Prev, NVDA_MA7]
        feature_columns = []
        for s in stocks:
            feature_columns.append(f"{s}_Prev")
            feature_columns.append(f"{s}_MA7")
        
        # Create the input array (zeros for other stocks, input values for the selected stock)
        input_features = np.zeros(len(feature_columns))
        stock_idx = stocks.index(stock)
        input_features[stock_idx * 2] = prev_price      # {stock}_Prev
        input_features[stock_idx * 2 + 1] = ma7         # {stock}_MA7
        
        # Make the prediction
        prediction = model.predict([input_features])[0]
        
        # The prediction array contains predicted prices for all stocks; we only need the selected stock
        predicted_price = prediction[stock_idx]
        
        return jsonify({"stock": stock, "predicted_price": round(predicted_price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)