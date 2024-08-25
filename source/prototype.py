import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import yfinance as yf
from flask import Flask, send_file # type: ignore
import io

app = Flask(__name__)

# Step 1: Load Historical Data
def load_data(crypto_symbol='BTC-USD', start_date='2015-01-01', end_date='2024-01-01'):
    df = yf.download(crypto_symbol, start=start_date, end=end_date)
    return df['Close'].values.reshape(-1, 1)

# Step 2: Preprocess Data
def preprocess_data(data, training_data_size=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * training_data_size)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, test_data, scaler

# Step 3: Build and Train the Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Predict Prices
def predict_prices(model, test_data, scaler):
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

# Step 5: Plot Results and Serve as Image
@app.route('/')
def index():
    # Load and preprocess data
    data = load_data()
    x_train, y_train, test_data, scaler = preprocess_data(data)

    # Build and train the LSTM model
    model = build_model(x_train.shape)
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Make predictions
    predicted_prices = predict_prices(model, test_data, scaler)
    actual_prices = scaler.inverse_transform(test_data[60:])

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(predicted_prices, color='green', label='Predicted Prices')
    plt.title('Cryptocurrency Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the plot to free resources

    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
