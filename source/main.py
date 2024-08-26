import pandas as pd # type: ignore
import numpy as np # type: ignore

import plotly.graph_objs as go
import matplotlib.pyplot as plt # type: ignore

from matplotlib.pylab import rcParams # type: ignore
rcParams['figure.figsize'] = 20, 10
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input as KerasInput # type: ignore

from sklearn.preprocessing import MinMaxScaler # type: ignore
import yfinance as yf # type: ignore

import dash # type: ignore
from dash import dcc # type: ignore
from dash import html # type: ignore
from dash.dependencies import Input, Output # type: ignore

app = dash.Dash()
server = app.server

# Constants
crypto_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']
start_date = '2015-01-01'
end_date = '2024-08-01'

# Function to load data of multiple cryptocurrencies
def load_data():
    data = {}
    for symbol in crypto_symbols:
        df = yf.download(symbol, start=start_date, end=end_date)
        data[symbol] = df
    return data

# Function to preprocess data
def preprocess_data(df):
    df = df.sort_index(ascending=True)
    new_dataset = pd.DataFrame(index=df.index, columns=['Close'])
    new_dataset['Close'] = df['Close']
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(new_dataset[['Close']].values)
    
    return new_dataset, scaled_data, scaler

# Function to prepare data for LSTM model
def prepare_lstm_data(scaled_data):
    train_data = scaled_data[0:987, :]
    valid_data = scaled_data[987:, :]

    x_train_data, y_train_data = [], []
    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
    
    return x_train_data, y_train_data, valid_data

# Function to build and train LSTM model
def build_and_train_model(x_train_data, y_train_data):
    model = Sequential()
    model.add(KerasInput(shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
    
    return model

# Function to make predictions
def make_predictions(model, valid_data, scaler, new_dataset):
    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_closing_price = model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
    
    return predicted_closing_price

# Function to update validation data with predictions
def update_validation_with_predictions(valid_data, predictions, new_dataset):
    valid_df = new_dataset.iloc[-len(predictions):].copy()
    valid_df['Predictions'] = predictions
    return valid_df

# Load data
data = load_data()

# Preprocess data for each cryptocurrency
btc_data, btc_scaled_data, btc_scaler = preprocess_data(data['BTC-USD'])
eth_data, eth_scaled_data, eth_scaler = preprocess_data(data['ETH-USD'])
ada_data, ada_scaled_data, ada_scaler = preprocess_data(data['ADA-USD'])

# Prepare data for LSTM model
x_train_btc, y_train_btc, valid_btc = prepare_lstm_data(btc_scaled_data)
x_train_eth, y_train_eth, valid_eth = prepare_lstm_data(eth_scaled_data)
x_train_ada, y_train_ada, valid_ada = prepare_lstm_data(ada_scaled_data)

# Build and train models for each cryptocurrency
btc_model = build_and_train_model(x_train_btc, y_train_btc)
eth_model = build_and_train_model(x_train_eth, y_train_eth)
ada_model = build_and_train_model(x_train_ada, y_train_ada)

# Make predictions for each cryptocurrency
btc_predictions = make_predictions(btc_model, valid_btc, btc_scaler, btc_data)
eth_predictions = make_predictions(eth_model, valid_eth, eth_scaler, eth_data)
ada_predictions = make_predictions(ada_model, valid_ada, ada_scaler, ada_data)

# Update validation datasets with predictions
valid_btc = update_validation_with_predictions(valid_btc, btc_predictions, btc_data)
valid_eth = update_validation_with_predictions(valid_eth, eth_predictions, eth_data)
valid_ada = update_validation_with_predictions(valid_ada, ada_predictions, ada_data)

# App layout with additional tabs for ETH and ADA
app.layout = html.Div([
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Bitcoin Price Data', children=[
            html.Div([
                html.H2("Actual BTC Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data BTC",
                    figure={
                        "data": [
                            go.Scatter(
                                x=btc_data.index,
                                y=btc_data["Close"],
                                mode='lines',  # Use 'lines' to show continuous data
                                name='Actual BTC Price'
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual BTC Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted BTC Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data BTC",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_btc.index,
                                y=valid_btc["Predictions"],
                                mode='lines',  # Use 'lines' to show continuous data
                                name='Predicted BTC Price'
                            )
                        ],
                        "layout": go.Layout(
                            title='Predicted BTC Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Ethereum Price Data', children=[
            html.Div([
                html.H2("Actual ETH Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data ETH",
                    figure={
                        "data": [
                            go.Scatter(
                                x=eth_data.index,
                                y=eth_data["Close"],
                                mode='lines',  # Use 'lines' to show continuous data
                                name='Actual ETH Price'
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual ETH Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted ETH Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ETH",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_eth.index,
                                y=valid_eth["Predictions"],
                                mode='lines',  # Use 'lines' to show continuous data
                                name='Predicted ETH Price'
                            )
                        ],
                        "layout": go.Layout(
                            title='Predicted ETH Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Cardano Price Data', children=[
            html.Div([
                html.H2("Actual ADA Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data ADA",
                    figure={
                        "data": [
                            go.Scatter(
                                x=ada_data.index,
                                y=ada_data["Close"],
                                mode='lines',  # Use 'lines' to show continuous data
                                name='Actual ADA Price'
                            )
                        ],
                        "layout": go.Layout(
                            title='Actual ADA Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted ADA Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ADA",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_ada.index,
                                y=valid_ada["Predictions"],
                                mode='lines',  # Use 'lines' to show continuous data
                                name='Predicted ADA Price'
                            )
                        ],
                        "layout": go.Layout(
                            title='Predicted ADA Prices',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Cryptocurrency Data', children=[
            html.Div([
                html.H1("Cryptocurrency Prices High vs Lows", 
                    style = {'textAlign': 'center'}),
                dcc.Dropdown(id = 'crypto-dropdown',
                    options = [{'label': symbol, 'value': symbol} for symbol in crypto_symbols], 
                    multi=True,
                    value=['BTC-USD'],
                    style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='crypto-highlow'),
                html.H1("Cryptocurrency Market Volume", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id = 'crypto-dropdown2',
                    options = [{'label': symbol, 'value': symbol} for symbol in crypto_symbols], 
                    multi = True,
                    value=['BTC-USD'],
                    style = {"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id = 'crypto-volume')
            ], className = "container"),
        ])
    ])
])

@app.callback(Output('crypto-highlow', 'figure'), [Input('crypto-dropdown', 'value')])
def update_crypto_graph(selected_symbols):
    traces = []
    for symbol in selected_symbols:
        crypto_data = data[symbol]
        traces.append(
            go.Scatter(
                x=crypto_data.index,
                y=crypto_data['High'],
                mode='lines',
                name=f'High {symbol}',
                opacity=0.7
            )
        )
        traces.append(
            go.Scatter(
                x=crypto_data.index,
                y=crypto_data['Low'],
                mode='lines',
                name=f'Low {symbol}',
                opacity=0.6
            )
        )
    figure = {
        'data': traces,
        'layout': go.Layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            title=f"High and Low Prices for Selected Cryptocurrencies",
            xaxis={"title": "Date",
                'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                    {'step': 'all'}])},
                'rangeslider': {'visible': True}, 'type': 'date'},
            yaxis={"title": "Price (USD)"}
        )
    }
    return figure

@app.callback(Output('crypto-volume', 'figure'), [Input('crypto-dropdown2', 'value')])
def update_crypto_volume_graph(selected_symbols):
    traces = []
    for symbol in selected_symbols:
        crypto_data = data[symbol]
        traces.append(
            go.Scatter(
                x=crypto_data.index,
                y=crypto_data['Volume'],
                mode='lines',
                name=f'Volume {symbol}',
                opacity=0.7
            )
        )
    figure = {
        'data': traces,
        'layout': go.Layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            title=f"Market Volume for Selected Cryptocurrencies",
            xaxis={"title": "Date",
                'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                    {'step': 'all'}])},
                'rangeslider': {'visible': True}, 'type': 'date'},
            yaxis={"title": "Volume"}
        )
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
