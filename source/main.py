import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import plotly.graph_objs as go
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pylab import rcParams  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input as KerasInput  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import yfinance as yf  # type: ignore
import dash  # type: ignore
from dash import dcc  # type: ignore
from dash import html  # type: ignore
from dash.dependencies import Input, Output  # type: ignore

# Configure plot size
rcParams['figure.figsize'] = 20, 10

app = dash.Dash()
server = app.server

# Declare constants
start_date = '2015-01-01'
end_date = '2024-01-01'
crypto_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']

# Step 1: Load yfinance data
def load_data(symbols):
    dfs = {}
    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date)
        dfs[symbol] = df
    return dfs

# Load data for multiple cryptocurrencies
crypto_data = load_data(crypto_symbols)
df_btc = crypto_data['BTC-USD']

# Data preparation for LSTM (use BTC-USD for LSTM training as an example)
df = df_btc.copy()
df = df.sort_index(ascending=True)

new_dataset = pd.DataFrame(index=df.index, columns=['Close'])
new_dataset['Close'] = df['Close']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_dataset[['Close']].values)

train_data = scaled_data[0:987, :]
valid_data = scaled_data[987:, :]

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# Step 3: Build and Train LSTM model
lstm_model = Sequential()
lstm_model.add(KerasInput(shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
lstm_model.save('saved_model.keras')

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

train = new_dataset[:987]
valid = new_dataset[987:].copy()
valid.loc[:, 'Predictions'] = predicted_closing_price

# Define layout and callbacks
app.layout = html.Div([
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Bitcoin Price Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid.index,
                                y=valid["Close"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Cryptocurrency Data', children=[
            html.Div([
                html.H1("Cryptocurrency Price Data", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='crypto-dropdown',
                    options=[{'label': symbol, 'value': symbol} for symbol in crypto_symbols],
                    value='BTC-USD',
                    style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}
                ),
                dcc.Graph(id='crypto-highlow'),
                dcc.Graph(id='crypto-volume')
            ], className="container"),
        ])
    ])
])

@app.callback(
    Output('crypto-highlow', 'figure'),
    [Input('crypto-dropdown', 'value')]
)
def update_crypto_graph(selected_symbol):
    df_selected = crypto_data[selected_symbol]
    figure = {
        'data': [
            go.Scatter(
                x=df_selected.index,
                y=df_selected['High'],
                mode='lines',
                name=f'High {selected_symbol}'
            ),
            go.Scatter(
                x=df_selected.index,
                y=df_selected['Low'],
                mode='lines',
                name=f'Low {selected_symbol}'
            )
        ],
        'layout': go.Layout(
            title=f"High and Low Prices for {selected_symbol} Over Time",
            xaxis={"title": "Date"},
            yaxis={"title": "Price (USD)"}
        )
    }
    return figure

@app.callback(
    Output('crypto-volume', 'figure'),
    [Input('crypto-dropdown', 'value')]
)
def update_crypto_volume_graph(selected_symbol):
    df_selected = crypto_data[selected_symbol]
    figure = {
        'data': [
            go.Scatter(
                x=df_selected.index,
                y=df_selected['Volume'],
                mode='lines',
                name=f'Volume {selected_symbol}'
            )
        ],
        'layout': go.Layout(
            title=f"Market Volume for {selected_symbol} Over Time",
            xaxis={"title": "Date"},
            yaxis={"title": "Volume"}
        )
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
