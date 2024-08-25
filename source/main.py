import pandas as pd # type: ignore
import numpy as np # type: ignore

import plotly.graph_objs as go

import matplotlib.pyplot as plt # type: ignore

from matplotlib.pylab import rcParams # type: ignore
rcParams['figure.figsize']=20,10
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

# Declare constants
start_date = '2015-01-01'
end_date = '2024-01-01'

# Step 1: Load yfinance data
def load_data(crypto_symbol='BTC-USD'):
    df = yf.download(crypto_symbol, start=start_date, end=end_date)
    return df

df = load_data()

# Step 2: Prepare data for training

# Sort index
df = df.sort_index(ascending=True)

# Create DataFrame
new_dataset = pd.DataFrame(index=df.index, columns=['Close'])
new_dataset['Close'] = df['Close']

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the scaler on the 'Close' data
scaled_data = scaler.fit_transform(new_dataset[['Close']].values)

# Split the data into training and validation sets
train_data = scaled_data[0:987, :]
valid_data = scaled_data[987:, :]

# Prepare training data for LSTM
x_train_data, y_train_data = [], []

# Create sequences for training data
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

# Convert lists to numpy arrays
x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

# Reshape data to be compatible with LSTM [samples, time steps, features]
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# Output shapes for verification
print(f"x_train_data shape: {x_train_data.shape}")
print(f"y_train_data shape: {y_train_data.shape}")

# Step 3: Buid and Train LTSM model

# Initialize the Sequential model
lstm_model = Sequential()

# Add Input layer to specify the input shape
# units=50: Number of LSTM units in this layer
# return_sequences=True: Returns the full sequence of outputs for each input sequence
# Input(shape=(60, 1)) specifies that each input sequence has 60 time steps and 1 feature
lstm_model.add(KerasInput(shape=(x_train_data.shape[1], 1)))

# Add the first LSTM layer
lstm_model.add(LSTM(units=50, return_sequences=True))

# Add the second LSTM layer
# units=50: Number of LSTM units in this layer
# No return_sequences=True: Outputs the last output in the sequence
lstm_model.add(LSTM(units=50))

# Add a Dense layer
# Dense(1): Output layer with 1 unit (predicting a single value)
lstm_model.add(Dense(1))

# Prepare input data for making predictions
# Extract the last part of the new_dataset (for validation) and prepare it for the model
inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)  # Reshape for scaling
inputs_data = scaler.transform(inputs_data)  # Normalize using the same scaler

# Compile the model
# loss='mean_squared_error': Loss function to minimize (suitable for regression problems)
# optimizer='adam': Optimization algorithm for updating model weights
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

# Save the trained model to a file
lstm_model.save('saved_model.keras')

# Prepare input data for prediction
X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict closing prices using the trained model
predicted_closing_price = lstm_model.predict(X_test)

# Inverse transform the predicted values to original scale
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# Define the train and validation datasets
# Split the new_dataset into training and validation sets
# `train_data` contains data from the beginning up to index 987
train = new_dataset[:987]

# `valid_data` contains data from index 987 onwards
# Make a deep copy of the slice to avoid modifying a view and causing SettingWithCopyWarning
valid = new_dataset[987:].copy()

# Ensure proper DataFrame assignment using .loc
# Assign the predicted closing prices to the 'Predictions' column of valid_data
# This operation avoids the SettingWithCopyWarning by ensuring we are modifying a copy of the DataFrame
valid.loc[:, 'Predictions'] = predicted_closing_price


app.layout = html.Div([
    html.H1("Cryptocurrency Price Analysis Dashboard", style = {"textAlign": "center"}),
    dcc.Tabs(id = "tabs", children = [
    dcc.Tab(label = 'Bitcoin Price Data',children = [
        html.Div([
            html.H2("Actual closing price",style = {"textAlign": "center"}),
                dcc.Graph(
                    id = "Actual Data",
                    figure = {
                        "data":[
                            go.Scatter(
                                x = valid.index,
                                y = valid["Close"],
                                mode = 'markers'
                            )
                        ],
                        "layout":go.Layout(
                            title = 'scatter plot',
                            xaxis = {'title':'Date'},
                            yaxis = {'title':'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id = "Predicted Data",
                    figure = {
                        "data":[
                            go.Scatter(
                                x = valid.index,
                                y = valid["Predictions"],
                                mode = 'markers'
                            )
                        ],
                        "layout":go.Layout(
                            title = 'scatter plot',
                            xaxis = {'title':'Date'},
                            yaxis = {'title':'Closing Rate'}
                        )
                    }
                )                
            ])                
        ]),
        dcc.Tab(label='Facebook Close Data', children=[
            html.Div([
                html.H1("Facebook Stocks High vs Lows", 
                    style = {'textAlign': 'center'}),
                dcc.Dropdown(id = 'my-dropdown',
                    options = [
                        {'label': 'Tesla', 'value': 'TSLA'},
                        {'label': 'Apple','value': 'AAPL'}, 
                        {'label': 'Facebook', 'value': 'FB'}, 
                        {'label': 'Microsoft','value': 'MSFT'}
                    ], 
                    multi=True,value=['FB'],
                    style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Facebook Market Volume", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id = 'my-dropdown2',
                    options = [
                        {'label': 'Tesla', 'value': 'TSLA'},
                        {'label': 'Apple','value': 'AAPL'}, 
                        {'label': 'Facebook', 'value': 'FB'},
                        {'label': 'Microsoft','value': 'MSFT'}
                    ], 
                    multi = True,value=['FB'],
                    style = {"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id = 'volume')
            ], className = "container"),
        ])
    ])
])
@app.callback(Output('highlow', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(
                x = df[df["Close"] == stock]["Date"],
                y = df[df["Close"] == stock]["High"],
                mode = 'lines', 
                opacity = 0.7, 
                name = f'High {dropdown[stock]}',
                textposition='bottom center'
            )
        )
        trace2.append(
            go.Scatter(
                x = df[df["Close"] == stock]["Date"],
                y = df[df["Close"] == stock]["Low"],
                mode = 'lines', 
                opacity=0.6,
                name = f'Low {dropdown[stock]}',
                textposition='bottom center'
            )
        )
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {
            'data': data,
            'layout': go.Layout(
                colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height = 600,
                title = f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                xaxis = {
                    "title":"Date",
                    'rangeselector': {
                        'buttons': list([
                        {
                            'count': 1, 'label': '1M', 
                            'step': 'month', 
                            'stepmode': 'backward'
                        },
                        {
                            'count': 6, 'label': '6M', 
                            'step': 'month', 
                            'stepmode': 'backward'
                        },
                        {'step': 'all'}])
                    },
                    'rangeslider': {'visible': True}, 'type': 'date'
                },
                yaxis = {"title":"Price (USD)"}
            )
        }
    return figure
@app.callback(Output('volume', 'figure'), [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(
                x = df[df["Close"] == stock]["Date"],
                y = df[df["Close"] == stock]["Volume"],
                mode = 'lines', 
                opacity = 0.7,
                name = f'Volume {dropdown[stock]}', 
                textposition = 'bottom center'
            )
        )
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {
            'data': data, 
            'layout': go.Layout(
                colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height = 600,
                title = f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                xaxis = {
                    "title":"Date",
                    'rangeselector': {
                        'buttons': list([
                            {
                                'count': 1, 
                                'label': '1M', 
                                'step': 'month', 
                                'stepmode': 'backward'
                            },
                            {
                                'count': 6, 'label': '6M',
                                'step': 'month', 
                                'stepmode': 'backward'
                            },
                            {'step': 'all'}
                        ])
                    },
                    'rangeslider': {'visible': True}, 'type': 'date'
                },
                yaxis = {"title":"Transactions Volume"}
            )
        }
    return figure
if __name__=='__main__':
    app.run_server(debug=True)