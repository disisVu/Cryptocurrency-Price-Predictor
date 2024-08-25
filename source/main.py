# import pandas as pd
# import numpy as np

# import matplotlib.pyplot as plt
# %matplotlib inline

# from matplotlib.pylab import rcParams
# rcParams['figure.figsize']=20,10
# from keras.models import Sequential
# from keras.layers import LSTM,Dropout,Dense


# from sklearn.preprocessing import MinMaxScaler

# # Step 1: Load Historical Data
# def load_data(crypto_symbol='BTC-USD', start_date='2015-01-01', end_date='2024-01-01'):
#     df = yf.download(crypto_symbol, start=start_date, end=end_date)
#     return df['Close'].values.reshape(-1, 1)