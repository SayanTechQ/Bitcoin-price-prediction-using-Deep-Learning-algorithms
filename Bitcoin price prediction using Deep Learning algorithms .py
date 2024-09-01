#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().system('pip install mplfinance')


# # Importing required libraries 

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import LSTM


# # Data Loading and reading 

# In[3]:


df = pd.read_csv("BTC_USD.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# # checking Missing values 

# In[6]:


# Check for missing values in each column
missing_values = df.isnull().sum()

# Display the missing values
print(missing_values)


# # Exploratory Data Analysis

# In[41]:


# Data Information
df.info()


# In[43]:


# Data Description 
df.describe()


# In[8]:


# Line Plot of Closing Price Over Time
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(df['Date']), df['Close'], label='Closing Price', color='blue')
plt.title('Bitcoin Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.grid(True)
plt.show()


# In[9]:


# Histogram of Daily Returns
df['Daily Return'] = df['Close'].pct_change()
plt.figure(figsize=(10, 6))
plt.hist(df['Daily Return'].dropna(), bins=50, color='purple', alpha=0.75)
plt.title('Histogram of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[10]:


# Moving Average Plot
df['50-Day MA'] = df['Close'].rolling(window=50).mean()
df['200-Day MA'] = df['Close'].rolling(window=200).mean()
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(df['Date']), df['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(pd.to_datetime(df['Date']), df['50-Day MA'], label='50-Day MA', color='red')
plt.plot(pd.to_datetime(df['Date']), df['200-Day MA'], label='200-Day MA', color='green')
plt.title('Bitcoin Closing Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


# Box Plot of Daily Prices
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Open', 'High', 'Low', 'Close']], palette='Set3')
plt.title('Box Plot of Daily Bitcoin Prices')
plt.ylabel('Price (USD)')
plt.show()


# In[12]:


# Volume Traded Over Time
plt.figure(figsize=(10, 6))
plt.bar(pd.to_datetime(df['Date']), df['Volume'], color='orange', alpha=0.6)
plt.title('Bitcoin Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.show()


# In[16]:


# Candlestick Chart
df_candlestick = df.copy()
df_candlestick['Date'] = pd.to_datetime(df_candlestick['Date'])
df_candlestick.set_index('Date', inplace=True)
mpf.plot(df_candlestick, type='candle', style='charles', title='Bitcoin Candlestick Chart', ylabel='Price (USD)')


# In[17]:


# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Bitcoin Prices and Volume')
plt.show()


# # Data Preprocessing 

# In[18]:


# Selecting the relevant feature(s)
features = ['Close'] 
df_selected = df[features]


# In[19]:


# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the selected data
scaled_data = scaler.fit_transform(df_selected)

# Convert scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(scaled_data, columns=features, index=df.index)


# In[22]:


# Function to create sequences for time-series data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Set sequence length (e.g., 60 days)
sequence_length = 60

# Create sequences
X, y = create_sequences(df_scaled.values, sequence_length)


# In[23]:


# Split into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")


# # Recurrent Neural Network

# In[25]:


# Define the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
rnn_model.add(Dense(units=1))

# Compile the model
rnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
rnn_model.summary()


# In[26]:


# Train the RNN model
rnn_history = rnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)


# In[27]:


# Evaluate the RNN model
rnn_loss = rnn_model.evaluate(X_test, y_test)
print(f"RNN Model Test Loss: {rnn_loss}")


# In[28]:


# Make predictions
rnn_predictions = rnn_model.predict(X_test)

# Inverse transform the predictions and actual values to original scale
rnn_predictions = scaler.inverse_transform(rnn_predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual Bitcoin Price')
plt.plot(rnn_predictions, color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using RNN')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price (USD)')
plt.legend()
plt.show()


# # Long Short-Term Memory model

# In[29]:


# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(LSTM(units=50, activation='tanh'))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
lstm_model.summary()


# In[30]:


# Train the LSTM model
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)


# In[31]:


# Evaluate the LSTM model
lstm_loss = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Model Test Loss: {lstm_loss}")


# In[32]:


# Make predictions
lstm_predictions = lstm_model.predict(X_test)

# Inverse transform the predictions and actual values to original scale
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual Bitcoin Price')
plt.plot(lstm_predictions, color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price (USD)')
plt.legend()
plt.show()


# # Model Comparison Graphs

# In[33]:


# Plot training and validation loss for RNN
plt.figure(figsize=(14, 6))
plt.plot(rnn_history.history['loss'], label='RNN Training Loss')
plt.plot(rnn_history.history['val_loss'], label='RNN Validation Loss')
plt.title('RNN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation loss for LSTM
plt.figure(figsize=(14, 6))
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[34]:


# Predicted vs. Actual Prices for RNN
plt.figure(figsize=(6, 6))
plt.scatter(y_test_rescaled, rnn_predictions, alpha=0.5)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('RNN: Predicted vs. Actual Prices')
plt.show()

# Predicted vs. Actual Prices for LSTM
plt.figure(figsize=(6, 6))
plt.scatter(y_test_rescaled, lstm_predictions, alpha=0.5)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('LSTM: Predicted vs. Actual Prices')
plt.show()


# In[35]:


# Residuals for RNN
rnn_residuals = y_test_rescaled - rnn_predictions

plt.figure(figsize=(14, 6))
plt.plot(rnn_residuals, label='RNN Residuals', color='blue')
plt.title('RNN Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.grid(True)
plt.legend()
plt.show()

# Residuals for LSTM
lstm_residuals = y_test_rescaled - lstm_predictions

plt.figure(figsize=(14, 6))
plt.plot(lstm_residuals, label='LSTM Residuals', color='green')
plt.title('LSTM Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.grid(True)
plt.legend()
plt.show()


# In[36]:


# Choose a specific timeframe to zoom in
zoom_start = -100  # Last 100 points
zoom_end = None

plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled[zoom_start:zoom_end], label='Actual Prices', color='blue')
plt.plot(rnn_predictions[zoom_start:zoom_end], label='RNN Predicted Prices', color='red', linestyle='dotted')
plt.plot(lstm_predictions[zoom_start:zoom_end], label='LSTM Predicted Prices', color='green', linestyle='dotted')
plt.title('Zoomed-in: Predictions vs. Actual Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# In[37]:


# Error distribution for RNN
plt.figure(figsize=(10, 6))
plt.hist(rnn_residuals, bins=50, alpha=0.7, color='blue', label='RNN Residuals')
plt.title('RNN Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Error distribution for LSTM
plt.figure(figsize=(10, 6))
plt.hist(lstm_residuals, bins=50, alpha=0.7, color='green', label='LSTM Residuals')
plt.title('LSTM Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[39]:


# Calculate cumulative returns for actual and predicted prices
actual_returns = y_test_rescaled / y_test_rescaled[0]
rnn_predicted_returns = rnn_predictions / y_test_rescaled[0]
lstm_predicted_returns = lstm_predictions / y_test_rescaled[0]

plt.figure(figsize=(14, 6))
plt.plot(actual_returns, label='Actual Cumulative Returns', color='blue')
plt.plot(rnn_predicted_returns, label='RNN Predicted Cumulative Returns', color='red')
plt.plot(lstm_predicted_returns, label='LSTM Predicted Cumulative Returns', color='green')
plt.title('Cumulative Returns Comparison')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


# In[40]:


# Comparison of RNN and LSTM predictions 
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
plt.plot(rnn_predictions, label='RNN Predicted Prices', color='red', linestyle='dotted')
plt.plot(lstm_predictions, label='LSTM Predicted Prices', color='green', linestyle='dotted')
plt.title('Comparison of RNN and LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# In[ ]:




