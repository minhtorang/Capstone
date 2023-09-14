import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the data
data = pd.read_excel('Dataset/data.xlsx')  # Replace with your data file
prices = data['Giá Cà Phê'].values.astype(float)
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices.reshape(-1, 1))


# Define a function to create sequences from the time series data
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


window_size = 10  # Adjust the window size as needed
X, y = create_sequences(prices, window_size)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(window_size, 1)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Create a Streamlit web app
st.title("Coffee Price Prediction")

# Input for user to enter the last 'window_size' prices for prediction
user_input = st.text_input("Enter the last {} prices separated by commas".format(window_size))

# Convert user input to a sequence for prediction
if user_input:
    user_input = [float(x.strip()) for x in user_input.split(",")]
    if len(user_input) == window_size:
        user_input = np.array(user_input).reshape(1, window_size, 1)
        scaled_input = scaler.transform(user_input.reshape(-1, 1))
        predicted_price = model.predict(scaled_input)
        predicted_price = scaler.inverse_transform(predicted_price)
        st.write("Predicted Price:", predicted_price[0, 0])
    else:
        st.write("Please enter {} prices.".format(window_size))

# Plot historical prices
st.subheader("Historical Coffee Prices")
plt.figure(figsize=(12, 6))
plt.plot(data['Ngày'], scaler.inverse_transform(prices), label='Actual Prices')
plt.xlabel("Ngày")
plt.ylabel("Giá Cà Phê")
plt.xticks(rotation=45)
plt.legend()
st.pyplot()



# Save the trained model (optional)
# model.save('coffee_price_lstm_model.h5')
