import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_excel('Dataset/coffee.xlsx', parse_dates=['Ngày'])
df = df.set_index('Ngày')

# Normalize the data
scaler = MinMaxScaler()
df['Giá Cà Phê'] = scaler.fit_transform(df['Giá Cà Phê'].values.reshape(-1, 1))

# Create sequences with a lookback of 60 days
lookback = 60

data = []
target = []
for i in range(len(df) - lookback):
    data.append(df.iloc[i:i+lookback].values)
    target.append(df.iloc[i+lookback].values)

data = np.array(data)
target = np.array(target)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(data))

x_train, y_train = data[:split_index], target[:split_index]
x_test, y_test = data[split_index:], target[split_index:]

# Build an LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(lookback, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Save the trained model as an HDF5 (.h5) file
# model.save("coffee_price_prediction_model.h5")

# Make predictions for the 30th day
predicted_price = model.predict(np.array([x_test[-1]]))
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

# Calculate the date for the 30th day
last_date = df.index[-1]
date_30th_day = last_date + pd.Timedelta(days=30)

# Create a Streamlit web application
st.title("Coffee Price Prediction")

st.write("Predicted Coffee Price for the 30th Day:")
st.write(predicted_price, "VND")

# Plot the actual vs. predicted prices with the 30th day date on the x-axis
st.write("Actual vs. Predicted Prices")
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(x_test):], scaler.inverse_transform(y_test), label='Actual Prices')
plt.plot(date_30th_day, predicted_price, 'ro', label=f'Predicted Price ({date_30th_day.strftime("%Y-%m-%d")})')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)
