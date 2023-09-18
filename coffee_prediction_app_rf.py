import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Define the load_data function at the global scope
@st.cache_resource
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    data['Ngày'] = pd.to_datetime(data['Ngày'])
    data = data.set_index('Ngày')  # Set 'Ngày' as the index column
    return data

# Define the prepare_data function at the global scope
def prepare_data(data):
    X = data[['Giá Diesel 1 lít', 'Precipitation (mm)']]
    y = data['Giá Cà Phê']
    return X, y

# Define the train_rf_model function at the global scope
def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model

# Main function for Streamlit app
def main():
    st.title('Coffee Price Prediction App')

    # Create a file uploader input
    st.subheader('Upload an Excel file:')
    uploaded_file = st.file_uploader('Choose a file', type=['xlsx'])

    if uploaded_file is not None:
        # Load data from the uploaded file
        data = load_data(uploaded_file)

        # Prepare data
        X, y = prepare_data(data)

        # Train the Random Forest model
        model = train_rf_model(X, y)

        # Use the last available data point to predict 'Giá Diesel 1 lít' and 'Precipitation (mm)' for the next day
        last_data_point = data.iloc[-1]
        last_diesel_price = last_data_point['Giá Diesel 1 lít']
        last_precipitation = last_data_point['Precipitation (mm)']

        # Calculate the date for the next day
        next_day_date = last_data_point.name + timedelta(days=1)  # Add one day to the last data point's date

        # Create input data for the next day
        input_data = pd.DataFrame({'Giá Diesel 1 lít': [last_diesel_price],
                                   'Precipitation (mm)': [last_precipitation]})

        # Make predictions for 'Giá Diesel 1 lít' and 'Precipitation (mm)' for the next day
        predicted_diesel_price = model.predict(input_data)
        predicted_precipitation = model.predict(input_data)

        # Use the predicted values to predict 'Giá Cà Phê' for the next day
        input_data['Giá Diesel 1 lít'] = predicted_diesel_price
        input_data['Precipitation (mm)'] = predicted_precipitation
        predicted_coffee_price = model.predict(input_data)

        # Display the predicted values with the next day's date
        st.subheader('Predicted Values for the Next Day')
        st.write(f'Date for the Next Day: {next_day_date.strftime("%Y-%m-%d")}')
        st.write(f'Predicted Giá Diesel 1 lít for the next day: {predicted_diesel_price[0]:.2f}')
        st.write(f'Predicted Precipitation (mm) for the next day: {predicted_precipitation[0]:.2f}')
        st.write(f'Predicted Giá Cà Phê for the next day: {predicted_coffee_price[0]:.2f} VND')

        # Create a time series plot for predicted coffee prices with improved date formatting
        st.subheader('Predicted Coffee Price Over Time')
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Giá Cà Phê'], label='Actual Giá Cà Phê', linestyle='-')
        ax.plot(next_day_date, predicted_coffee_price[0], label='Predicted Giá Cà Phê', marker='x', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Giá Cà Phê (VND)')
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Format the date as desired
        plt.xticks(rotation=45)  # Rotate the date labels for readability
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
