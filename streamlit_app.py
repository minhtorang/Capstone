import numpy as np
import pandas as pd
import streamlit as st
import time as t
import pathlib
import os.path
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from datetime import time, datetime

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    with st.spinner('Loading data...'):
        t.sleep(2)
    st.success('Success!')
    # EDA and preprocessing for time series data
    df = pd.read_csv(uploaded_file, index_col="Ngày", parse_dates=True).asfreq('D')
    df = pd.read_csv('./coffee_data.csv', index_col="Ngày", parse_dates=True).asfreq('D')
    df['Tên_mặt_hàng'].fillna('Cà phê Robusta nhân xô', inplace=True)
    df['Đơn_vị_tính'].fillna('VNĐ/kg', inplace=True)
    df['Loại_giá'].fillna('Thu mua', inplace=True)
    df['Giá'].fillna(df['Giá'].resample('1D').mean().ffill(), inplace=True)
    df['Loại_tiền'].fillna('VNĐ', inplace=True)
    df['Đơn_vị_tính'] = 'VNĐ/kg'
    
    def separate_year_month_day(date_obj):
        # Extract year, month, day from the datetime object
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
    
        return datetime(year, month, day)

    date_range = st.slider(
    "What date range do you want to see?",
    min_value=separate_year_month_day(df.index[0]),
    value=(separate_year_month_day(df.index[0]), separate_year_month_day(df.index[-1]))
    )

    st.write(df.loc[date_range[0]:date_range[1]])


