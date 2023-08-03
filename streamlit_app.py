import numpy as np
import pandas as pd
import streamlit as st
import time as t
import pathlib
import os.path
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from datetime import time, datetime, timedelta

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    def is_leap_year(year: int) -> bool:
        """_summary_

        Args:
            year (_int_): The year.

        Returns:
            bool: True if the year is leap year, False otherwise.
        """
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    def get_number_of_days_in_month(date_obj: datetime) -> int:
        """_summary_

        Args:
            year (_int_): The year.
            month (_int_): The month.

        Returns:
            int: The number of days in the month.
        """
        year = date_obj.year
        month = date_obj.month
        
        if month == 2:
            return 29 if is_leap_year(year) else 28
        elif month in [4, 6, 9, 11]:
            return 30
        else:
            return 31
    
    if uploaded_file:
        # Date options for the user to select from
        option = st.selectbox(
            'Select Date Range',
            ('Day', 'Week', 'Month')
            )
        
        if option == 'Day':
            frequency = 'D'
        elif option == 'Week':
            frequency = 'W-MON'
        else:
            frequency = 'MS'
        
        # EDA and preprocessing for time series data
        df = pd.read_csv(uploaded_file, index_col="Ngày", parse_dates=True).asfreq(frequency)
        df['Tên_mặt_hàng'].fillna('Cà phê Robusta nhân xô', inplace=True)
        df['Đơn_vị_tính'].fillna('VNĐ/kg', inplace=True)
        df['Loại_giá'].fillna('Thu mua', inplace=True)
        df['Giá'].fillna(df['Giá'].resample(frequency).mean().ffill(), inplace=True)
        df['Loại_tiền'].fillna('VNĐ', inplace=True)
        df['Đơn_vị_tính'] = 'VNĐ/kg'
        
        if frequency == 'D':
            day = 1
        elif frequency == 'W-MON':
            day = 7
        else:
            day = get_number_of_days_in_month(df.index[0])
    
        def separate_year_month_day(date_obj: datetime) -> datetime:
            """_summary_

            Args:
                date_obj (_date object_): date object.

            Returns:
                datetime: year, month, day.
            """
            # Extract year, month, day from the datetime object
            year = date_obj.year
            month = date_obj.month
            day = date_obj.day
    
            return datetime(year, month, day)
        
        date_range = st.slider(
        "What date range do you want to see?",
        min_value=separate_year_month_day(df.index[0]),
        step=timedelta(days=date_range[0]),
        value=(separate_year_month_day(df.index[0]), separate_year_month_day(df.index[-1]))
        )

        st.write(df.loc[date_range[0]:date_range[1]])
    
if __name__ == "__main__":
    main()


