import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from datetime import time, datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import altair as alt
pd.options.display.float_format = '{:.2f}'.format

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

def read_data(uploaded_file: type, frequency: str, day: int) -> pd.DataFrame:
    """_summary_
    Args:
        uploaded_file (_csv file_): csv file.
        frequency (_string_): frequency of time series data.
        day (_int_): number of days.

        Returns:
            dataframe: time series data.
    """
    df = pd.read_csv(uploaded_file, index_col="Ngày", parse_dates=True).asfreq(frequency)
    df['Giá'].fillna(df['Giá'].resample(frequency).mean().ffill(), inplace=True)
    column_to_keep = ["Giá"]
    df = df[column_to_keep]
    return df  
    
def plotTimeSeries(y, lags=None, figsize=(12, 7), style='bmh'):
    def to_series(y):
        try:
            # Try to treat y as pandas.Series
            y = pd.Series(y)
        except:
            # If it's already a Series or not convertable, don't do anything
            pass
        return y
    
    y = to_series(y)        
        
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        plt.tight_layout()
        st.pyplot()
        

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        # Date options for the user to select from
        
        with st.sidebar:
            option = st.selectbox(
                label="Select the frequency of the time series:",
                options=('Day', 'Week'),
                )
            
            if option == 'Day':
                frequency = 'D'
                day = 1
            elif option == 'Week':
                frequency = 'W-MON'
                day = 7
        
        try:
            df = read_data(uploaded_file, frequency, day)
            with st.sidebar: 
                date_range = st.slider(
                "What date range do you want to see?",
                min_value=separate_year_month_day(df.index[0]),
                step=timedelta(days=day),
                value=(separate_year_month_day(df.index[0]), separate_year_month_day(df.index[-1]))
                )
            plotTimeSeries(df.loc[date_range[0]:date_range[1]], lags=10)
            
            
        except Exception as e:
            st.error(f"Error: {e}")
            
        
            
        
    
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()


