import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, timedelta

# List of stock tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE"]

# Define the timeframes
timeframes = {
    "Today": 1,
    "5 Days": 5,
    "Monthly": 30,
    "3 Months": 90
}

# Function to plot stock data
def plot_stock_data(ticker, days):
    stock = yf.Ticker(ticker)
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    df = stock.history(start=start_date, end=end_date)
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'])
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    return plt

# Streamlit App
st.title('Stock Data Visualization')

# Create a DataFrame to hold the data
data = []

for ticker in tickers:
    row = [ticker]
    for timeframe, days in timeframes.items():
        fig = plot_stock_data(ticker, days)
        row.append(fig)
    data.append(row)

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=["Ticker", "Today", "5 Days", "Monthly", "3 Months"])

# Display the DataFrame
st.write("Stock Data")
for i, row in df.iterrows():
    st.write(f"## {row['Ticker']}")
    cols = st.columns(len(timeframes) + 1)
    cols[0].write("Ticker")
    cols[0].write(row['Ticker'])
    for j, timeframe in enumerate(timeframes.keys()):
        cols[j+1].write(timeframe)
        cols[j+1].pyplot(row[timeframe].gcf())
    
    with st.expander("More Information about " + row['Ticker']):
        stock = yf.Ticker(row['Ticker'])
        info = stock.info
        st.write(f"**Name:** {info['shortName']}")
        st.write(f"**Industry:** {info['industry']}")
        st.write(f"**Market Cap:** ${info['marketCap']:,}")
        st.write(f"**PE Ratio:** {info['forwardPE']}")
        st.write(f"**Dividend Yield:** {info['dividendYield']}%")
