import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import norm
import tweepy
from textblob import TextBlob

# Set page config
st.set_page_config(page_title="Advanced Bitcoin Analyzer", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Correlation Analysis", "Technical Indicators", "Monte Carlo Simulation", "Trading Strategy Backtester"])

# Fetch data
@st.cache_data
def get_data(ticker, period="1y"):
    data = yf.Ticker(ticker).history(period=period)
    return data

btc_data = get_data("BTC-USD")
gold_data = get_data("GC=F")
sp500_data = get_data("^GSPC")

# Technical Indicators Functions
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

# Home page
if page == "Home":
    st.title("Advanced Bitcoin Analyzer")
    st.markdown("*Dedicated to Breth - \"I hate bitcoin\"*")

    # Bitcoin price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=btc_data.index,
        open=btc_data['Open'],
        high=btc_data['High'],
        low=btc_data['Low'],
        close=btc_data['Close'],
        name='Bitcoin Price'
    ))
    fig.update_layout(title="Bitcoin Price (Last 1 Year)", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig)

    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(f"Current Price: ${btc_data['Close'].iloc[-1]:.2f}")
    st.write(f"1-Year High: ${btc_data['High'].max():.2f}")
    st.write(f"1-Year Low: ${btc_data['Low'].min():.2f}")
    st.write(f"1-Year Return: {((btc_data['Close'].iloc[-1] / btc_data['Close'].iloc[0]) - 1) * 100:.2f}%")

# Sentiment Analysis
elif page == "Sentiment Analysis":
    st.title("Bitcoin Sentiment Analysis")

    st.write("To implement sentiment analysis, you need to set up Twitter API credentials.")
    st.write("Once set up, this section will analyze recent tweets about Bitcoin and display sentiment scores.")

# Correlation Analysis
elif page == "Correlation Analysis":
    st.title("Correlation Analysis")

    # Combine data
    combined_data = pd.DataFrame({
        'Bitcoin': btc_data['Close'],
        'Gold': gold_data['Close'],
        'S&P 500': sp500_data['Close']
    })

    # Calculate correlation
    correlation = combined_data.corr()

    # Plot correlation heatmap
    fig = px.imshow(correlation, text_auto=True, aspect="auto")
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig)

    st.write("This heatmap shows the correlation between Bitcoin, Gold, and the S&P 500.")

# Technical Indicators
elif page == "Technical Indicators":
    st.title("Technical Indicators")

    # Calculate indicators
    btc_data['RSI'] = calculate_rsi(btc_data['Close'])
    btc_data['MACD'], btc_data['Signal'] = calculate_macd(btc_data['Close'])
    btc_data['BB_upper'], btc_data['BB_middle'], btc_data['BB_lower'] = calculate_bollinger_bands(btc_data['Close'])

    # Plot indicators
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Close'], name='Bitcoin Price'))
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['BB_upper'], name='Upper BB', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['BB_lower'], name='Lower BB', line=dict(dash='dash')))
    fig.update_layout(title="Bitcoin Price with Bollinger Bands")
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['RSI'], name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="Relative Strength Index (RSI)")
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['MACD'], name='MACD'))
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Signal'], name='Signal'))
    fig.update_layout(title="Moving Average Convergence Divergence (MACD)")
    st.plotly_chart(fig)

# Monte Carlo Simulation
elif page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation")

    # Parameters
    days = st.slider("Number of days to simulate", 30, 365, 90)
    simulations = st.slider("Number of simulations", 100, 10000, 1000)

    # Calculate daily returns
    returns = np.log(1 + btc_data['Close'].pct_change())

    # Run simulation
    last_price = btc_data['Close'].iloc[-1]
    simulation_df = pd.DataFrame()

    for i in range(simulations):
        count = 0
        price_series = []
        price = last_price

        for j in range(days):
            price = price * (1 + np.random.normal(returns.mean(), returns.std()))
            price_series.append(price)
            count += 1

        simulation_df[i] = price_series

    # Plot results
    fig = go.Figure()
    for i in range(simulations):
        fig.add_trace(go.Scatter(y=simulation_df[i], mode='lines', line=dict(width=0.5), name=f'Simulation {i}'))
    fig.update_layout(title=f"Monte Carlo Simulation ({simulations} runs, {days} days)")
    st.plotly_chart(fig)

    # Calculate statistics
    st.subheader("Simulation Statistics")
    st.write(f"Mean Final Price: ${simulation_df.iloc[-1].mean():.2f}")
    st.write(f"Median Final Price: ${simulation_df.iloc[-1].median():.2f}")
    st.write(f"95% VaR: ${np.percentile(simulation_df.iloc[-1], 5):.2f}")

# Trading Strategy Backtester
elif page == "Trading Strategy Backtester":
    st.title("Simple Trading Strategy Backtester")

    # Strategy parameters
    st.subheader("Strategy Parameters")
    short_window = st.slider("Short window", 10, 50, 20)
    long_window = st.slider("Long window", 50, 200, 100)

    # Calculate moving averages
    btc_data['Short_MA'] = btc_data['Close'].rolling(window=short_window).mean()
    btc_data['Long_MA'] = btc_data['Close'].rolling(window=long_window).mean()

    # Generate signals
    btc_data['Signal'] = 0
    btc_data['Signal'][short_window:] = np.where(btc_data['Short_MA'][short_window:] > btc_data['Long_MA'][short_window:], 1, 0)
    btc_data['Position'] = btc_data['Signal'].diff()

    # Calculate returns
    btc_data['Returns'] = btc_data['Close'].pct_change()
    btc_data['Strategy_Returns'] = btc_data['Signal'] * btc_data['Returns']

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Close'], name='Bitcoin Price'))
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Short_MA'], name=f'{short_window}-day MA'))
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Long_MA'], name=f'{long_window}-day MA'))
    fig.update_layout(title="Bitcoin Price with Moving Averages")
    st.plotly_chart(fig)

    # Performance metrics
    st.subheader("Strategy Performance")
    cumulative_returns = (1 + btc_data['Strategy_Returns']).cumprod()
    st.write(f"Total Return: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
    st.write(f"Sharpe Ratio: {(btc_data['Strategy_Returns'].mean() / btc_data['Strategy_Returns'].std()) * np.sqrt(252):.2f}")

    # Plot cumulative returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_data.index, y=cumulative_returns, name='Cumulative Returns'))
    fig.update_layout(title="Cumulative Returns of Trading Strategy")
    st.plotly_chart(fig)

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.caption("Disclaimer: This app is for educational purposes only. Cryptocurrency investments are highly speculative and volatile. Always do your own research before making any investment decisions.")
