import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the new 'ta' library classes
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

def fetch_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Check if data is empty
    if df.empty:
        print("No data found. Check ticker symbol.")
        return None
    
    # Flatten columns if yfinance returns multi-level index (common issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- Feature Engineering (The "Major Project" Part) ---
    
    # 1. RSI (Relative Strength Index)
    # Measures if a stock is overbought (expensive) or oversold (cheap)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # 2. MACD (Moving Average Convergence Divergence)
    # Shows trend direction (up or down)
    macd_indicator = MACD(close=df['Close'])
    df['MACD'] = macd_indicator.macd()
    
    # 3. EMA (Exponential Moving Average)
    # Smooths out the price noise to show the true trend
    ema_indicator = EMAIndicator(close=df['Close'], window=20)
    df['EMA_20'] = ema_indicator.ema_indicator()

    # Drop NaN values created by indicators (first 20-30 rows will be empty)
    df.dropna(inplace=True)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

# Test it
if __name__ == "__main__":
    ticker_symbol = "RELIANCE.NS" 
    data = fetch_data(ticker_symbol, "2020-01-01", "2024-01-01")

    if data is not None:
        print(data.head())
        # Save to CSV for Phase 2
        data.to_csv("stock_data.csv")
        print("Data saved to 'stock_data.csv'")
        
        # Visualize to confirm
        plt.figure(figsize=(10,5))
        plt.plot(data['Close'], label='Close Price')
        plt.plot(data['EMA_20'], label='EMA 20', alpha=0.7)
        plt.title(f'{ticker_symbol} Price with Technical Indicators')
        plt.legend()
        plt.show()