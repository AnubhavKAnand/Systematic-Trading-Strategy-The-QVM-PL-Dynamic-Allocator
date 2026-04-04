import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_nifty_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical OHLCV data for Indian stocks and formats it 
    into the 'long' panel format required by our Alpha Engine.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # yfinance requires '.NS' for National Stock Exchange of India
    ns_tickers = [f"{ticker}.NS" for ticker in tickers]
    
    # Download adjusted data (crucial for accurate momentum calculations)
    raw_data = yf.download(ns_tickers, start=start_date, end=end_date, auto_adjust=True)
    
    if raw_data.empty:
        raise ValueError("No data downloaded. Check your internet connection or ticker symbols.")

    # yfinance returns a MultiIndex DataFrame if multiple tickers are passed.
    # We need to melt it down to our required Long format.
    
    # Isolate Close and Volume
    closes = raw_data['Close'].stack().reset_index()
    closes.columns = ['Date', 'Ticker', 'Close']
    
    volumes = raw_data['Volume'].stack().reset_index()
    volumes.columns = ['Date', 'Ticker', 'Volume']
    
    # Merge them together
    df = pd.merge(closes, volumes, on=['Date', 'Ticker'])
    
    # Clean up ticker names (remove the .NS suffix for cleaner output)
    df['Ticker'] = df['Ticker'].str.replace('.NS', '')
    
    return df

def add_mock_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hackathon Workaround: Injects proxy fundamental data so the QVM 
    Alpha Engine and PyTorch optimizer can run without NaN errors.
    """
    print("Injecting proxy fundamental data for pipeline testing...")
    np.random.seed(42) # For reproducibility
    
    # PE Ratio: Typically between 10 and 50, with some noise
    df['PE_Ratio'] = np.random.uniform(10, 50, size=len(df))
    
    # ROIC (Return on Invested Capital): Typically 5% to 25%
    df['ROIC'] = np.random.uniform(0.05, 0.25, size=len(df))
    
    # Debt to Equity: Typically 0.1 to 2.0
    df['Debt_to_Equity'] = np.random.uniform(0.1, 2.0, size=len(df))
    
    return df

if __name__ == "__main__":
    # A representative list of highly liquid NIFTY stocks for initial testing.
    # Expand this list to the full NIFTY 500 constituents for your final run.
    test_tickers = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", 
        "ITC", "SBIN", "BHARTIARTL", "LT", "BAJFINANCE",
        "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO"
    ]
    
    # Fetch 3 years of data
    final_df = fetch_nifty_data(
        tickers=test_tickers, 
        start_date="2021-01-01", 
        end_date="2024-01-01"
    )
    
    # Add fundamentals to satisfy the QVM engine
    final_df = add_mock_fundamentals(final_df)
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = "data/nifty500_daily.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Dataset saved to {output_path}")
    print(f"Total Rows: {len(final_df)}")
    print("You can now run 'python main.py' to test your PyTorch Allocator and Backtester.")