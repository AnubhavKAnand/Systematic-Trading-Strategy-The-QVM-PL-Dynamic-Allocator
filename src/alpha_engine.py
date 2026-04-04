import pandas as pd
import numpy as np

# Suppress SettingWithCopyWarning for cleaner output during fast iterations
pd.options.mode.chained_assignment = None

def filter_universe(df: pd.DataFrame, min_price: float = 50.0, adv_window: int = 30) -> pd.DataFrame:
    """
    Filters the NIFTY 500 universe for liquidity and minimum price constraints.
    Expects df with columns: ['Date', 'Ticker', 'Close', 'Volume']
    """
    # Sort values to ensure rolling calculations are correct
    df = df.sort_values(by=['Ticker', 'Date'])
    
    # Calculate 30-day Average Daily Volume (ADV) in INR
    df['Traded_Value'] = df['Close'] * df['Volume']
    df['ADV_30'] = df.groupby('Ticker')['Traded_Value'].transform(
        lambda x: x.rolling(window=adv_window, min_periods=15).mean()
    )
    
    # Cross-sectional rank of ADV for each date
    df['ADV_Rank'] = df.groupby('Date')['ADV_30'].rank(pct=True)
    
    # Filter logic: Keep top 90% by volume, and price >= min_price
    # We shift the filter by 1 day to avoid look-ahead bias in trading
    mask = (df['ADV_Rank'].groupby(df['Ticker']).shift(1) > 0.10) & \
           (df['Close'].groupby(df['Ticker']).shift(1) >= min_price)
           
    return df[mask].copy()

def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes raw Quality, Value, and Momentum factors.
    Expects fundamental columns: ['EPS', 'PE_Ratio', 'ROIC', 'Debt_to_Equity']
    """
    df = df.sort_values(by=['Ticker', 'Date'])
    
    # --- MOMENTUM (12M - 1M) ---
    # 252 days = 1 Year, 21 days = 1 Month
    df['Ret_12M'] = df.groupby('Ticker')['Close'].pct_change(252)
    df['Ret_1M'] = df.groupby('Ticker')['Close'].pct_change(21)
    
    # Traditional cross-sectional momentum (ignoring the most recent month for mean reversion)
    df['Momentum_Raw'] = df['Ret_12M'] - df['Ret_1M']
    
    # --- VALUE ---
    # Earnings Yield = 1 / P/E Ratio (Inverting handles negative/zero PE better)
    df['Value_EY'] = 1.0 / df['PE_Ratio'].replace(0, np.nan)
    
    # --- QUALITY ---
    # High Return on Invested Capital, Low Debt
    df['Quality_ROIC'] = df['ROIC']
    df['Quality_LowDebt'] = -1.0 * df['Debt_to_Equity'] # Negative so higher score is better
    
    return df

def standardize_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-scores the raw factors cross-sectionally and computes the final Alpha score.
    """
    # Helper function for cross-sectional Z-score handling NaNs
    def z_score(series):
        return (series - series.mean()) / series.std()

    # Apply Z-scoring grouped by Date
    factors = ['Momentum_Raw', 'Value_EY', 'Quality_ROIC', 'Quality_LowDebt']
    
    for factor in factors:
        df[f'{factor}_Z'] = df.groupby('Date')[factor].transform(z_score)
        
        # Handle outliers by winsorizing at +/- 3 standard deviations
        df[f'{factor}_Z'] = df[f'{factor}_Z'].clip(lower=-3.0, upper=3.0)

    # Combine internal factor groups
    df['Quality_Z'] = (df['Quality_ROIC_Z'] + df['Quality_LowDebt_Z']) / 2.0
    
    # Final Alpha Composite Score (Weights: 40% Momentum, 30% Value, 30% Quality)
    df['Alpha_Score'] = (0.40 * df['Momentum_Raw_Z']) + \
                        (0.30 * df['Value_EY_Z']) + \
                        (0.30 * df['Quality_Z'])
                        
    return df

def generate_alpha_signals(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to process raw data into tradable alpha scores.
    """
    print("Filtering universe...")
    df_filtered = filter_universe(raw_data)
    
    print("Calculating raw factors...")
    df_factors = calculate_factors(df_filtered)
    
    print("Scoring and ranking...")
    df_scored = standardize_and_score(df_factors)
    
    # Drop rows where Alpha_Score couldn't be calculated (e.g., due to missing data)
    df_scored = df_scored.dropna(subset=['Alpha_Score'])
    
    return df_scored[['Date', 'Ticker', 'Close', 'Alpha_Score']]