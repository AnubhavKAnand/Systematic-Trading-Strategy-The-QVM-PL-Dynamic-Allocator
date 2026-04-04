import pandas as pd
from src.alpha_engine import generate_alpha_signals
from src.backtester import run_systematic_backtest, calculate_performance_metrics
from src.visualize import generate_tear_sheet

if __name__ == "__main__":
    print("Loading NIFTY 500 dataset...")
    raw_data = pd.read_csv('data/nifty500_daily.csv', parse_dates=['Date'])
    
    # 1. Generate Alpha Scores
    print("Running QVM Alpha Engine...")
    alpha_signals = generate_alpha_signals(raw_data)
    
    # Create a pivot table of daily prices for the backtester
    daily_prices = raw_data.pivot(index='Date', columns='Ticker', values='Close')
    
    # 2. Run the Walk-Forward Backtest (incorporates the ML Allocator inside)
    equity_curve = run_systematic_backtest(
        daily_prices=daily_prices, 
        alpha_signals=alpha_signals, 
        initial_capital=1000000.0, 
        t_cost_rate=0.0015
    )
    
    # 3. Output Results
    calculate_performance_metrics(equity_curve)
    generate_tear_sheet(equity_curve)