### How to Run the Strategy
1. Install the required dependencies: `pip install -r requirements.txt`
2. Ensure the historical dataset (`nifty500_daily.csv`) is located in the `data/` directory.
    (downloaded from yfinance library)
3. Execute the main pipeline: `python main.py`

*The script will automatically run the QVM alpha engine, optimize the portfolio weights using the PyTorch allocator, and output the final institutional metrics (CAGR, Sharpe Ratio, Max Drawdown) to the terminal.*