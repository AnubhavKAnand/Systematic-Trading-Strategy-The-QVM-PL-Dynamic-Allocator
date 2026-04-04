import pandas as pd
import numpy as np
from src.ml_allocater import optimize_portfolio_weights
from sklearn.covariance import LedoitWolf

def run_systematic_backtest(
    daily_prices: pd.DataFrame, 
    alpha_signals: pd.DataFrame, 
    initial_capital: float = 1000000.0, 
    t_cost_rate: float = 0.0015,
    risk_free_rate: float = 0.07 # 7% Indian cash rate
):
    """
    Simulates the portfolio performance over time with transaction costs.
    daily_prices expects a pivot table of Close prices: Index=Date, Columns=Tickers
    """
    # Extract end-of-month rebalancing dates
    # Assuming the index is a datetime object
    eom_dates = daily_prices.resample('M').last().index
    
    portfolio_value = initial_capital
    current_weights = pd.Series(dtype=float)
    
    # To store daily portfolio values for performance metrics
    portfolio_history = []
    
    # --- NEW: Regime Filter Engineering ---
    # Create an equal-weight proxy of the broader market
    market_proxy = daily_prices.mean(axis=1)
    # Calculate the 200-day Moving Average of the market
    market_ma_200 = market_proxy.rolling(window=200).mean()

    print("Starting Walk-Forward Backtest...")
    
    for i in range(len(eom_dates) - 1):
        rebalance_date = eom_dates[i]
        next_rebalance = eom_dates[i+1]
        
        # 1. Isolate signals for the current rebalance date
        current_signals = alpha_signals[alpha_signals['Date'] == rebalance_date]
        
        if current_signals.empty:
            continue
            
        # Select Top 30 stocks based on Alpha Score
        top_30 = current_signals.nlargest(30, 'Alpha_Score')
        tickers = top_30['Ticker'].tolist()
        alphas = top_30['Alpha_Score'].values
        
        # 2. Calculate 60-day Covariance Matrix for the ML Allocator
        # We look backward from the rebalance date to avoid look-ahead bias
        historical_prices = daily_prices.loc[:rebalance_date, tickers].tail(60)
        returns_matrix = historical_prices.pct_change().dropna()
        #cov_matrix = returns_matrix.cov().values * 252 # Annualized
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_matrix).covariance_ * 252
        # 3. Optimize Weights using your PyTorch PL-Stable Allocator (Code 2)
        # optimal_weights = optimize_portfolio_weights(alphas, cov_matrix)
        # For simulation purposes here, let's assume it returns a numpy array:
        optimal_weights = optimize_portfolio_weights(alphas, cov_matrix, epochs=100)
        
        target_weights = pd.Series(optimal_weights, index=tickers)
        # --- NEW: Volatility Targeting (Target Vol) ---
        target_volatility = 0.15  # Target 15% annualized portfolio volatility
        
        # Calculate trailing 20-day covariance for immediate realized risk
        recent_returns = daily_prices.loc[:rebalance_date, tickers].tail(20).pct_change().dropna()
        if not recent_returns.empty:
            recent_cov = recent_returns.cov().values * 252
            
            # Calculate expected variance of the new portfolio weights
            port_variance = np.dot(target_weights.values, np.dot(recent_cov, target_weights.values))
            realized_vol = np.sqrt(port_variance)
            
            # Scale exposure inversely to realized volatility
            # Cap at 1.0 so we do not use leverage (fixed capital constraint)
            if realized_vol > 0:
                exposure_scalar = min(target_volatility / realized_vol, 1.0)
            else:
                exposure_scalar = 1.0
        else:
            exposure_scalar = 1.0
            
        target_weights = target_weights * exposure_scalar
        cash_weight = 1.0 - target_weights.sum()
        # 4. Calculate Turnover and Transaction Costs
        # Align old weights and new weights to find the absolute change
        weight_df = pd.DataFrame({'old': current_weights, 'new': target_weights}).fillna(0)
        turnover = (weight_df['new'] - weight_df['old']).abs().sum() / 2.0
        
        cost_in_rupees = turnover * portfolio_value * t_cost_rate
        portfolio_value -= cost_in_rupees # Deduct costs from capital
        
        # 5. Step Forward: Calculate returns until the next rebalance
        # Get the daily returns of the selected stocks for the next month
        forward_returns = daily_prices.loc[rebalance_date:next_rebalance, tickers].pct_change().dropna()
        
        # Multiply daily stock returns by our fixed target weights to get daily portfolio return
        daily_portfolio_returns = forward_returns.dot(target_weights)
        
        # Add the daily risk-free rate return for the cash portion
        daily_cash_return = (risk_free_rate / 252) * cash_weight
        daily_total_return = daily_portfolio_returns + daily_cash_return
        
        # Compound the portfolio value daily
        for date, ret in daily_portfolio_returns.items():
            portfolio_value *= (1 + ret)
            # Get the benchmark (NIFTY 500 equal-weight proxy) daily return
            bmark_ret = daily_prices.loc[date].mean() if date in daily_prices.index else 0
            
            portfolio_history.append({
                'Date': date, 
                'Portfolio_Value': portfolio_value,
                'Strategy_Return': ret,
                'Equity_Exposure': exposure_scalar, # The Vol Target scalar we added
                'Benchmark_Price': market_proxy.loc[date] # The proxy we created
            })
            
        # Update current weights for the next loop's turnover calculation
        current_weights = target_weights

    print("Backtest Complete.")
    return pd.DataFrame(portfolio_history).set_index('Date')

def calculate_performance_metrics(equity_curve: pd.DataFrame, risk_free_rate: float = 0.07):
    """
    Calculates key institutional risk/return metrics.
    Assumes 7% Indian risk-free rate.
    """
    df = equity_curve.copy()
    df['Daily_Return'] = df['Portfolio_Value'].pct_change()
    
    # 1. CAGR (Compound Annual Growth Rate)
    total_return = df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (total_return ** (1 / years)) - 1
    
    # 2. Sharpe Ratio
    excess_returns = df['Daily_Return'] - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / df['Daily_Return'].std())
    
    # 3. Maximum Drawdown
    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] - df['Peak']) / df['Peak']
    max_drawdown = df['Drawdown'].min()
    
    print("--- Strategy Performance ---")
    print(f"CAGR:           {cagr * 100:.2f}%")
    print(f"Sharpe Ratio:   {sharpe_ratio:.2f}")
    print(f"Max Drawdown:   {max_drawdown * 100:.2f}%")
    
    return df