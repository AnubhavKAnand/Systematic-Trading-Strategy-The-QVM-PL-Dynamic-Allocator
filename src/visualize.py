import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_tear_sheet(portfolio_df: pd.DataFrame, save_path: str = "tearsheet.png"):
    """
    Generates an institutional 3-panel tear sheet:
    1. Equity Curve vs Benchmark
    2. Underwater Plot (Drawdowns)
    3. Dynamic Exposure Map
    """
    # Set professional style
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('NIFTY 500 QVM Strategy with PL-Stable ML Allocator & Vol Targeting', fontsize=16, fontweight='bold')

    df = portfolio_df.copy()
    
    # Normalize starting values to 100 for easy comparison
    df['Strategy_Norm'] = (df['Portfolio_Value'] / df['Portfolio_Value'].iloc[0]) * 100
    df['Benchmark_Norm'] = (df['Benchmark_Price'] / df['Benchmark_Price'].iloc[0]) * 100

    # --- Panel 1: Equity Curve ---
    ax1 = axes[0]
    ax1.plot(df.index, df['Strategy_Norm'], label='Systematic QVM Strategy', color='#00ffcc', linewidth=2)
    ax1.plot(df.index, df['Benchmark_Norm'], label='NIFTY 500 Benchmark', color='#aaaaaa', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Cumulative Return (Base 100)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('Cumulative Performance', loc='left')

    # --- Panel 2: Underwater Plot (Drawdown) ---
    ax2 = axes[1]
    df['Strat_Peak'] = df['Strategy_Norm'].cummax()
    df['Strat_DD'] = (df['Strategy_Norm'] - df['Strat_Peak']) / df['Strat_Peak'] * 100
    
    df['Bench_Peak'] = df['Benchmark_Norm'].cummax()
    df['Bench_DD'] = (df['Benchmark_Norm'] - df['Bench_Peak']) / df['Bench_Peak'] * 100

    ax2.fill_between(df.index, df['Strat_DD'], 0, color='#ff3366', alpha=0.5, label='Strategy Drawdown')
    ax2.plot(df.index, df['Bench_DD'], color='#aaaaaa', linewidth=1, alpha=0.7, label='Benchmark Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('Drawdown Profile', loc='left')

    # --- Panel 3: Exposure Map ---
    ax3 = axes[2]
    # We plot the Equity Exposure vs Cash (1 - Exposure)
    ax3.fill_between(df.index, df['Equity_Exposure'] * 100, 0, color='#3399ff', alpha=0.6, label='Equity (Top 30 NIFTY 500)')
    ax3.fill_between(df.index, 100, df['Equity_Exposure'] * 100, color='#555555', alpha=0.4, label='Risk-Free Cash')
    ax3.set_ylabel('Capital Allocation (%)')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_title('Dynamic Volatility Targeting Exposure', loc='left')

    # Format x-axis dates nicely
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Tear sheet saved successfully to {save_path}")
    plt.show()