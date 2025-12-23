import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(filepath):
    """Load CSV and prepare data for analysis"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Calculate additional features
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Price_Range'] = df['High'] - df['Low']
    df['Moving_Avg_20'] = df['Close'].rolling(window=20).mean()
    df['Moving_Avg_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    df['Cumulative_Return'] = (1 + df['Close'].pct_change()).cumprod() - 1

    return df


def print_summary_statistics(df):
    """Print comprehensive summary statistics"""
    print("=" * 80)
    print("STOCK DATA - SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nData Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total Trading Days: {len(df)}")

    print("\n--- PRICE STATISTICS ---")
    print(
        f"Open:  Mean=${df['Open'].mean():.2f}, Std=${df['Open'].std():.2f}, Min=${df['Open'].min():.2f}, Max=${df['Open'].max():.2f}")
    print(
        f"High:  Mean=${df['High'].mean():.2f}, Std=${df['High'].std():.2f}, Min=${df['High'].min():.2f}, Max=${df['High'].max():.2f}")
    print(
        f"Low:   Mean=${df['Low'].mean():.2f}, Std=${df['Low'].std():.2f}, Min=${df['Low'].min():.2f}, Max=${df['Low'].max():.2f}")
    print(
        f"Close: Mean=${df['Close'].mean():.2f}, Std=${df['Close'].std():.2f}, Min=${df['Close'].min():.2f}, Max=${df['Close'].max():.2f}")

    print("\n--- RETURN STATISTICS ---")
    print(f"Average Daily Return: {df['Daily_Return'].mean():.4f}%")
    print(f"Daily Return Std Dev (Volatility): {df['Daily_Return'].std():.4f}%")
    print(f"Best Day: {df['Daily_Return'].max():.2f}%")
    print(f"Worst Day: {df['Daily_Return'].min():.2f}%")
    print(f"Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")

    print("\n--- VOLUME STATISTICS ---")
    print(f"Average Volume: {df['Volume'].mean():,.0f}")
    print(f"Max Volume: {df['Volume'].max():,.0f}")
    print(f"Min Volume: {df['Volume'].min():,.0f}")

    print("\n--- PRICE RANGE STATISTICS ---")
    print(f"Average Daily Range: ${df['Price_Range'].mean():.2f}")
    print(f"Max Daily Range: ${df['Price_Range'].max():.2f}")
    print(f"Min Daily Range: ${df['Price_Range'].min():.2f}")
    print("=" * 80)


def create_visualizations(df):
    """Create comprehensive visualizations"""

    # Figure 1: Price Time Series with Moving Averages
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(df.index, df['Close'], label='Close Price', linewidth=1.5, alpha=0.8)
    axes[0].plot(df.index, df['Moving_Avg_20'], label='20-Day MA', linewidth=1.5, linestyle='--')
    axes[0].plot(df.index, df['Moving_Avg_50'], label='50-Day MA', linewidth=1.5, linestyle='--')
    axes[0].set_title('Stock Price with Moving Averages', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(df.index, df['Volume'], alpha=0.6, color='steelblue')
    axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Volume')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_price_and_volume.png', dpi=300, bbox_inches='tight')
    print("Saved: 01_price_and_volume.png")

    # Figure 2: OHLC Candlestick-style visualization
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sample data for visibility (every 10th point if dataset is large)
    plot_df = df[::max(1, len(df) // 200)]

    colors = ['green' if close >= open_ else 'red'
              for close, open_ in zip(plot_df['Close'], plot_df['Open'])]

    ax.bar(plot_df.index, plot_df['High'] - plot_df['Low'],
           width=0.5, bottom=plot_df['Low'], color=colors, alpha=0.3)
    ax.bar(plot_df.index, abs(plot_df['Close'] - plot_df['Open']),
           width=2, bottom=plot_df[['Open', 'Close']].min(axis=1),
           color=colors, alpha=0.8)

    ax.set_title('OHLC Price Visualization', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('02_ohlc_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: 02_ohlc_chart.png")

    # Figure 3: Daily Returns Distribution and Time Series
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df.index, df['Daily_Return'], linewidth=0.8, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Daily Returns Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Daily Return (%)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df['Daily_Return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['Daily_Return'].mean(), color='r', linestyle='--',
                       linewidth=2, label=f'Mean: {df["Daily_Return"].mean():.3f}%')
    axes[0, 1].set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Daily Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df.index, df['Cumulative_Return'] * 100, linewidth=1.5)
    axes[1, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Cumulative Return (%)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df.index, df['Volatility_20'], linewidth=1.5, color='orange')
    axes[1, 1].set_title('20-Day Rolling Volatility', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Volatility (%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_returns_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: 03_returns_analysis.png")

    # Figure 4: Price Range and Correlation Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df.index, df['Price_Range'], linewidth=1, color='purple')
    axes[0, 0].set_title('Daily Price Range (High - Low)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price Range ($)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(df['Volume'], df['Price_Range'], alpha=0.3, s=10)
    axes[0, 1].set_title('Volume vs Price Range', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Volume')
    axes[0, 1].set_ylabel('Price Range ($)')
    axes[0, 1].grid(True, alpha=0.3)

    # Correlation matrix
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Price_Range']
    corr_matrix = df[corr_cols].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=axes[1, 0], square=True, cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Correlation Matrix', fontsize=12, fontweight='bold')

    # Box plots for OHLC
    box_data = [df['Open'], df['High'], df['Low'], df['Close']]
    axes[1, 1].boxplot(box_data, labels=['Open', 'High', 'Low', 'Close'])
    axes[1, 1].set_title('OHLC Price Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('04_range_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved: 04_range_correlation.png")

    # Figure 5: Monthly and Yearly Analysis
    df_monthly = df.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    df_monthly['Monthly_Return'] = df_monthly['Close'].pct_change() * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df_monthly.index, df_monthly['Close'], marker='o', linewidth=2)
    axes[0, 0].set_title('Monthly Closing Prices', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Close Price ($)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(df_monthly.index, df_monthly['Monthly_Return'],
                   color=['green' if x > 0 else 'red' for x in df_monthly['Monthly_Return']])
    axes[0, 1].set_title('Monthly Returns', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # Yearly returns
    df_yearly = df.resample('Y').agg({
        'Open': 'first',
        'Close': 'last'
    })
    df_yearly['Yearly_Return'] = ((df_yearly['Close'] / df_yearly['Open']) - 1) * 100

    axes[1, 0].bar(df_yearly.index.year, df_yearly['Yearly_Return'],
                   color=['green' if x > 0 else 'red' for x in df_yearly['Yearly_Return']])
    axes[1, 0].set_title('Yearly Returns', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Return (%)')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # Day of week analysis
    df_dow = df.copy()
    df_dow['DayOfWeek'] = df_dow.index.dayofweek
    dow_returns = df_dow.groupby('DayOfWeek')['Daily_Return'].mean()
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    axes[1, 1].bar(range(len(dow_returns)), dow_returns.values,
                   tick_label=[dow_labels[i] for i in dow_returns.index])
    axes[1, 1].set_title('Average Returns by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Average Return (%)')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_temporal_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: 05_temporal_analysis.png")

    plt.show()


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("STOCK MARKET EXPLORATORY DATA ANALYSIS")
    print("=" * 80 + "\n")

    # Get file path from user
    filepath = input("Enter the path to your CSV file: ").strip()

    try:
        # Load and prepare data
        print("\nLoading and preparing data...")
        df = load_and_prepare_data(filepath)

        # Print summary statistics
        print_summary_statistics(df)

        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(df)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("All visualizations have been saved as PNG files in the current directory.")
        print("=" * 80 + "\n")

    except FileNotFoundError:
        print(f"\nError: File '{filepath}' not found. Please check the path and try again.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")


if __name__ == "__main__":
    main()