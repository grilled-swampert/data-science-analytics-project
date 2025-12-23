# Financial Risk Framework

A comprehensive machine learning framework for financial volatility analysis and prediction using signal decomposition, deep learning, and reinforcement learning techniques.

## Overview

This project implements an advanced volatility prediction system that combines multiple state-of-the-art methodologies:

- **Signal Decomposition**: CEEMDAN and VMD for multi-scale time series analysis
- **Deep Learning**: GRU and LSTM networks for temporal pattern recognition
- **Statistical Models**: GARCH framework for volatility modeling
- **Ensemble Learning**: Q-Learning based adaptive weight optimization
- **Feature Engineering**: Technical indicators and entropy-based component analysis

## Features

### Data Acquisition & Processing
- Automated download of financial time series data via `yfinance`
- Support for major market indices:
  - S&P 500 (^GSPC)
  - CBOE Volatility Index (^VIX)
  - NASDAQ Composite (^IXIC)
  - Dow Jones Industrial Average (^DJI)
- Configurable lookback periods and data intervals
- Robust CSV/Excel serialization for reproducibility

### Technical Indicators
- **Daily Returns**: Percentage change in closing prices
- **Price Range**: Intraday volatility measures
- **Moving Averages**: 20-day and 50-day SMAs
- **Realized Volatility**: Rolling standard deviation metrics
- **Cumulative Returns**: Long-term growth tracking

### Signal Decomposition Methods

#### CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)
- Extracts Intrinsic Mode Functions (IMFs) at distinct frequency scales
- Separates high-frequency noise, medium-frequency cycles, and low-frequency trends
- Adaptive decomposition without pre-specified mode counts

#### VMD (Variational Mode Decomposition)
- Formulates decomposition as a constrained variational problem
- Superior frequency separation with minimal mode mixing
- ADMM optimization for robust component extraction

### Deep Learning Architecture

#### GRU (Gated Recurrent Units)
- Efficient handling of long-term dependencies
- Update and reset gates for selective memory retention
- Reduced parameter count compared to LSTMs

#### LSTM (Long Short-Term Memory)
- Explicit memory cells with three-gate control
- Captures complex temporal dependencies across extended sequences
- Ideal for multi-cycle volatility patterns

### Volatility Models

#### ARCH/GARCH Framework
- Time-varying volatility modeling
- Captures volatility clustering phenomena
- Parsimonious GARCH(1,1) specification

#### Hybrid GARCH-Deep Learning
- Combines statistical foundations with neural network flexibility
- GARCH predictions as additional input features
- Learns nonlinear patterns beyond parametric constraints

### Ensemble Optimization

#### Q-Learning Based Weighting
- Adaptive forecast combination through reinforcement learning
- Temporal difference learning for weight optimization
- Epsilon-greedy exploration strategy
- Multi-metric reward function (MAPE, RMSE, MAE)

## Methodology Pipeline

1. **Data Extraction**: Historical market data via `StockDataDownloader`
2. **Feature Engineering**: Technical indicators and transformations
3. **Signal Decomposition**: CEEMDAN/VMD for multi-scale extraction
4. **Entropy Analysis**: Sample entropy for component characterization
5. **K-Means Clustering**: Grouping IMFs into frequency bands
6. **Co-IMF Integration**: Composite signal formation
7. **Model Training**: Specialized GRU/LSTM for each Co-IMF
8. **GARCH Augmentation**: Statistical feature enrichment
9. **Ensemble Weighting**: Q-learning optimization
10. **Performance Evaluation**: Multi-metric assessment

## Performance Metrics

- **R² (Coefficient of Determination)**: Explained variance
- **RMSE (Root Mean Squared Error)**: Forecast error magnitude
- **MAE (Mean Absolute Error)**: Average absolute deviation
- **MAPE (Mean Absolute Percentage Error)**: Scale-independent errors
- **NRMSE (Normalized RMSE)**: Reconstruction quality (<0.1%)

## Key Equations

### Log Returns
```
rt = ln(Pt / Pt-1)
```

### Realized Volatility
```
RVt(w) = sqrt(Σ ri²)
```

### GARCH(1,1)
```
σt² = ω + α·εt-1² + β·σt-1²
```

### Q-Learning Update
```
Q(st, at) ← Q(st, at) + α[rt + γ·max Q(st+1, a') - Q(st, at)]
```

## Technical Details

### Data Normalization
- MinMaxScaler for bounded range transformation [0,1] or [-1,1]
- Prevents information leakage via training-exclusive fitting
- Improves gradient flow and numerical stability

### Sliding Window Construction
- Transforms time series into supervised learning format
- Configurable lookback windows (30-60 days typical)
- Multi-step ahead prediction capability

### Training Optimization
- **Early Stopping**: Monitors validation metrics to prevent overfitting
- **Learning Rate Reduction**: Adaptive scheduling on plateaus
- **Patience Counters**: Distinguishes plateaus from convergence

## Use Cases

- Risk management and portfolio optimization
- Volatility forecasting for derivatives pricing
- Market regime detection
- Trading strategy development
- Financial stress testing

## Requirements

- Python 3.8+
- TensorFlow/Keras
- yfinance
- numpy, pandas
- scikit-learn
- PyEMD (for CEEMDAN)
- vmdpy (for VMD)

## Installation

```bash
git clone https://github.com/grilled-swampert/financial-risk-framework.git
cd financial-risk-framework
pip install -r requirements.txt
```

## Usage

```python
# Download data
from download_data import StockDataDownloader
downloader = StockDataDownloader(tickers=['^GSPC', '^VIX'])
data = downloader.download(period='5y', interval='1d')

# Run EDA
from eda import load_and_prepare_data
df = load_and_prepare_data('data.csv')

# Train models and generate predictions
# (See documentation for detailed examples)
```

## Research Foundation

This framework implements methodologies from contemporary research in:
- Time series decomposition
- Financial econometrics
- Deep learning for sequential data
- Reinforcement learning for ensemble methods

## Acknowledgments

Built with contributions from academic research in financial time series analysis and modern machine learning architectures.
