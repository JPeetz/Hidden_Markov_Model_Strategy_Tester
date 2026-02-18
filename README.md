# Regime-Based Trading App

**Copyright: Joerg Peetz**

A professional cryptocurrency trading strategy backtester using **Hidden Markov Models (HMM)** for market regime detection combined with an **8-confirmation voting system** for trade entry signals.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **HMM Regime Detection**: Uses a 7-component Gaussian Hidden Markov Model to classify market conditions into Bull, Bear, or Sideways regimes
- **8-Point Confirmation System**: Multiple technical indicators must align before entering trades
- **Multiple Trading Modes**: Conservative, Moderate, Aggressive, and fully Custom configurations
- **Risk Management**: Trailing stops, stop losses, take profits, and cooldown periods
- **Multi-Asset Support**: Trade BTC, ETH, SOL, XRP, ADA, BNB, DOGE, SHIB, or add custom pairs
- **Interactive Dashboard**: Real-time candlestick charts with regime overlays using Plotly
- **Fully Customizable**: Every strategy parameter and confirmation threshold can be adjusted

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Trading Modes](#trading-modes)
- [Strategy Parameters](#strategy-parameters)
- [Confirmation Conditions](#confirmation-conditions)
- [Understanding the Dashboard](#understanding-the-dashboard)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/regime-trading-app.git
cd regime-trading-app
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python3 -m streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## Quick Start

1. **Select a Market**: Choose from the dropdown (BTC-USD, ETH-USD, etc.)
2. **Choose a Trading Mode**: Start with "Conservative" for safer settings
3. **For Full Customization**: Check **"Enable CUSTOM Mode"** in the sidebar to unlock all parameter sliders and confirmation threshold controls
4. **Run the Backtest**: The system will automatically fetch data, train the HMM, and run the strategy
5. **Analyze Results**: Review the performance metrics, trade history, and regime chart

---

## How It Works

### Hidden Markov Model (HMM) Regime Detection

The strategy uses a **Gaussian HMM with 7 hidden states** to detect market regimes. The model is trained on three features:

| Feature | Description |
|---------|-------------|
| **Returns** | Log returns of the closing price |
| **Range** | `(High - Low) / Close` - Normalized price range |
| **Volume Volatility** | Rolling standard deviation of volume changes |

After training, the 7 states are classified into three regimes based on their characteristics:

- **🟢 Bull Regime**: States with positive mean returns and moderate volatility
- **🔴 Bear Regime**: States with negative mean returns
- **🟡 Sideways Regime**: States with near-zero returns or high volatility

### 8-Point Confirmation System

Before entering a trade, up to 8 technical conditions are checked:

1. **RSI** - Relative Strength Index below threshold (not overbought)
2. **Momentum** - Price momentum above minimum threshold
3. **Volatility** - Market volatility below maximum threshold
4. **ADX** - Average Directional Index above minimum (trending market)
5. **Volume > SMA** - Volume above its 20-period moving average
6. **Price > EMA50** - Price above 50-period Exponential Moving Average
7. **Price > EMA200** - Price above 200-period Exponential Moving Average
8. **MACD > Signal** - MACD line above Signal line

The strategy only enters a trade when:
- The market is in a **Bull regime**
- At least **N confirmations** are met (configurable per mode)
- The **cooldown period** has elapsed since the last exit

---

## Trading Modes

| Mode | Leverage | Required Confirmations | Trailing Stop | Risk Level |
|------|----------|------------------------|---------------|------------|
| **Conservative** | 2.5x | 7 of 8 | 10% | Low |
| **Moderate** | 3.5x | 6 of 8 | 7% | Medium |
| **Aggressive** | 5.0x | 5 of 8 | 5% | High |
| **Custom** | User-defined | User-defined | User-defined | Variable |

---

## Strategy Parameters

### Main Parameters

| Parameter | Description | Default (Conservative) |
|-----------|-------------|------------------------|
| **Leverage** | Multiplier applied to returns (for simulation) | 2.5x |
| **Required Confirmations** | Minimum confirmations needed to enter | 7 |
| **Cooldown (hours)** | Minimum wait time after exiting before new entry | 48 |

### Risk Management Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Trailing Stop (%)** | Activates after position is profitable; trails the highest price | 10% |
| **Stop Loss (%)** | Maximum allowed loss before forced exit | 15% |
| **Take Profit (%)** | Target profit level for automatic exit | 50% |

### How Trailing Stop Works

1. When a position is opened, the trailing stop is **inactive**
2. Once the position reaches a **profit**, the trailing stop **activates**
3. It tracks the **highest price** reached during the trade
4. If price drops by the trailing stop percentage from the high, the position is **closed**

Example with 10% trailing stop:
- Entry at $50,000
- Price rises to $60,000 (trailing stop now at $54,000)
- Price rises to $65,000 (trailing stop now at $58,500)
- Price drops to $58,000 → **Exit triggered** (below $58,500)

---

## Confirmation Conditions

> **IMPORTANT**: To access all customizable parameters (strategy settings AND confirmation thresholds), you must **enable CUSTOM mode** by checking the "Enable CUSTOM Mode" checkbox in the sidebar. The preset modes (Conservative, Moderate, Aggressive) use fixed parameters.

In **Custom mode**, you can adjust every confirmation threshold:

### Numeric Thresholds

| Condition | Parameter | Description | Default |
|-----------|-----------|-------------|---------|
| **RSI Maximum** | `rsi_max` | RSI must be below this value (avoid overbought) | 90 |
| **Momentum Minimum** | `momentum_min` | 24-period momentum must exceed this (%) | 1.0% |
| **Volatility Maximum** | `volatility_max` | Volatility must be below this (%) | 6.0% |
| **ADX Minimum** | `adx_min` | ADX must exceed this (trending market) | 25 |

### Boolean Conditions

| Condition | Description | Default |
|-----------|-------------|---------|
| **Volume > SMA(20)** | Current volume exceeds 20-period average | ✅ Enabled |
| **Price > EMA 50** | Price above 50-period EMA (short-term trend) | ✅ Enabled |
| **Price > EMA 200** | Price above 200-period EMA (long-term trend) | ✅ Enabled |
| **MACD > Signal** | MACD histogram positive (bullish momentum) | ✅ Enabled |

### Understanding Each Indicator

#### RSI (Relative Strength Index)
- Measures momentum on a 0-100 scale
- Above 70 = Overbought (potential reversal down)
- Below 30 = Oversold (potential reversal up)
- **Strategy use**: Avoid entering when RSI is too high

#### Momentum
- Percentage price change over 24 periods
- Positive = Price trending up
- **Strategy use**: Require positive momentum before entry

#### Volatility
- Rolling standard deviation of returns (annualized)
- High volatility = Uncertain/risky market
- **Strategy use**: Avoid extremely volatile conditions

#### ADX (Average Directional Index)
- Measures trend strength (not direction)
- Above 25 = Strong trend
- Below 20 = Weak/no trend
- **Strategy use**: Only trade when market is trending

#### Volume
- Trading volume compared to its moving average
- Above average = Strong conviction in moves
- **Strategy use**: Confirm moves with volume

#### EMA (Exponential Moving Average)
- Price above EMA = Bullish bias
- EMA50 = Short-term trend
- EMA200 = Long-term trend (the "golden cross" indicator)

#### MACD (Moving Average Convergence Divergence)
- Difference between fast (12) and slow (26) EMAs
- MACD > Signal = Bullish momentum
- **Strategy use**: Confirm bullish momentum

---

## Understanding the Dashboard

### Main Sections

1. **Sidebar** (Left)
   - Market selection
   - Trading mode checkboxes (Aggressive, Moderate, Custom)
   - **Enable CUSTOM mode** to unlock all parameter controls:
     - Leverage, Required Confirmations, Cooldown
     - Trailing Stop, Stop Loss, Take Profit
     - All 8 confirmation condition thresholds

2. **Performance Metrics** (Top)
   - Total Return: Overall strategy performance
   - Buy & Hold: Comparison benchmark
   - Max Drawdown: Largest peak-to-trough decline
   - Sharpe Ratio: Risk-adjusted return
   - Win Rate: Percentage of profitable trades
   - Profit Factor: Gross profit / Gross loss

3. **Regime Chart** (Middle)
   - Candlestick chart with colored backgrounds
   - 🟢 Green = Bull regime
   - 🔴 Red = Bear regime
   - 🟡 Yellow = Sideways regime
   - 🔺 Green triangles = Entry points
   - 🔻 Red triangles = Exit points

4. **Trade History** (Bottom)
   - Detailed log of all trades
   - Entry/exit prices and times
   - P&L for each trade
   - Exit reasons (regime change, trailing stop, etc.)

5. **Current Status**
   - Live regime detection
   - Current confirmation status
   - Regime distribution over time

---

## Project Structure

```
regime_trading_app/
├── app.py              # Streamlit dashboard (main entry point)
├── backtester.py       # Strategy logic, HMM training, backtesting engine
├── data_loader.py      # Data fetching and feature calculation
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

### Module Descriptions

#### `data_loader.py`
- Fetches hourly OHLCV data from Yahoo Finance
- Calculates technical indicators (RSI, MACD, EMA, ADX, etc.)
- Handles data caching for faster reloads

#### `backtester.py`
- `ConfirmationThresholds`: Dataclass for customizable thresholds
- `TradingConfig`: Trading mode configurations
- `ConfirmationSystem`: 8-point entry signal system
- `RegimeDetector`: HMM training and regime classification
- `Backtester`: Main backtesting engine with risk management

#### `app.py`
- Streamlit UI components
- Interactive charts with Plotly
- Real-time parameter adjustment
- Performance visualization

---

## Advanced Usage

### Adding Custom Market Pairs

1. In the sidebar, find "Add Custom Pair"
2. Enter the Yahoo Finance ticker (e.g., "AVAX-USD")
3. Click "Add Pair"
4. Select it from the dropdown

### Optimizing Parameters

1. Start with **Conservative** mode
2. Run backtests on different timeframes
3. Switch to **Custom** mode
4. Gradually adjust parameters:
   - Lower confirmations = More trades, potentially lower quality
   - Higher leverage = Higher risk/reward
   - Tighter trailing stop = Lock in profits faster, more whipsaws

### Interpreting Results

- **Sharpe Ratio > 1.0**: Good risk-adjusted returns
- **Win Rate > 50%** with **Profit Factor > 1.5**: Solid strategy
- **Max Drawdown < 30%**: Reasonable risk exposure

---

## Troubleshooting

### "No data available"
- Yahoo Finance may have temporary issues
- Try a different market pair
- Wait a few minutes and retry

### "Not enough data for HMM training"
- Hourly data is limited to ~730 days on Yahoo Finance
- The app needs at least 500 candles after feature calculation

### App runs slowly
- HMM training can take 10-30 seconds
- Subsequent runs use cached data (faster)

---

## Disclaimer

⚠️ **This software is for educational and research purposes only.**

- This is NOT financial advice
- Past performance does NOT guarantee future results
- Cryptocurrency trading involves substantial risk of loss
- Never trade with money you cannot afford to lose
- Always do your own research (DYOR)

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

- [hmmlearn](https://hmmlearn.readthedocs.io/) - Hidden Markov Model implementation
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data API
- [Streamlit](https://streamlit.io/) - Web app framework
- [Plotly](https://plotly.com/) - Interactive charting library
