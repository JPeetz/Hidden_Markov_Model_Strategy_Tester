"""
Backtester Module
HMM Regime Detection + Strategy Logic + Risk Management
Supports: Conservative, Moderate, and Aggressive trading modes

Copyright: Joerg Peetz
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import timedelta
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class TradingMode(Enum):
    """Trading mode presets."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class ConfirmationThresholds:
    """Configurable thresholds for confirmation conditions."""
    rsi_max: float = 90.0  # RSI must be below this (not overbought)
    momentum_min: float = 1.0  # Momentum must be above this %
    volatility_max: float = 6.0  # Volatility must be below this %
    adx_min: float = 25.0  # ADX must be above this (trend strength)
    volume_above_sma: bool = True  # Volume must be above SMA
    price_above_ema50: bool = True  # Price must be above EMA50
    price_above_ema200: bool = True  # Price must be above EMA200
    macd_above_signal: bool = True  # MACD must be above signal line

    def to_dict(self) -> Dict:
        return {
            'rsi_max': self.rsi_max,
            'momentum_min': self.momentum_min,
            'volatility_max': self.volatility_max,
            'adx_min': self.adx_min,
            'volume_above_sma': self.volume_above_sma,
            'price_above_ema50': self.price_above_ema50,
            'price_above_ema200': self.price_above_ema200,
            'macd_above_signal': self.macd_above_signal
        }


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    mode: TradingMode = TradingMode.CONSERVATIVE
    leverage: float = 2.5
    required_confirmations: int = 7
    trailing_stop_pct: Optional[float] = None  # e.g., 5.0 for 5%
    stop_loss_pct: Optional[float] = None  # e.g., 10.0 for 10%
    take_profit_pct: Optional[float] = None  # e.g., 20.0 for 20%
    cooldown_hours: int = 48
    initial_capital: float = 10000.0
    confirmation_thresholds: ConfirmationThresholds = None

    def __post_init__(self):
        if self.confirmation_thresholds is None:
            self.confirmation_thresholds = ConfirmationThresholds()

    @classmethod
    def conservative(cls) -> 'TradingConfig':
        """Conservative mode: Lower leverage, strict confirmations."""
        return cls(
            mode=TradingMode.CONSERVATIVE,
            leverage=2.5,
            required_confirmations=7,
            trailing_stop_pct=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            cooldown_hours=48
        )

    @classmethod
    def moderate(cls) -> 'TradingConfig':
        """Moderate mode: Medium leverage, relaxed confirmations."""
        return cls(
            mode=TradingMode.MODERATE,
            leverage=3.5,
            required_confirmations=6,
            trailing_stop_pct=7.0,  # 7% trailing stop
            stop_loss_pct=15.0,  # 15% stop loss
            take_profit_pct=None,
            cooldown_hours=36
        )

    @classmethod
    def aggressive(cls) -> 'TradingConfig':
        """Aggressive mode: Higher leverage, relaxed confirmations, trailing stop."""
        return cls(
            mode=TradingMode.AGGRESSIVE,
            leverage=5.0,
            required_confirmations=5,
            trailing_stop_pct=5.0,  # 5% trailing stop to protect gains
            stop_loss_pct=10.0,  # 10% stop loss
            take_profit_pct=None,
            cooldown_hours=24
        )

    @classmethod
    def custom(
        cls,
        leverage: float = 2.5,
        required_confirmations: int = 7,
        trailing_stop_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        cooldown_hours: int = 48,
        initial_capital: float = 10000.0,
        confirmation_thresholds: ConfirmationThresholds = None
    ) -> 'TradingConfig':
        """Custom mode with user-defined parameters."""
        return cls(
            mode=TradingMode.CUSTOM,
            leverage=leverage,
            required_confirmations=required_confirmations,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            cooldown_hours=cooldown_hours,
            initial_capital=initial_capital,
            confirmation_thresholds=confirmation_thresholds or ConfirmationThresholds()
        )


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_price: float = 0.0  # For trailing stop tracking


@dataclass
class BacktestResult:
    """Results from backtesting."""
    trades: List[Trade]
    equity_curve: pd.Series
    regimes: pd.Series
    signals: pd.Series
    total_return: float
    buy_hold_return: float
    alpha: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    avg_trade_duration: float
    config: TradingConfig = None


class RegimeDetector:
    """
    Hidden Markov Model based regime detection.
    Uses 7 components to identify market states.
    """

    def __init__(self, n_components: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
            random_state=random_state,
            verbose=False
        )
        self.bull_state: Optional[int] = None
        self.bear_state: Optional[int] = None
        self.state_returns: Dict[int, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the HMM model and identify Bull/Bear states.

        Args:
            df: DataFrame with Returns, Range, Volume_Volatility columns

        Returns:
            Array of regime states
        """
        # Prepare features
        features = df[['Returns', 'Range', 'Volume_Volatility']].values

        # Handle any remaining NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit the model
        self.model.fit(features)
        self.fitted = True

        # Predict states
        states = self.model.predict(features)

        # Identify Bull and Bear states based on average returns
        self._identify_states(df['Returns'].values, states)

        return states

    def _identify_states(self, returns: np.ndarray, states: np.ndarray):
        """Identify Bull (highest return) and Bear (lowest return) states."""
        self.state_returns = {}

        for state in range(self.n_components):
            mask = states == state
            if mask.sum() > 0:
                self.state_returns[state] = returns[mask].mean()

        # Bull state = highest average return
        self.bull_state = max(self.state_returns, key=self.state_returns.get)

        # Bear state = lowest average return
        self.bear_state = min(self.state_returns, key=self.state_returns.get)

        print(f"\nRegime Analysis:")
        print(f"  Bull State: {self.bull_state} (avg return: {self.state_returns[self.bull_state]*100:.4f}%)")
        print(f"  Bear State: {self.bear_state} (avg return: {self.state_returns[self.bear_state]*100:.4f}%)")
        print(f"  All state returns: {[(s, f'{r*100:.4f}%') for s, r in sorted(self.state_returns.items())]}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict states for new data."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        features = df[['Returns', 'Range', 'Volume_Volatility']].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(features)

    def is_bullish(self, state: int) -> bool:
        """Check if state is the bull state."""
        return state == self.bull_state

    def is_bearish(self, state: int) -> bool:
        """Check if state is the bear/crash state."""
        return state == self.bear_state

    def get_regime_name(self, state: int) -> str:
        """Get human-readable regime name."""
        if state == self.bull_state:
            return "BULL"
        elif state == self.bear_state:
            return "BEAR"
        else:
            # Categorize other states based on return
            ret = self.state_returns.get(state, 0)
            if ret > 0.0001:
                return "MILD_BULL"
            elif ret < -0.0001:
                return "MILD_BEAR"
            else:
                return "NEUTRAL"


class ConfirmationSystem:
    """
    Configurable confirmation voting system for trade entry.
    All thresholds and conditions are customizable.
    """

    TOTAL_CONFIRMATIONS = 8

    def __init__(
        self,
        required_confirmations: int = 7,
        thresholds: ConfirmationThresholds = None
    ):
        self.required_confirmations = required_confirmations
        self.thresholds = thresholds or ConfirmationThresholds()

    def check_confirmations(self, row: pd.Series) -> Tuple[int, List[str], List[Dict]]:
        """
        Check all confirmation conditions with configurable thresholds.

        Returns:
            Tuple of (confirmation_count, list_of_passed_conditions, list_of_condition_details)
        """
        t = self.thresholds
        confirmations = []
        passed = []
        details = []

        # 1. RSI < threshold (not overbought)
        rsi_ok = row['RSI'] < t.rsi_max
        confirmations.append(rsi_ok)
        details.append({
            'name': 'RSI',
            'condition': f'< {t.rsi_max}',
            'value': row['RSI'],
            'passed': rsi_ok
        })
        if rsi_ok:
            passed.append(f"RSI={row['RSI']:.1f}<{t.rsi_max}")

        # 2. Momentum > threshold
        momentum_ok = row['Momentum'] > t.momentum_min
        confirmations.append(momentum_ok)
        details.append({
            'name': 'Momentum',
            'condition': f'> {t.momentum_min}%',
            'value': row['Momentum'],
            'passed': momentum_ok
        })
        if momentum_ok:
            passed.append(f"Momentum={row['Momentum']:.2f}%>{t.momentum_min}%")

        # 3. Volatility < threshold
        vol_ok = row['Volatility'] < t.volatility_max
        confirmations.append(vol_ok)
        details.append({
            'name': 'Volatility',
            'condition': f'< {t.volatility_max}%',
            'value': row['Volatility'],
            'passed': vol_ok
        })
        if vol_ok:
            passed.append(f"Vol={row['Volatility']:.2f}%<{t.volatility_max}%")

        # 4. Volume > 20-period SMA (if enabled)
        if t.volume_above_sma:
            vol_above_sma = row['Volume'] > row['Volume_SMA']
        else:
            vol_above_sma = True  # Always pass if disabled
        confirmations.append(vol_above_sma)
        details.append({
            'name': 'Volume>SMA',
            'condition': 'enabled' if t.volume_above_sma else 'disabled',
            'value': row['Volume'] / row['Volume_SMA'] if row['Volume_SMA'] > 0 else 0,
            'passed': vol_above_sma
        })
        if vol_above_sma:
            passed.append("Vol>SMA20")

        # 5. ADX > threshold (strong trend)
        adx_ok = row['ADX'] > t.adx_min
        confirmations.append(adx_ok)
        details.append({
            'name': 'ADX',
            'condition': f'> {t.adx_min}',
            'value': row['ADX'],
            'passed': adx_ok
        })
        if adx_ok:
            passed.append(f"ADX={row['ADX']:.1f}>{t.adx_min}")

        # 6. Price > EMA 50 (if enabled)
        if t.price_above_ema50:
            above_ema50 = row['Close'] > row['EMA50']
        else:
            above_ema50 = True  # Always pass if disabled
        confirmations.append(above_ema50)
        details.append({
            'name': 'Price>EMA50',
            'condition': 'enabled' if t.price_above_ema50 else 'disabled',
            'value': row['Close'] / row['EMA50'] if row['EMA50'] > 0 else 0,
            'passed': above_ema50
        })
        if above_ema50:
            passed.append("Price>EMA50")

        # 7. Price > EMA 200 (if enabled)
        if t.price_above_ema200:
            above_ema200 = row['Close'] > row['EMA200']
        else:
            above_ema200 = True  # Always pass if disabled
        confirmations.append(above_ema200)
        details.append({
            'name': 'Price>EMA200',
            'condition': 'enabled' if t.price_above_ema200 else 'disabled',
            'value': row['Close'] / row['EMA200'] if row['EMA200'] > 0 else 0,
            'passed': above_ema200
        })
        if above_ema200:
            passed.append("Price>EMA200")

        # 8. MACD > Signal Line (if enabled)
        if t.macd_above_signal:
            macd_bullish = row['MACD'] > row['MACD_Signal']
        else:
            macd_bullish = True  # Always pass if disabled
        confirmations.append(macd_bullish)
        details.append({
            'name': 'MACD>Signal',
            'condition': 'enabled' if t.macd_above_signal else 'disabled',
            'value': row['MACD'] - row['MACD_Signal'],
            'passed': macd_bullish
        })
        if macd_bullish:
            passed.append("MACD>Signal")

        return sum(confirmations), passed, details


class Backtester:
    """
    Main backtesting engine with risk management.
    Supports multiple trading modes and risk controls.
    """

    def __init__(self, config: TradingConfig = None):
        if config is None:
            config = TradingConfig.conservative()

        self.config = config
        self.initial_capital = config.initial_capital
        self.leverage = config.leverage
        self.cooldown_hours = config.cooldown_hours
        self.regime_detector = RegimeDetector(n_components=7)
        self.confirmation_system = ConfirmationSystem(
            config.required_confirmations,
            config.confirmation_thresholds
        )

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest simulation.

        Args:
            df: DataFrame with OHLCV and calculated features

        Returns:
            BacktestResult with all metrics
        """
        print("\n" + "="*60)
        print(f"STARTING BACKTEST - {self.config.mode.value.upper()} MODE")
        print("="*60)
        print(f"  Leverage: {self.config.leverage}x")
        print(f"  Required Confirmations: {self.config.required_confirmations}/8")
        print(f"  Trailing Stop: {self.config.trailing_stop_pct}%" if self.config.trailing_stop_pct else "  Trailing Stop: Disabled")
        print(f"  Stop Loss: {self.config.stop_loss_pct}%" if self.config.stop_loss_pct else "  Stop Loss: Disabled")
        print(f"  Take Profit: {self.config.take_profit_pct}%" if self.config.take_profit_pct else "  Take Profit: Disabled")

        # Fit HMM and get regimes
        print("\nFitting HMM model...")
        states = self.regime_detector.fit(df)
        df = df.copy()
        df['Regime'] = states

        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = cash, 1 = long
        entry_price = 0.0
        entry_time = None
        last_exit_time = None
        max_price_since_entry = 0.0  # For trailing stop
        trades: List[Trade] = []
        equity_curve = []
        signals = []

        print(f"\nRunning simulation with ${self.initial_capital:,.0f} capital...")

        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_price = row['Close']
            current_regime = row['Regime']
            is_bullish = self.regime_detector.is_bullish(current_regime)
            is_bearish = self.regime_detector.is_bearish(current_regime)

            # Check cooldown
            in_cooldown = False
            if last_exit_time is not None:
                cooldown_end = last_exit_time + timedelta(hours=self.cooldown_hours)
                in_cooldown = timestamp < cooldown_end

            # Current signal
            signal = "CASH"
            exit_reason = None

            if position == 1:
                # Update max price for trailing stop
                max_price_since_entry = max(max_price_since_entry, current_price)

                # === EXIT LOGIC ===
                # Check all exit conditions

                # 1. Bear Regime Exit
                if is_bearish:
                    exit_reason = "BEAR_REGIME"

                # 2. Trailing Stop Check
                elif self.config.trailing_stop_pct is not None:
                    trailing_stop_price = max_price_since_entry * (1 - self.config.trailing_stop_pct / 100)
                    if current_price <= trailing_stop_price:
                        exit_reason = "TRAILING_STOP"

                # 3. Stop Loss Check
                if exit_reason is None and self.config.stop_loss_pct is not None:
                    stop_loss_price = entry_price * (1 - self.config.stop_loss_pct / 100)
                    if current_price <= stop_loss_price:
                        exit_reason = "STOP_LOSS"

                # 4. Take Profit Check
                if exit_reason is None and self.config.take_profit_pct is not None:
                    take_profit_price = entry_price * (1 + self.config.take_profit_pct / 100)
                    if current_price >= take_profit_price:
                        exit_reason = "TAKE_PROFIT"

                # Execute exit if triggered
                if exit_reason:
                    exit_price = current_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_leveraged = pnl_pct * self.leverage
                    pnl_dollar = capital * pnl_leveraged

                    capital = capital * (1 + pnl_leveraged)

                    trade = Trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=timestamp,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl=pnl_dollar,
                        pnl_pct=pnl_leveraged * 100,
                        max_price=max_price_since_entry
                    )
                    trades.append(trade)

                    position = 0
                    last_exit_time = timestamp
                    max_price_since_entry = 0.0
                    signal = "EXIT"

                    if len(trades) <= 10 or len(trades) % 50 == 0:
                        print(f"  [{timestamp}] EXIT ({exit_reason}) @ ${exit_price:,.2f} | PnL: {pnl_leveraged*100:+.2f}%")
                else:
                    signal = "LONG"

            else:
                # === ENTRY LOGIC ===
                if not in_cooldown and is_bullish:
                    # Check confirmations
                    conf_count, passed, _ = self.confirmation_system.check_confirmations(row)

                    if conf_count >= self.config.required_confirmations:
                        position = 1
                        entry_price = current_price
                        entry_time = timestamp
                        max_price_since_entry = current_price
                        signal = "ENTRY"

                        if len(trades) <= 10 or len(trades) % 50 == 0:
                            print(f"  [{timestamp}] ENTRY @ ${entry_price:,.2f} | Confirmations: {conf_count}/{ConfirmationSystem.TOTAL_CONFIRMATIONS}")

            # Track equity
            if position == 1:
                unrealized_pnl = (current_price - entry_price) / entry_price * self.leverage
                current_equity = capital * (1 + unrealized_pnl)
            else:
                current_equity = capital

            equity_curve.append(current_equity)
            signals.append(signal)

        # Close any open position at the end
        if position == 1:
            final_price = df.iloc[-1]['Close']
            pnl_pct = (final_price - entry_price) / entry_price
            pnl_leveraged = pnl_pct * self.leverage
            pnl_dollar = capital * pnl_leveraged
            capital = capital * (1 + pnl_leveraged)

            trade = Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=df.index[-1],
                exit_price=final_price,
                exit_reason="END_OF_DATA",
                pnl=pnl_dollar,
                pnl_pct=pnl_leveraged * 100,
                max_price=max_price_since_entry
            )
            trades.append(trade)
            equity_curve[-1] = capital

        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=df.index)
        regime_series = pd.Series(states, index=df.index)
        signal_series = pd.Series(signals, index=df.index)

        metrics = self._calculate_metrics(df, equity_series, trades)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            regimes=regime_series,
            signals=signal_series,
            config=self.config,
            **metrics
        )

    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        equity: pd.Series,
        trades: List[Trade]
    ) -> Dict:
        """Calculate performance metrics."""
        # Total return
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100

        # Buy and hold return
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

        # Alpha
        alpha = total_return - buy_hold_return

        # Win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Sharpe ratio (hourly returns annualized)
        returns = equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe_ratio = 0

        # Average trade duration
        durations = []
        for t in trades:
            if t.exit_time is not None:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"  Total Return:      {total_return:+.2f}%")
        print(f"  Buy & Hold Return: {buy_hold_return:+.2f}%")
        print(f"  Alpha:             {alpha:+.2f}%")
        print(f"  Win Rate:          {win_rate:.1f}%")
        print(f"  Max Drawdown:      {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio:      {sharpe_ratio:.2f}")
        print(f"  Total Trades:      {len(trades)}")
        print(f"  Avg Trade Duration: {avg_duration:.1f} hours")
        print("="*60)

        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'avg_trade_duration': avg_duration
        }


def run_backtest(
    df: pd.DataFrame,
    mode: str = "conservative",
    **custom_params
) -> BacktestResult:
    """
    Convenience function to run backtest with specified mode.

    Args:
        df: DataFrame with OHLCV and features
        mode: "conservative", "moderate", "aggressive", or "custom"
        **custom_params: Parameters for custom mode (leverage, required_confirmations, etc.)

    Returns:
        BacktestResult
    """
    if mode == "conservative":
        config = TradingConfig.conservative()
    elif mode == "moderate":
        config = TradingConfig.moderate()
    elif mode == "aggressive":
        config = TradingConfig.aggressive()
    elif mode == "custom":
        config = TradingConfig.custom(**custom_params)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    backtester = Backtester(config)
    return backtester.run(df)


if __name__ == "__main__":
    from data_loader import fetch_btc_data

    # Fetch data
    df = fetch_btc_data()

    # Test all modes
    for mode in ["conservative", "moderate", "aggressive"]:
        print(f"\n\n{'#'*60}")
        print(f"TESTING {mode.upper()} MODE")
        print('#'*60)

        result = run_backtest(df, mode=mode)

        print(f"\nTRADE LOG (Last 5 trades):")
        print("-" * 80)
        for trade in result.trades[-5:]:
            print(f"  {trade.entry_time} -> {trade.exit_time}")
            print(f"    Entry: ${trade.entry_price:,.2f} | Exit: ${trade.exit_price:,.2f}")
            print(f"    PnL: {trade.pnl_pct:+.2f}% (${trade.pnl:+,.2f}) | Reason: {trade.exit_reason}")
            print()
