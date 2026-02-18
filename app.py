"""
Regime-Based Trading Dashboard
Streamlit app with Plotly visualization
Supports multiple trading modes, market pairs, and fully customizable parameters

Copyright: Joerg Peetz
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_loader import fetch_btc_data
from backtester import (
    Backtester, TradingConfig, TradingMode,
    ConfirmationSystem, ConfirmationThresholds, run_backtest
)

# Page config
st.set_page_config(
    page_title="Regime Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default market pairs
DEFAULT_PAIRS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
    "BNB-USD",
    "DOGE-USD",
    "SHIB-USD"
]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .signal-long {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .signal-cash {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .regime-bull {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .regime-bear {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .regime-neutral {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
    }
    .mode-conservative {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        display: inline-block;
    }
    .mode-moderate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        display: inline-block;
    }
    .mode-aggressive {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        display: inline-block;
    }
    .mode-custom {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for custom pairs
if 'custom_pairs' not in st.session_state:
    st.session_state.custom_pairs = []


def get_all_pairs():
    """Get all available market pairs including custom ones."""
    return DEFAULT_PAIRS + st.session_state.custom_pairs


@st.cache_data(ttl=3600)
def load_data(ticker: str = "BTC-USD"):
    """Load and cache data for any ticker."""
    return fetch_btc_data(ticker=ticker, days=730)


def run_backtest_with_config(df, config: TradingConfig):
    """Run backtest with specific configuration."""
    backtester = Backtester(config)
    return backtester.run(df), backtester.regime_detector


def get_current_signal(df, result, regime_detector, config):
    """Determine current trading signal."""
    latest = df.iloc[-1]
    current_regime = result.regimes.iloc[-1]

    # Check if in cooldown
    last_trade = result.trades[-1] if result.trades else None
    in_cooldown = False
    cooldown_end = None

    if last_trade and last_trade.exit_time:
        cooldown_end = last_trade.exit_time + timedelta(hours=config.cooldown_hours)
        if datetime.now() < cooldown_end.to_pydatetime().replace(tzinfo=None):
            in_cooldown = True

    # Get confirmations using the config's thresholds
    conf_system = ConfirmationSystem(
        config.required_confirmations,
        config.confirmation_thresholds
    )
    conf_count, passed_conditions, details = conf_system.check_confirmations(latest)
    total_confirmations = ConfirmationSystem.TOTAL_CONFIRMATIONS

    # Determine signal
    is_bullish = regime_detector.is_bullish(current_regime)
    regime_name = regime_detector.get_regime_name(current_regime)

    if in_cooldown:
        signal = "COOLDOWN"
        signal_detail = f"Cooldown until {cooldown_end}"
    elif is_bullish and conf_count >= config.required_confirmations:
        signal = "LONG"
        signal_detail = f"{conf_count}/{total_confirmations} confirmations"
    else:
        signal = "CASH"
        if not is_bullish:
            signal_detail = f"Regime: {regime_name}"
        else:
            signal_detail = f"Only {conf_count}/{total_confirmations} confirmations (need {config.required_confirmations})"

    return signal, signal_detail, regime_name, conf_count, details


def create_regime_chart(df, result, regime_detector, ticker):
    """Create candlestick chart with regime-colored background."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price with Regime Detection", "Equity Curve", "Volume")
    )

    # Add regime background shapes
    prev_regime = None
    start_idx = 0

    for i, regime in enumerate(result.regimes):
        if regime != prev_regime and prev_regime is not None:
            if regime_detector.is_bullish(prev_regime):
                color = "rgba(0, 200, 100, 0.15)"
            elif regime_detector.is_bearish(prev_regime):
                color = "rgba(255, 50, 50, 0.15)"
            else:
                color = "rgba(150, 150, 150, 0.08)"

            fig.add_vrect(
                x0=df.index[start_idx],
                x1=df.index[i-1],
                fillcolor=color,
                layer="below",
                line_width=0,
                row=1, col=1
            )
            start_idx = i
        prev_regime = regime

    # Add final regime block
    if prev_regime is not None:
        if regime_detector.is_bullish(prev_regime):
            color = "rgba(0, 200, 100, 0.15)"
        elif regime_detector.is_bearish(prev_regime):
            color = "rgba(255, 50, 50, 0.15)"
        else:
            color = "rgba(150, 150, 150, 0.08)"

        fig.add_vrect(
            x0=df.index[start_idx],
            x1=df.index[-1],
            fillcolor=color,
            layer="below",
            line_width=0,
            row=1, col=1
        )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Add EMAs
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='#ff9800', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA200'],
            mode='lines',
            name='EMA 200',
            line=dict(color='#2196f3', width=1)
        ),
        row=1, col=1
    )

    # Add trade markers
    for trade in result.trades:
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#00ff00'),
                name='Entry',
                showlegend=False,
                hovertemplate=f"Entry<br>${trade.entry_price:,.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Exit marker
        if trade.exit_time:
            color = '#ff0000' if trade.pnl < 0 else '#00ff00'
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color=color),
                    name='Exit',
                    showlegend=False,
                    hovertemplate=f"Exit ({trade.exit_reason})<br>${trade.exit_price:,.2f}<br>PnL: {trade.pnl_pct:+.2f}%<extra></extra>"
                ),
                row=1, col=1
            )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode='lines',
            name='Strategy Equity',
            line=dict(color='#9c27b0', width=2),
            fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.1)'
        ),
        row=2, col=1
    )

    # Buy and hold comparison
    initial_capital = result.config.initial_capital if result.config else 10000
    buy_hold = initial_capital * (df['Close'] / df['Close'].iloc[0])
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=buy_hold,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#607d8b', width=1, dash='dash')
        ),
        row=2, col=1
    )

    # Volume bars
    colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,30,1)',
        font=dict(color='white')
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig


def main():
    # Header
    st.markdown('<p class="main-header">Regime-Based Trading Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">HMM-Powered Market Regime Detection with Fully Customizable Parameters</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Market Pair Selection
        st.subheader("Market Pair")
        all_pairs = get_all_pairs()
        selected_ticker = st.selectbox(
            "Select Trading Pair",
            options=all_pairs,
            index=0
        )

        # Add custom pair
        with st.expander("Add Custom Pair"):
            new_pair = st.text_input(
                "Enter ticker symbol (e.g., AVAX-USD)",
                placeholder="TICKER-USD"
            ).upper()
            if st.button("Add Pair"):
                if new_pair and new_pair not in all_pairs:
                    st.session_state.custom_pairs.append(new_pair)
                    st.success(f"Added {new_pair}")
                    st.rerun()
                elif new_pair in all_pairs:
                    st.warning(f"{new_pair} already exists")

        st.divider()

        # Trading Mode Selection
        st.subheader("Trading Mode")

        use_aggressive = st.checkbox("Enable AGGRESSIVE Mode", value=False, help="5x leverage, 5 confirmations, 5% trailing stop")
        use_moderate = st.checkbox("Enable MODERATE Mode", value=False, help="3.5x leverage, 6 confirmations, 7% trailing stop")
        use_custom = st.checkbox("Enable CUSTOM Mode", value=False, help="Manually configure all parameters")

        # Determine active mode
        if use_custom:
            active_mode = "custom"
        elif use_aggressive:
            active_mode = "aggressive"
        elif use_moderate:
            active_mode = "moderate"
        else:
            active_mode = "conservative"

        # Display active mode
        mode_colors = {
            "conservative": "mode-conservative",
            "moderate": "mode-moderate",
            "aggressive": "mode-aggressive",
            "custom": "mode-custom"
        }
        st.markdown(f'<div class="{mode_colors[active_mode]}">Active: {active_mode.upper()}</div>', unsafe_allow_html=True)

        st.divider()

        # Custom Parameters
        st.subheader("Strategy Parameters")

        if active_mode == "custom":
            leverage = st.slider("Leverage", min_value=1.0, max_value=10.0, value=2.5, step=0.5,
                               help="Multiplier applied to gains/losses")
            required_confirmations = st.slider("Required Confirmations", min_value=1, max_value=8, value=7,
                                              help="Minimum confirmations needed to enter a trade")
            cooldown_hours = st.slider("Cooldown (hours)", min_value=1, max_value=72, value=48,
                                      help="Wait time after exit before next entry")
            initial_capital = st.number_input("Initial Capital ($)", min_value=100, max_value=1000000, value=10000, step=1000,
                                             help="Starting capital for backtesting")

            st.subheader("Risk Management")
            enable_trailing_stop = st.checkbox("Enable Trailing Stop", value=False,
                                              help="Exit when price drops X% from highest point since entry")
            trailing_stop_pct = st.slider("Trailing Stop %", min_value=1.0, max_value=20.0, value=5.0, step=0.5) if enable_trailing_stop else None

            enable_stop_loss = st.checkbox("Enable Stop Loss", value=False,
                                          help="Exit when price drops X% from entry price")
            stop_loss_pct = st.slider("Stop Loss %", min_value=1.0, max_value=30.0, value=10.0, step=0.5) if enable_stop_loss else None

            enable_take_profit = st.checkbox("Enable Take Profit", value=False,
                                            help="Exit when price rises X% from entry price")
            take_profit_pct = st.slider("Take Profit %", min_value=5.0, max_value=100.0, value=20.0, step=1.0) if enable_take_profit else None

            st.divider()

            # Customizable Confirmation Conditions
            st.subheader("Confirmation Conditions")
            st.caption("Adjust thresholds for entry signals")

            rsi_max = st.slider("RSI Maximum", min_value=50.0, max_value=100.0, value=90.0, step=1.0,
                               help="RSI must be below this value (avoid overbought)")
            momentum_min = st.slider("Momentum Minimum (%)", min_value=-5.0, max_value=10.0, value=1.0, step=0.5,
                                    help="24-hour momentum must be above this")
            volatility_max = st.slider("Volatility Maximum (%)", min_value=1.0, max_value=20.0, value=6.0, step=0.5,
                                      help="Hourly volatility must be below this")
            adx_min = st.slider("ADX Minimum", min_value=10.0, max_value=50.0, value=25.0, step=1.0,
                               help="Trend strength must be above this")

            volume_above_sma = st.checkbox("Volume > SMA(20)", value=True,
                                          help="Volume must exceed 20-period average")
            price_above_ema50 = st.checkbox("Price > EMA 50", value=True,
                                           help="Price must be above 50-period EMA")
            price_above_ema200 = st.checkbox("Price > EMA 200", value=True,
                                            help="Price must be above 200-period EMA")
            macd_above_signal = st.checkbox("MACD > Signal", value=True,
                                           help="MACD must be above signal line")

            # Create confirmation thresholds
            confirmation_thresholds = ConfirmationThresholds(
                rsi_max=rsi_max,
                momentum_min=momentum_min,
                volatility_max=volatility_max,
                adx_min=adx_min,
                volume_above_sma=volume_above_sma,
                price_above_ema50=price_above_ema50,
                price_above_ema200=price_above_ema200,
                macd_above_signal=macd_above_signal
            )

            # Build custom config
            config = TradingConfig.custom(
                leverage=leverage,
                required_confirmations=required_confirmations,
                trailing_stop_pct=trailing_stop_pct if enable_trailing_stop else None,
                stop_loss_pct=stop_loss_pct if enable_stop_loss else None,
                take_profit_pct=take_profit_pct if enable_take_profit else None,
                cooldown_hours=cooldown_hours,
                initial_capital=initial_capital,
                confirmation_thresholds=confirmation_thresholds
            )
        else:
            # Show preset parameters (read-only)
            if active_mode == "conservative":
                config = TradingConfig.conservative()
            elif active_mode == "moderate":
                config = TradingConfig.moderate()
            else:  # aggressive
                config = TradingConfig.aggressive()

            t = config.confirmation_thresholds
            st.info(f"""
            **{active_mode.upper()} Mode:**
            - Leverage: {config.leverage}x
            - Confirmations: {config.required_confirmations}/{ConfirmationSystem.TOTAL_CONFIRMATIONS}
            - Trailing Stop: {config.trailing_stop_pct}%{' (enabled)' if config.trailing_stop_pct else ' (disabled)'}
            - Stop Loss: {config.stop_loss_pct}%{' (enabled)' if config.stop_loss_pct else ' (disabled)'}
            - Cooldown: {config.cooldown_hours}h

            **Confirmation Thresholds:**
            - RSI < {t.rsi_max}
            - Momentum > {t.momentum_min}%
            - Volatility < {t.volatility_max}%
            - ADX > {t.adx_min}
            """)

        st.divider()

        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # Load data and run backtest
    with st.spinner(f"Loading {selected_ticker} data and running backtest..."):
        try:
            df = load_data(selected_ticker)
            result, regime_detector = run_backtest_with_config(df, config)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("The selected ticker might not have sufficient hourly data available. Try a different pair.")
            return

    # Get current signal
    signal, signal_detail, regime_name, conf_count, conf_details = get_current_signal(
        df, result, regime_detector, config
    )
    total_confirmations = ConfirmationSystem.TOTAL_CONFIRMATIONS

    # Top section - Current Signal, Regime, and Mode
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if signal == "LONG":
            st.markdown(f"""
            <div class="signal-long">
                <div style="font-size: 2rem;">📈 LONG</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{signal_detail}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-cash">
                <div style="font-size: 2rem;">💵 {signal}</div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">{signal_detail}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if "BULL" in regime_name:
            css_class = "regime-bull"
            emoji = "🐂"
        elif "BEAR" in regime_name:
            css_class = "regime-bear"
            emoji = "🐻"
        else:
            css_class = "regime-neutral"
            emoji = "😐"

        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size: 1.2rem;">Current Regime</div>
            <div style="font-size: 2rem; font-weight: bold;">{emoji} {regime_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        latest_price = df['Close'].iloc[-1]
        price_change = df['Returns'].iloc[-1] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.2rem;">{selected_ticker} Price</div>
            <div style="font-size: 2rem; font-weight: bold;">${latest_price:,.2f}</div>
            <div style="font-size: 0.9rem;">{price_change:+.2f}% (1H)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="{mode_colors[active_mode]}" style="padding: 1rem; text-align: center;">
            <div style="font-size: 1.2rem;">Trading Mode</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{active_mode.upper()}</div>
            <div style="font-size: 0.9rem;">{config.leverage}x | {config.required_confirmations}/{total_confirmations}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Metrics row
    st.subheader("Performance Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        color = "normal" if result.total_return >= 0 else "inverse"
        st.metric(
            "Total Return",
            f"{result.total_return:+.2f}%",
            delta=f"${config.initial_capital * result.total_return / 100:+,.0f}",
            delta_color=color
        )

    with m2:
        color = "normal" if result.alpha >= 0 else "inverse"
        st.metric(
            "Alpha vs B&H",
            f"{result.alpha:+.2f}%",
            delta=f"B&H: {result.buy_hold_return:+.1f}%",
            delta_color=color
        )

    with m3:
        st.metric(
            "Win Rate",
            f"{result.win_rate:.1f}%",
            delta=f"{result.total_trades} trades"
        )

    with m4:
        st.metric(
            "Max Drawdown",
            f"{result.max_drawdown:.2f}%",
            delta_color="inverse"
        )

    with m5:
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}",
        )

    with m6:
        st.metric(
            "Avg Trade Duration",
            f"{result.avg_trade_duration:.1f}h"
        )

    st.divider()

    # Main chart
    st.subheader(f"Price Chart with Regime Detection - {selected_ticker}")
    fig = create_regime_chart(df, result, regime_detector, selected_ticker)
    st.plotly_chart(fig, use_container_width=True)

    # Current confirmations with dynamic thresholds
    st.divider()
    st.subheader(f"Current Confirmation Status (Need {config.required_confirmations}/{total_confirmations})")

    conf_cols = st.columns(8)
    t = config.confirmation_thresholds

    for i, detail in enumerate(conf_details):
        with conf_cols[i]:
            if detail['passed']:
                st.success(f"✅ {detail['name']}\n{detail['condition']}")
            else:
                st.error(f"❌ {detail['name']}\n{detail['condition']}")

    # Active Risk Management Display
    if config.trailing_stop_pct or config.stop_loss_pct or config.take_profit_pct:
        st.divider()
        st.subheader("Active Risk Management")
        rm_cols = st.columns(3)

        with rm_cols[0]:
            if config.trailing_stop_pct:
                st.info(f"🔄 **Trailing Stop:** {config.trailing_stop_pct}%")
            else:
                st.warning("🔄 Trailing Stop: Disabled")

        with rm_cols[1]:
            if config.stop_loss_pct:
                st.error(f"🛑 **Stop Loss:** {config.stop_loss_pct}%")
            else:
                st.warning("🛑 Stop Loss: Disabled")

        with rm_cols[2]:
            if config.take_profit_pct:
                st.success(f"🎯 **Take Profit:** {config.take_profit_pct}%")
            else:
                st.warning("🎯 Take Profit: Disabled")

    # Trade log
    st.divider()
    st.subheader("Recent Trades")

    if result.trades:
        trade_data = []
        for t in result.trades[-20:]:
            trade_data.append({
                "Entry Time": t.entry_time,
                "Entry Price": f"${t.entry_price:,.2f}",
                "Exit Time": t.exit_time,
                "Exit Price": f"${t.exit_price:,.2f}" if t.exit_price else "-",
                "PnL %": f"{t.pnl_pct:+.2f}%",
                "PnL $": f"${t.pnl:+,.2f}",
                "Exit Reason": t.exit_reason
            })

        trade_df = pd.DataFrame(trade_data)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed yet.")

    # Exit reason breakdown
    if result.trades:
        st.divider()
        st.subheader("Exit Reason Breakdown")

        exit_reasons = {}
        for trade in result.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "total_pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["total_pnl"] += trade.pnl

        reason_cols = st.columns(len(exit_reasons))
        for i, (reason, stats) in enumerate(exit_reasons.items()):
            with reason_cols[i]:
                st.metric(
                    reason,
                    f"{stats['count']} trades",
                    f"${stats['total_pnl']:+,.0f}"
                )

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: {len(df)} hourly candles from {df.index[0].date()} to {df.index[-1].date()} | Mode: {active_mode.upper()}")


if __name__ == "__main__":
    main()
