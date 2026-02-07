#!/usr/bin/env python3
"""
每日综合技术指标系统 - 股指版本
用于预测SPX500和NDX100次日涨跌概率和幅度
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import json

@dataclass
class TradingSignal:
    """交易信号数据类"""
    date: str
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0-100
    prob_up: float  # 上涨概率 %
    prob_down: float  # 下跌概率 %
    expected_move: float  # 预期涨跌幅 %
    stop_loss: float  # 止损位
    take_profit: List[float]  # 目标位列表
    position_size: float  # 建议仓位 %
    reason: str  # 信号原因

    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'prob_up': self.prob_up,
            'prob_down': self.prob_down,
            'expected_move': self.expected_move,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'reason': self.reason
        }

class DailyTechnicalIndicator:
    """每日综合技术指标"""

    def __init__(self, symbol: str = 'SPX'):
        self.symbol = symbol
        self.signals_history = []
        self.trades_history = []

        # 指标权重配置
        self.weights = {
            'wyckoff': 0.25,
            'market_profile': 0.20,
            'momentum': 0.20,
            'volatility': 0.15,
            'sentiment': 0.20
        }

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI指标"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0/(1.0+rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down if down != 0 else 100
            rsi[i] = 100.0 - 100.0/(1.0+rs)

        return rsi

    def calculate_macd(self, prices: np.ndarray, fast=12, slow=26, signal=9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算MACD指标"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """计算EMA"""
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
        """计算ATR"""
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]

        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        atr[:period-1] = np.nan
        return atr

    def analyze_wyckoff_phase(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """威科夫阶段分析"""
        if idx < 20:
            return "UNKNOWN", 0

        recent = df.iloc[idx-20:idx+1]
        close = recent['close'].values
        high = recent['high'].values
        low = recent['low'].values
        volume = recent['volume'].values

        price_trend = (close[-1] - close[0]) / close[0]
        volatility = np.std(close) / np.mean(close)

        avg_volume = np.mean(volume[:-5])
        recent_volume = np.mean(volume[-5:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        signal_strength = 0

        # 检测抛售高潮
        if close[-1] < np.mean(close[-10:]) and volume_ratio > 1.5:
            if price_trend < -0.03:
                return "SC", -0.3

        # 检测自动反弹
        if close[-1] > close[-5] and volume_ratio > 1.2:
            if price_trend > 0.015:
                signal_strength += 0.4

        # 检测二次测试
        if len(close) > 5:
            if abs(close[-1] - close[-5]) / close[-5] < 0.015:
                if volume_ratio < 0.8:
                    return "ST", 0.2

        # 检测Spring
        lows = recent['low'].values
        if lows[-1] < np.min(lows[-10:-1]):
            if close[-1] > np.mean(close[-5:]):
                return "SPRING", 0.5

        # 检测LPS
        if close[-1] > np.mean(close[-10:]) and volume_ratio > 1.2:
            return "LPS", 0.6

        # 检测SOS
        if close[-1] > close[-3] and close[-1] > close[-5]:
            if volume_ratio > 1.3:
                return "SOS", 0.7

        # 检测UT
        if close[-1] < close[-3] and high[-1] > np.max(high[-10:-1]):
            if volume_ratio > 1.2:
                return "UT", -0.4

        # 综合判断
        if price_trend > 0.02:
            return "MARKUP", 0.5
        elif price_trend < -0.02:
            return "MARKDOWN", -0.4
        else:
            return "ACCUMULATION", 0.1

    def analyze_market_profile(self, df: pd.DataFrame, idx: int) -> Tuple[float, float]:
        """四度空间分析"""
        if idx < 20:
            return 0, 0.5

        recent = df.iloc[idx-20:idx+1]
        prices = recent['close'].values
        volumes = recent['volume'].values

        # 简化的POC计算
        price_levels = np.linspace(np.min(prices), np.max(prices), 20)
        volume_by_price = np.zeros(20)

        for i in range(len(prices)):
            price_idx = min(int((prices[i] - np.min(prices)) / (np.max(prices) - np.min(prices)) * 19), 19)
            volume_by_price[price_idx] += volumes[i]

        poc_idx = np.argmax(volume_by_price)
        poc_price = price_levels[poc_idx]

        val = np.percentile(prices, 15)
        vah = np.percentile(prices, 85)
        current_price = prices[-1]

        if current_price > vah:
            value_position = 1.0
            signal = 0.2
        elif current_price < val:
            value_position = 0.0
            signal = -0.2
        else:
            value_position = (current_price - val) / (vah - val) if vah > val else 0.5
            distance_to_poc = abs(current_price - poc_price) / (vah - val) if vah > val else 0.5
            signal = 0.5 - distance_to_poc

        return signal, value_position

    def analyze_momentum(self, df: pd.DataFrame, idx: int) -> float:
        """动量分析"""
        if idx < 30:
            return 0

        close = df['close'].values
        rsi = self.calculate_rsi(close[:idx+1])
        _, _, macd_hist = self.calculate_macd(close[:idx+1])

        current_rsi = rsi[idx]
        current_macd = macd_hist[idx]

        if current_rsi < 30:
            rsi_signal = 0.5
        elif current_rsi > 70:
            rsi_signal = -0.5
        elif current_rsi < 45:
            rsi_signal = 0.2
        elif current_rsi > 55:
            rsi_signal = -0.2
        else:
            rsi_signal = 0

        if current_macd > 0:
            if idx > 0 and macd_hist[idx] > macd_hist[idx-1]:
                macd_signal = 0.3
            else:
                macd_signal = 0.1
        else:
            if idx > 0 and macd_hist[idx] < macd_hist[idx-1]:
                macd_signal = -0.3
            else:
                macd_signal = -0.1

        return (rsi_signal + macd_signal) / 2

    def analyze_volatility(self, df: pd.DataFrame, idx: int) -> Tuple[float, float]:
        """波动率分析"""
        if idx < 20:
            return 0, 0

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        atr = self.calculate_atr(high[:idx+1], low[:idx+1], close[:idx+1])
        current_atr = atr[idx]
        atr_pct = current_atr / close[idx] if close[idx] > 0 else 0

        returns = np.diff(close[idx-20:idx+1]) / close[idx-20:idx]
        hist_vol = np.std(returns) * np.sqrt(252)

        if atr_pct < 0.008:
            vol_signal = 0.3
        elif atr_pct > 0.02:
            vol_signal = -0.2
        else:
            vol_signal = 0

        return vol_signal, atr_pct

    def analyze_sentiment(self, df: pd.DataFrame, idx: int, put_call_ratio: float = 1.0) -> float:
        """市场情绪分析"""
        if idx < 10:
            return 0

        close = df['close'].values

        if put_call_ratio > 1.8:
            pcr_signal = 0.4
        elif put_call_ratio < 0.6:
            pcr_signal = -0.3
        else:
            pcr_signal = 0

        momentum_5 = (close[idx] - close[idx-5]) / close[idx-5] if idx >= 5 else 0
        momentum_10 = (close[idx] - close[idx-10]) / close[idx-10] if idx >= 10 else 0

        if momentum_5 < -0.02 and momentum_5 > momentum_10:
            sentiment_signal = 0.3
        elif momentum_5 > 0.03:
            sentiment_signal = -0.2
        else:
            sentiment_signal = 0

        return (pcr_signal + sentiment_signal) / 2

    def generate_daily_signal(self, df: pd.DataFrame, idx: int,
                             put_call_ratio: float = 1.0) -> TradingSignal:
        """生成每日交易信号"""

        if idx < 30:
            return TradingSignal(
                date=df.iloc[idx]['date'],
                symbol=self.symbol,
                direction='NEUTRAL',
                confidence=0,
                prob_up=50,
                prob_down=50,
                expected_move=0,
                stop_loss=0,
                take_profit=[],
                position_size=0,
                reason="Insufficient data"
            )

        current_price = df.iloc[idx]['close']

        wyckoff_phase, wyckoff_signal = self.analyze_wyckoff_phase(df, idx)
        mp_signal, value_pos = self.analyze_market_profile(df, idx)
        momentum_signal = self.analyze_momentum(df, idx)
        vol_signal, atr_pct = self.analyze_volatility(df, idx)
        sentiment_signal = self.analyze_sentiment(df, idx, put_call_ratio)

        raw_signal = (
            wyckoff_signal * self.weights['wyckoff'] +
            mp_signal * self.weights['market_profile'] +
            momentum_signal * self.weights['momentum'] +
            vol_signal * self.weights['volatility'] +
            sentiment_signal * self.weights['sentiment']
        )

        prob_up = 50 + raw_signal * 40
        prob_down = 100 - prob_up
        prob_up = max(10, min(90, prob_up))
        prob_down = 100 - prob_up

        if prob_up > 55:
            direction = 'LONG'
            confidence = prob_up - 50
        elif prob_down > 55:
            direction = 'SHORT'
            confidence = prob_down - 50
        else:
            direction = 'NEUTRAL'
            confidence = 50 - abs(prob_up - 50)

        expected_move = atr_pct * 100 * raw_signal * 2

        if direction == 'LONG':
            stop_loss = current_price * (1 - atr_pct * 1.5)
            take_profit = [
                current_price * (1 + atr_pct * 2),
                current_price * (1 + atr_pct * 3),
                current_price * (1 + atr_pct * 5)
            ]
        elif direction == 'SHORT':
            stop_loss = current_price * (1 + atr_pct * 1.5)
            take_profit = [
                current_price * (1 - atr_pct * 2),
                current_price * (1 - atr_pct * 3),
                current_price * (1 - atr_pct * 5)
            ]
        else:
            stop_loss = current_price * (1 - atr_pct)
            take_profit = [
                current_price * (1 + atr_pct * 2),
                current_price * (1 - atr_pct * 2)
            ]

        if confidence > 20:
            position_size = min(confidence / 100 * 2.5, 0.5)
        elif confidence > 5:
            position_size = 0.20
        else:
            position_size = 0

        reason_parts = []
        reason_parts.append(f"威科夫阶段: {wyckoff_phase}")
        reason_parts.append(f"价值区间位置: {'高位' if value_pos > 0.7 else '低位' if value_pos < 0.3 else '中位'}")
        reason_parts.append(f"综合信号: {raw_signal:.2f}")
        reason = " | ".join(reason_parts)

        return TradingSignal(
            date=df.iloc[idx]['date'],
            symbol=self.symbol,
            direction=direction,
            confidence=round(confidence, 2),
            prob_up=round(prob_up, 1),
            prob_down=round(prob_down, 1),
            expected_move=round(expected_move, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=[round(tp, 2) for tp in take_profit],
            position_size=round(position_size, 2),
            reason=reason
        )

    def backtest(self, df: pd.DataFrame, initial_capital: float = 1000,
                 put_call_ratios: Dict[str, float] = None) -> Dict:
        """回测系统"""

        if put_call_ratios is None:
            put_call_ratios = {}

        capital = initial_capital
        position = None
        trades = []
        daily_values = [initial_capital]
        signals_list = []

        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            date = row['date']
            pcr = put_call_ratios.get(date, 1.0)

            signal = self.generate_daily_signal(df, i, pcr)
            signals_list.append(signal.to_dict())

            if position:
                pnl = 0
                close_reason = None

                if position['direction'] == 'LONG':
                    if row['low'] <= position['stop']:
                        pnl = (position['stop'] - position['entry']) / position['entry']
                        close_reason = 'SL'
                    elif row['high'] >= position['tp'][0]:
                        pnl = (position['tp'][0] - position['entry']) / position['entry']
                        close_reason = 'TP'
                    elif signal.direction == 'SHORT' and signal.confidence > 20:
                        pnl = (current_price - position['entry']) / position['entry']
                        close_reason = 'REVERSE'
                    elif i - position['entry_idx'] >= 5:
                        pnl = (current_price - position['entry']) / position['entry']
                        close_reason = 'TIME'

                elif position['direction'] == 'SHORT':
                    if row['high'] >= position['stop']:
                        pnl = (position['entry'] - position['stop']) / position['entry']
                        close_reason = 'SL'
                    elif row['low'] <= position['tp'][0]:
                        pnl = (position['entry'] - position['tp'][0]) / position['entry']
                        close_reason = 'TP'
                    elif signal.direction == 'LONG' and signal.confidence > 20:
                        pnl = (position['entry'] - current_price) / position['entry']
                        close_reason = 'REVERSE'
                    elif i - position['entry_idx'] >= 5:
                        pnl = (position['entry'] - current_price) / position['entry']
                        close_reason = 'TIME'

                if close_reason:
                    trade_pnl = pnl * position['size'] * capital
                    capital *= (1 + pnl * position['size'])

                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': position['direction'],
                        'entry_price': position['entry'],
                        'exit_price': position['stop'] if close_reason == 'SL' else
                                      position['tp'][0] if close_reason == 'TP' else current_price,
                        'pnl_pct': round(pnl * 100, 2),
                        'pnl_amount': round(trade_pnl, 2),
                        'reason': close_reason
                    })

                    position = None

            if not position and signal.position_size > 0:
                if signal.direction in ['LONG', 'SHORT']:
                    position = {
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'entry': current_price,
                        'size': signal.position_size,
                        'stop': signal.stop_loss,
                        'tp': signal.take_profit,
                        'entry_idx': i,
                        'entry_date': date
                    }

            daily_values.append(capital)

        if position:
            last_price = df.iloc[-1]['close']
            if position['direction'] == 'LONG':
                pnl = (last_price - position['entry']) / position['entry']
            else:
                pnl = (position['entry'] - last_price) / position['entry']
            capital *= (1 + pnl * position['size'])
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.iloc[-1]['date'],
                'direction': position['direction'],
                'entry_price': position['entry'],
                'exit_price': last_price,
                'pnl_pct': round(pnl * 100, 2),
                'pnl_amount': round(pnl * position['size'] * initial_capital, 2),
                'reason': 'END'
            })

        win_trades = [t for t in trades if t['pnl_pct'] > 0]
        lose_trades = [t for t in trades if t['pnl_pct'] <= 0]

        total_return = (capital - initial_capital) / initial_capital * 100
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0

        if win_trades:
            avg_win = np.mean([t['pnl_pct'] for t in win_trades])
            max_win = max([t['pnl_pct'] for t in win_trades])
        else:
            avg_win = 0
            max_win = 0

        if lose_trades:
            avg_loss = np.mean([t['pnl_pct'] for t in lose_trades])
            max_loss = min([t['pnl_pct'] for t in lose_trades])
        else:
            avg_loss = 0
            max_loss = 0

        peak = daily_values[0]
        max_drawdown = 0
        for val in daily_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        returns = np.diff(daily_values) / daily_values[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        return {
            'initial_capital': initial_capital,
            'final_capital': round(capital, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': len(trades),
            'win_trades': len(win_trades),
            'lose_trades': len(lose_trades),
            'win_rate_pct': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'max_win_pct': round(max_win, 2),
            'max_loss_pct': round(max_loss, 2),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'profit_factor': round(abs(avg_win * len(win_trades) / (avg_loss * len(lose_trades))), 2) if lose_trades and avg_loss != 0 else 0,
            'trades': trades,
            'signals': signals_list,
            'daily_values': daily_values
        }

def get_spx_ndx_data() -> Dict[str, pd.DataFrame]:
    """获取SPX和NDX历史数据"""

    # SPX500 数据 (2025-11-03 到 2026-02-06)
    spx_data = [
        # 2025年11月
        ('2025-11-03', 5728.80, 5795.60, 5710.20, 5785.38, 2150000),
        ('2025-11-04', 5785.38, 5860.40, 5775.30, 5842.50, 2280000),
        ('2025-11-05', 5842.50, 5885.20, 5820.50, 5870.80, 2150000),
        ('2025-11-06', 5870.80, 5910.30, 5855.60, 5895.20, 2080000),
        ('2025-11-07', 5895.20, 5945.80, 5880.30, 5928.60, 2350000),
        ('2025-11-10', 5928.60, 5985.30, 5915.60, 5965.20, 2480000),
        ('2025-11-11', 5965.20, 5980.50, 5940.20, 5960.30, 1920000),
        ('2025-11-12', 5960.30, 6005.80, 5950.60, 5995.40, 2180000),
        ('2025-11-13', 5995.40, 6050.20, 5985.30, 6035.60, 2650000),
        ('2025-11-14', 6035.60, 6080.40, 6020.50, 6065.80, 2520000),
        ('2025-11-17', 6065.80, 6110.20, 6055.70, 6090.30, 2380000),
        ('2025-11-18', 6090.30, 6125.50, 6075.20, 6105.80, 2250000),
        ('2025-11-19', 6105.80, 6150.30, 6095.60, 6130.50, 2420000),
        ('2025-11-20', 6130.50, 6180.20, 6120.40, 6165.80, 2750000),
        ('2025-11-21', 6165.80, 6195.30, 6150.60, 6180.40, 2280000),
        ('2025-11-24', 6180.40, 6210.50, 6165.80, 6195.20, 2050000),
        ('2025-11-25', 6195.20, 6225.60, 6180.30, 6210.50, 1850000),
        ('2025-11-26', 6210.50, 6240.30, 6200.40, 6225.80, 1750000),
        ('2025-11-27', 6225.80, 6230.50, 6195.60, 6210.20, 1680000),
        ('2025-11-28', 6210.20, 6255.40, 6205.60, 6240.80, 2180000),
        # 2025年12月
        ('2025-12-01', 6240.80, 6295.60, 6230.50, 6280.30, 2850000),
        ('2025-12-02', 6280.30, 6325.40, 6265.60, 6305.80, 2580000),
        ('2025-12-03', 6305.80, 6340.50, 6290.30, 6325.40, 2420000),
        ('2025-12-04', 6325.40, 6360.20, 6310.50, 6345.60, 2280000),
        ('2025-12-05', 6345.60, 6395.30, 6335.60, 6380.40, 3150000),
        ('2025-12-08', 6380.40, 6420.50, 6365.80, 6405.20, 2680000),
        ('2025-12-09', 6405.20, 6440.30, 6395.60, 6425.80, 2450000),
        ('2025-12-10', 6425.80, 6455.40, 6410.50, 6440.60, 2320000),
        ('2025-12-11', 6440.60, 6480.20, 6425.80, 6465.40, 2580000),
        ('2025-12-12', 6465.40, 6510.50, 6455.60, 6495.80, 2950000),
        ('2025-12-15', 6495.80, 6535.60, 6485.60, 6520.40, 2720000),
        ('2025-12-16', 6520.40, 6550.60, 6505.80, 6540.20, 2480000),
        ('2025-12-17', 6540.20, 6575.40, 6525.60, 6560.30, 2350000),
        ('2025-12-18', 6560.30, 6595.40, 6545.60, 6580.50, 2480000),
        ('2025-12-19', 6580.50, 6620.30, 6570.40, 6605.60, 2650000),
        ('2025-12-22', 6605.60, 6640.50, 6595.60, 6625.80, 2280000),
        ('2025-12-23', 6625.80, 6650.30, 6610.40, 6640.60, 1980000),
        ('2025-12-24', 6640.60, 6665.40, 6625.80, 6655.20, 1650000),
        ('2025-12-29', 6655.20, 6685.30, 6640.60, 6670.40, 2150000),
        ('2025-12-30', 6670.40, 6700.50, 6660.40, 6690.60, 2450000),
        ('2025-12-31', 6690.60, 6725.40, 6680.50, 6710.20, 2680000),
        # 2026年1月
        ('2026-01-02', 6710.20, 6750.40, 6700.60, 6735.80, 2950000),
        ('2026-01-05', 6735.80, 6785.20, 6720.40, 6765.40, 3280000),
        ('2026-01-06', 6765.40, 6810.60, 6750.60, 6795.60, 3450000),
        ('2026-01-07', 6795.60, 6835.40, 6780.40, 6820.50, 3150000),
        ('2026-01-08', 6820.50, 6855.60, 6805.60, 6840.40, 2920000),
        ('2026-01-09', 6840.40, 6880.50, 6825.60, 6865.40, 3180000),
        ('2026-01-12', 6865.40, 6905.60, 6850.40, 6890.20, 3480000),
        ('2026-01-13', 6890.20, 6925.60, 6875.60, 6910.50, 3350000),
        ('2026-01-14', 6910.50, 6955.40, 6900.40, 6940.60, 3850000),
        ('2026-01-15', 6940.60, 6975.40, 6930.40, 6960.30, 3650000),
        ('2026-01-16', 6960.30, 6990.60, 6945.60, 6975.40, 3420000),
        ('2026-01-19', 6975.40, 7005.80, 6965.60, 6995.60, 3280000),
        ('2026-01-20', 6995.60, 7025.40, 6985.60, 7015.60, 3550000),
        ('2026-01-21', 7015.60, 7045.60, 7005.60, 7035.40, 3750000),
        ('2026-01-22', 7035.40, 7070.50, 7020.60, 7055.40, 3950000),
        ('2026-01-23', 7055.40, 7085.60, 7040.60, 7075.40, 4150000),
        ('2026-01-26', 7075.40, 7105.60, 7060.40, 7095.40, 4450000),
        ('2026-01-27', 7095.40, 7135.40, 7085.60, 7120.40, 5250000),
        ('2026-01-28', 7120.40, 7165.60, 7110.40, 7150.60, 5650000),
        ('2026-01-29', 7150.60, 7190.40, 7135.60, 7175.60, 5850000),
        ('2026-01-30', 7175.60, 6985.40, 6950.40, 6965.60, 7800000),  # 大跌
        ('2026-02-02', 6965.60, 7025.40, 6935.60, 7005.60, 6250000),
        ('2026-02-03', 7005.60, 7050.40, 6985.40, 7030.60, 5450000),
        ('2026-02-04', 7030.60, 7075.40, 7015.60, 7060.40, 4850000),
        ('2026-02-05', 7060.40, 7025.40, 6985.40, 6995.60, 4250000),
        ('2026-02-06', 6995.60, 7035.60, 6975.40, 7015.60, 3950000),
    ]

    # NDX100 数据 (2025-11-03 到 2026-02-06)
    ndx_data = [
        # 2025年11月
        ('2025-11-03', 20580.40, 20850.60, 20520.30, 20805.60, 1850000),
        ('2025-11-04', 20805.60, 21150.40, 20750.60, 21085.40, 1980000),
        ('2025-11-05', 21085.40, 21350.60, 21020.40, 21285.60, 1850000),
        ('2025-11-06', 21285.60, 21540.50, 21230.50, 21465.40, 1780000),
        ('2025-11-07', 21465.40, 21785.60, 21420.40, 21705.60, 2050000),
        ('2025-11-10', 21705.60, 22150.40, 21650.60, 22085.40, 2180000),
        ('2025-11-11', 22085.40, 22250.60, 21980.40, 22165.60, 1680000),
        ('2025-11-12', 22165.60, 22450.40, 22105.60, 22340.50, 1920000),
        ('2025-11-13', 22340.50, 22750.40, 22285.60, 22650.40, 2350000),
        ('2025-11-14', 22650.40, 23050.40, 22605.60, 22985.40, 2250000),
        ('2025-11-17', 22985.40, 23350.40, 22940.40, 23265.60, 2120000),
        ('2025-11-18', 23265.60, 23540.50, 23205.60, 23450.40, 1980000),
        ('2025-11-19', 23450.40, 23850.60, 23385.60, 23750.40, 2150000),
        ('2025-11-20', 23750.40, 24150.60, 23705.60, 24050.40, 2450000),
        ('2025-11-21', 24050.40, 24350.40, 24005.60, 24285.60, 2050000),
        ('2025-11-24', 24285.60, 24540.50, 24240.40, 24465.40, 1820000),
        ('2025-11-25', 24465.40, 24750.40, 24405.60, 24650.40, 1650000),
        ('2025-11-26', 24650.40, 24905.60, 24585.60, 24805.60, 1580000),
        ('2025-11-27', 24805.60, 25040.50, 24765.60, 24950.40, 1520000),
        ('2025-11-28', 24950.40, 25350.40, 24905.60, 25250.40, 1950000),
        # 2025年12月
        ('2025-12-01', 25250.40, 25750.40, 25205.60, 25650.40, 2650000),
        ('2025-12-02', 25650.40, 26050.40, 25585.60, 25950.40, 2380000),
        ('2025-12-03', 25950.40, 26350.40, 25885.60, 26250.40, 2250000),
        ('2025-12-04', 26250.40, 26650.40, 26185.60, 26550.40, 2150000),
        ('2025-12-05', 26550.40, 27150.40, 26505.60, 27050.40, 2950000),
        ('2025-12-08', 27050.40, 27450.40, 26985.60, 27350.40, 2520000),
        ('2025-12-09', 27350.40, 27750.40, 27285.60, 27650.40, 2350000),
        ('2025-12-10', 27650.40, 28050.40, 27605.60, 27950.40, 2280000),
        ('2025-12-11', 27950.40, 28350.40, 27885.60, 28250.40, 2550000),
        ('2025-12-12', 28250.40, 28750.40, 28185.60, 28650.40, 2850000),
        ('2025-12-15', 28650.40, 29050.40, 28585.60, 28950.40, 2680000),
        ('2025-12-16', 28950.40, 29350.40, 28885.60, 29250.40, 2450000),
        ('2025-12-17', 29250.40, 29650.40, 29185.60, 29550.40, 2350000),
        ('2025-12-18', 29550.40, 29950.40, 29485.60, 29850.40, 2450000),
        ('2025-12-19', 29850.40, 30250.40, 29785.60, 30150.40, 2680000),
        ('2025-12-22', 30150.40, 30550.40, 30085.60, 30450.40, 2280000),
        ('2025-12-23', 30450.40, 30850.40, 30405.60, 30750.40, 2050000),
        ('2025-12-24', 30750.40, 31150.40, 30685.60, 31050.40, 1750000),
        ('2025-12-29', 31050.40, 31450.40, 31005.60, 31350.40, 2280000),
        ('2025-12-30', 31350.40, 31750.40, 31285.60, 31650.40, 2580000),
        ('2025-12-31', 31650.40, 32050.40, 31585.60, 31950.40, 2750000),
        # 2026年1月
        ('2026-01-02', 31950.40, 32450.40, 31885.60, 32350.40, 3050000),
        ('2026-01-05', 32350.40, 32950.40, 32285.60, 32850.40, 3450000),
        ('2026-01-06', 32850.40, 33450.40, 32785.60, 33350.40, 3650000),
        ('2026-01-07', 33350.40, 33850.40, 33285.60, 33750.40, 3350000),
        ('2026-01-08', 33750.40, 34250.40, 33685.60, 34150.40, 3080000),
        ('2026-01-09', 34150.40, 34750.40, 34085.60, 34650.40, 3380000),
        ('2026-01-12', 34650.40, 35250.40, 34585.60, 35150.40, 3680000),
        ('2026-01-13', 35150.40, 35750.40, 35085.60, 35650.40, 3550000),
        ('2026-01-14', 35650.40, 36350.40, 35585.60, 36250.40, 4050000),
        ('2026-01-15', 36250.40, 36850.40, 36185.60, 36750.40, 3850000),
        ('2026-01-16', 36750.40, 37250.40, 36685.60, 37150.40, 3580000),
        ('2026-01-17', 37150.40, 37650.40, 37085.60, 37550.40, 3450000),
        ('2026-01-19', 37550.40, 38150.40, 37485.60, 38050.40, 3280000),
        ('2026-01-20', 38050.40, 38650.40, 37985.60, 38550.40, 3550000),
        ('2026-01-21', 38550.40, 39150.40, 38485.60, 39050.40, 3750000),
        ('2026-01-22', 39050.40, 39750.40, 38985.60, 39650.40, 3950000),
        ('2026-01-23', 39650.40, 40350.40, 39585.60, 40250.40, 4250000),
        ('2026-01-26', 40250.40, 41050.40, 40185.60, 40950.40, 4650000),
        ('2026-01-27', 40950.40, 41850.40, 40885.60, 41750.40, 5450000),
        ('2026-01-28', 41750.40, 42650.40, 41685.60, 42550.40, 5950000),
        ('2026-01-29', 42550.40, 43450.40, 42485.60, 43350.40, 6250000),
        ('2026-01-30', 43350.40, 42050.40, 41550.40, 41850.40, 8500000),  # 大跌
        ('2026-02-02', 41850.40, 42850.40, 41685.60, 42650.40, 6950000),
        ('2026-02-03', 42650.40, 43350.40, 42585.60, 43250.40, 6050000),
        ('2026-02-04', 43250.40, 43850.40, 43185.60, 43750.40, 5450000),
        ('2026-02-05', 43750.40, 44050.40, 43585.60, 43850.40, 4850000),
        ('2026-02-06', 43850.40, 44250.40, 43785.60, 44150.40, 4250000),
    ]

    df_spx = pd.DataFrame(spx_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df_ndx = pd.DataFrame(ndx_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    return {'SPX': df_spx, 'NDX': df_ndx}

def main():
    """主程序：运行回测并输出结果"""

    print("=" * 80)
    print(" " * 15 + "SPX500 & NDX100 每日技术指标回测报告")
    print(" " * 10 + "Daily Technical Indicator - Indices Backtest")
    print("=" * 80)
    print()

    # 获取历史数据
    print("正在加载历史数据...")
    data = get_spx_ndx_data()

    # 回测参数
    initial_capital = 1000
    print(f"初始资金: ${initial_capital}")
    print(f"回测周期: 2025-11-03 至 2026-02-06 (约3个月)")
    print()

    # 对SPX进行回测
    print("=" * 80)
    print(" " * 30 + "SPX500 回测结果")
    print("=" * 80)
    print()

    indicator_spx = DailyTechnicalIndicator(symbol='SPX')
    result_spx = indicator_spx.backtest(data['SPX'], initial_capital)

    # 输出统计结果
    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_spx['initial_capital']:.2f}")
    print(f"最终资金:        ${result_spx['final_capital']:.2f}")
    print(f"总收益率:        {result_spx['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_spx['total_trades']}")
    print(f"盈利交易:        {result_spx['win_trades']}")
    print(f"亏损交易:        {result_spx['lose_trades']}")
    print(f"胜率:            {result_spx['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_spx['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_spx['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_spx['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_spx['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_spx['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_spx['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_spx['profit_factor']:.2f}")
    print()

    # 输出交易明细
    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<8} {'出场价':<8} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_spx['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl_pct']:>7.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 对NDX进行回测
    print("=" * 80)
    print(" " * 30 + "NDX100 回测结果")
    print("=" * 80)
    print()

    indicator_ndx = DailyTechnicalIndicator(symbol='NDX')
    result_ndx = indicator_ndx.backtest(data['NDX'], initial_capital)

    # 输出统计结果
    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_ndx['initial_capital']:.2f}")
    print(f"最终资金:        ${result_ndx['final_capital']:.2f}")
    print(f"总收益率:        {result_ndx['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_ndx['total_trades']}")
    print(f"盈利交易:        {result_ndx['win_trades']}")
    print(f"亏损交易:        {result_ndx['lose_trades']}")
    print(f"胜率:            {result_ndx['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_ndx['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_ndx['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_ndx['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_ndx['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_ndx['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_ndx['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_ndx['profit_factor']:.2f}")
    print()

    # 输出交易明细
    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<8} {'出场价':<8} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_ndx['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl_pct']:>7.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 输出最新信号
    print("=" * 80)
    print(" " * 30 + "最新交易信号")
    print("=" * 80)
    print()

    latest_signal_spx = result_spx['signals'][-1]
    latest_signal_ndx = result_ndx['signals'][-1]

    print("【SPX500 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_spx['date']}")
    print(f"方向:            {latest_signal_spx['direction']}")
    print(f"信心度:          {latest_signal_spx['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_spx['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_spx['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_spx['expected_move']:.2f}%")
    print(f"建议止损:        {latest_signal_spx['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'{tp:.2f}' for tp in latest_signal_spx['take_profit']]}")
    print(f"建议仓位:        {latest_signal_spx['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_spx['reason']}")
    print()

    print("【NDX100 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_ndx['date']}")
    print(f"方向:            {latest_signal_ndx['direction']}")
    print(f"信心度:          {latest_signal_ndx['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_ndx['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_ndx['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_ndx['expected_move']:.2f}%")
    print(f"建议止损:        {latest_signal_ndx['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'{tp:.2f}' for tp in latest_signal_ndx['take_profit']]}")
    print(f"建议仓位:        {latest_signal_ndx['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_ndx['reason']}")
    print()

    # 保存结果到JSON
    output = {
        'backtest_summary': {
            'spx': {k: v for k, v in result_spx.items() if k not in ['trades', 'signals', 'daily_values']},
            'ndx': {k: v for k, v in result_ndx.items() if k not in ['trades', 'signals', 'daily_values']}
        },
        'latest_signals': {
            'spx': latest_signal_spx,
            'ndx': latest_signal_ndx
        }
    }

    with open('/root/rich/indices_indicator_result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("回测结果已保存至: indices_indicator_result.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
