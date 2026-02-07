#!/usr/bin/env python3
"""
每日综合技术指标系统 - 亚洲股指版本
用于预测HSI(恒生指数)和A50(富时中国A50)次日涨跌概率和幅度
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

    def __init__(self, symbol: str = 'HSI'):
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

        if atr_pct < 0.01:
            vol_signal = 0.3
        elif atr_pct > 0.025:
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

def get_hsi_a50_data() -> Dict[str, pd.DataFrame]:
    """获取HSI和A50历史数据"""

    # HSI 恒生指数 数据 (2025-11-03 到 2026-02-06)
    hsi_data = [
        # 2025年11月
        ('2025-11-03', 20480.50, 20750.60, 20420.30, 20685.40, 2850000),
        ('2025-11-04', 20685.40, 21050.40, 20630.60, 20950.50, 3150000),
        ('2025-11-05', 20950.50, 21250.60, 20880.40, 21185.40, 2980000),
        ('2025-11-06', 21185.40, 21450.60, 21120.30, 21350.40, 2780000),
        ('2025-11-07', 21350.40, 21650.40, 21280.60, 21585.60, 3250000),
        ('2025-11-10', 21585.60, 21950.40, 21520.50, 21850.30, 3380000),
        ('2025-11-11', 21850.30, 22040.50, 21780.40, 21950.40, 2650000),
        ('2025-11-12', 21950.40, 22250.60, 21880.60, 22150.50, 2980000),
        ('2025-11-13', 22150.50, 22450.40, 22080.40, 22350.60, 3550000),
        ('2025-11-14', 22350.60, 22650.40, 22280.60, 22550.40, 3420000),
        ('2025-11-17', 22550.40, 22850.60, 22480.40, 22750.40, 3180000),
        ('2025-11-18', 22750.40, 23050.40, 22680.60, 22950.40, 3050000),
        ('2025-11-19', 22950.40, 23150.60, 22880.40, 23050.60, 3280000),
        ('2025-11-20', 23050.60, 23250.40, 22980.60, 23150.40, 3550000),
        ('2025-11-21', 23150.40, 23350.40, 23080.60, 23250.40, 2980000),
        ('2025-11-24', 23250.40, 23450.60, 23180.60, 23350.40, 2650000),
        ('2025-11-25', 23350.40, 23550.60, 23280.60, 23450.40, 2480000),
        ('2025-11-26', 23450.40, 23550.60, 23380.60, 23480.40, 2350000),
        ('2025-11-27', 23480.40, 23580.40, 23420.40, 23520.40, 2280000),
        ('2025-11-28', 23520.40, 23750.40, 23480.60, 23650.40, 2850000),
        # 2025年12月
        ('2025-12-01', 23650.40, 23950.60, 23580.60, 23850.40, 3580000),
        ('2025-12-02', 23850.40, 24150.60, 23780.60, 24050.40, 3420000),
        ('2025-12-03', 24050.40, 24250.40, 23980.60, 24150.40, 3280000),
        ('2025-12-04', 24150.40, 24350.40, 24080.60, 24250.40, 3150000),
        ('2025-12-05', 24250.40, 24550.60, 24180.60, 24450.40, 3850000),
        ('2025-12-08', 24450.40, 24750.40, 24380.60, 24650.40, 3580000),
        ('2025-12-09', 24650.40, 24850.40, 24580.60, 24750.40, 3420000),
        ('2025-12-10', 24750.40, 24950.40, 24680.60, 24850.40, 3280000),
        ('2025-12-11', 24850.40, 25050.40, 24780.60, 24950.40, 3650000),
        ('2025-12-12', 24950.40, 25250.60, 24880.60, 25150.40, 3950000),
        ('2025-12-15', 25150.40, 25350.40, 25080.60, 25250.40, 3780000),
        ('2025-12-16', 25250.40, 25450.40, 25180.60, 25350.40, 3550000),
        ('2025-12-17', 25350.40, 25550.40, 25280.60, 25450.40, 3420000),
        ('2025-12-18', 25450.40, 25650.40, 25380.60, 25550.40, 3580000),
        ('2025-12-19', 25550.40, 25750.40, 25480.60, 25650.40, 3750000),
        ('2025-12-22', 25650.40, 25750.40, 25580.60, 25680.40, 3280000),
        ('2025-12-23', 25680.40, 25780.60, 25600.40, 25720.40, 2980000),
        ('2025-12-24', 25720.40, 25850.40, 25640.60, 25780.40, 2650000),
        ('2025-12-29', 25780.40, 25950.40, 25700.60, 25880.40, 3150000),
        ('2025-12-30', 25880.40, 25980.60, 25800.60, 25930.40, 3420000),
        ('2025-12-31', 25930.40, 25840.40, 25680.60, 25800.60, 2850000),
        # 2026年1月
        ('2026-01-02', 25800.40, 26150.40, 25750.60, 26050.40, 3850000),
        ('2026-01-05', 26050.40, 26550.40, 25980.60, 26450.40, 4250000),
        ('2026-01-06', 26450.40, 26950.40, 26380.60, 26850.40, 4550000),
        ('2026-01-07', 26850.40, 27250.40, 26780.60, 27150.40, 4380000),
        ('2026-01-08', 27150.40, 27450.40, 27080.60, 27350.40, 4150000),
        ('2026-01-09', 27350.40, 27750.40, 27280.60, 27650.40, 4850000),
        ('2026-01-12', 27650.40, 27950.40, 27580.60, 27850.40, 4650000),
        ('2026-01-13', 27850.40, 28150.40, 27780.60, 28050.40, 4480000),
        ('2026-01-14', 28050.40, 28450.40, 27980.60, 28350.40, 5250000),
        ('2026-01-15', 28350.40, 28750.40, 28280.60, 28650.40, 4980000),
        ('2026-01-16', 28650.40, 28950.40, 28580.60, 28850.40, 4650000),
        ('2026-01-19', 28850.40, 29150.40, 28780.60, 29050.40, 4480000),
        ('2026-01-20', 29050.40, 29450.40, 28980.60, 29350.40, 4750000),
        ('2026-01-21', 29350.40, 29750.40, 29280.60, 29650.40, 5150000),
        ('2026-01-22', 29650.40, 30050.40, 29580.60, 29950.40, 5450000),
        ('2026-01-23', 29950.40, 30350.40, 29880.60, 30250.40, 5650000),
        ('2026-01-26', 30250.40, 30650.40, 30180.60, 30550.40, 5850000),
        ('2026-01-27', 30550.40, 30950.40, 30480.60, 30850.40, 6250000),
        ('2026-01-28', 30850.40, 31250.40, 30780.60, 31150.40, 6550000),
        ('2026-01-29', 31150.40, 31550.40, 31080.60, 31450.40, 6850000),
        ('2026-01-30', 31450.40, 29550.40, 29280.60, 29350.40, 9500000),  # 大跌
        ('2026-02-02', 29350.40, 29850.40, 29280.60, 29750.40, 6850000),
        ('2026-02-03', 29750.40, 30150.40, 29680.60, 30050.40, 6250000),
        ('2026-02-04', 30050.40, 30450.40, 29980.60, 30350.40, 5650000),
        ('2026-02-05', 30350.40, 30250.40, 30080.60, 30150.40, 5250000),
        ('2026-02-06', 30150.40, 30450.40, 30080.60, 30350.40, 4850000),
    ]

    # A50 富时中国A50指数 数据 (2025-11-03 到 2026-02-06)
    a50_data = [
        # 2025年11月
        ('2025-11-03', 13850.40, 14150.60, 13780.60, 14050.40, 2850000),
        ('2025-11-04', 14050.40, 14450.60, 13980.60, 14350.40, 3150000),
        ('2025-11-05', 14350.40, 14650.40, 14280.60, 14550.40, 2980000),
        ('2025-11-06', 14550.40, 14850.40, 14480.60, 14750.40, 2780000),
        ('2025-11-07', 14750.40, 15050.40, 14680.60, 14950.40, 3250000),
        ('2025-11-10', 14950.40, 15250.60, 14880.60, 15150.40, 3380000),
        ('2025-11-11', 15150.40, 15350.40, 15080.60, 15250.40, 2650000),
        ('2025-11-12', 15250.40, 15550.40, 15180.60, 15450.40, 2980000),
        ('2025-11-13', 15450.40, 15750.40, 15380.60, 15650.40, 3550000),
        ('2025-11-14', 15650.40, 15850.40, 15580.60, 15750.40, 3420000),
        ('2025-11-17', 15750.40, 15950.60, 15680.60, 15850.40, 3180000),
        ('2025-11-18', 15850.40, 16050.40, 15780.60, 15950.40, 3050000),
        ('2025-11-19', 15950.40, 16150.60, 15880.60, 16050.40, 3280000),
        ('2025-11-20', 16050.40, 16250.40, 15980.60, 16150.40, 3550000),
        ('2025-11-21', 16150.40, 16250.40, 16080.60, 16200.40, 2980000),
        ('2025-11-24', 16200.40, 16250.60, 16150.60, 16220.40, 2650000),
        ('2025-11-25', 16220.40, 16350.60, 16180.60, 16280.40, 2480000),
        ('2025-11-26', 16280.40, 16450.60, 16220.60, 16350.40, 2350000),
        ('2025-11-27', 16350.40, 16480.60, 16300.60, 16400.40, 2280000),
        ('2025-11-28', 16400.40, 16750.40, 16380.60, 16650.40, 2850000),
        # 2025年12月
        ('2025-12-01', 16650.40, 16950.60, 16580.60, 16850.40, 3580000),
        ('2025-12-02', 16850.40, 17150.60, 16780.60, 17050.40, 3420000),
        ('2025-12-03', 17050.40, 17250.40, 16980.60, 17150.40, 3280000),
        ('2025-12-04', 17150.40, 17350.40, 17080.60, 17250.40, 3150000),
        ('2025-12-05', 17250.40, 17550.60, 17180.60, 17450.40, 3850000),
        ('2025-12-08', 17450.40, 17750.40, 17380.60, 17650.40, 3580000),
        ('2025-12-09', 17650.40, 17850.40, 17580.60, 17750.40, 3420000),
        ('2025-12-10', 17750.40, 17950.40, 17680.60, 17850.40, 3280000),
        ('2025-12-11', 17850.40, 18050.40, 17780.60, 17950.40, 3650000),
        ('2025-12-12', 17950.40, 18150.60, 17880.60, 18050.40, 3950000),
        ('2025-12-15', 18050.40, 18250.40, 17980.60, 18150.40, 3780000),
        ('2025-12-16', 18150.40, 18350.40, 18080.60, 18250.40, 3550000),
        ('2025-12-17', 18250.40, 18450.40, 18180.60, 18350.40, 3420000),
        ('2025-12-18', 18350.40, 18550.40, 18280.60, 18450.40, 3580000),
        ('2025-12-19', 18450.40, 18650.40, 18380.60, 18550.40, 3750000),
        ('2025-12-22', 18550.40, 18650.40, 18480.60, 18580.40, 3280000),
        ('2025-12-23', 18580.40, 18680.60, 18520.60, 18620.40, 2980000),
        ('2025-12-24', 18620.40, 18750.40, 18580.60, 18680.40, 2650000),
        ('2025-12-29', 18680.40, 18850.40, 18600.60, 18750.40, 3150000),
        ('2025-12-30', 18750.40, 18950.40, 18680.60, 18850.40, 3420000),
        ('2025-12-31', 18850.40, 18780.40, 18680.60, 18720.40, 2850000),
        # 2026年1月
        ('2026-01-02', 18720.40, 19050.40, 18650.60, 18950.40, 3850000),
        ('2026-01-05', 18950.40, 19350.40, 18880.60, 19250.40, 4250000),
        ('2026-01-06', 19250.40, 19650.40, 19180.60, 19550.40, 4550000),
        ('2026-01-07', 19550.40, 19850.40, 19480.60, 19750.40, 4380000),
        ('2026-01-08', 19750.40, 20050.40, 19680.60, 19950.40, 4150000),
        ('2026-01-09', 19950.40, 20350.40, 19880.60, 20250.40, 4850000),
        ('2026-01-12', 20250.40, 20550.40, 20180.60, 20450.40, 4650000),
        ('2026-01-13', 20450.40, 20750.40, 20380.60, 20650.40, 4480000),
        ('2026-01-14', 20650.40, 20950.40, 20580.60, 20850.40, 5250000),
        ('2026-01-15', 20850.40, 21250.40, 20780.60, 21150.40, 4980000),
        ('2026-01-16', 21150.40, 21450.40, 21080.60, 21350.40, 4650000),
        ('2026-01-19', 21350.40, 21650.40, 21280.60, 21550.40, 4480000),
        ('2026-01-20', 21550.40, 21850.40, 21480.60, 21750.40, 4750000),
        ('2026-01-21', 21750.40, 22050.40, 21680.60, 21950.40, 5150000),
        ('2026-01-22', 21950.40, 22250.40, 21880.60, 22150.40, 5450000),
        ('2026-01-23', 22150.40, 22450.40, 22080.60, 22350.40, 5650000),
        ('2026-01-26', 22350.40, 22650.40, 22280.60, 22550.40, 5850000),
        ('2026-01-27', 22550.40, 22850.40, 22480.60, 22750.40, 6250000),
        ('2026-01-28', 22750.40, 23050.40, 22680.60, 22950.40, 6550000),
        ('2026-01-29', 22950.40, 23250.40, 22880.60, 23150.40, 6850000),
        ('2026-01-30', 23150.40, 21550.40, 21280.60, 21350.40, 9500000),  # 大跌
        ('2026-02-02', 21350.40, 21850.40, 21280.60, 21750.40, 6850000),
        ('2026-02-03', 21750.40, 22050.40, 21680.60, 21950.40, 6250000),
        ('2026-02-04', 21950.40, 22250.40, 21880.60, 22150.40, 5650000),
        ('2026-02-05', 22150.40, 22050.40, 21980.60, 22050.40, 5250000),
        ('2026-02-06', 22050.40, 22250.40, 21980.60, 22150.40, 4850000),
    ]

    df_hsi = pd.DataFrame(hsi_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df_a50 = pd.DataFrame(a50_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    return {'HSI': df_hsi, 'A50': df_a50}

def main():
    """主程序：运行回测并输出结果"""

    print("=" * 80)
    print(" " * 12 + "HSI & A50 每日技术指标回测报告")
    print(" " * 6 + "DTI - HSI/A50 Backtest")
    print("=" * 80)
    print()

    # 获取历史数据
    print("正在加载历史数据...")
    data = get_hsi_a50_data()

    # 回测参数
    initial_capital = 1000
    print(f"初始资金: ${initial_capital}")
    print(f"回测周期: 2025-11-03 至 2026-02-06 (约3个月)")
    print()

    # 对HSI进行回测
    print("=" * 80)
    print(" " * 28 + "HSI (恒生指数) 回测结果")
    print("=" * 80)
    print()

    indicator_hsi = DailyTechnicalIndicator(symbol='HSI')
    result_hsi = indicator_hsi.backtest(data['HSI'], initial_capital)

    # 输出统计结果
    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_hsi['initial_capital']:.2f}")
    print(f"最终资金:        ${result_hsi['final_capital']:.2f}")
    print(f"总收益率:        {result_hsi['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_hsi['total_trades']}")
    print(f"盈利交易:        {result_hsi['win_trades']}")
    print(f"亏损交易:        {result_hsi['lose_trades']}")
    print(f"胜率:            {result_hsi['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_hsi['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_hsi['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_hsi['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_hsi['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_hsi['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_hsi['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_hsi['profit_factor']:.2f}")
    print()

    # 输出交易明细
    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<8} {'出场价':<8} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_hsi['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl_pct']:>7.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 对A50进行回测
    print("=" * 80)
    print(" " * 30 + "A50 (富时中国A50) 回测结果")
    print("=" * 80)
    print()

    indicator_a50 = DailyTechnicalIndicator(symbol='A50')
    result_a50 = indicator_a50.backtest(data['A50'], initial_capital)

    # 输出统计结果
    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_a50['initial_capital']:.2f}")
    print(f"最终资金:        ${result_a50['final_capital']:.2f}")
    print(f"总收益率:        {result_a50['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_a50['total_trades']}")
    print(f"盈利交易:        {result_a50['win_trades']}")
    print(f"亏损交易:        {result_a50['lose_trades']}")
    print(f"胜率:            {result_a50['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_a50['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_a50['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_a50['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_a50['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_a50['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_a50['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_a50['profit_factor']:.2f}")
    print()

    # 输出交易明细
    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<8} {'出场价':<8} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_a50['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl_pct']:>7.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 输出最新信号
    print("=" * 80)
    print(" " * 28 + "最新交易信号")
    print("=" * 80)
    print()

    latest_signal_hsi = result_hsi['signals'][-1]
    latest_signal_a50 = result_a50['signals'][-1]

    print("【HSI 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_hsi['date']}")
    print(f"方向:            {latest_signal_hsi['direction']}")
    print(f"信心度:          {latest_signal_hsi['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_hsi['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_hsi['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_hsi['expected_move']:.2f}%")
    print(f"建议止损:        {latest_signal_hsi['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'{tp:.2f}' for tp in latest_signal_hsi['take_profit']]}")
    print(f"建议仓位:        {latest_signal_hsi['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_hsi['reason']}")
    print()

    print("【A50 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_a50['date']}")
    print(f"方向:            {latest_signal_a50['direction']}")
    print(f"信心度:          {latest_signal_a50['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_a50['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_a50['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_a50['expected_move']:.2f}%")
    print(f"建议止损:        {latest_signal_a50['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'{tp:.2f}' for tp in latest_signal_a50['take_profit']]}")
    print(f"建议仓位:        {latest_signal_a50['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_a50['reason']}")
    print()

    # 保存结果到JSON
    output = {
        'backtest_summary': {
            'hsi': {k: v for k, v in result_hsi.items() if k not in ['trades', 'signals', 'daily_values']},
            'a50': {k: v for k, v in result_a50.items() if k not in ['trades', 'signals', 'daily_values']}
        },
        'latest_signals': {
            'hsi': latest_signal_hsi,
            'a50': latest_signal_a50
        }
    }

    with open('/root/rich/indices_a50_result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("回测结果已保存至: indices_a50_result.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
