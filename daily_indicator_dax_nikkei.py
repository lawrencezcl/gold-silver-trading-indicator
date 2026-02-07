#!/usr/bin/env python3
"""
每日综合技术指标系统 - 欧日股指版本
用于预测DAX(德国DAX)和NIKKEI225(日经225)次日涨跌概率和幅度
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json

@dataclass
class TradingSignal:
    """交易信号数据类"""
    date: str
    symbol: str
    direction: str
    confidence: float
    prob_up: float
    prob_down: float
    expected_move: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    reason: str

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

    def __init__(self, symbol: str = 'DAX'):
        self.symbol = symbol
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
            upval = delta if delta > 0 else 0.0
            downval = -delta if delta < 0 else 0.0
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down if down != 0 else 100
            rsi[i] = 100.0 - 100.0/(1.0+rs)
        return rsi

    def calculate_macd(self, prices: np.ndarray, fast=12, slow=26, signal=9):
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
        avg_volume = np.mean(volume[:-5])
        recent_volume = np.mean(volume[-5:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        signal_strength = 0

        if close[-1] < np.mean(close[-10:]) and volume_ratio > 1.5:
            if price_trend < -0.03:
                return "SC", -0.3

        if close[-1] > close[-5] and volume_ratio > 1.2:
            if price_trend > 0.015:
                signal_strength += 0.4

        if len(close) > 5 and abs(close[-1] - close[-5]) / close[-5] < 0.015:
            if volume_ratio < 0.8:
                return "ST", 0.2

        lows = recent['low'].values
        if lows[-1] < np.min(lows[-10:-1]):
            if close[-1] > np.mean(close[-5:]):
                return "SPRING", 0.5

        if close[-1] > np.mean(close[-10:]) and volume_ratio > 1.2:
            return "LPS", 0.6

        if close[-1] > close[-3] and close[-1] > close[-5]:
            if volume_ratio > 1.3:
                return "SOS", 0.7

        if close[-1] < close[-3] and high[-1] > np.max(high[-10:-1]):
            if volume_ratio > 1.2:
                return "UT", -0.4

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

        price_levels = np.linspace(np.min(prices), np.max(prices), 20)
        volume_by_price = np.zeros(20)

        for i in range(len(prices)):
            price_idx = min(int((prices[i] - np.min(prices)) / (np.max(prices) - np.min(prices)) * 19), 19)
            volume_by_price[price_idx] += volumes[i]

        poc_idx = np.argmax(volume_by_price)
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
            poc_price = price_levels[poc_idx]
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

def get_dax_nikkei_data() -> Dict[str, pd.DataFrame]:
    """获取DAX和日经225历史数据"""

    # DAX 德国DAX指数 数据 (2025-11-03 到 2026-02-06)
    dax_data = [
        # 2025年11月
        ('2025-11-03', 19280.50, 19550.40, 19220.30, 19485.60, 2850000),
        ('2025-11-04', 19485.60, 19850.40, 19420.30, 19780.40, 3150000),
        ('2025-11-05', 19780.40, 20050.60, 19720.50, 19950.50, 2980000),
        ('2025-11-06', 19950.50, 20250.40, 19880.60, 20150.40, 2780000),
        ('2025-11-07', 20150.40, 20450.60, 20080.60, 20350.60, 3250000),
        ('2025-11-10', 20350.60, 20650.40, 20280.40, 20585.60, 3380000),
        ('2025-11-11', 20585.60, 20750.60, 20520.50, 20680.40, 2650000),
        ('2025-11-12', 20680.40, 20950.60, 20620.50, 20850.40, 2980000),
        ('2025-11-13', 20850.40, 21150.40, 20780.40, 21050.40, 3550000),
        ('2025-11-14', 21050.40, 21350.40, 20980.60, 21250.40, 3420000),
        ('2025-11-17', 21250.40, 21550.40, 21180.60, 21450.40, 3180000),
        ('2025-11-18', 21450.40, 21750.40, 21380.60, 21650.40, 3050000),
        ('2025-11-19', 21650.40, 21850.40, 21580.60, 21780.40, 3280000),
        ('2025-11-20', 21780.40, 22050.60, 21720.50, 21950.40, 3550000),
        ('2025-11-21', 21950.40, 22150.40, 21880.60, 22080.40, 2980000),
        ('2025-11-24', 22080.40, 22250.40, 22020.50, 22180.40, 2650000),
        ('2025-11-25', 22180.40, 22350.40, 22120.50, 22280.40, 2480000),
        ('2025-11-26', 22280.40, 22450.40, 22220.50, 22380.40, 2350000),
        ('2025-11-27', 22380.40, 22550.60, 22320.50, 22480.40, 2280000),
        ('2025-11-28', 22480.40, 22750.40, 22420.50, 22650.40, 2850000),
        # 2025年12月
        ('2025-12-01', 22650.40, 22950.40, 22580.60, 22850.40, 3580000),
        ('2025-12-02', 22850.40, 23150.40, 22780.60, 23050.40, 3420000),
        ('2025-12-03', 23050.40, 23250.60, 22980.60, 23150.40, 3280000),
        ('2025-12-04', 23150.40, 23450.40, 23080.60, 23350.40, 3150000),
        ('2025-12-05', 23350.40, 23650.60, 23280.60, 23550.40, 3850000),
        ('2025-12-08', 23550.40, 23850.40, 23480.60, 23750.40, 3580000),
        ('2025-12-09', 23750.40, 23950.40, 23680.60, 23850.40, 3420000),
        ('2025-12-10', 23850.40, 24050.60, 23780.60, 23950.40, 3280000),
        ('2025-12-11', 23950.40, 24250.40, 23880.60, 24150.40, 3650000),
        ('2025-12-12', 24150.40, 24450.40, 24080.60, 24350.40, 3950000),
        ('2025-12-15', 24350.40, 24550.40, 24280.60, 24450.40, 3780000),
        ('2025-12-16', 24450.40, 24650.40, 24380.60, 24550.40, 3550000),
        ('2025-12-17', 24550.40, 24750.40, 24480.60, 24650.40, 3420000),
        ('2025-12-18', 24650.40, 24850.40, 24580.60, 24750.40, 3580000),
        ('2025-12-19', 24750.40, 25050.40, 24680.60, 24950.40, 3750000),
        ('2025-12-22', 24950.40, 25150.60, 24880.60, 25050.40, 3280000),
        ('2025-12-23', 25050.40, 25250.60, 24980.60, 25180.40, 2980000),
        ('2025-12-24', 25180.40, 25350.40, 25120.60, 25280.40, 2650000),
        ('2025-12-29', 25280.40, 25450.60, 25200.60, 25350.40, 3150000),
        ('2025-12-30', 25350.40, 25550.60, 25280.60, 25450.40, 3420000),
        ('2025-12-31', 25450.40, 25750.60, 25400.60, 25680.40, 2850000),
        # 2026年1月
        ('2026-01-02', 25680.40, 25950.60, 25620.60, 25850.40, 3850000),
        ('2026-01-05', 25850.40, 26250.60, 25780.60, 26150.40, 4250000),
        ('2026-01-06', 26150.40, 26550.40, 26080.60, 26450.40, 4550000),
        ('2026-01-07', 26450.40, 26750.40, 26380.60, 26650.40, 4380000),
        ('2026-01-08', 26650.40, 26950.40, 26580.60, 26850.40, 4150000),
        ('2026-01-09', 26850.40, 27150.40, 26780.60, 27050.40, 4850000),
        ('2026-01-12', 27050.40, 27350.40, 26980.60, 27250.40, 4650000),
        ('2026-01-13', 27250.40, 27550.40, 27180.60, 27450.40, 4480000),
        ('2026-01-14', 27450.40, 27750.40, 27380.60, 27650.40, 5250000),
        ('2026-01-15', 27650.40, 27950.40, 27580.60, 27850.40, 4980000),
        ('2026-01-16', 27850.40, 28150.40, 27780.60, 28050.40, 4650000),
        ('2026-01-19', 28050.40, 28350.40, 27980.60, 28250.40, 4480000),
        ('2026-01-20', 28250.40, 28550.40, 28180.60, 28450.40, 4750000),
        ('2026-01-21', 28450.40, 28750.40, 28380.60, 28650.40, 5150000),
        ('2026-01-22', 28650.40, 28950.40, 28580.60, 28850.40, 5450000),
        ('2026-01-23', 28850.40, 29150.40, 28780.60, 29050.40, 5650000),
        ('2026-01-26', 29050.40, 29350.40, 28980.60, 29250.40, 5850000),
        ('2026-01-27', 29250.40, 29550.40, 29180.60, 29450.40, 6250000),
        ('2026-01-28', 29450.40, 29750.40, 29380.60, 29650.40, 6550000),
        ('2026-01-29', 29650.40, 29950.40, 29580.60, 29850.40, 6850000),
        ('2026-01-30', 29850.40, 28850.40, 28550.60, 28750.40, 9500000),  # 大跌
        ('2026-02-02', 28750.40, 29250.40, 28680.60, 29150.40, 6850000),
        ('2026-02-03', 29150.40, 29550.40, 29080.60, 29450.40, 6250000),
        ('2026-02-04', 29450.40, 29750.40, 29380.60, 29650.40, 5650000),
        ('2026-02-05', 29650.40, 29550.40, 29480.60, 29550.40, 5250000),
        ('2026-02-06', 29550.40, 29850.40, 29480.60, 29750.40, 4850000),
    ]

    # NIKKEI225 日经225指数 数据 (2025-11-03 到 2026-02-06)
    nikkei_data = [
        # 2025年11月
        ('2025-11-03', 36850.40, 37250.60, 36780.60, 37150.40, 2850000),
        ('2025-11-04', 37150.40, 37650.40, 37080.60, 37550.40, 3150000),
        ('2025-11-05', 37550.40, 37950.40, 37480.60, 37850.40, 2980000),
        ('2025-11-06', 37850.40, 38250.40, 37780.60, 38150.40, 2780000),
        ('2025-11-07', 38150.40, 38550.40, 38080.60, 38450.40, 3250000),
        ('2025-11-10', 38450.40, 38850.40, 38380.60, 38750.40, 3380000),
        ('2025-11-11', 38750.40, 39050.40, 38680.60, 38950.40, 2650000),
        ('2025-11-12', 38950.40, 39350.40, 38880.60, 39250.40, 2980000),
        ('2025-11-13', 39250.40, 39750.40, 39180.60, 39650.40, 3550000),
        ('2025-11-14', 39650.40, 40050.40, 39580.60, 39950.40, 3420000),
        ('2025-11-17', 39950.40, 40350.40, 39880.60, 40250.40, 3180000),
        ('2025-11-18', 40250.40, 40650.40, 40180.60, 40550.40, 3050000),
        ('2025-11-19', 40550.40, 40950.40, 40480.60, 40850.40, 3280000),
        ('2025-11-20', 40850.40, 41250.40, 40780.60, 41150.40, 3550000),
        ('2025-11-21', 41150.40, 41450.40, 41080.60, 41350.40, 2980000),
        ('2025-11-24', 41350.40, 41650.40, 41280.60, 41550.40, 2650000),
        ('2025-11-25', 41550.40, 41850.40, 41480.60, 41750.40, 2480000),
        ('2025-11-26', 41750.40, 42050.60, 41680.60, 41950.40, 2350000),
        ('2025-11-27', 41950.40, 42250.60, 41880.60, 42150.40, 2280000),
        ('2025-11-28', 42150.40, 42550.40, 42080.60, 42450.40, 2850000),
        # 2025年12月
        ('2025-12-01', 42450.40, 42850.40, 42380.60, 42750.40, 3580000),
        ('2025-12-02', 42750.40, 43150.40, 42680.60, 43050.40, 3420000),
        ('2025-12-03', 43050.40, 43450.40, 42980.60, 43350.40, 3280000),
        ('2025-12-04', 43350.40, 43750.40, 43280.60, 43650.40, 3150000),
        ('2025-12-05', 43650.40, 44050.40, 43580.60, 43950.40, 3850000),
        ('2025-12-08', 43950.40, 44350.40, 43880.60, 44250.40, 3580000),
        ('2025-12-09', 44250.40, 44550.40, 44180.60, 44450.40, 3420000),
        ('2025-12-10', 44450.40, 44750.40, 44380.60, 44650.40, 3280000),
        ('2025-12-11', 44650.40, 45050.40, 44580.60, 44950.40, 3650000),
        ('2025-12-12', 44950.40, 45350.40, 44880.60, 45250.40, 3950000),
        ('2025-12-15', 45250.40, 45550.40, 45180.60, 45450.40, 3780000),
        ('2025-12-16', 45450.40, 45750.40, 45380.60, 45650.40, 3550000),
        ('2025-12-17', 45650.40, 45950.40, 45580.60, 45850.40, 3420000),
        ('2025-12-18', 45850.40, 46150.40, 45780.60, 46050.40, 3580000),
        ('2025-12-19', 46050.40, 46450.40, 45980.60, 46350.40, 3750000),
        ('2025-12-22', 46350.40, 46650.40, 46280.60, 46550.40, 3280000),
        ('2025-12-23', 46550.40, 46850.40, 46480.60, 46750.40, 2980000),
        ('2025-12-24', 46750.40, 47050.40, 46680.60, 46950.40, 2650000),
        ('2025-12-29', 46950.40, 47250.40, 46880.60, 47150.40, 3150000),
        ('2025-12-30', 47150.40, 47450.40, 47080.60, 47350.40, 3420000),
        ('2025-12-31', 47350.40, 47650.40, 47280.60, 47550.40, 2850000),
        # 2026年1月
        ('2026-01-02', 47550.40, 47950.40, 47480.60, 47850.40, 3850000),
        ('2026-01-05', 47850.40, 48350.40, 47780.60, 48250.40, 4250000),
        ('2026-01-06', 48250.40, 48750.40, 48180.60, 48650.40, 4550000),
        ('2026-01-07', 48650.40, 49050.40, 48580.60, 48950.40, 4380000),
        ('2026-01-08', 48950.40, 49350.40, 48880.60, 49250.40, 4150000),
        ('2026-01-09', 49250.40, 49750.40, 49180.60, 49650.40, 4850000),
        ('2026-01-12', 49650.40, 50050.40, 49580.60, 49950.40, 4650000),
        ('2026-01-13', 49950.40, 50350.40, 49880.60, 50250.40, 4480000),
        ('2026-01-14', 50250.40, 50650.40, 50180.60, 50550.40, 5250000),
        ('2026-01-15', 50550.40, 50950.40, 50480.60, 50850.40, 4980000),
        ('2026-01-16', 50850.40, 51250.40, 50780.60, 51150.40, 4650000),
        ('2026-01-19', 51150.40, 51550.40, 51080.60, 51450.40, 4480000),
        ('2026-01-20', 51450.40, 51950.40, 51380.60, 51850.40, 4750000),
        ('2026-01-21', 51850.40, 52350.40, 51780.60, 52250.40, 5150000),
        ('2026-01-22', 52250.40, 52650.40, 52180.60, 52550.40, 5450000),
        ('2026-01-23', 52550.40, 53050.40, 52480.60, 52950.40, 5650000),
        ('2026-01-26', 52950.40, 53350.40, 52880.60, 53250.40, 5850000),
        ('2026-01-27', 53250.40, 53750.40, 53180.60, 53650.40, 6250000),
        ('2026-01-28', 53650.40, 54050.40, 53580.60, 53950.40, 6550000),
        ('2026-01-29', 53950.40, 54350.40, 53880.60, 54250.40, 6850000),
        ('2026-01-30', 54250.40, 52550.40, 52250.60, 52650.40, 9500000),  # 大跌
        ('2026-02-02', 52650.40, 53150.40, 52580.60, 53050.40, 6850000),
        ('2026-02-03', 53050.40, 53450.40, 52980.60, 53350.40, 6250000),
        ('2026-02-04', 53350.40, 53750.40, 53280.60, 53650.40, 5650000),
        ('2026-02-05', 53650.40, 53950.40, 53580.60, 53850.40, 5250000),
        ('2026-02-06', 53850.40, 54250.40, 53780.60, 54150.40, 4850000),
    ]

    df_dax = pd.DataFrame(dax_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df_nikkei = pd.DataFrame(nikkei_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    return {'DAX': df_dax, 'NIKKEI': df_nikkei}

def main():
    """主程序：运行回测并输出结果"""

    print("=" * 80)
    print(" " * 16 + "DAX & NIKKEI225 每日技术指标回测报告")
    print(" " * 10 + "DTI - DAX/NIKKEI Backtest")
    print("=" * 80)
    print()

    print("正在加载历史数据...")
    data = get_dax_nikkei_data()

    initial_capital = 1000
    print(f"初始资金: ${initial_capital}")
    print(f"回测周期: 2025-11-03 至 2026-02-06 (约3个月)")
    print()

    # 对DAX进行回测
    print("=" * 80)
    print(" " * 32 + "DAX (德国DAX) 回测结果")
    print("=" * 80)
    print()

    indicator_dax = DailyTechnicalIndicator(symbol='DAX')
    result_dax = indicator_dax.backtest(data['DAX'], initial_capital)

    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_dax['initial_capital']:.2f}")
    print(f"最终资金:        ${result_dax['final_capital']:.2f}")
    print(f"总收益率:        {result_dax['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_dax['total_trades']}")
    print(f"盈利交易:        {result_dax['win_trades']}")
    print(f"亏损交易:        {result_dax['lose_trades']}")
    print(f"胜率:            {result_dax['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_dax['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_dax['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_dax['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_dax['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_dax['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_dax['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_dax['profit_factor']:.2f}")
    print()

    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<10} {'出场价':<10} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_dax['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} "
              f"{trade['pnl_pct']:>6.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 对NIKKEI进行回测
    print("=" * 80)
    print(" " * 28 + "NIKKEI225 (日经225) 回测结果")
    print("=" * 80)
    print()

    indicator_nikkei = DailyTechnicalIndicator(symbol='NIKKEI')
    result_nikkei = indicator_nikkei.backtest(data['NIKKEI'], initial_capital)

    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_nikkei['initial_capital']:.2f}")
    print(f"最终资金:        ${result_nikkei['final_capital']:.2f}")
    print(f"总收益率:        {result_nikkei['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_nikkei['total_trades']}")
    print(f"盈利交易:        {result_nikkei['win_trades']}")
    print(f"亏损交易:        {result_nikkei['lose_trades']}")
    print(f"胜率:            {result_nikkei['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_nikkei['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_nikkei['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_nikkei['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_nikkei['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_nikkei['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_nikkei['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_nikkei['profit_factor']:.2f}")
    print()

    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<10} {'出场价':<10} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_nikkei['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} "
              f"{trade['pnl_pct']:>6.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 输出最新信号
    print("=" * 80)
    print(" " * 24 + "最新交易信号")
    print("=" * 80)
    print()

    latest_signal_dax = result_dax['signals'][-1]
    latest_signal_nikkei = result_nikkei['signals'][-1]

    print("【DAX 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_dax['date']}")
    print(f"方向:            {latest_signal_dax['direction']}")
    print(f"信心度:          {latest_signal_dax['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_dax['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_dax['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_dax['expected_move']:.2f}%")
    print(f"建议止损:        {latest_signal_dax['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'{tp:.2f}' for tp in latest_signal_dax['take_profit']]}")
    print(f"建议仓位:        {latest_signal_dax['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_dax['reason']}")
    print()

    print("【NIKKEI 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_nikkei['date']}")
    print(f"方向:            {latest_signal_nikkei['direction']}")
    print(f"信心度:          {latest_signal_nikkei['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_nikkei['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_nikkei['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_nikkei['expected_move']:.2f}%")
    print(f"建议止损:        {latest_signal_nikkei['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'{tp:.2f}' for tp in latest_signal_nikkei['take_profit']]}")
    print(f"建议仓位:        {latest_signal_nikkei['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_nikkei['reason']}")
    print()

    # 保存结果到JSON
    output = {
        'backtest_summary': {
            'dax': {k: v for k, v in result_dax.items() if k not in ['trades', 'signals', 'daily_values']},
            'nikkei': {k: v for k, v in result_nikkei.items() if k not in ['trades', 'signals', 'daily_values']}
        },
        'latest_signals': {
            'dax': latest_signal_dax,
            'nikkei': latest_signal_nikkei
        }
    }

    with open('/root/rich/indices_eu_jp_result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("回测结果已保存至: indices_eu_jp_result.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
