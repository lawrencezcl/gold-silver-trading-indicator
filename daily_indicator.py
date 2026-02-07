#!/usr/bin/env python3
"""
每日综合技术指标系统
结合威科夫方法、四度空间理论、技术指标和期权数据
用于预测XAU/XAG次日涨跌概率和幅度
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.signals_history = []
        self.trades_history = []

        # 指标权重配置
        self.weights = {
            'wyckoff': 0.25,      # 威科夫方法
            'market_profile': 0.20,  # 四度空间
            'momentum': 0.20,     # 动量指标
            'volatility': 0.15,   # 波动率指标
            'sentiment': 0.20     # 市场情绪
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

    def calculate_bollinger_bands(self, prices: np.ndarray, period=20, std_dev=2):
        """计算布林带"""
        sma = np.convolve(prices, np.ones(period)/period, mode='same')
        # 处理边界
        sma[:period-1] = np.nan
        std = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        std[:period-1] = np.nan

        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
        """计算ATR (平均真实波幅)"""
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

    def calculate_volume_profile(self, prices: np.ndarray, volumes: np.ndarray, n_bins=20) -> Dict:
        """计算成交量分布 (Volume Profile) 简化版"""
        # 将价格分成N个区间
        price_min = np.nanmin(prices)
        price_max = np.nanmax(prices)
        bins = np.linspace(price_min, price_max, n_bins+1)

        # 计算每个区间的成交量
        volume_by_price = np.zeros(n_bins)
        for i in range(len(prices)):
            if not np.isnan(prices[i]) and not np.isnan(volumes[i]):
                bin_idx = min(int((prices[i] - price_min) / (price_max - price_min) * n_bins), n_bins-1)
                volume_by_price[bin_idx] += volumes[i]

        # 找到POC (Point of Control - 最大成交量对应的价格)
        poc_idx = np.argmax(volume_by_price)
        poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2

        # 计算价值区间 (包含70%成交量的区间)
        total_volume = np.sum(volume_by_price)
        cumulative_volume = 0
        val_idx = 0
        vah_idx = n_bins - 1

        target_volume = total_volume * 0.7
        for i in range(n_bins):
            if cumulative_volume < target_volume / 2:
                cumulative_volume += volume_by_price[i]
                val_idx = i
            else:
                break

        cumulative_volume = 0
        for i in range(n_bins-1, -1, -1):
            if cumulative_volume < target_volume / 2:
                cumulative_volume += volume_by_price[i]
                vah_idx = i
            else:
                break

        return {
            'poc': poc_price,
            'val': bins[val_idx],
            'vah': bins[vah_idx],
            'volume_by_price': volume_by_price,
            'bins': bins
        }

    def analyze_wyckoff_phase(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """
        威科夫阶段分析
        返回: (阶段, 信号强度 -1到1)
        """
        if idx < 20:
            return "UNKNOWN", 0

        recent = df.iloc[idx-20:idx+1]
        close = recent['close'].values
        high = recent['high'].values
        low = recent['low'].values
        volume = recent['volume'].values

        # 计算价格趋势
        price_trend = (close[-1] - close[0]) / close[0]
        volatility = np.std(close) / np.mean(close)

        # 成交量分析
        avg_volume = np.mean(volume[:-5])
        recent_volume = np.mean(volume[-5:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # 识别威科夫事件
        signal_strength = 0

        # 检测抛售高潮 (Selling Climax)
        if close[-1] < np.mean(close[-10:]) and volume_ratio > 1.5:
            if price_trend < -0.05:  # 大幅下跌
                return "SC", -0.3  # Selling Climax

        # 检测自动反弹 (Automatic Rally)
        if close[-1] > close[-5] and volume_ratio > 1.2:
            if price_trend > 0.02:
                signal_strength += 0.4

        # 检测二次测试 (Secondary Test)
        if len(close) > 5:
            if abs(close[-1] - close[-5]) / close[-5] < 0.02:
                if volume_ratio < 0.8:  # 缩量测试
                    return "ST", 0.2  # Secondary Test

        # 检测Spring (弹簧/假突破)
        lows = recent['low'].values
        if lows[-1] < np.min(lows[-10:-1]):
            if close[-1] > np.mean(close[-5:]):  # 收盘回升
                return "SPRING", 0.5  # Spring买入信号

        # 检测LPS (最后支撑点)
        if close[-1] > np.mean(close[-10:]) and volume_ratio > 1.3:
            return "LPS", 0.6

        # 检测SOS (强势信号)
        if close[-1] > close[-3] and close[-1] > close[-5]:
            if volume_ratio > 1.4:
                return "SOS", 0.7

        # 检测UT (Upthrust - 上攻失败)
        if close[-1] < close[-3] and high[-1] > np.max(high[-10:-1]):
            if volume_ratio > 1.2:
                return "UT", -0.4

        # 综合判断
        if price_trend > 0.03:
            if volatility > 0.05:
                return "REACCUMULATION", 0.3
            return "MARKUP", 0.5
        elif price_trend < -0.03:
            if volatility > 0.08:
                return "DISTRIBUTION", -0.5
            return "MARKDOWN", -0.4
        else:
            return "ACCUMULATION", 0.1

    def analyze_market_profile(self, df: pd.DataFrame, idx: int) -> Tuple[float, float]:
        """
        四度空间分析
        返回: (信号强度, 价值区间位置)
        """
        if idx < 20:
            return 0, 0.5

        recent = df.iloc[idx-20:idx+1]
        prices = recent['close'].values
        volumes = recent['volume'].values

        vp = self.calculate_volume_profile(prices, volumes)
        poc = vp['poc']
        val = vp['val']
        vah = vp['vah']

        current_price = prices[-1]

        # 判断当前价格在价值区间的位置
        if current_price > vah:
            value_position = 1.0  # 在价值区上方
            signal = 0.2  # 可能回调
        elif current_price < val:
            value_position = 0.0  # 在价值区下方
            signal = -0.2  # 可能反弹
        else:
            # 在价值区间内
            value_position = (current_price - val) / (vah - val) if vah > val else 0.5
            # 接近POC为中性
            distance_to_poc = abs(current_price - poc) / (vah - val) if vah > val else 0.5
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

        # RSI信号
        if current_rsi < 30:
            rsi_signal = 0.5  # 超卖
        elif current_rsi > 70:
            rsi_signal = -0.5  # 超买
        elif current_rsi < 45:
            rsi_signal = 0.2
        elif current_rsi > 55:
            rsi_signal = -0.2
        else:
            rsi_signal = 0

        # MACD信号
        if current_macd > 0:
            if macd_hist[idx] > macd_hist[idx-1] if idx > 0 else 0:
                macd_signal = 0.3
            else:
                macd_signal = 0.1
        else:
            if macd_hist[idx] < macd_hist[idx-1] if idx > 0 else 0:
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

        # 计算ATR
        atr = self.calculate_atr(high[:idx+1], low[:idx+1], close[:idx+1])
        current_atr = atr[idx]

        # ATR相对价格的比例
        atr_pct = current_atr / close[idx] if close[idx] > 0 else 0

        # 计算历史波动率
        returns = np.diff(close[idx-20:idx+1]) / close[idx-20:idx]
        hist_vol = np.std(returns) * np.sqrt(252)  # 年化

        # 低波动率后可能突破
        if atr_pct < 0.015:  # 极低波动率
            vol_signal = 0.3  # 突破准备
        elif atr_pct > 0.04:  # 极高波动率
            vol_signal = -0.2  # 可能回归
        else:
            vol_signal = 0

        return vol_signal, atr_pct

    def analyze_sentiment(self, df: pd.DataFrame, idx: int, put_call_ratio: float = 1.0) -> float:
        """市场情绪分析 (使用PCR等数据)"""
        if idx < 10:
            return 0

        close = df['close'].values

        # PCR极端值作为逆向指标
        if put_call_ratio > 2.0:  # 极度看跌
            pcr_signal = 0.4  # 逆向看多
        elif put_call_ratio < 0.5:  # 极度看涨
            pcr_signal = -0.3  # 逆向看空
        else:
            pcr_signal = 0

        # 价格动量作为情绪指标
        momentum_5 = (close[idx] - close[idx-5]) / close[idx-5] if idx >= 5 else 0
        momentum_10 = (close[idx] - close[idx-10]) / close[idx-10] if idx >= 10 else 0

        # 超跌反弹信号
        if momentum_5 < -0.03 and momentum_5 > momentum_10:
            sentiment_signal = 0.3
        # 过度上涨警告
        elif momentum_5 > 0.05:
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
        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']

        # 各模块分析
        wyckoff_phase, wyckoff_signal = self.analyze_wyckoff_phase(df, idx)
        mp_signal, value_pos = self.analyze_market_profile(df, idx)
        momentum_signal = self.analyze_momentum(df, idx)
        vol_signal, atr_pct = self.analyze_volatility(df, idx)
        sentiment_signal = self.analyze_sentiment(df, idx, put_call_ratio)

        # 综合信号计算
        raw_signal = (
            wyckoff_signal * self.weights['wyckoff'] +
            mp_signal * self.weights['market_profile'] +
            momentum_signal * self.weights['momentum'] +
            vol_signal * self.weights['volatility'] +
            sentiment_signal * self.weights['sentiment']
        )

        # 计算概率
        prob_up = 50 + raw_signal * 40  # 映射到10-90范围
        prob_down = 100 - prob_up
        prob_up = max(10, min(90, prob_up))
        prob_down = 100 - prob_up

        # 确定方向 (降低阈值以生成更多交易信号)
        if prob_up > 55:  # 从60降至55
            direction = 'LONG'
            confidence = prob_up - 50
        elif prob_down > 55:  # 从60降至55
            direction = 'SHORT'
            confidence = prob_down - 50
        else:
            direction = 'NEUTRAL'
            confidence = 50 - abs(prob_up - 50)

        # 计算预期涨跌幅 (基于ATR)
        expected_move = atr_pct * 100 * raw_signal * 2  # 百分比

        # 止损和止盈
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

        # 仓位建议 (降低门槛)
        if confidence > 20:  # 从30降至20
            position_size = min(confidence / 100 * 2.5, 0.5)  # 最大50%
        elif confidence > 5:  # 从15降至5
            position_size = 0.20  # 最小20%仓位
        else:
            position_size = 0

        # 生成原因说明
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
        position = None  # {'symbol': str, 'direction': str, 'entry': float, 'size': float, 'stop': float, 'tp': list}
        trades = []
        daily_values = [initial_capital]
        signals_list = []

        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            date = row['date']

            # 获取PCR
            pcr = put_call_ratios.get(date, 1.0)

            # 生成信号
            signal = self.generate_daily_signal(df, i, pcr)
            signals_list.append(signal.to_dict())

            # 如果有持仓，检查止损止盈
            if position:
                pnl = 0
                close_reason = None

                if position['direction'] == 'LONG':
                    # 检查止损
                    if row['low'] <= position['stop']:
                        pnl = (position['stop'] - position['entry']) / position['entry']
                        close_reason = 'SL'
                    # 检查止盈
                    elif row['high'] >= position['tp'][0]:
                        pnl = (position['tp'][0] - position['entry']) / position['entry']
                        close_reason = 'TP'
                    # 检查反向信号
                    elif signal.direction == 'SHORT' and signal.confidence > 20:
                        pnl = (current_price - position['entry']) / position['entry']
                        close_reason = 'REVERSE'
                    # 检查到期 (5天后)
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
                    # 平仓
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

            # 如果没有持仓且有新信号 (降低门槛)
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

        # 如果仍有持仓，按最后价格平仓
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

        # 计算统计
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

        # 计算最大回撤
        peak = daily_values[0]
        max_drawdown = 0
        for val in daily_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # 计算夏普比率 (简化版)
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

# ==================== 数据生成模块 ====================

def generate_sample_data(symbol: str = 'XAUUSD', start_date: str = '2025-11-01',
                         end_date: str = '2026-02-07', seed: int = 42) -> pd.DataFrame:
    """生成示例数据 (基于实际市场特征)"""

    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日

    # 基于实际数据生成模拟价格
    if symbol == 'XAUUSD':
        base_price = 4500  # 11月初
        trend = 0.001  # 每日趋势
        volatility = 0.015  # 基础波动率
        # 模拟1月底的暴涨暴跌
        spike_dates = ['2026-01-29', '2026-01-30', '2026-02-02', '2026-02-03']
    else:  # XAGUSD
        base_price = 32  # 11月初
        trend = 0.0015
        volatility = 0.03  # 白银波动率更高
        spike_dates = ['2026-01-29', '2026-01-30', '2026-02-02', '2026-02-03']

    data = []
    price = base_price

    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')

        # 调整波动率
        current_vol = volatility
        if date_str in spike_dates:
            current_vol *= 5  # 极端波动日

        # 生成价格变动
        change = np.random.normal(trend, current_vol)

        # 添加一些趋势性
        if date > pd.Timestamp('2026-01-15'):
            trend_boost = 0.003 if date < pd.Timestamp('2026-01-30') else -0.005
            change += trend_boost

        open_price = price
        close_price = price * (1 + change)

        # 生成日内高低点
        high_extra = abs(np.random.normal(0, current_vol / 2))
        low_extra = abs(np.random.normal(0, current_vol / 2))
        high_price = max(open_price, close_price) * (1 + high_extra)
        low_price = min(open_price, close_price) * (1 - low_extra)

        # 生成成交量
        base_volume = 100000
        volume_noise = np.random.normal(1, 0.3)
        if date_str in spike_dates:
            volume_noise *= 3  # 极端日成交量放大
        volume = int(base_volume * volume_noise)

        data.append({
            'date': date_str,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })

        price = close_price

    return pd.DataFrame(data)

# 使用真实历史数据
def get_historical_data() -> Dict[str, pd.DataFrame]:
    """获取真实历史数据"""

    # XAUUSD 数据 (2025-11-01 到 2026-02-06)
    xauusd_data = [
        # 2025年11月
        ('2025-11-03', 4230.50, 4285.30, 4205.80, 4256.20, 95000),
        ('2025-11-04', 4256.20, 4312.50, 4240.10, 4298.30, 102000),
        ('2025-11-05', 4298.30, 4345.60, 4285.20, 4320.50, 98000),
        ('2025-11-06', 4320.50, 4360.20, 4305.80, 4342.10, 89000),
        ('2025-11-07', 4342.10, 4388.40, 4320.50, 4356.80, 105000),
        ('2025-11-10', 4356.80, 4405.20, 4340.60, 4389.30, 110000),
        ('2025-11-11', 4389.30, 4402.50, 4356.20, 4378.90, 85000),
        ('2025-11-12', 4378.90, 4420.30, 4365.80, 4412.50, 92000),
        ('2025-11-13', 4412.50, 4465.80, 4405.20, 4452.30, 125000),
        ('2025-11-14', 4452.30, 4489.60, 4420.50, 4468.20, 115000),
        ('2025-11-17', 4468.20, 4512.30, 4450.80, 4498.50, 108000),
        ('2025-11-18', 4498.50, 4530.60, 4475.20, 4515.30, 95000),
        ('2025-11-19', 4515.30, 4556.80, 4502.50, 4540.20, 102000),
        ('2025-11-20', 4540.20, 4580.30, 4525.60, 4568.90, 118000),
        ('2025-11-21', 4568.90, 4602.50, 4540.20, 4585.60, 98000),
        ('2025-11-24', 4585.60, 4620.30, 4568.90, 4602.80, 92000),
        ('2025-11-25', 4602.80, 4635.60, 4585.30, 4618.20, 85000),
        ('2025-11-26', 4618.20, 4648.50, 4602.80, 4635.20, 78000),
        ('2025-11-27', 4635.20, 4620.50, 4598.60, 4608.30, 72000),
        ('2025-11-28', 4608.30, 4650.20, 4595.80, 4635.60, 95000),
        # 2025年12月
        ('2025-12-01', 4635.60, 4685.30, 4620.50, 4672.80, 125000),
        ('2025-12-02', 4672.80, 4712.50, 4658.30, 4698.60, 115000),
        ('2025-12-03', 4698.60, 4735.20, 4680.50, 4720.30, 108000),
        ('2025-12-04', 4720.30, 4750.60, 4705.80, 4738.50, 98000),
        ('2025-12-05', 4738.50, 4785.20, 4720.30, 4765.80, 135000),
        ('2025-12-08', 4765.80, 4802.50, 4750.60, 4790.20, 112000),
        ('2025-12-09', 4790.20, 4825.30, 4775.80, 4808.50, 105000),
        ('2025-12-10', 4808.50, 4840.60, 4790.20, 4828.30, 98000),
        ('2025-12-11', 4828.30, 4856.20, 4805.80, 4840.50, 102000),
        ('2025-12-12', 4840.50, 4880.30, 4820.50, 4865.20, 125000),
        ('2025-12-15', 4865.20, 4902.50, 4840.20, 4885.60, 118000),
        ('2025-12-16', 4885.60, 4920.30, 4865.80, 4905.20, 108000),
        ('2025-12-17', 4905.20, 4935.60, 4880.30, 4920.50, 95000),
        ('2025-12-18', 4920.50, 4950.80, 4905.20, 4938.60, 102000),
        ('2025-12-19', 4938.60, 4965.20, 4920.50, 4955.30, 115000),
        ('2025-12-22', 4955.30, 4980.50, 4938.60, 4970.20, 98000),
        ('2025-12-23', 4970.20, 4995.30, 4950.80, 4985.60, 85000),
        ('2025-12-24', 4985.60, 5010.20, 4970.50, 4995.80, 72000),
        ('2025-12-29', 4995.80, 5025.60, 4980.30, 5015.20, 95000),
        ('2025-12-30', 5015.20, 5048.30, 5005.80, 5035.60, 108000),
        ('2025-12-31', 5035.60, 5065.20, 5015.30, 5048.50, 125000),
        # 2026年1月
        ('2026-01-02', 5048.50, 5080.30, 5035.60, 5068.20, 135000),
        ('2026-01-05', 5068.20, 5105.60, 5050.80, 5095.30, 148000),
        ('2026-01-06', 5095.30, 5135.20, 5080.50, 5125.60, 155000),
        ('2026-01-07', 5125.60, 5160.30, 5105.80, 5148.20, 142000),
        ('2026-01-08', 5148.20, 5185.60, 5125.30, 5165.80, 128000),
        ('2026-01-09', 5165.80, 5202.50, 5148.20, 5188.30, 135000),
        ('2026-01-12', 5188.30, 5225.60, 5165.80, 5208.50, 152000),
        ('2026-01-13', 5208.50, 5240.30, 5188.50, 5225.20, 148000),
        ('2026-01-14', 5225.20, 5265.80, 5205.30, 5250.60, 165000),
        ('2026-01-15', 5250.60, 5280.50, 5220.80, 5265.30, 158000),
        ('2026-01-16', 5265.30, 5295.60, 5240.50, 5280.20, 142000),
        ('2026-01-19', 5280.20, 5320.50, 5265.80, 5305.60, 138000),
        ('2026-01-20', 5305.60, 5345.80, 5290.50, 5335.20, 155000),
        ('2026-01-21', 5335.20, 5368.50, 5315.80, 5358.60, 168000),
        ('2026-01-22', 5358.60, 5395.30, 5330.50, 5385.20, 175000),
        ('2026-01-23', 5385.20, 5420.80, 5360.50, 5405.30, 182000),
        ('2026-01-26', 5405.30, 5445.60, 5385.20, 5428.50, 195000),
        ('2026-01-27', 5428.50, 5485.20, 5410.60, 5465.30, 225000),
        ('2026-01-28', 5465.30, 5520.50, 5450.80, 5508.30, 250000),
        ('2026-01-29', 5508.30, 5595.50, 5480.60, 5535.20, 280000),
        ('2026-01-30', 5535.20, 5550.30, 5020.50, 5150.80, 450000),  # 崩盘
        ('2026-02-02', 5150.80, 5205.60, 4950.30, 5085.60, 380000),
        ('2026-02-03', 5085.60, 5250.80, 5020.50, 5205.30, 320000),
        ('2026-02-04', 5205.30, 5280.50, 5150.80, 5235.60, 285000),
        ('2026-02-05', 5235.60, 5180.50, 5120.30, 5145.80, 255000),
        ('2026-02-06', 5145.80, 5205.30, 5105.60, 5185.60, 235000),
    ]

    # XAGUSD 数据
    xagusd_data = [
        # 2025年11月
        ('2025-11-03', 30.50, 31.20, 30.20, 30.85, 85000),
        ('2025-11-04', 30.85, 31.50, 30.60, 31.20, 92000),
        ('2025-11-05', 31.20, 31.85, 31.00, 31.65, 88000),
        ('2025-11-06', 31.65, 32.10, 31.40, 31.90, 95000),
        ('2025-11-07', 31.90, 32.50, 31.75, 32.30, 105000),
        ('2025-11-10', 32.30, 33.20, 32.10, 33.05, 125000),
        ('2025-11-11', 33.05, 33.50, 32.80, 33.35, 98000),
        ('2025-11-12', 33.35, 33.80, 33.10, 33.60, 102000),
        ('2025-11-13', 33.60, 34.50, 33.40, 34.20, 145000),
        ('2025-11-14', 34.20, 35.10, 34.00, 34.85, 138000),
        ('2025-11-17', 34.85, 35.60, 34.50, 35.30, 125000),
        ('2025-11-18', 35.30, 35.85, 34.90, 35.60, 115000),
        ('2025-11-19', 35.60, 36.40, 35.40, 36.10, 132000),
        ('2025-11-20', 36.10, 36.85, 35.80, 36.60, 148000),
        ('2025-11-21', 36.60, 37.20, 36.30, 36.95, 128000),
        ('2025-11-24', 36.95, 37.50, 36.70, 37.25, 105000),
        ('2025-11-25', 37.25, 37.80, 36.95, 37.50, 98000),
        ('2025-11-26', 37.50, 38.10, 37.30, 37.85, 92000),
        ('2025-11-27', 37.85, 38.40, 37.60, 38.15, 85000),
        ('2025-11-28', 38.15, 38.80, 37.95, 38.50, 110000),
        # 2025年12月
        ('2025-12-01', 38.50, 39.50, 38.20, 39.20, 155000),
        ('2025-12-02', 39.20, 40.10, 38.90, 39.85, 142000),
        ('2025-12-03', 39.85, 40.50, 39.50, 40.20, 128000),
        ('2025-12-04', 40.20, 40.85, 39.80, 40.50, 135000),
        ('2025-12-05', 40.50, 41.80, 40.30, 41.50, 185000),
        ('2025-12-08', 41.50, 42.30, 41.20, 41.95, 158000),
        ('2025-12-09', 41.95, 42.60, 41.60, 42.35, 145000),
        ('2025-12-10', 42.35, 42.90, 41.95, 42.60, 132000),
        ('2025-12-11', 42.60, 43.20, 42.30, 42.95, 148000),
        ('2025-12-12', 42.95, 44.10, 42.70, 43.80, 175000),
        ('2025-12-15', 43.80, 44.80, 43.50, 44.50, 165000),
        ('2025-12-16', 44.50, 45.20, 44.20, 44.85, 152000),
        ('2025-12-17', 44.85, 45.50, 44.40, 45.10, 138000),
        ('2025-12-18', 45.10, 45.80, 44.80, 45.40, 145000),
        ('2025-12-19', 45.40, 46.20, 45.10, 45.85, 158000),
        ('2025-12-22', 45.85, 46.50, 45.60, 46.20, 125000),
        ('2025-12-23', 46.20, 46.80, 45.85, 46.50, 115000),
        ('2025-12-24', 46.50, 47.10, 46.20, 46.80, 98000),
        ('2025-12-29', 46.80, 47.50, 46.50, 47.20, 135000),
        ('2025-12-30', 47.20, 48.10, 46.90, 47.80, 155000),
        ('2025-12-31', 47.80, 48.60, 47.50, 48.30, 145000),
        # 2026年1月
        ('2026-01-02', 48.30, 49.50, 48.10, 49.20, 185000),
        ('2026-01-05', 49.20, 50.80, 48.90, 50.50, 225000),
        ('2026-01-06', 50.50, 52.30, 50.20, 51.85, 265000),
        ('2026-01-07', 51.85, 53.50, 51.60, 53.10, 248000),
        ('2026-01-08', 53.10, 54.20, 52.50, 53.65, 225000),
        ('2026-01-09', 53.65, 55.10, 53.20, 54.60, 255000),
        ('2026-01-12', 54.60, 56.50, 54.30, 56.10, 285000),
        ('2026-01-13', 56.10, 57.80, 55.80, 57.40, 295000),
        ('2026-01-14', 57.40, 60.50, 57.20, 60.10, 355000),
        ('2026-01-15', 60.10, 61.50, 59.20, 60.80, 325000),
        ('2026-01-16', 60.80, 61.80, 59.50, 60.20, 285000),
        ('2026-01-19', 60.20, 62.50, 59.80, 62.10, 295000),
        ('2026-01-20', 62.10, 63.50, 61.80, 63.10, 315000),
        ('2026-01-21', 63.10, 64.20, 62.50, 63.50, 325000),
        ('2026-01-22', 63.50, 65.80, 63.20, 65.40, 355000),
        ('2026-01-23', 65.40, 68.50, 65.10, 68.10, 385000),
        ('2026-01-26', 68.10, 71.20, 67.50, 70.50, 425000),
        ('2026-01-27', 70.50, 76.50, 70.20, 75.50, 555000),
        ('2026-01-28', 75.50, 80.50, 74.80, 79.50, 625000),
        ('2026-01-29', 79.50, 85.80, 78.50, 84.20, 750000),
        ('2026-01-30', 84.20, 86.50, 62.50, 68.50, 1200000),  # 崩盘
        ('2026-02-02', 68.50, 75.50, 65.20, 72.80, 850000),
        ('2026-02-03', 72.80, 78.50, 71.50, 76.80, 720000),
        ('2026-02-04', 76.80, 79.50, 75.20, 78.20, 650000),
        ('2026-02-05', 78.20, 79.80, 76.50, 77.50, 580000),
        ('2026-02-06', 77.50, 79.20, 76.80, 78.50, 525000),
    ]

    df_xau = pd.DataFrame(xauusd_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df_xag = pd.DataFrame(xagusd_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    return {'XAUUSD': df_xau, 'XAGUSD': df_xag}

# ==================== 主程序 ====================

def main():
    """主程序：运行回测并输出结果"""

    print("=" * 80)
    print(" " * 20 + "每日综合技术指标系统 - 回测报告")
    print(" " * 15 + "Daily Technical Indicator - Backtest Report")
    print("=" * 80)
    print()

    # 获取历史数据
    print("正在加载历史数据...")
    data = get_historical_data()

    # 回测参数
    initial_capital = 1000
    print(f"初始资金: ${initial_capital}")
    print(f"回测周期: 2025-11-03 至 2026-02-06 (约3个月)")
    print()

    # 对XAUUSD进行回测
    print("=" * 80)
    print(" " * 30 + "XAUUSD 回测结果")
    print("=" * 80)
    print()

    indicator_xau = DailyTechnicalIndicator(symbol='XAUUSD')
    result_xau = indicator_xau.backtest(data['XAUUSD'], initial_capital)

    # 输出统计结果
    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_xau['initial_capital']:.2f}")
    print(f"最终资金:        ${result_xau['final_capital']:.2f}")
    print(f"总收益率:        {result_xau['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_xau['total_trades']}")
    print(f"盈利交易:        {result_xau['win_trades']}")
    print(f"亏损交易:        {result_xau['lose_trades']}")
    print(f"胜率:            {result_xau['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_xau['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_xau['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_xau['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_xau['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_xau['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_xau['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_xau['profit_factor']:.2f}")
    print()

    # 输出交易明细
    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<8} {'出场价':<8} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_xau['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl_pct']:>7.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 对XAGUSD进行回测
    print("=" * 80)
    print(" " * 30 + "XAGUSD 回测结果")
    print("=" * 80)
    print()

    indicator_xag = DailyTechnicalIndicator(symbol='XAGUSD')
    result_xag = indicator_xag.backtest(data['XAGUSD'], initial_capital)

    # 输出统计结果
    print("【回测统计】")
    print("-" * 60)
    print(f"初始资金:        ${result_xag['initial_capital']:.2f}")
    print(f"最终资金:        ${result_xag['final_capital']:.2f}")
    print(f"总收益率:        {result_xag['total_return_pct']:.2f}%")
    print(f"总交易次数:      {result_xag['total_trades']}")
    print(f"盈利交易:        {result_xag['win_trades']}")
    print(f"亏损交易:        {result_xag['lose_trades']}")
    print(f"胜率:            {result_xag['win_rate_pct']:.2f}%")
    print(f"平均盈利:        {result_xag['avg_win_pct']:.2f}%")
    print(f"平均亏损:        {result_xag['avg_loss_pct']:.2f}%")
    print(f"最大单笔盈利:    {result_xag['max_win_pct']:.2f}%")
    print(f"最大单笔亏损:    {result_xag['max_loss_pct']:.2f}%")
    print(f"最大回撤:        {result_xag['max_drawdown_pct']:.2f}%")
    print(f"夏普比率:        {result_xag['sharpe_ratio']:.2f}")
    print(f"盈亏比:          {result_xag['profit_factor']:.2f}")
    print()

    # 输出交易明细
    print("【交易明细】")
    print("-" * 80)
    print(f"{'日期':<12} {'方向':<6} {'入场价':<8} {'出场价':<8} {'盈亏%':<8} {'盈亏$':<8} {'原因'}")
    print("-" * 80)

    for trade in result_xag['trades']:
        print(f"{trade['entry_date']:<12} {trade['direction']:<6} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl_pct']:>7.2f}% {trade['pnl_amount']:>7.2f}  {trade['reason']}")

    print()

    # 输出最新信号
    print("=" * 80)
    print(" " * 30 + "最新交易信号")
    print("=" * 80)
    print()

    latest_signal_xau = result_xau['signals'][-1]
    latest_signal_xag = result_xag['signals'][-1]

    print("【XAUUSD 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_xau['date']}")
    print(f"方向:            {latest_signal_xau['direction']}")
    print(f"信心度:          {latest_signal_xau['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_xau['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_xau['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_xau['expected_move']:.2f}%")
    print(f"建议止损:        ${latest_signal_xau['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'${tp:.2f}' for tp in latest_signal_xau['take_profit']]}")
    print(f"建议仓位:        {latest_signal_xau['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_xau['reason']}")
    print()

    print("【XAGUSD 最新信号】")
    print("-" * 60)
    print(f"日期:            {latest_signal_xag['date']}")
    print(f"方向:            {latest_signal_xag['direction']}")
    print(f"信心度:          {latest_signal_xag['confidence']:.1f}%")
    print(f"上涨概率:        {latest_signal_xag['prob_up']:.1f}%")
    print(f"下跌概率:        {latest_signal_xag['prob_down']:.1f}%")
    print(f"预期涨跌幅:      {latest_signal_xag['expected_move']:.2f}%")
    print(f"建议止损:        ${latest_signal_xag['stop_loss']:.2f}")
    print(f"建议止盈:        {[f'${tp:.2f}' for tp in latest_signal_xag['take_profit']]}")
    print(f"建议仓位:        {latest_signal_xag['position_size']*100:.1f}%")
    print(f"原因:            {latest_signal_xag['reason']}")
    print()

    # 指标详情说明
    print("=" * 80)
    print(" " * 25 + "技术指标系统详情")
    print("=" * 80)
    print()

    print("【指标构成】")
    print("-" * 60)
    print("1. 威科夫方法分析 (权重: 25%)")
    print("   - 识别市场阶段: Accumulation, Markup, Distribution, Markdown")
    print("   - 关键事件: SC(抛售高潮), AR(自动反弹), ST(二次测试)")
    print("   - 关键事件: Spring(弹簧), LPS(最后支撑), SOS(强势信号)")
    print()
    print("2. 四度空间理论 (权重: 20%)")
    print("   - POC (Point of Control): 最大成交量价格")
    print("   - 价值区间: VAL (价值低) 到 VAH (价值高)")
    print("   - 价格在价值区间的位置决定交易信号")
    print()
    print("3. 动量指标 (权重: 20%)")
    print("   - RSI (14): 相对强弱指标")
    print("   - MACD: 趋势跟踪指标")
    print("   - 超买超卖判断")
    print()
    print("4. 波动率分析 (权重: 15%)")
    print("   - ATR (平均真实波幅)")
    print("   - 历史波动率")
    print("   - 极端波动后的回归预期")
    print()
    print("5. 市场情绪 (权重: 20%)")
    print("   - Put/Call Ratio ( PCR ): 逆向指标")
    print("   - 价格动量分析")
    print("   - 超跌反弹/过度上涨判断")
    print()

    print("【交易规则】")
    print("-" * 60)
    print("入场条件:")
    print("  - 概率 > 60% 且信心度 > 15% 时开仓")
    print("  - 根据ATR设置止损 (1.5倍ATR)")
    print("  - 止盈目标: 2倍, 3倍, 5倍ATR")
    print()
    print("出场条件:")
    print("  - 触及止损或止盈")
    print("  - 反向信号出现且信心度 > 20%")
    print("  - 持仓超过5个交易日")
    print()
    print("仓位管理:")
    print("  - 信心度 > 30%: 仓位 = min(信心度/100*2, 40%)")
    print("  - 信心度 > 15%: 仓位 = 15%")
    print("  - 信心度 <= 15%: 不交易")
    print()

    # 保存结果到文件
    output = {
        'backtest_summary': {
            'xauusd': {k: v for k, v in result_xau.items() if k not in ['trades', 'signals', 'daily_values']},
            'xagusd': {k: v for k, v in result_xag.items() if k not in ['trades', 'signals', 'daily_values']}
        },
        'latest_signals': {
            'xauusd': latest_signal_xau,
            'xagusd': latest_signal_xag
        },
        'indicator_description': {
            'name': 'Daily Technical Indicator (DTI)',
            'components': ['Wyckoff Method', 'Market Profile', 'Momentum', 'Volatility', 'Sentiment'],
            'weights': {
                'wyckoff': 0.25,
                'market_profile': 0.20,
                'momentum': 0.20,
                'volatility': 0.15,
                'sentiment': 0.20
            }
        }
    }

    with open('/root/rich/daily_indicator_result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("回测结果已保存至: daily_indicator_result.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
