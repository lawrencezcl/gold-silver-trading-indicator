#!/usr/bin/env python3
"""
Daily Technical Indicator (DTI) - Crypto Edition
分析加密货币: BTC, ETH, BNB

威科夫方法 + Market Profile + 动量指标 + 波动率分析 + 情绪指标
"""

import json
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 加密货币历史数据 (2025-11-03 至 2026-02-06)
# ==========================================

# BTC (Bitcoin) - 历史数据模拟
# 2025年11月: 约95,000-100,000美元区间
# 2025年12月: 突破100,000,涨至108,000
# 2026年1月: 继续上涨至115,000附近
# 1月30日: 市场回调,跌至110,000
btc_data = [
    # 2025年11月
    {"date": "2025-11-03", "open": 95250.40, "high": 95850.40, "low": 94850.40, "close": 95350.40, "volume": 28.5, "pcr": 0.85},
    {"date": "2025-11-04", "open": 95350.40, "high": 96200.40, "low": 95100.40, "close": 95950.40, "volume": 32.1, "pcr": 0.82},
    {"date": "2025-11-05", "open": 95950.40, "high": 96500.40, "low": 95650.40, "close": 96350.40, "volume": 29.8, "pcr": 0.80},
    {"date": "2025-11-06", "open": 96350.40, "high": 97000.40, "low": 96000.40, "close": 96850.40, "volume": 35.2, "pcr": 0.78},
    {"date": "2025-11-07", "open": 96850.40, "high": 97500.40, "low": 96500.40, "close": 97250.40, "volume": 31.5, "pcr": 0.75},
    {"date": "2025-11-10", "open": 97250.40, "high": 97850.40, "low": 96850.40, "close": 97650.40, "volume": 28.9, "pcr": 0.77},
    {"date": "2025-11-11", "open": 97650.40, "high": 98350.40, "low": 97350.40, "close": 98150.40, "volume": 33.4, "pcr": 0.73},
    {"date": "2025-11-12", "open": 98150.40, "high": 98850.40, "low": 97850.40, "close": 98650.40, "volume": 36.8, "pcr": 0.71},
    {"date": "2025-11-13", "open": 98650.40, "high": 99250.40, "low": 98250.40, "close": 99050.40, "volume": 32.2, "pcr": 0.69},
    {"date": "2025-11-14", "open": 99050.40, "high": 99750.40, "low": 98650.40, "close": 99550.40, "volume": 38.5, "pcr": 0.67},
    {"date": "2025-11-17", "open": 99550.40, "high": 100250.40, "low": 99250.40, "close": 100050.40, "volume": 42.1, "pcr": 0.65},
    {"date": "2025-11-18", "open": 100050.40, "high": 100850.40, "low": 99750.40, "close": 100350.40, "volume": 45.3, "pcr": 0.68},
    {"date": "2025-11-19", "open": 100350.40, "high": 101050.40, "low": 99950.40, "close": 100850.40, "volume": 39.7, "pcr": 0.66},
    {"date": "2025-11-20", "open": 100850.40, "high": 101550.40, "low": 100450.40, "close": 101250.40, "volume": 41.2, "pcr": 0.64},
    {"date": "2025-11-21", "open": 101250.40, "high": 101950.40, "low": 100850.40, "close": 101650.40, "volume": 37.6, "pcr": 0.62},
    {"date": "2025-11-24", "open": 101650.40, "high": 102350.40, "low": 101250.40, "close": 102050.40, "volume": 43.8, "pcr": 0.60},
    {"date": "2025-11-25", "open": 102050.40, "high": 102750.40, "low": 101650.40, "close": 102450.40, "volume": 46.2, "pcr": 0.63},
    {"date": "2025-11-26", "open": 102450.40, "high": 103150.40, "low": 102050.40, "close": 102850.40, "volume": 38.9, "pcr": 0.61},
    {"date": "2025-11-27", "open": 102850.40, "high": 103550.40, "low": 102450.40, "close": 103250.40, "volume": 44.5, "pcr": 0.59},
    {"date": "2025-11-28", "open": 103250.40, "high": 103950.40, "low": 102850.40, "close": 103650.40, "volume": 47.1, "pcr": 0.57},
    # 2025年12月 - 突破上涨
    {"date": "2025-12-01", "open": 103650.40, "high": 104450.40, "low": 103250.40, "close": 104150.40, "volume": 52.3, "pcr": 0.55},
    {"date": "2025-12-02", "open": 104150.40, "high": 104950.40, "low": 103750.40, "close": 104650.40, "volume": 48.7, "pcr": 0.53},
    {"date": "2025-12-03", "open": 104650.40, "high": 105550.40, "low": 104250.40, "close": 105150.40, "volume": 55.2, "pcr": 0.51},
    {"date": "2025-12-04", "open": 105150.40, "high": 105950.40, "low": 104750.40, "close": 105650.40, "volume": 51.8, "pcr": 0.54},
    {"date": "2025-12-05", "open": 105650.40, "high": 106550.40, "low": 105250.40, "close": 106150.40, "volume": 58.4, "pcr": 0.52},
    {"date": "2025-12-08", "open": 106150.40, "high": 107050.40, "low": 105750.40, "close": 106750.40, "volume": 62.1, "pcr": 0.50},
    {"date": "2025-12-09", "open": 106750.40, "high": 107650.40, "low": 106350.40, "close": 107350.40, "volume": 54.3, "pcr": 0.48},
    {"date": "2025-12-10", "open": 107350.40, "high": 108250.40, "low": 106950.40, "close": 107950.40, "volume": 66.5, "pcr": 0.46},
    {"date": "2025-12-11", "open": 107950.40, "high": 108750.40, "low": 107550.40, "close": 108450.40, "volume": 59.2, "pcr": 0.49},
    {"date": "2025-12-12", "open": 108450.40, "high": 109250.40, "low": 108050.40, "close": 108950.40, "volume": 71.8, "pcr": 0.47},
    {"date": "2025-12-15", "open": 108950.40, "high": 109750.40, "low": 108550.40, "close": 109450.40, "volume": 64.3, "pcr": 0.45},
    {"date": "2025-12-16", "open": 109450.40, "high": 110250.40, "low": 109050.40, "close": 109950.40, "volume": 68.9, "pcr": 0.43},
    {"date": "2025-12-17", "open": 109950.40, "high": 110650.40, "low": 109550.40, "close": 110350.40, "volume": 72.4, "pcr": 0.46},
    {"date": "2025-12-18", "open": 110350.40, "high": 111050.40, "low": 109950.40, "close": 110750.40, "volume": 65.7, "pcr": 0.44},
    {"date": "2025-12-19", "open": 110750.40, "high": 111450.40, "low": 110350.40, "close": 111150.40, "volume": 77.2, "pcr": 0.42},
    {"date": "2025-12-22", "open": 111150.40, "high": 111850.40, "low": 110750.40, "close": 111550.40, "volume": 69.8, "pcr": 0.40},
    {"date": "2025-12-23", "open": 111550.40, "high": 112250.40, "low": 111150.40, "close": 111950.40, "volume": 81.5, "pcr": 0.38},
    {"date": "2025-12-24", "open": 111950.40, "high": 112550.40, "low": 111550.40, "close": 112250.40, "volume": 74.1, "pcr": 0.41},
    {"date": "2025-12-26", "open": 112250.40, "high": 112950.40, "low": 111850.40, "close": 112650.40, "volume": 85.3, "pcr": 0.39},
    {"date": "2025-12-29", "open": 112650.40, "high": 113350.40, "low": 112250.40, "close": 113050.40, "volume": 78.6, "pcr": 0.37},
    {"date": "2025-12-30", "open": 113050.40, "high": 113750.40, "low": 112650.40, "close": 113450.40, "volume": 88.2, "pcr": 0.35},
    {"date": "2025-12-31", "open": 113450.40, "high": 114150.40, "low": 113050.40, "close": 113850.40, "volume": 82.4, "pcr": 0.38},
    # 2026年1月 - 继续上涨后回调
    {"date": "2026-01-02", "open": 113850.40, "high": 114550.40, "low": 113450.40, "close": 114250.40, "volume": 91.5, "pcr": 0.36},
    {"date": "2026-01-05", "open": 114250.40, "high": 114950.40, "low": 113850.40, "close": 114650.40, "volume": 84.7, "pcr": 0.34},
    {"date": "2026-01-06", "open": 114650.40, "high": 115350.40, "low": 114250.40, "close": 115050.40, "volume": 95.3, "pcr": 0.32},
    {"date": "2026-01-07", "open": 115050.40, "high": 115750.40, "low": 114650.40, "close": 115450.40, "volume": 88.9, "pcr": 0.35},
    {"date": "2026-01-08", "open": 115450.40, "high": 116150.40, "low": 115050.40, "close": 115850.40, "volume": 102.4, "pcr": 0.33},
    {"date": "2026-01-09", "open": 115850.40, "high": 116550.40, "low": 115450.40, "close": 116250.40, "volume": 94.6, "pcr": 0.31},
    {"date": "2026-01-12", "open": 116250.40, "high": 116850.40, "low": 115850.40, "close": 116550.40, "volume": 108.7, "pcr": 0.29},
    {"date": "2026-01-13", "open": 116550.40, "high": 117250.40, "low": 116150.40, "close": 116950.40, "volume": 99.8, "pcr": 0.32},
    {"date": "2026-01-14", "open": 116950.40, "high": 117550.40, "low": 116550.40, "close": 117250.40, "volume": 112.3, "pcr": 0.30},
    {"date": "2026-01-15", "open": 117250.40, "high": 117850.40, "low": 116950.40, "close": 117550.40, "volume": 104.5, "pcr": 0.28},
    {"date": "2026-01-16", "open": 117550.40, "high": 118150.40, "low": 117150.40, "close": 117950.40, "volume": 118.2, "pcr": 0.31},
    {"date": "2026-01-19", "open": 117950.40, "high": 118650.40, "low": 117550.40, "close": 118350.40, "volume": 109.4, "pcr": 0.29},
    {"date": "2026-01-20", "open": 118350.40, "high": 118950.40, "low": 117950.40, "close": 118650.40, "volume": 122.8, "pcr": 0.27},
    {"date": "2026-01-21", "open": 118650.40, "high": 119350.40, "low": 118350.40, "close": 119050.40, "volume": 115.3, "pcr": 0.30},
    {"date": "2026-01-22", "open": 119050.40, "high": 119650.40, "low": 118650.40, "close": 119350.40, "volume": 128.5, "pcr": 0.28},
    {"date": "2026-01-23", "open": 119350.40, "high": 119950.40, "low": 118950.40, "close": 119650.40, "volume": 119.7, "pcr": 0.26},
    {"date": "2026-01-26", "open": 119650.40, "high": 120250.40, "low": 119350.40, "close": 119950.40, "volume": 135.2, "pcr": 0.24},
    {"date": "2026-01-27", "open": 119950.40, "high": 120550.40, "low": 119650.40, "close": 120250.40, "volume": 124.8, "pcr": 0.29},
    {"date": "2026-01-28", "open": 120250.40, "high": 119450.40, "low": 117850.40, "close": 118150.40, "volume": 185.3, "pcr": 1.85},
    {"date": "2026-01-29", "open": 118150.40, "high": 118850.40, "low": 117350.40, "close": 117650.40, "volume": 142.7, "pcr": 1.65},
    {"date": "2026-01-30", "open": 117650.40, "high": 118650.40, "low": 117250.40, "close": 118350.40, "volume": 138.9, "pcr": 0.95},
    {"date": "2026-02-02", "open": 118350.40, "high": 119250.40, "low": 117850.40, "close": 118950.40, "volume": 115.2, "pcr": 0.55},
    {"date": "2026-02-03", "open": 118950.40, "high": 119650.40, "low": 118550.40, "close": 119350.40, "volume": 108.4, "pcr": 0.48},
    {"date": "2026-02-04", "open": 119350.40, "high": 120050.40, "low": 118950.40, "close": 119750.40, "volume": 125.7, "pcr": 0.42},
    {"date": "2026-02-05", "open": 119750.40, "high": 120450.40, "low": 119350.40, "close": 120150.40, "volume": 118.3, "pcr": 0.38},
    {"date": "2026-02-06", "open": 120150.40, "high": 120850.40, "low": 119750.40, "close": 120450.40, "volume": 132.5, "pcr": 0.35},
]

# ETH (Ethereum) - 历史数据模拟
# ETH价格约在BTC的1/100左右波动
eth_data = []
eth_base = 3500
for i, btc in enumerate(btc_data):
    btc_close = btc["close"]
    # ETH/BTC比率在0.028-0.032之间波动
    ratio = 0.028 + (i % 10) * 0.0004
    if i > 70:  # 1月底回调
        ratio = 0.027 + (i % 8) * 0.0003
    eth_close = btc_close * ratio
    eth_high = eth_close * (1 + np.random.uniform(0.003, 0.008))
    eth_low = eth_close * (1 - np.random.uniform(0.003, 0.008))
    eth_open = (eth_high + eth_low) / 2
    eth_volume = btc["volume"] * 120  # ETH成交量更大

    eth_data.append({
        "date": btc["date"],
        "open": round(eth_open, 2),
        "high": round(eth_high, 2),
        "low": round(eth_low, 2),
        "close": round(eth_close, 2),
        "volume": round(eth_volume, 1),
        "pcr": round(btc["pcr"] * 0.95, 2)
    })

# BNB (Binance Coin) - 历史数据模拟
# BNB价格相对独立,在600-750区间波动
bnb_data = []
bnb_base = 620
for i, btc in enumerate(btc_data):
    # BNB与BTC相关性较弱
    trend = 1 + (i % 20 - 10) * 0.002  # 小幅波动
    if i < 40:  # 11月-12月上涨
        trend = 1 + (i / 40) * 0.15
    elif i < 70:  # 1月继续上涨
        trend = 1.15 + ((i - 40) / 30) * 0.08
    else:  # 回调
        trend = 1.18 - ((i - 70) * 0.005)

    bnb_close = bnb_base * trend
    bnb_high = bnb_close * (1 + np.random.uniform(0.005, 0.015))
    bnb_low = bnb_close * (1 - np.random.uniform(0.005, 0.015))
    bnb_open = (bnb_high + bnb_low) / 2
    bnb_volume = np.random.uniform(8, 25)

    bnb_data.append({
        "date": btc["date"],
        "open": round(bnb_open, 2),
        "high": round(bnb_high, 2),
        "low": round(bnb_low, 2),
        "close": round(bnb_close, 2),
        "volume": round(bnb_volume, 1),
        "pcr": round(np.random.uniform(0.8, 1.2), 2)
    })

# ==========================================
# DTI 分析框架
# ==========================================

def calculate_sma(data, period):
    """计算简单移动平均线"""
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def calculate_ema(data, period):
    """计算指数移动平均线"""
    if len(data) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = data[0]
    for price in data[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    return ema

def calculate_rsi(closes, period=14):
    """计算RSI指标"""
    if len(closes) < period + 1:
        return 50

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(closes):
    """计算MACD指标"""
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)

    if ema12 is None or ema26 is None:
        return {"macd": 0, "signal": 0, "histogram": 0}

    macd = ema12 - ema26

    # 简化的signal line计算
    signal_line_history = []
    for i in range(max(0, len(closes) - 9), len(closes)):
        e12 = calculate_ema(closes[:i+1], 12)
        e26 = calculate_ema(closes[:i+1], 26)
        if e12 and e26:
            signal_line_history.append(e12 - e26)

    signal = calculate_ema(signal_line_history, 9) if signal_line_history else 0
    histogram = macd - (signal or 0)

    return {"macd": macd, "signal": signal, "histogram": histogram}

def calculate_atr(data, period=14):
    """计算ATR (Average True Range)"""
    if len(data) < period + 1:
        return sum([d["high"] - d["low"] for d in data[-period:]]) / period

    trs = []
    for i in range(1, len(data)):
        high = data[i]["high"]
        low = data[i]["low"]
        prev_close = data[i - 1]["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

    return sum(trs[-period:]) / period

def analyze_wyckoff_phase(data):
    """
    分析威科夫市场阶段
    返回: (阶段, 信号强度, 描述)
    """
    closes = [d["close"] for d in data]
    volumes = [d["volume"] for d in data]

    if len(closes) < 20:
        return ("NEUTRAL", 0, "数据不足")

    sma20 = calculate_sma(closes, 20)
    sma50 = calculate_sma(closes, 50) if len(closes) >= 50 else sma20
    current_price = closes[-1]

    # 计算价格趋势
    short_trend = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
    medium_trend = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0

    # 计算成交量趋势
    vol_avg = sum(volumes[-20:]) / 20
    vol_current = volumes[-1] / vol_avg if vol_avg > 0 else 1

    # RSI用于判断超买超卖
    rsi = calculate_rsi(closes)

    # 判断阶段
    phase = "NEUTRAL"
    signal_strength = 0

    if rsi > 70:
        if current_price > sma20 and short_trend > 0:
            phase = "DISTRIBUTION"  # 派发
            signal_strength = -0.3
        else:
            phase = "MARKDOWN"  # 下跌
            signal_strength = -0.5
    elif rsi < 30:
        if current_price < sma20 and short_trend < 0:
            phase = "ACCUMULATION"  # 积累
            signal_strength = 0.4
        else:
            phase = "REACCUMULATION"  # 再积累
            signal_strength = 0.3
    else:
        if current_price > sma20 and medium_trend > 0.02:
            phase = "MARKUP"  # 上涨
            signal_strength = 0.25
        elif current_price < sma20 and medium_trend < -0.02:
            phase = "MARKDOWN"  # 下跌
            signal_strength = -0.35
        elif short_trend > 0 and vol_current > 1.2:
            phase = "MARKUP"  # 突破
            signal_strength = 0.35
        elif short_trend < 0 and vol_current > 1.3:
            phase = "DISTRIBUTION"  # 下跌突破
            signal_strength = -0.4

    return (phase, signal_strength, f"趋势: {short_trend:.2%}")

def analyze_market_profile(data):
    """
    分析Market Profile / Volume Profile
    返回: (信号强度, 价值区间位置, POC)
    """
    closes = [d["close"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    volumes = [d["volume"] for d in data]

    if len(closes) < 20:
        return (0, "中部", closes[-1])

    # 计算价值区间
    period_highs = highs[-20:]
    period_lows = lows[-20:]

    vah = sum(period_highs) / len(period_highs)  # Value Area High
    val = sum(period_lows) / len(period_lows)    # Value Area Low
    poc = (vah + val) / 2                         # Point of Control

    current_price = closes[-1]

    # 判断价格在价值区间的位置
    value_range = vah - val
    if value_range == 0:
        value_position = "中部"
    else:
        position_ratio = (current_price - val) / value_range
        if position_ratio > 0.7:
            value_position = "高位"
        elif position_ratio < 0.3:
            value_position = "低位"
        else:
            value_position = "中部"

    # 信号强度
    signal_strength = 0
    if value_position == "高位" and current_price > vah:
        signal_strength = -0.2  # 可能有回调
    elif value_position == "低位" and current_price < val:
        signal_strength = 0.3   # 可能反弹
    elif value_position == "高位" and closes[-1] > closes[-5]:
        signal_strength = 0.2   # 强势上涨
    elif value_position == "低位" and closes[-1] < closes[-5]:
        signal_strength = -0.25 # 弱势下跌

    return (signal_strength, value_position, poc)

def analyze_momentum(data):
    """
    分析动量指标 (RSI, MACD)
    返回: (信号强度, 超买超卖状态)
    """
    closes = [d["close"] for d in data]

    if len(closes) < 14:
        return (0, "中性")

    rsi = calculate_rsi(closes, 14)
    macd_result = calculate_macd(closes)

    # 超买超卖判断
    if rsi > 70:
        overbought_status = "超买"
    elif rsi < 30:
        overbought_status = "超卖"
    elif rsi > 60:
        overbought_status = "偏多"
    elif rsi < 40:
        overbought_status = "偏空"
    else:
        overbought_status = "中性"

    # 信号强度
    signal_strength = 0

    # RSI信号
    if rsi < 30:
        signal_strength += 0.3  # 超卖反弹信号
    elif rsi > 70:
        signal_strength -= 0.3  # 超买回调信号
    elif 45 <= rsi <= 55:
        signal_strength += 0.1  # 中性偏多

    # MACD信号
    macd_val = macd_result["macd"] or 0
    signal_val = macd_result["signal"] or 0

    if macd_result["histogram"] and macd_result["histogram"] > 0:
        if macd_val > signal_val:
            signal_strength += 0.15  # 金叉
    elif macd_result["histogram"] and macd_result["histogram"] < 0:
        if macd_val < signal_val:
            signal_strength -= 0.15  # 死叉

    return (signal_strength, overbought_status)

def analyze_volatility(data):
    """
    分析波动率 (ATR)
    返回: (信号强度, 波动率水平)
    """
    atr = calculate_atr(data, 14)
    closes = [d["close"] for d in data]

    if len(closes) < 14:
        return (0, "中等")

    current_price = closes[-1]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    # 波动率分类
    if atr_pct > 3:
        volatility_level = "极高"
    elif atr_pct > 2:
        volatility_level = "高"
    elif atr_pct > 1:
        volatility_level = "中等"
    else:
        volatility_level = "低"

    # 信号强度 - 波动率太高时减仓,低时可以加仓
    signal_strength = 0
    if atr_pct < 1:
        signal_strength = 0.1   # 低波动,可能突破
    elif atr_pct > 3:
        signal_strength = -0.15 # 高波动,谨慎

    return (signal_strength, volatility_level)

def analyze_sentiment(data):
    """
    分析市场情绪 (Put/Call Ratio)
    返回: (信号强度, 情绪状态)
    """
    pcr = data[-1]["pcr"]

    # PCR解读: 高PCR表示看跌期权多,市场悲观(反向指标)
    if pcr > 1.3:
        sentiment_status = "极度恐慌"
        signal_strength = 0.3  # 恐慌时反向买入
    elif pcr > 1.0:
        sentiment_status = "恐慌"
        signal_strength = 0.15
    elif pcr < 0.5:
        sentiment_status = "极度贪婪"
        signal_strength = -0.25 # 贪婪时反向卖出
    elif pcr < 0.7:
        sentiment_status = "贪婪"
        signal_strength = -0.1
    else:
        sentiment_status = "中性"
        signal_strength = 0

    return (signal_strength, sentiment_status)

def generate_signal(data):
    """
    生成综合交易信号
    返回: {方向, 上涨概率, 下跌概率, 信心度, 预期涨跌, 止损, 止盈, 仓位, 原因}
    """
    # 各组件权重
    W_WYCKOFF = 0.25
    W_MARKET_PROFILE = 0.20
    W_MOMENTUM = 0.20
    W_VOLATILITY = 0.15
    W_SENTIMENT = 0.20

    # 获取各组件分析
    wyckoff_phase, wyckoff_signal, wyckoff_desc = analyze_wyckoff_phase(data)
    mp_signal, value_position, poc = analyze_market_profile(data)
    mom_signal, overbought = analyze_momentum(data)
    vol_signal, volatility = analyze_volatility(data)
    sent_signal, sentiment = analyze_sentiment(data)

    # 计算综合信号
    raw_signal = (
        wyckoff_signal * W_WYCKOFF +
        mp_signal * W_MARKET_PROFILE +
        mom_signal * W_MOMENTUM +
        vol_signal * W_VOLATILITY +
        sent_signal * W_SENTIMENT
    )

    # 转换为概率
    # raw_signal范围约 -1 到 1
    # 映射到概率: raw_signal=1 -> 80% up, 20% down
    #              raw_signal=-1 -> 20% up, 80% down
    #              raw_signal=0 -> 50% up, 50% down

    prob_up = 50 + raw_signal * 30
    prob_down = 100 - prob_up

    prob_up = max(20, min(80, prob_up))
    prob_down = 100 - prob_up

    # 信心度 = raw_signal的绝对值 * 100
    confidence = abs(raw_signal) * 100

    # 确定方向
    if prob_up >= 55 and confidence >= 5:
        direction = "LONG"
    elif prob_down >= 55 and confidence >= 5:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # 预期涨跌幅
    atr = calculate_atr(data, 14)
    current_price = data[-1]["close"]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 2

    expected_move = raw_signal * atr_pct * 2

    # 止损止盈
    stop_loss = current_price - (atr * 1.5) if direction == "LONG" else current_price + (atr * 1.5)
    if direction == "NEUTRAL":
        stop_loss = current_price - (atr * 1.5)

    tp1 = current_price + (atr * 2) if direction == "LONG" else current_price - (atr * 2)
    tp2 = current_price + (atr * 3) if direction == "LONG" else current_price - (atr * 3)
    tp3 = current_price + (atr * 5) if direction == "LONG" else current_price - (atr * 5)

    if direction == "NEUTRAL":
        tp1 = current_price + (atr * 2)
        tp2 = current_price - (atr * 1.5)

    # 仓位建议
    if confidence >= 20:
        position_size = 0.5
    elif confidence >= 15:
        position_size = 0.3
    elif confidence >= 10:
        position_size = 0.2
    elif confidence >= 5:
        position_size = 0.1
    else:
        position_size = 0.05

    # 原因
    reason = f"威科夫阶段: {wyckoff_phase} | 价值区间位置: {value_position} | 综合信号: {raw_signal:.2f}"

    return {
        "direction": direction,
        "prob_up": round(prob_up, 1),
        "prob_down": round(prob_down, 1),
        "confidence": round(confidence, 1),
        "expected_move": round(expected_move, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": [round(tp1, 2), round(tp2, 2), round(tp3, 2)] if direction != "NEUTRAL" else [round(tp1, 2), round(tp2, 2)],
        "position_size": position_size,
        "reason": reason
    }

# ==========================================
# 回测引擎
# ==========================================

def run_backtest(data, symbol="SYMBOL"):
    """运行回测"""
    capital = 1000
    position = None
    trades = []

    for i in range(30, len(data) - 1):  # 从第30天开始,留一天用于执行
        current_data = data[:i+1]
        today = data[i]
        next_day = data[i+1]

        # 生成信号
        signal = generate_signal(current_data)

        # 如果没有持仓,检查入场
        if position is None:
            if signal["direction"] == "LONG" and signal["confidence"] >= 5:
                entry_price = today["close"]
                atr = calculate_atr(current_data, 14)

                position = {
                    "entry_date": today["date"],
                    "entry_price": entry_price,
                    "direction": "LONG",
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "capital_used": capital * signal["position_size"],
                    "position_size": signal["position_size"],
                    "days_held": 0
                }
            elif signal["direction"] == "SHORT" and signal["confidence"] >= 5:
                entry_price = today["close"]
                atr = calculate_atr(current_data, 14)

                position = {
                    "entry_date": today["date"],
                    "entry_price": entry_price,
                    "direction": "SHORT",
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "capital_used": capital * signal["position_size"],
                    "position_size": signal["position_size"],
                    "days_held": 0
                }

        # 如果有持仓,检查出场
        elif position is not None:
            position["days_held"] += 1

            exit_reason = None
            exit_price = None

            if position["direction"] == "LONG":
                # 检查止损
                if next_day["low"] <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "SL"
                # 检查止盈1
                elif next_day["high"] >= position["take_profit"][0]:
                    exit_price = position["take_profit"][0]
                    exit_reason = "TP"
                # 检查反向信号
                elif signal["direction"] == "SHORT":
                    exit_price = next_day["close"]
                    exit_reason = "REVERSE"
                # 时间止损 (5天)
                elif position["days_held"] >= 5:
                    exit_price = next_day["close"]
                    exit_reason = "TIME"
                else:
                    exit_price = next_day["close"]
                    exit_reason = None

            else:  # SHORT
                if next_day["high"] >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "SL"
                elif next_day["low"] <= position["take_profit"][0]:
                    exit_price = position["take_profit"][0]
                    exit_reason = "TP"
                elif signal["direction"] == "LONG":
                    exit_price = next_day["close"]
                    exit_reason = "REVERSE"
                elif position["days_held"] >= 5:
                    exit_price = next_day["close"]
                    exit_reason = "TIME"
                else:
                    exit_price = next_day["close"]
                    exit_reason = None

            if exit_reason:
                # 计算盈亏
                if position["direction"] == "LONG":
                    pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                else:
                    pnl_pct = (position["entry_price"] - exit_price) / position["entry_price"] * 100

                pnl_amount = position["capital_used"] * pnl_pct / 100

                trade = {
                    "entry_date": position["entry_date"],
                    "exit_date": next_day["date"],
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_amount": pnl_amount,
                    "reason": exit_reason
                }
                trades.append(trade)

                capital += pnl_amount
                position = None

    # 处理最后持仓
    if position:
        last_day = data[-1]
        if position["direction"] == "LONG":
            pnl_pct = (last_day["close"] - position["entry_price"]) / position["entry_price"] * 100
        else:
            pnl_pct = (position["entry_price"] - last_day["close"]) / position["entry_price"] * 100

        pnl_amount = position["capital_used"] * pnl_pct / 100

        trade = {
            "entry_date": position["entry_date"],
            "exit_date": last_day["date"],
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": last_day["close"],
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
            "reason": "END"
        }
        trades.append(trade)
        capital += pnl_amount

    # 计算统计
    win_trades = [t for t in trades if t["pnl_pct"] > 0]
    lose_trades = [t for t in trades if t["pnl_pct"] <= 0]

    total_trades = len(trades)
    win_rate = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0

    avg_win = sum([t["pnl_pct"] for t in win_trades]) / len(win_trades) if win_trades else 0
    avg_loss = sum([t["pnl_pct"] for t in lose_trades]) / len(lose_trades) if lose_trades else 0

    max_win = max([t["pnl_pct"] for t in win_trades]) if win_trades else 0
    max_loss = min([t["pnl_pct"] for t in lose_trades]) if lose_trades else 0

    # 计算最大回撤
    peak = 1000
    max_drawdown = 0
    current_capital = 1000

    for t in trades:
        current_capital += t["pnl_amount"]
        if current_capital > peak:
            peak = current_capital
        drawdown = (peak - current_capital) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # 计算夏普比率 (简化)
    returns = [t["pnl_pct"] for t in trades]
    avg_return = sum(returns) / len(returns) if returns else 0
    return_std = np.std(returns) if returns else 1
    sharpe_ratio = (avg_return / return_std) if return_std > 0 else 0

    profit_factor = abs(sum([t["pnl_amount"] for t in win_trades]) / sum([t["pnl_amount"] for t in lose_trades])) if lose_trades and sum([t["pnl_amount"] for t in lose_trades]) < 0 else 0

    return {
        "symbol": symbol,
        "initial_capital": 1000,
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - 1000) / 1000 * 100, 2),
        "total_trades": total_trades,
        "win_trades": len(win_trades),
        "lose_trades": len(lose_trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_win_pct": round(max_win, 2),
        "max_loss_pct": round(max_loss, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "trades": trades
    }

# ==========================================
# 主程序
# ==========================================

def main():
    print("=" * 80)
    print("                BTC & ETH & BNB 每日技术指标回测报告")
    print("          DTI - Crypto Backtest (BTC/ETH/BNB)")
    print("=" * 80)
    print()

    print("正在加载历史数据...")
    print(f"初始资金: $1000")
    print(f"回测周期: {btc_data[0]['date']} 至 {btc_data[-1]['date']} (约3个月)")
    print()

    # 运行回测
    btc_result = run_backtest(btc_data, "BTC")
    eth_result = run_backtest(eth_data, "ETH")
    bnb_result = run_backtest(bnb_data, "BNB")

    # 打印结果
    symbols = [
        ("BTC (比特币)", btc_result),
        ("ETH (以太坊)", eth_result),
        ("BNB (币安币)", bnb_result)
    ]

    for symbol_name, result in symbols:
        print("=" * 80)
        print(f"                            {symbol_name} 回测结果")
        print("=" * 80)
        print()
        print("【回测统计】")
        print("-" * 60)
        print(f"初始资金:        ${result['initial_capital']:.2f}")
        print(f"最终资金:        ${result['final_capital']:.2f}")
        print(f"总收益率:        {result['total_return_pct']}%")
        print(f"总交易次数:      {result['total_trades']}")
        print(f"盈利交易:        {result['win_trades']}")
        print(f"亏损交易:        {result['lose_trades']}")
        print(f"胜率:            {result['win_rate_pct']}%")
        print(f"平均盈利:        {result['avg_win_pct']}%")
        print(f"平均亏损:        {result['avg_loss_pct']}%")
        print(f"最大单笔盈利:    {result['max_win_pct']}%")
        print(f"最大单笔亏损:    {result['max_loss_pct']}%")
        print(f"最大回撤:        {result['max_drawdown_pct']}%")
        print(f"夏普比率:        {result['sharpe_ratio']}")
        print(f"盈亏比:          {result['profit_factor']}")
        print()
        print("【交易明细】")
        print("-" * 78)
        print(f"{'日期':<12} {'方向':<6} {'入场价':<12} {'出场价':<12} {'盈亏%':<8} {'盈亏$':<8} {'原因':<6}")
        print("-" * 78)

        for t in result["trades"]:
            pnl_sign = "+" if t["pnl_pct"] >= 0 else ""
            print(f"{t['entry_date']:<12} {t['direction']:<6} "
                  f"{t['entry_price']:<12.2f} {t['exit_price']:<12.2f} "
                  f"{pnl_sign}{t['pnl_pct']:<7.2f}% ${pnl_sign}{t['pnl_amount']:<7.2f} {t['reason']:<6}")
        print()

    # 生成最新信号
    print("=" * 80)
    print("                        最新交易信号")
    print("=" * 80)
    print()

    latest_signals = {
        "BTC": generate_signal(btc_data),
        "ETH": generate_signal(eth_data),
        "BNB": generate_signal(bnb_data)
    }

    for symbol, signal in latest_signals.items():
        print(f"【{symbol} 最新信号】")
        print("-" * 60)
        print(f"日期:            {btc_data[-1]['date']}")
        print(f"方向:            {signal['direction']}")
        print(f"信心度:          {signal['confidence']}%")
        print(f"上涨概率:        {signal['prob_up']}%")
        print(f"下跌概率:        {signal['prob_down']}%")
        print(f"预期涨跌幅:      {signal['expected_move']:.2f}%")
        print(f"建议止损:        {signal['stop_loss']:.2f}")
        print(f"建议止盈:        {signal['take_profit']}")
        print(f"建议仓位:        {signal['position_size']*100}%")
        print(f"原因:            {signal['reason']}")
        print()

    # 保存结果
    output = {
        "backtest_summary": {
            "btc": {k: v for k, v in btc_result.items() if k != "trades"},
            "eth": {k: v for k, v in eth_result.items() if k != "trades"},
            "bnb": {k: v for k, v in bnb_result.items() if k != "trades"}
        },
        "latest_signals": {
            "btc": {
                "date": btc_data[-1]["date"],
                "symbol": "BTC",
                **latest_signals["BTC"]
            },
            "eth": {
                "date": eth_data[-1]["date"],
                "symbol": "ETH",
                **latest_signals["ETH"]
            },
            "bnb": {
                "date": bnb_data[-1]["date"],
                "symbol": "BNB",
                **latest_signals["BNB"]
            }
        }
    }

    with open("/root/rich/crypto_result.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("回测结果已保存至: crypto_result.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
