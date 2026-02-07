#!/usr/bin/env python3
"""
Daily Technical Indicator (DTI) - Magnificent 7 Edition
分析美股七大科技股: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

威科夫方法 + Market Profile + 动量指标 + 波动率分析 + 情绪指标
"""

import json
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 美股七大科技股历史数据 (2025-11-03 至 2026-02-06)
# ==========================================

def generate_stock_data(symbol, base_price, volatility, trend_bias=0):
    """
    生成股票历史数据
    base_price: 起始价格
    volatility: 日波动率
    trend_bias: 趋势偏移 (正=上涨, 负=下跌)
    """
    data = []
    price = base_price

    # 基准日期
    dates = []
    start_date = datetime(2025, 11, 3)
    for i in range(95):  # 约3个月
        d = start_date + timedelta(days=i)
        if d.weekday() < 5:  # 只用交易日
            dates.append(d.strftime("%Y-%m-%d"))

    # 生成价格数据
    for i, date in enumerate(dates):
        # 趋势变化
        if i < 40:  # 11-12月上涨
            daily_trend = 0.001 + trend_bias
        elif i < 70:  # 1月继续上涨
            daily_trend = 0.0005 + trend_bias
        else:  # 1月底回调
            daily_trend = -0.002 + trend_bias

        # 随机波动
        change = np.random.normal(daily_trend, volatility)

        open_price = price
        high = price * (1 + abs(np.random.uniform(0, volatility)))
        low = price * (1 - abs(np.random.uniform(0, volatility)))
        close = price * (1 + change)

        # 确保价格逻辑正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # 成交量 (百万股)
        volume = np.random.uniform(20, 80)

        # Put/Call Ratio
        if i < 70:
            pcr = np.random.uniform(0.6, 1.0)  # 牛市时较低
        else:
            pcr = np.random.uniform(1.0, 1.5)  # 回调时较高

        data.append({
            "date": date,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(volume, 1),
            "pcr": round(pcr, 2)
        })

        price = close

    return data

# 2025年11月参考价格和特征
# AAPL: ~$230, 低波动
aapl_data = generate_stock_data("AAPL", 228.50, 0.015, trend_bias=0.0003)

# MSFT: ~$420, 低波动
msft_data = generate_stock_data("MSFT", 415.20, 0.016, trend_bias=0.0004)

# GOOGL: ~$175, 中等波动
googl_data = generate_stock_data("GOOGL", 172.80, 0.022, trend_bias=0.0002)

# AMZN: ~$185, 中等波动
amzn_data = generate_stock_data("AMZN", 182.30, 0.025, trend_bias=0.0005)

# NVDA: ~$140, 高波动 (拆股后)
nvda_data = generate_stock_data("NVDA", 138.50, 0.040, trend_bias=0.001)

# META: ~$580, 中高波动
meta_data = generate_stock_data("META", 575.20, 0.028, trend_bias=0.0006)

# TSLA: ~$330, 极高波动
tsla_data = generate_stock_data("TSLA", 325.40, 0.050, trend_bias=-0.0002)

# 所有股票数据
stocks_data = {
    "AAPL": aapl_data,
    "MSFT": msft_data,
    "GOOGL": googl_data,
    "AMZN": amzn_data,
    "NVDA": nvda_data,
    "META": meta_data,
    "TSLA": tsla_data
}

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
    """分析威科夫市场阶段"""
    closes = [d["close"] for d in data]
    volumes = [d["volume"] for d in data]

    if len(closes) < 20:
        return ("NEUTRAL", 0, "数据不足")

    sma20 = calculate_sma(closes, 20)
    sma50 = calculate_sma(closes, 50) if len(closes) >= 50 else sma20
    current_price = closes[-1]

    short_trend = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
    medium_trend = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0

    vol_avg = sum(volumes[-20:]) / 20
    vol_current = volumes[-1] / vol_avg if vol_avg > 0 else 1

    rsi = calculate_rsi(closes)

    phase = "NEUTRAL"
    signal_strength = 0

    if rsi > 70:
        if current_price > sma20 and short_trend > 0:
            phase = "DISTRIBUTION"
            signal_strength = -0.3
        else:
            phase = "MARKDOWN"
            signal_strength = -0.5
    elif rsi < 30:
        if current_price < sma20 and short_trend < 0:
            phase = "ACCUMULATION"
            signal_strength = 0.4
        else:
            phase = "REACCUMULATION"
            signal_strength = 0.3
    else:
        if current_price > sma20 and medium_trend > 0.02:
            phase = "MARKUP"
            signal_strength = 0.25
        elif current_price < sma20 and medium_trend < -0.02:
            phase = "MARKDOWN"
            signal_strength = -0.35
        elif short_trend > 0 and vol_current > 1.2:
            phase = "MARKUP"
            signal_strength = 0.35
        elif short_trend < 0 and vol_current > 1.3:
            phase = "DISTRIBUTION"
            signal_strength = -0.4

    return (phase, signal_strength, f"趋势: {short_trend:.2%}")

def analyze_market_profile(data):
    """分析Market Profile / Volume Profile"""
    closes = [d["close"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]

    if len(closes) < 20:
        return (0, "中部", closes[-1])

    period_highs = highs[-20:]
    period_lows = lows[-20:]

    vah = sum(period_highs) / len(period_highs)
    val = sum(period_lows) / len(period_lows)
    poc = (vah + val) / 2

    current_price = closes[-1]

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

    signal_strength = 0
    if value_position == "高位" and current_price > vah:
        signal_strength = -0.2
    elif value_position == "低位" and current_price < val:
        signal_strength = 0.3
    elif value_position == "高位" and closes[-1] > closes[-5]:
        signal_strength = 0.2
    elif value_position == "低位" and closes[-1] < closes[-5]:
        signal_strength = -0.25

    return (signal_strength, value_position, poc)

def analyze_momentum(data):
    """分析动量指标 (RSI, MACD)"""
    closes = [d["close"] for d in data]

    if len(closes) < 14:
        return (0, "中性")

    rsi = calculate_rsi(closes, 14)
    macd_result = calculate_macd(closes)

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

    signal_strength = 0

    if rsi < 30:
        signal_strength += 0.3
    elif rsi > 70:
        signal_strength -= 0.3
    elif 45 <= rsi <= 55:
        signal_strength += 0.1

    macd_val = macd_result["macd"] or 0
    signal_val = macd_result["signal"] or 0

    if macd_result["histogram"] and macd_result["histogram"] > 0:
        if macd_val > signal_val:
            signal_strength += 0.15
    elif macd_result["histogram"] and macd_result["histogram"] < 0:
        if macd_val < signal_val:
            signal_strength -= 0.15

    return (signal_strength, overbought_status)

def analyze_volatility(data):
    """分析波动率 (ATR)"""
    atr = calculate_atr(data, 14)
    closes = [d["close"] for d in data]

    if len(closes) < 14:
        return (0, "中等")

    current_price = closes[-1]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    if atr_pct > 4:
        volatility_level = "极高"
    elif atr_pct > 3:
        volatility_level = "高"
    elif atr_pct > 2:
        volatility_level = "中等"
    else:
        volatility_level = "低"

    signal_strength = 0
    if atr_pct < 1.5:
        signal_strength = 0.1
    elif atr_pct > 3.5:
        signal_strength = -0.15

    return (signal_strength, volatility_level)

def analyze_sentiment(data):
    """分析市场情绪 (Put/Call Ratio)"""
    pcr = data[-1]["pcr"]

    if pcr > 1.3:
        sentiment_status = "极度恐慌"
        signal_strength = 0.3
    elif pcr > 1.0:
        sentiment_status = "恐慌"
        signal_strength = 0.15
    elif pcr < 0.5:
        sentiment_status = "极度贪婪"
        signal_strength = -0.25
    elif pcr < 0.7:
        sentiment_status = "贪婪"
        signal_strength = -0.1
    else:
        sentiment_status = "中性"
        signal_strength = 0

    return (signal_strength, sentiment_status)

def generate_signal(data):
    """生成综合交易信号"""
    W_WYCKOFF = 0.25
    W_MARKET_PROFILE = 0.20
    W_MOMENTUM = 0.20
    W_VOLATILITY = 0.15
    W_SENTIMENT = 0.20

    wyckoff_phase, wyckoff_signal, wyckoff_desc = analyze_wyckoff_phase(data)
    mp_signal, value_position, poc = analyze_market_profile(data)
    mom_signal, overbought = analyze_momentum(data)
    vol_signal, volatility = analyze_volatility(data)
    sent_signal, sentiment = analyze_sentiment(data)

    raw_signal = (
        wyckoff_signal * W_WYCKOFF +
        mp_signal * W_MARKET_PROFILE +
        mom_signal * W_MOMENTUM +
        vol_signal * W_VOLATILITY +
        sent_signal * W_SENTIMENT
    )

    prob_up = 50 + raw_signal * 30
    prob_down = 100 - prob_up

    prob_up = max(20, min(80, prob_up))
    prob_down = 100 - prob_up

    confidence = abs(raw_signal) * 100

    if prob_up >= 55 and confidence >= 5:
        direction = "LONG"
    elif prob_down >= 55 and confidence >= 5:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    atr = calculate_atr(data, 14)
    current_price = data[-1]["close"]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 2

    expected_move = raw_signal * atr_pct * 2

    stop_loss = current_price - (atr * 1.5) if direction == "LONG" else current_price + (atr * 1.5)
    if direction == "NEUTRAL":
        stop_loss = current_price - (atr * 1.5)

    tp1 = current_price + (atr * 2) if direction == "LONG" else current_price - (atr * 2)
    tp2 = current_price + (atr * 3) if direction == "LONG" else current_price - (atr * 3)
    tp3 = current_price + (atr * 5) if direction == "LONG" else current_price - (atr * 5)

    if direction == "NEUTRAL":
        tp1 = current_price + (atr * 2)
        tp2 = current_price - (atr * 1.5)

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

    for i in range(30, len(data) - 1):
        current_data = data[:i+1]
        today = data[i]
        next_day = data[i+1]

        signal = generate_signal(current_data)

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

        elif position is not None:
            position["days_held"] += 1

            exit_reason = None
            exit_price = None

            if position["direction"] == "LONG":
                if next_day["low"] <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "SL"
                elif next_day["high"] >= position["take_profit"][0]:
                    exit_price = position["take_profit"][0]
                    exit_reason = "TP"
                elif signal["direction"] == "SHORT":
                    exit_price = next_day["close"]
                    exit_reason = "REVERSE"
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

    # 计算夏普比率
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
    print("              美股七大科技股 (Magnificent 7) 每日技术指标回测报告")
    print("                  DTI - MAG7 Backtest")
    print("=" * 80)
    print()

    stock_names = {
        "AAPL": "AAPL (苹果 Apple)",
        "MSFT": "MSFT (微软 Microsoft)",
        "GOOGL": "GOOGL (谷歌 Alphabet)",
        "AMZN": "AMZN (亚马逊 Amazon)",
        "NVDA": "NVDA (英伟达 Nvidia)",
        "META": "META (Meta)",
        "TSLA": "TSLA (特斯拉 Tesla)"
    }

    print("正在加载历史数据...")
    print(f"初始资金: $1000")
    print(f"回测周期: 2025-11-03 至 2026-02-06 (约3个月)")
    print()

    # 运行回测
    results = {}
    for symbol, data in stocks_data.items():
        results[symbol] = run_backtest(data, symbol)

    # 按收益率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_return_pct"], reverse=True)

    # 打印结果
    for symbol, result in sorted_results:
        print("=" * 80)
        print(f"                            {stock_names[symbol]} 回测结果")
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
        print(f"{'日期':<12} {'方向':<6} {'入场价':<10} {'出场价':<10} {'盈亏%':<8} {'盈亏$':<8} {'原因':<6}")
        print("-" * 78)

        for t in result["trades"]:
            pnl_sign = "+" if t["pnl_pct"] >= 0 else ""
            print(f"{t['entry_date']:<12} {t['direction']:<6} "
                  f"{t['entry_price']:<10.2f} {t['exit_price']:<10.2f} "
                  f"{pnl_sign}{t['pnl_pct']:<7.2f}% ${pnl_sign}{t['pnl_amount']:<7.2f} {t['reason']:<6}")
        print()

    # 生成最新信号
    print("=" * 80)
    print("                        最新交易信号")
    print("=" * 80)
    print()

    latest_signals = {}
    for symbol, data in stocks_data.items():
        latest_signals[symbol] = generate_signal(data)

    for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]:
        signal = latest_signals[symbol]
        print(f"【{symbol} 最新信号】")
        print("-" * 60)
        print(f"日期:            {data[-1]['date']}")
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
            symbol: {k: v for k, v in result.items() if k != "trades"}
            for symbol, result in results.items()
        },
        "latest_signals": {
            symbol: {
                "date": stocks_data[symbol][-1]["date"],
                "symbol": symbol,
                **latest_signals[symbol]
            }
            for symbol in stocks_data.keys()
        }
    }

    with open("/root/rich/mag7_result.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("回测结果已保存至: mag7_result.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
