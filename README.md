# Daily Technical Indicator (DTI) System

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English Documentation

### Overview

The **Daily Technical Indicator (DTI)** is a comprehensive trading analysis system that combines multiple technical analysis methodologies to generate trading signals for financial instruments.

**Supported Instruments:**
- Gold (XAUUSD)
- Silver (XAGUSD)
- S&P 500 (SPX500)
- Nasdaq 100 (NDX100)

### Features

- **Wyckoff Method Analysis** (25% weight): Identifies market phases (Accumulation, Markup, Distribution, Markdown)
- **Market Profile Analysis** (20% weight): Volume Profile with POC, VAL, VAH
- **Momentum Indicators** (20% weight): RSI, MACD
- **Volatility Analysis** (15% weight): ATR, Historical Volatility
- **Sentiment Analysis** (20% weight): Put/Call Ratio, Price Momentum

### Installation

```bash
# Clone the repository
git clone https://github.com/lawrencezcl/gold-silver-trading-indicator.git
cd gold-silver-trading-indicator

# Install dependencies
pip install numpy pandas
```

### Usage

#### Running the Backtest

```bash
# For Gold and Silver
python3 daily_indicator.py

# For Indices (SPX/NDX)
python3 daily_indicator_indices.py
```

#### Output Files

- `daily_indicator_result.json` - Backtest results in JSON format
- `DTI_BACKTEST_REPORT.md` - Detailed analysis report
- `XAUXAG20260207.md` - Daily market analysis for Gold/Silver
- `SPXNDX+20260207.md` - Daily market analysis for Indices

### Trading Rules

**Entry Conditions:**
- Probability > 55%
- Confidence > 5%

**Stop Loss:** 1.5x ATR

**Take Profit:**
- TP1: 2x ATR
- TP2: 3x ATR
- TP3: 5x ATR

**Exit Conditions:**
1. Stop Loss or Take Profit hit
2. Reverse signal with confidence > 20%
3. Position held > 5 trading days

### Backtest Results (3 Months)

| Instrument | Return | Win Rate | Sharpe Ratio |
|------------|--------|----------|--------------|
| XAUUSD     | +2.37% | 87.50%   | 4.41         |
| XAGUSD     | +9.57% | 90.00%   | 4.74         |
| SPX500     | +1.38% | 85.71%   | 3.99         |
| NDX100     | +4.40% | 100.00%  | 5.38         |

### Risk Disclaimer

This system is for educational purposes only. Past performance does not guarantee future results. Always use proper risk management and never risk more than you can afford to lose.

### License

MIT License

---

<a name="中文"></a>
## 中文文档

### 概述

**每日技术指标系统 (DTI)** 是一个综合性的交易分析系统，结合多种技术分析方法为金融工具生成交易信号。

**支持品种:**
- 黄金 (XAUUSD)
- 白银 (XAGUSD)
- 标普500 (SPX500)
- 纳斯达克100 (NDX100)

### 功能特点

- **威科夫方法分析** (权重25%): 识别市场阶段(吸筹/上涨/派发/下跌)
- **四度空间理论** (权重20%): 成交量分布POC/VAL/VAH分析
- **动量指标** (权重20%): RSI、MACD
- **波动率分析** (权重15%): ATR、历史波动率
- **市场情绪** (权重20%): Put/Call比率、价格动量

### 安装

```bash
# 克隆仓库
git clone https://github.com/lawrencezcl/gold-silver-trading-indicator.git
cd gold-silver-trading-indicator

# 安装依赖
pip install numpy pandas
```

### 使用方法

#### 运行回测

```bash
# 贵金属分析(黄金/白银)
python3 daily_indicator.py

# 股指分析(SPX/NDX)
python3 daily_indicator_indices.py
```

#### 输出文件

- `daily_indicator_result.json` - 回测结果JSON格式
- `DTI_BACKTEST_REPORT.md` - 详细分析报告
- `XAUXAG20260207.md` - 黄金白银每日市场分析
- `SPXNDX+20260207.md` - 股指每日市场分析

### 交易规则

**入场条件:**
- 上涨/下跌概率 > 55%
- 信心度 > 5%

**止损:** 1.5倍ATR

**止盈:**
- TP1: 2倍ATR
- TP2: 3倍ATR
- TP3: 5倍ATR

**出场条件:**
1. 触及止损或止盈
2. 反向信号且信心度 > 20%
3. 持仓超过5个交易日

### 回测结果 (3个月)

| 品种 | 收益率 | 胜率 | 夏普比率 |
|------|--------|------|----------|
| XAUUSD | +2.37% | 87.50% | 4.41 |
| XAGUSD | +9.57% | 90.00% | 4.74 |
| SPX500 | +1.38% | 85.71% | 3.99 |
| NDX100 | +4.40% | 100.00% | 5.38 |

### 风险提示

本系统仅供教育研究使用。历史表现不代表未来收益。请务必做好风险管理，切勿投入超出承受能力的资金。

### 许可证

MIT License

---

## Project Structure

```
gold-silver-trading-indicator/
├── README.md                          # This file
├── daily_indicator.py                 # DTI for Gold/Silver
├── daily_indicator_indices.py         # DTI for Indices
├── daily_indicator_result.json        # Gold/Silver backtest results
├── indices_indicator_result.json      # Indices backtest results
├── DTI_BACKTEST_REPORT.md             # Detailed backtest report
├── XAUXAG20260207.md                  # Gold/Silver daily analysis
└── SPXNDX+20260207.md                 # Indices daily analysis
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue on GitHub.
