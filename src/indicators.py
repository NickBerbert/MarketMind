import numpy as np


def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50)


def calcular_macd(prices, fast=12, slow=26):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return (ema_fast - ema_slow).fillna(0)


def calcular_bollinger_bands(prices, window=20, std_dev=2):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper.fillna(prices), lower.fillna(prices)


def calcular_atr(high, low, close, period=14):
    """Average True Range — measures volatility."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = tr1.combine(tr2, max).combine(tr3, max)
    return tr.rolling(window=period).mean().fillna(tr)


def calcular_stochastic(high, low, close, period=14):
    """Stochastic %K — momentum oscillator (0-100)."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    denom = highest_high - lowest_low
    stoch_k = ((close - lowest_low) / denom.where(denom != 0, 1)) * 100
    return stoch_k.fillna(50)


def calcular_obv(close, volume):
    """On-Balance Volume — cumulative volume direction indicator."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv.fillna(0)


def calcular_williams_r(high, low, close, period=14):
    """Williams %R — overbought/oversold oscillator (-100 to 0)."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = highest_high - lowest_low
    wr = ((highest_high - close) / denom.where(denom != 0, 1)) * -100
    return wr.fillna(-50)


def calcular_roc(prices, period=10):
    """Rate of Change — percentage price change over n periods."""
    return prices.pct_change(period).fillna(0) * 100
