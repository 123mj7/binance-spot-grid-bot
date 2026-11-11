# -*- coding: utf-8 -*-
"""
strategy_engine.py —— 可无限扩展的策略生成器(Strategy Generator)
==================================================================
纯 pandas 实现, 不依赖 pandas_ta。三大扩展点全部用装饰器注册, 新指标/新算子
上线时不需要改动引擎代码:

  1. 指标注册表 INDICATORS:  @register_indicator("XXX") 添加新指标
  2. 算子注册表 OPERATORS :  @register_operator("op")    添加新算子(比较/交叉)
  3. 自定义序列注入       :  engine.register_series("名字", series)
                             之后条件里可以直接引用该名字

内置指标: EMA(5/10/20/50/100), RSI(14), MACD(12,26,9), ATR(14),
          BOLL(20, 2倍标准差), ADX(14), VOLUME(成交量/均量/量比)

内置算子: > < >= <= == != cross_above cross_below between
          (还支持中文别名: 金叉/死叉)

条件语法: 三元组 (left, op, right)
  left/right 支持: 指标名(如 'EMA20'/'RSI')、原始列名(如 'close')、
  数字常量(如 30)、任意可调用对象(如 lambda df: ...)、自定义序列名

组合语法:
  平铺列表  = 组内全部 AND : [('EMA20','>','EMA50'), ('RSI','<',30), ('ADX','>',25)]
  嵌套列表  = 组间 OR     : [ [条件A1, 条件A2], [条件B1] ]  即 (A1&A2) OR B1

用法示例:
    df   = 你的 OHLCV DataFrame(列名小写: open/high/low/close/volume)
    eng  = StrategyEngine(df)

    buy  = eng.build_signal([('EMA20', '>', 'EMA50'),
                             ('RSI', '<', 30),
                             ('ADX', '>', 25)])          # 动态生成买入信号
    sell = eng.build_signal([('RSI', '>', 70)])           # 动态生成卖出信号

    out = eng.generate(entry_conditions=buy_conds, exit_conditions=sell_conds)
    # out 附带 buy_signal / sell_signal 两个布尔列, 可直接喂给回测循环

自检(不依赖网络, 用随机数据): python strategy_engine.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ==================== 注册表(扩展点) ====================
INDICATORS: dict = {}   # name -> callable(df, name) -> Series
OPERATORS: dict = {}    # name -> callable(left, right) -> bool Series


def register_indicator(*names):
    """把函数注册为一个或多个指标名。例: @register_indicator('SAR')"""
    def deco(fn):
        for n in names:
            INDICATORS[n] = fn
        return fn
    return deco


def register_operator(*names):
    """把函数注册为一个或多个算子名。例: @register_operator('pct_change')"""
    def deco(fn):
        for n in names:
            OPERATORS[n] = fn
        return fn
    return deco


def _num_suffix(name: str, prefix: str, default: int) -> int:
    s = name.replace(prefix, "")
    return int(s) if s else default


# ==================== 内置指标 ====================

@register_indicator("EMA5", "EMA10", "EMA20", "EMA50", "EMA100")
def _ema(df: pd.DataFrame, name: str) -> pd.Series:
    span = _num_suffix(name, "EMA", 20)
    return df["close"].ewm(span=span, adjust=False).mean()


@register_indicator("RSI14", "RSI")
def _rsi(df: pd.DataFrame, name: str) -> pd.Series:
    period = _num_suffix(name, "RSI", 14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where((avg_loss != 0.0) | (avg_gain == 0.0), 100.0)
    return rsi


@register_indicator("MACD", "MACD_SIGNAL", "MACD_HIST")
def _macd(df: pd.DataFrame, name: str) -> pd.Series:
    macd = (df["close"].ewm(span=12, adjust=False).mean()
            - df["close"].ewm(span=26, adjust=False).mean())
    signal = macd.ewm(span=9, adjust=False).mean()
    return {"MACD": macd, "MACD_SIGNAL": signal, "MACD_HIST": macd - signal}[name]


@register_indicator("ATR14", "ATR")
def _atr(df: pd.DataFrame, name: str) -> pd.Series:
    period = _num_suffix(name, "ATR", 14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


@register_indicator("BOLL_MID", "BOLL_UPPER", "BOLL_LOWER")
def _boll(df: pd.DataFrame, name: str) -> pd.Series:
    mid = df["close"].rolling(20, min_periods=20).mean()
    std = df["close"].rolling(20, min_periods=20).std(ddof=0)
    return {"BOLL_MID": mid, "BOLL_UPPER": mid + 2 * std, "BOLL_LOWER": mid - 2 * std}[name]


@register_indicator("ADX14", "ADX", "PLUS_DI", "MINUS_DI")
def _adx(df: pd.DataFrame, name: str) -> pd.Series:
    period = _num_suffix(name, "ADX", 14)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, min_periods=period,
                                  adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, min_periods=period,
                                    adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return {"ADX14": adx, "ADX": adx, "PLUS_DI": plus_di, "MINUS_DI": minus_di}[name]


@register_indicator("VOLUME", "VOLUME_SMA20", "VOLUME_RATIO")
def _volume(df: pd.DataFrame, name: str) -> pd.Series:
    vol = df["volume"]
    sma = vol.rolling(20, min_periods=20).mean()
    return {"VOLUME": vol, "VOLUME_SMA20": sma, "VOLUME_RATIO": vol / sma}[name]


# ==================== 内置算子 ====================

@register_operator(">")
def _gt(l, r):
    return l > r


@register_operator("<")
def _lt(l, r):
    return l < r


@register_operator(">=")
def _ge(l, r):
    return l >= r


@register_operator("<=")
def _le(l, r):
    return l <= r


@register_operator("==")
def _eq(l, r):
    return l == r


@register_operator("!=")
def _ne(l, r):
    return l != r


@register_operator("cross_above", "crossup", "金叉")
def _cross_above(left, right):
    """left 上穿 right(上一根还在下方/相等, 本根到上方)。"""
    left = pd.Series(left)
    if not isinstance(right, pd.Series):
        right = pd.Series(right, index=left.index)
    return (left > right) & (left.shift(1) <= right.shift(1))


@register_operator("cross_below", "crossdown", "死叉")
def _cross_below(left, right):
    """left 下穿 right。"""
    left = pd.Series(left)
    if not isinstance(right, pd.Series):
        right = pd.Series(right, index=left.index)
    return (left < right) & (left.shift(1) >= right.shift(1))


@register_operator("between")
def _between(left, right):
    """right 为 (下界, 上界) 二元组: 下界 <= left <= 上界。"""
    lo, hi = sorted(right)
    return (left >= lo) & (left <= hi)


# ==================== 引擎 ====================


class StrategyEngine:
    """把条件列表动态编译成买入/卖出布尔信号。

    扩展点:
      - 新指标:  用 @register_indicator 注册后, 条件里直接用名字引用
      - 新算子:  用 @register_operator 注册后, 条件里直接用符号引用
      - 新序列:  engine.register_series("名字", series) 注入外部算好的序列
      - 复杂值:  条件项里直接放 callable, 如 ('MY_LAMBDA', '>', 0)
                 其中 MY_LAMBDA = lambda df: df['close'].pct_change(24)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cache: dict = {}        # 指标缓存: name -> Series
        self._series_pool: dict = {}  # 自定义序列池: name -> Series

    # ---------- 序列解析 ----------
    def _resolve(self, token):
        """把 token 解析成与 df 等长的 Series, 或数值常量。"""
        if callable(token):                       # 用户自定义函数
            token = token(self.df)
        if isinstance(token, (pd.Series, np.ndarray)):
            return pd.Series(np.asarray(token, dtype=float), index=self.df.index)
        if isinstance(token, (int, float, np.number)):
            return float(token)
        if isinstance(token, (list, tuple)):      # 交给算子自行处理(如 between)
            return token
        if isinstance(token, str):
            name = token.strip()
            try:                                  # "0.5" 这类数字字符串
                return float(name)
            except ValueError:
                pass
            if name in self.df.columns:           # 原始 OHLCV 列
                return self.df[name]
            if name.lower() in self.df.columns:   # 兼容列名大小写
                return self.df[name.lower()]
            if name in INDICATORS:                # 已注册指标(带缓存)
                return self.get(name)
            if name in self._series_pool:         # 自定义序列
                return self._series_pool[name]
            raise KeyError(
                f"无法解析 '{name}': 不是原始列/已注册指标/自定义序列/数字。"
                f" 已注册指标: {sorted(INDICATORS)}")
        raise TypeError(f"不支持的条件项类型: {type(token)!r}")

    # ---------- 指标计算 ----------
    def get(self, name: str) -> pd.Series:
        """按名计算(并缓存)一个指标。"""
        if name not in self._cache:
            if name not in INDICATORS:
                raise KeyError(f"未注册的指标: {name}")
            series = INDICATORS[name](self.df, name)
            if not isinstance(series, pd.Series):
                series = pd.Series(series, index=self.df.index, dtype=float)
            self._cache[name] = series
        return self._cache[name]

    def register_series(self, name: str, series) -> "StrategyEngine":
        """注入外部序列(如用 TA-Lib/自定义函数算好的列), 供条件引用。"""
        self._series_pool[name] = pd.Series(
            np.asarray(series, dtype=float), index=self.df.index)
        return self

    def add_indicators(self, names=None) -> pd.DataFrame:
        """返回附上指标列的 df 副本; names=None 时计算全部已注册指标。"""
        names = list(INDICATORS) if names is None else names
        out = self.df.copy()
        for n in names:
            out[n] = self.get(n)
        return out

    # ---------- 条件编译 ----------
    def build_condition_group(self, group) -> pd.Series:
        """一个条件组(3 元组列表): 组内全部 AND。空组 = 恒真。"""
        if not group:
            return pd.Series(True, index=self.df.index)
        mask = None
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(f"条件必须是 (left, op, right) 三元组: {item!r}")
            left, op, right = item
            if op not in OPERATORS:
                raise KeyError(f"未注册的算子: '{op}'. 已注册: {sorted(OPERATORS)}")
            piece = pd.Series(np.asarray(OPERATORS[op](self._resolve(left),
                                                       self._resolve(right)),
                                        dtype=bool),
                              index=self.df.index)
            mask = piece if mask is None else (mask & piece)
        return mask.fillna(False)

    def build_signal(self, conditions) -> pd.Series:
        """把条件列表编译成布尔信号 Series。

        平铺列表: [c1, c2, c3]          -> c1 AND c2 AND c3
        嵌套列表: [[c1, c2], [c3]]      -> (c1 AND c2) OR c3
        """
        if not conditions:
            return pd.Series(False, index=self.df.index)
        if isinstance(conditions[0], list):       # 嵌套 = 组间 OR
            pieces = [self.build_condition_group(g) for g in conditions]
            return pd.concat(pieces, axis=1).any(axis=1).fillna(False)
        return self.build_condition_group(conditions)

    def generate(self, entry_conditions, exit_conditions=None) -> pd.DataFrame:
        """一步生成: df 副本 + buy_signal / sell_signal 布尔列。"""
        out = self.df.copy()
        out["buy_signal"] = self.build_signal(entry_conditions).astype(bool)
        if exit_conditions is not None:
            out["sell_signal"] = self.build_signal(exit_conditions).astype(bool)
        return out


# ==================== 模块级工具 ====================

def list_indicators():
    return sorted(INDICATORS)


def list_operators():
    return sorted(OPERATORS)


# ==================== 自检(直接运行本文件时执行, 不依赖网络) ====================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    base = pd.DataFrame({
        "open": close,
        "close": close * (1 + rng.normal(0, 0.005, n)),
    })
    df = pd.DataFrame({
        "open": base["open"],
        "high": base[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.003, n))),
        "low": base[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.003, n))),
        "close": base["close"],
        "volume": rng.uniform(100, 1000, n),
    })

    eng = StrategyEngine(df)
    buy = eng.build_signal([("EMA20", ">", "EMA50"), ("RSI", "<", 30), ("ADX", ">", 25)])
    sell = eng.build_signal([("RSI", ">", 70)])
    print("已注册指标:", list_indicators())
    print("已注册算子:", list_operators())
    print(f"买入信号触发 {int(buy.sum())} 次 / 卖出信号触发 {int(sell.sum())} 次")
    print(eng.add_indicators(["EMA20", "EMA50", "RSI", "ADX"]).tail(3).round(4))
