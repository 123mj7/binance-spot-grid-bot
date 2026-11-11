# -*- coding: utf-8 -*-
"""
futures_backtest.py —— 合约趋势策略研究引擎(海龟式)
====================================================
与现货机器人无关的纯研究引擎, 模拟:
  - 永续合约: 资金费率(每8小时结算, 可正可负) + 爆仓(隔离保证金) + 杠杆
  - 海龟式趋势: Donchian N 根突破进场 / M 根反向突破离场 + 2xATR 硬止损
  - 仓位: 每笔风险 = 账户净值 x risk_pct, 名义额 = 风险预算 / (2xATR), 受杠杆封顶
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")

FEE = 0.0005          # 合约 taker 费率 0.05%
MAINT = 0.005         # 维持保证金率 0.5%(爆仓判定用)


def to_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    bucket = df1h["time"].dt.floor("4h")
    g = df1h.assign(b=bucket).groupby("b")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
    }).reset_index().rename(columns={"b": "time"})
    return out


def atr_series(df4h: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df4h["high"].to_numpy(), df4h["low"].to_numpy(), df4h["close"].to_numpy()
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    return atr


def turtle(df4h: pd.DataFrame, entry_n: int = 120, exit_n: int = 60,
           atr_n: int = 14, risk_pct: float = 0.01, lev: float = 2.0,
           funding_8h: float = 0.0, long_short: bool = True) -> dict:
    """海龟式趋势(4h K线)。
    账户模型: cash + 持仓未实现盈亏; 开仓只扣手续费并锁保证金;
    平仓结算盈亏; 爆仓损失全部保证金(隔离); 资金费每 2 根 4h 结算一次。"""
    h = df4h["high"].to_numpy(dtype=float)
    l = df4h["low"].to_numpy(dtype=float)
    c = df4h["close"].to_numpy(dtype=float)
    n = len(c)
    atr = atr_series(df4h, atr_n)
    cash = 100.0
    qty = 0.0
    entry_px = 0.0
    side = 0                       # 1 多 / -1 空
    margin = 0.0
    entries = closes = wins = liq = 0
    eq_arr = np.empty(n)
    for i in range(n):
        # 1) 资金费(每8小时 = 2根4h)
        if qty > 0 and i % 2 == 0 and funding_8h != 0.0:
            cash -= side * qty * c[i] * funding_8h
        # 2) 爆仓检查(隔离保证金: 亏光保证金出局)
        if qty > 0:
            liq_hit = (side == 1 and c[i] <= entry_px * (1.0 - 1.0 / lev + MAINT)) or \
                      (side == -1 and c[i] >= entry_px * (1.0 + 1.0 / lev - MAINT))
            if liq_hit:
                cash -= margin
                qty, side, margin = 0.0, 0, 0.0
                liq += 1
        # 3) 离场(反向突破 或 2xATR 止损)
        if qty > 0 and i >= exit_n:
            if side == 1:
                exit_sig = (c[i] < min(l[i - exit_n + 1:i + 1])
                            or c[i] <= entry_px - 2 * atr[i])
            else:
                exit_sig = (c[i] > max(h[i - exit_n + 1:i + 1])
                            or c[i] >= entry_px + 2 * atr[i])
            if exit_sig:
                pnl = side * (c[i] - entry_px) * qty
                cash += pnl - qty * c[i] * FEE
                closes += 1
                wins += 1 if pnl > 0 else 0
                qty, side, margin = 0.0, 0, 0.0
        # 4) 进场(Donchian 突破)
        if qty <= 0 and i >= entry_n and cash > 10:
            up = c[i] > max(h[i - entry_n:i])
            dn = c[i] < min(l[i - entry_n:i])
            if up or (dn and long_short):
                side = 1 if up else -1
                entry_px = c[i]
                units = (cash * risk_pct) / (2 * atr[i])
                max_units = cash * lev / c[i]
                units = min(units, max_units)
                if units * c[i] >= 10:
                    qty = units
                    margin = qty * c[i] / lev
                    cash -= qty * c[i] * FEE      # 开仓手续费
                    entries += 1
        # 5) 净值(含未实现盈亏)
        eq_arr[i] = cash + (side * (c[i] - entry_px) * qty if qty > 0 else 0.0)
    # 期末持仓按收盘平掉
    if qty > 0:
        pnl = side * (c[-1] - entry_px) * qty
        cash += pnl - qty * c[-1] * FEE
        closes += 1
        wins += 1 if pnl > 0 else 0
        eq_arr[-1] = cash
    peak = np.maximum.accumulate(eq_arr)
    dd = float(np.min((eq_arr - peak) / peak)) * 100
    return {"ret": (eq_arr[-1] / 100 - 1) * 100, "dd": dd,
            "entries": entries, "closes": closes,
            "winrate": wins / max(1, closes) * 100,
            "liq": liq, "equity": eq_arr,
            "end": "持仓中" if qty > 0 else "空仓"}
