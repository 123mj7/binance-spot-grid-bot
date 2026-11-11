# -*- coding: utf-8 -*-
"""compound_test.py —— 复投版 vs 固定版(现版本), DOGE 2021-2026至今"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab
from beauty_filter_audit import run_fixed

FEE = 0.001
TP, DROP = 0.03, 0.03
MAX_ADDS = 9


def run_compound(df: pd.DataFrame, ema4h: np.ndarray, frac: float) -> dict:
    """复投版: 每开新轮按账户净值重新分档(轮预算 = 净值 x frac, 50 起 500 封顶)。"""
    prices = df["close"].to_numpy(dtype=float)
    n = len(prices)
    cash, qty = 50.0, 0.0
    budget, buy = 50.0, 5.0
    stop_line = 50.0 * 0.85
    last_buy, buys, cost = 0.0, 0, 0.0
    stopped = False
    rounds = rounds_tp = n_buys = n_sells = 0
    equity = np.empty(n)
    for i in range(n):
        p = prices[i]
        if stopped:
            equity[i] = cash
            continue
        if qty <= 0:
            e = ema4h[i]
            if not np.isnan(e) and p > e:
                budget = min(500.0, max(50.0, cash * frac))   # 复投核心
                buy = budget / 10.0
                stop_line = budget * 0.85
                fee = buy * FEE
                qty += (buy - fee) / p
                cash -= buy
                cost, buys, last_buy = buy, 1, p
                n_buys += 1
                rounds += 1
        else:
            if buys - 1 < MAX_ADDS and p <= last_buy * (1 - DROP):
                fee = buy * FEE
                qty += (buy - fee) / p
                cash -= buy
                cost += buy
                buys += 1
                last_buy = p
                n_buys += 1
            avg = cost / qty
            if p >= avg * (1 + TP):
                cash += qty * p * (1 - FEE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                rounds_tp += 1
            elif buys >= MAX_ADDS + 1 and cash + qty * p < stop_line:
                cash += qty * p * (1 - FEE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                stopped = True
        equity[i] = cash + qty * p
    peak = np.maximum.accumulate(equity)
    dd = float(np.min((equity - peak) / peak)) * 100 if peak.max() > 0 else 0.0
    return {"return_pct": (equity[-1] / 50.0 - 1) * 100, "max_dd_pct": dd,
            "rounds": rounds, "rounds_tp": rounds_tp,
            "n_buys": n_buys, "n_sells": n_sells, "equity": equity,
            "end_state": "止损停止" if stopped else ("持仓中" if qty > 0 else "空仓")}


d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
d2 = pd.read_csv("doge_usdt_1h_3y.csv")
d3 = pd.read_csv("doge_usdt_1h_2026.csv")
df = pd.concat([d1, d2, d3], ignore_index=True)
df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"], utc=True)
print(f"数据: {len(df)} 根 | {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")

ema = ab.compute_ema4h(df, period=60)
fixed = run_fixed(df, ema, "close", slip=0.001)     # 现版本(实盘口径 0.1% 滑点)

print()
print("=" * 84)
print(" 复投敏感性: 轮预算 = 净值 x 比例(50U 起, 500U 封顶), DOGE 2021-2026至今")
print("=" * 84)
for label, r in (("现版本·固定50U/轮(含0.1%滑点)", fixed),):
    eq = np.asarray(r["equity"])
    print(f"  {label:<30} 最终 {eq[-1]:8.2f} U | {r['return_pct']:+8.2f} % | "
          f"DD {r['max_dd_pct']:6.2f} %")
for frac in (0.3, 0.5, 0.8):
    r = run_compound(df, ema, frac)
    eq = np.asarray(r["equity"])
    print(f"  复投版·净值x{frac}(无滑点)        最终 {eq[-1]:8.2f} U | "
          f"{r['return_pct']:+8.2f} % | DD {r['max_dd_pct']:6.2f} % | "
          f"轮数 {r['rounds']:>4} | {r['end_state']}")
