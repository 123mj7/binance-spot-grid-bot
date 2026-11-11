# -*- coding: utf-8 -*-
"""gene_compare.py —— 同一段 DOGE 2021-2026 数据, 六种策略基因收益/回撤对比"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab
from beauty_filter_audit import run_fixed

d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
d2 = pd.read_csv("doge_usdt_1h_3y.csv")
d3 = pd.read_csv("doge_usdt_1h_2026.csv")
df = pd.concat([d1, d2, d3], ignore_index=True)
df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"], utc=True)
closes = df["close"].to_numpy(dtype=float)
n = len(closes)
FEE = 0.001
INIT = 100.0


def stats(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = float(np.min((eq - peak) / peak)) * 100
    return eq[-1], (eq[-1] / INIT - 1) * 100, dd


def bh():
    eq = INIT * closes / closes[0]
    return stats(eq)


def weekly_dca(per_week=2.0):
    cash = INIT
    qty = 0.0
    eq = np.empty(n)
    wk = df["time"].dt.isocalendar().week.to_numpy()
    yr = df["time"].dt.year.to_numpy()
    last_key = None
    for i in range(n):
        key = (yr[i], wk[i])
        if key != last_key:
            last_key = key
            spend = min(per_week, cash)
            if spend > 0:
                qty += (spend - spend * FEE) / closes[i]
                cash -= spend
        eq[i] = cash + qty * closes[i]
    return stats(eq)


def rsi_swing(lo=30, hi=70, period=14):
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    al = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rsi = 100 - 100 / (1 + ag / np.where(al == 0, 1e-12, al))
    cash = INIT
    qty = 0.0
    eq = np.empty(n)
    for i in range(n):
        if qty <= 0 and rsi[i] < lo:
            qty = (cash - cash * FEE) / closes[i]
            cash = 0.0
        elif qty > 0 and rsi[i] > hi:
            cash = qty * closes[i] * (1 - FEE)
            qty = 0.0
        eq[i] = cash + qty * closes[i]
    return stats(eq)


def trend_follow(fast=20, slow=60):
    s_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().to_numpy()
    s_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().to_numpy()
    cash = INIT
    qty = 0.0
    eq = np.empty(n)
    for i in range(n):
        if i < slow:
            eq[i] = cash
            continue
        if s_fast[i] > s_slow[i] and qty <= 0:
            qty = (cash - cash * FEE) / closes[i]
            cash = 0.0
        elif s_fast[i] < s_slow[i] and qty > 0:
            cash = qty * closes[i] * (1 - FEE)
            qty = 0.0
        eq[i] = cash + qty * closes[i]
    return stats(eq)


ema = ab.compute_ema4h(df, period=60)
grid = run_fixed(df, ema, "close", slip=0.001)
grid_eq = np.asarray(grid["equity"]) * (INIT / 50.0)   # 等比放大到 100U 口径

print("=" * 92)
print(f" 六种策略基因对比 | DOGE {df['time'].iloc[0].date()} ~ "
      f"{df['time'].iloc[-1].date()} | 初始 {INIT:.0f}U, 手续费 0.1%")
print("=" * 92)
rows = [
    ("① 网格基线(现版本, 含滑点)", stats(grid_eq)),
    ("② 买入持有(死拿)", bh()),
    ("③ 周定投(每周2U)", weekly_dca()),
    ("④ RSI14<30买/>70卖(全仓)", rsi_swing()),
    ("⑤ 趋势跟踪(4h EMA20/60交叉)", trend_follow()),
]
for label, (final, ret, dd) in rows:
    print(f"  {label:<28} 最终 {final:9.2f} U | {ret:+9.2f} % | 最大回撤 {dd:6.2f} %")
print()
print("  注: ①为等比放大到100U口径; ⑤纯交叉无止损, 震荡期会反复打脸;")
print("  所有基因共用同一段数据和费率, 收益越高回撤越大, 没有免费午餐。")
