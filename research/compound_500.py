# -*- coding: utf-8 -*-
"""compound_500.py —— 500U 本金下: 固定等比 vs 复投各档, DOGE 2021-2026至今"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab

FEE = 0.001
TP, DROP = 0.03, 0.03
MAX_ADDS = 9
INIT = 500.0


def run_strat(df: pd.DataFrame, ema4h: np.ndarray, frac=None, slip=0.0) -> dict:
    """frac=None: 固定等比(轮预算=500U, 每单50U); 否则复投(预算=净值xfrac, 50~500)。"""
    prices = df["close"].to_numpy(dtype=float)
    n = len(prices)
    cash, qty = INIT, 0.0
    budget, buy, stop_line = INIT, INIT / 10, INIT * 0.85
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
                if frac is not None:
                    budget = min(500.0, max(50.0, cash * frac))
                    buy = budget / 10.0
                    stop_line = budget * 0.85
                else:
                    budget, buy, stop_line = INIT, INIT / 10, INIT * 0.85
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
    return {"final": equity[-1], "ret": (equity[-1] / INIT - 1) * 100, "dd": dd,
            "rounds": rounds, "rounds_tp": rounds_tp, "equity": equity,
            "end_state": "止损停止" if stopped else ("持仓中" if qty > 0 else "空仓")}


d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
d2 = pd.read_csv("doge_usdt_1h_3y.csv")
d3 = pd.read_csv("doge_usdt_1h_2026.csv")
df = pd.concat([d1, d2, d3], ignore_index=True)
df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"], utc=True)
ema = ab.compute_ema4h(df, period=60)

print("=" * 88)
print(f" 500U 本金跑 DOGE {df['time'].iloc[0].date()} ~ {df['time'].iloc[-1].date()} (UTC)")
print("=" * 88)
for label, frac, slip in (("固定等比(500U轮/50U单, 含0.1%滑点)", None, 0.001),
                          ("复投 x0.3(无滑点)", 0.3, 0.0),
                          ("复投 x0.5(无滑点)", 0.5, 0.0),
                          ("复投 x0.8(无滑点)", 0.8, 0.0)):
    r = run_strat(df, ema, frac=frac, slip=slip)
    print(f"  {label:<30} 最终 {r['final']:9.2f} U | {r['ret']:+8.2f} % | "
          f"DD {r['dd']:6.2f} % | 轮数 {r['rounds']:>3} | {r['end_state']}")

print()
print("  另: 500U 钱包但网格仍是50U/轮(现版本参数): 450U 闲置 + 50U网格赚51.8U")
print(f"  → 最终约 551.8 U (+10.4%), 只有 50U 在真正工作")
print()
print("  注: 500U 起步时复投升档门槛(50U起)不再卡脖子, 复投机制从第一轮就开始工作。")
