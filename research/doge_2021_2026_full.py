# -*- coding: utf-8 -*-
"""doge_2021_2026_full.py —— 现版本基线跑 DOGE 2021-01-01 ~ 至今(2026-11)总账"""
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
print(f"数据: {len(df)} 根 | {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")

ema = ab.compute_ema4h(df, period=60)

for label, slip in (("原始口径(手续费0.1%, 无滑点)", 0.0),
                    ("实盘口径(另加双边滑点0.1%)", 0.001)):
    r = run_fixed(df, ema, "close", slip=slip)
    eq = np.asarray(r["equity"])
    print()
    print(f"--- {label} ---")
    print(f"  初始资金         : 50.00 USDT")
    print(f"  最终总资产       : {eq[-1]:.2f} USDT")
    print(f"  总盈亏           : {eq[-1]-50:+.2f} USDT ({r['return_pct']:+.2f} %)")
    print(f"  最大回撤         : {r['max_dd_pct']:.2f} %")
    print(f"  开轮/止盈轮      : {r['rounds']} / {r['rounds_tp']}")
    print(f"  买入/卖出次数    : {r['n_buys']} / {r['n_sells']}")
    print(f"  期末状态         : {r['end_state']}")
    years = df["time"].dt.year.to_numpy()
    print("  逐年总资产:")
    for y in sorted(set(years)):
        idx = np.where(years == y)[0]
        if len(idx):
            print(f"    {y} 末: {eq[idx[-1]]:.2f} USDT")

bh_start = df["close"].iloc[0]
bh_end = df["close"].iloc[-1]
print()
print(f"  对照·买入持有    : {bh_start:.6f} → {bh_end:.6f} "
      f"({(bh_end/bh_start-1)*100:+.2f} %)")
