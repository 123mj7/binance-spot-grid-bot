# -*- coding: utf-8 -*-
"""futures_v2.py —— 合约趋势 v2: 1h快周期 + 宽止损 + 费率过滤, 快速筛选"""
import itertools
import sys

import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
from futures_backtest import turtle

d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
d2 = pd.read_csv("doge_usdt_1h_3y.csv")
df = pd.concat([d1, d2], ignore_index=True)
df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
df["time"] = pd.to_datetime(df["time"], utc=True)
df5 = df[df["time"].dt.year <= 2025].reset_index(drop=True)
print(f"DOGE 五年 1h: {len(df5)} 根")

print("=" * 88)
print(" v2 快速筛选: 1h Donchian 突破, 宽止损 3~5xATR, 费率过滤(>0.01%不做多)")
print("=" * 88)
results = []
for entry_n, stop_mult, risk, fund in itertools.product(
        (24, 48, 96), (3.0, 4.0, 5.0), (0.005, 0.01), (0.0, 0.0001)):
    r = turtle(df5, entry_n=entry_n, exit_n=max(12, entry_n // 2),
               risk_pct=risk, lev=2.0, funding_8h=fund,
               stop_mult=stop_mult, funding_every=8, long_short=True)
    results.append((entry_n, stop_mult, risk, fund, r["ret"], r["dd"],
                    r["closes"], r["winrate"], r["liq"]))
    if r["closes"] >= 10:
        print(f"  突破{entry_n:>3} 止损{stop_mult:.0f}xATR 风险{risk*100:.1f}% "
              f"费率{fund*10000:.0f}万: {r['ret']:+9.2f}% DD {r['dd']:6.2f}% "
              f"交易{r['closes']:>3} 胜率{r['winrate']:.0f}% 爆仓{r['liq']}")
res = pd.DataFrame(results, columns=["entry", "stopx", "risk", "fund", "ret",
                                     "dd", "trades", "winrate", "liq"])
res.to_csv("futures_v2_screen.csv", index=False, encoding="utf-8-sig")
ok = res[(res["trades"] >= 10) & (res["ret"] > 0) & (res["dd"] > -50)]
print("-" * 88)
print(f"  初筛通过(≥10笔且正收益且DD≤50%): {len(ok)}/{len(res)} 组")
if len(ok):
    print(ok.sort_values("ret", ascending=False).head(8).to_string(index=False))
else:
    print("  → v2 无一组通过初筛: 1h 突破+宽止损在 DOGE 五年同样不成立")
