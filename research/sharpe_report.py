# -*- coding: utf-8 -*-
"""sharpe_report.py —— 基线策略夏普比率(小时级+月级年化, 分时段, 对照买入持有)"""
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

ema = ab.compute_ema4h(df, period=60)
r = run_fixed(df, ema, "close")
eq = np.asarray(r["equity"], dtype=float)
years = df["time"].dt.year.to_numpy()
times = df["time"].to_numpy()
bh = df["close"].to_numpy(dtype=float)


def hourly_sharpe(eq_arr, idx):
    e = eq_arr[idx]
    e = e[e > 0]
    ret = np.diff(np.log(e))
    ret = ret[np.isfinite(ret)]
    if len(ret) < 10 or ret.std() == 0:
        return 0.0, 0
    return (ret.mean() / ret.std()) * np.sqrt(8760), len(ret)


def monthly_sharpe(eq_arr, idx, times_arr):
    s = pd.Series(eq_arr[idx], index=times_arr[idx])
    m = s.resample("ME").last().dropna()
    ret = m.pct_change().dropna()
    ret = ret[np.isfinite(ret)]
    if len(ret) < 3 or ret.std() == 0:
        return 0.0, len(ret)
    return (ret.mean() / ret.std()) * np.sqrt(12), len(ret)


def report(label, idx):
    hs, n1 = hourly_sharpe(eq, idx)
    ms, n2 = monthly_sharpe(eq, idx, times)
    hb, _ = hourly_sharpe(bh, idx)
    mb, _ = monthly_sharpe(bh, idx, times)
    ret = (eq[idx][-1] / eq[idx][0] - 1) * 100 if len(idx) else 0
    print(f"  {label:<18} 收益 {ret:+8.2f}% | 策略夏普(小时年化) {hs:5.2f} "
          f"(月级 {ms:5.2f}) | 持有夏普 {hb:5.2f} (月级 {mb:5.2f})")


print("=" * 96)
print(f" 基线策略夏普比率 | DOGE {df['time'].iloc[0].date()} ~ "
      f"{df['time'].iloc[-1].date()} (UTC)")
print("=" * 96)
mask_all = np.ones(len(eq), dtype=bool)
report("2021~2026 全周期", mask_all)
report("2021~2025 五年", years <= 2025)
report("2022~2025 滚动前推检验期", (years >= 2022) & (years <= 2025))
report("2026 至今", years >= 2026)
print()
print("  注: 风险利率取 0; 策略大量时间空仓(净值走平), 会压低波动率分母,")
print("  使夏普偏高; 加密资产尾部风险大, 夏普只作参考, 请结合最大回撤一起看。")
