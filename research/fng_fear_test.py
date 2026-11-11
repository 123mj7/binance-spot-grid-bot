# -*- coding: utf-8 -*-
"""
fng_fear_test.py —— FNG 恐惧侧外样本验证: "极度恐惧放开趋势过滤"有用吗?
=========================================================================
问题: FNG <= 15(极度恐惧) 时无视 4h EMA60 过滤直接开仓, 在 2021-2025 五年
外样本里能不能赚到"带血的筹码"?

三条腿(止盈 3% / 加仓间距 3% / 42.5U 固定止损 / 加仓规则全部一致):
  基线                : 4h EMA60 过滤 + 3% 网格 + 固定止损
  恐惧放开            : FNG(前一日) <= 15 时无视 EMA60 直接开仓, 加仓照常
  恐惧放开+贪婪暂停    : 上面那条 + FNG >= 80 时暂停开新仓

防未来函数: 每根 1h K 线只用"前一天"的 FNG 值。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from fng_test import fetch_fng
from adaptive_backtest import (compute_ema4h, backtest_baseline,
                               _summarize, INITIAL_CASH)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

FEAR_BYPASS = 15.0     # FNG <= 15 放开 EMA60 过滤
GREED_GATE = 80.0      # FNG >= 80 暂停开新仓
DATA_PARTS = ["doge_usdt_1h_2021_22.csv", "doge_usdt_1h_3y.csv"]
PNG_OUT = "fng_fear_compare.png"
CSV_OUT = "fng_fear_results.csv"


def backtest_fng2(df, ema4h, fng_arr, use_fear=False, use_greedy=False,
                  tag="") -> dict:
    prices = df["close"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy(dtype=int)
    n = len(prices)
    cash, qty = INITIAL_CASH, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    stopped = False
    n_buys = n_sells = rounds = rounds_tp = n_fear_entry = 0
    rounds_years = []
    equity = np.empty(n)

    for i in range(n):
        p = prices[i]
        f = fng_arr[i]
        if stopped:
            equity[i] = cash
            continue
        if qty <= 0:                                   # 空仓 → 决定是否开仓
            e = ema4h[i]
            ok = (not np.isnan(e)) and p > e           # 默认: 趋势过滤
            if use_fear and (not np.isnan(f)) and f <= FEAR_BYPASS:
                ok = True                              # 极度恐惧: 放开过滤
                if not ((not np.isnan(e)) and p > e):
                    pass
            if use_greedy and (not np.isnan(f)) and f >= GREED_GATE:
                ok = False                             # 极度贪婪: 暂停开新仓
            if ok:
                fee = 5 * 0.001
                qty += (5 - fee) / p
                cash -= 5
                cost, buys, last_buy = 5.0, 1, p
                n_buys += 1
                rounds += 1
                rounds_years.append(int(years[i]))
                if use_fear and (not np.isnan(f)) and f <= FEAR_BYPASS:
                    n_fear_entry += 1
        else:                                          # 持仓中(规则不变)
            if buys - 1 < 9 and p <= last_buy * 0.97:
                fee = 5 * 0.001
                qty += (5 - fee) / p
                cash -= 5
                cost += 5
                buys += 1
                last_buy = p
                n_buys += 1
            avg = cost / qty
            if p >= avg * 1.03:
                cash += qty * p * 0.999
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                rounds_tp += 1
            elif qty > 0 and buys >= 10 and cash + qty * p < 42.5:
                cash += qty * p * 0.999
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                stopped = True
        equity[i] = cash + qty * p

    res = _summarize(equity, years, rounds_years, cash, qty, n_buys, n_sells,
                     rounds, rounds_tp, 0, stopped, tag)
    res["n_fear_entry"] = n_fear_entry
    return res


def main() -> None:
    print("=" * 82)
    print(" FNG 恐惧侧验证: 极度恐惧放开 EMA60 过滤有用吗? (2021-2025 五年)")
    print("=" * 82)

    parts = []
    for f in DATA_PARTS:
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["time"] = pd.to_datetime(d["time"], utc=True)
            parts.append(d)
    if not parts:
        sys.exit("[错误] 无本地 K 线缓存")
    df = (pd.concat(parts).drop_duplicates("ts").sort_values("ts")
          .reset_index(drop=True))
    print(f"[K线] {len(df)} 根 1h K 线 | "
          f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")

    fng = fetch_fng()
    fng_prev = fng.shift(1)
    day = df["time"].dt.normalize()
    fng_arr = day.map(fng_prev).to_numpy(dtype=float)
    print(f"[FNG] FNG<=15 天数: {int((fng <= FEAR_BYPASS).sum())} | "
          f"FNG>=80 天数: {int((fng >= GREED_GATE).sum())}")

    ema4h = compute_ema4h(df)
    legs = {
        "基线(无FNG)": backtest_baseline(df, ema4h, trailing=False),
        "恐惧放开(≤15)": backtest_fng2(df, ema4h, fng_arr,
                                      use_fear=True, use_greedy=False, tag="恐惧放开"),
        "恐惧放开+贪婪暂停": backtest_fng2(df, ema4h, fng_arr,
                                        use_fear=True, use_greedy=True,
                                        tag="恐惧放开+贪婪暂停"),
    }

    print()
    print(f"{'腿':<20}{'最终资产':>10}{'总收益%':>9}{'最大回撤%':>10}"
          f"{'轮数':>6}{'恐惧开仓':>10}{'止损触发':>10}")
    print("-" * 82)
    for k, r in legs.items():
        stopped = "是" if r.get("stopped") else "否"
        print(f"{k:<20}{r['final_equity']:>10.2f}{r['return_pct']:>9.2f}"
              f"{r['max_dd_pct']:>10.2f}{r['rounds_opened']:>6}"
              f"{r.get('n_fear_entry', 0):>10}{stopped:>10}")
    print("-" * 82)

    print()
    print(f"{'腿':<20}{'2021':>10}{'2022':>10}{'2023':>10}{'2024':>10}{'2025':>10}")
    print("-" * 82)
    for k, r in legs.items():
        vals = [r["yearly"].get(y, {}).get("return_pct", 0.0)
                for y in (2021, 2022, 2023, 2024, 2025)]
        print(f"{k:<20}" + "".join(f"{v:>10.2f}" for v in vals))
    print("-" * 82)

    base = legs["基线(无FNG)"]
    print()
    for k in ["恐惧放开(≤15)", "恐惧放开+贪婪暂停"]:
        r = legs[k]
        diff = r["return_pct"] - base["return_pct"]
        dd = r["max_dd_pct"] - base["max_dd_pct"]
        print(f" {k}: 相对基线 {('多赚' if diff >= 0 else '少赚')} {abs(diff):.2f} 个百分点, "
              f"回撤 {'多' if dd >= 0 else '少'} {abs(dd):.2f} 个百分点")

    if plt is not None:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(13, 6))
        t = df["time"]
        ax.plot(t, legs["基线(无FNG)"]["equity"], color="#1f77b4", linewidth=1.1,
                label="基线(无FNG)")
        ax.plot(t, legs["恐惧放开(≤15)"]["equity"], color="#2ca02c", linewidth=0.9,
                label="恐惧放开(≤15)")
        ax.plot(t, legs["恐惧放开+贪婪暂停"]["equity"], color="#d62728", linewidth=0.9,
                label="恐惧放开+贪婪暂停")
        ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8,
                   label="初始资金 50 USDT")
        for y in (2022, 2023, 2024, 2025):
            ax.axvline(pd.Timestamp(f"{y}-01-01", tz="UTC"), color="lightgray",
                       linestyle=":", linewidth=0.8)
        ax.set_title("FNG 恐惧侧外样本验证: 2021-2025 DOGE/USDT 网格")
        ax.set_xlabel("时间")
        ax.set_ylabel("总资产 (USDT)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PNG_OUT, dpi=150)
        plt.close(fig)
        print(f"\n [图表] 已保存 {PNG_OUT}")

    pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in ("equity", "yearly")}
                  for k, v in legs.items()}).T.to_csv(CSV_OUT, encoding="utf-8-sig")
    print(f" [数据] 已保存 {CSV_OUT}")


if __name__ == "__main__":
    main()
