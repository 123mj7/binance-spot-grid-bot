# -*- coding: utf-8 -*-
"""
adaptive_oos.py —— 外样本(OOS)检验: 胜出组合拉去跑 2021-2022
==============================================================
把三年窗口(2023-2025)选出的唯一胜出组合:
    【有路由(1h 震荡闸门 + 日线 EMA60) + 轮次峰值回撤 20% + 冷却 3 天】
原封不动地拉到 2021-01-01 ~ 2022-12-31 做样本外检验。
该区间 = 2021 疯牛 + 519 单日暴跌 + 2022 深熊 + LUNA/FTX 连环暴雷,
且多次出现 V 型暴跌反弹 —— 正好检验"冷却 3 天 vs 7 天"的机制差异。

对比腿(参数全部一致, 只变冷却):
  - 无路由-固定止损(旧基线)
  - 有路由-固定止损(旧基线)
  - 有路由-轮次20%-冷却3天  ★ 胜出组合
  - 有路由-轮次20%-冷却7天

输出:
  - 整体对比表 + 分年度收益表
  - 两次冷却机制的"止损/重武装"时间线对比(直接回答 Q2)
  - adaptive_oos_compare.png / adaptive_oos_results.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import regime_detect as rd
from adaptive_v3 import backtest_router_v3
from adaptive_backtest import (compute_ema4h, backtest_baseline as fixed_baseline,
                               backtest_router as fixed_router)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

START_DATE = "2021-01-01"
END_DATE = "2022-12-31"
CACHE_OOS = "doge_usdt_1h_2021_22.csv"
PNG_OUT = "adaptive_oos_compare.png"
CSV_OUT = "adaptive_oos_results.csv"
INITIAL_CASH = 50.0


def main() -> None:
    print("=" * 88)
    print(f" 外样本检验: 胜出组合跑 {START_DATE} ~ {END_DATE}(深熊 + 连环暴雷)")
    print("=" * 88)

    df = rd.fetch_history(START_DATE, END_DATE, cache_csv=CACHE_OOS)
    router = rd.compute_router(df)
    ema4h = compute_ema4h(df)

    legs = {}
    bf = fixed_baseline(df, ema4h, trailing=False)
    rf = fixed_router(df, router, trailing=False)
    legs["无路由-固定(旧)"] = {
        **bf, "tag": "无路由-固定(旧)", "n_stops": 1 if bf["stopped"] else 0,
        "rounds": bf["rounds_opened"]}
    legs["有路由-固定(旧)"] = {
        **rf, "tag": "有路由-固定(旧)", "n_stops": 1 if rf["stopped"] else 0,
        "rounds": rf["rounds_opened"]}
    legs["有路由-轮次20%-冷却3天"] = backtest_router_v3(df, router, 0.20, 3)
    legs["有路由-轮次20%-冷却7天"] = backtest_router_v3(df, router, 0.20, 7)

    doge_y = {}
    for y, g in df.groupby(df["time"].dt.year):
        doge_y[y] = (g["close"].iloc[-1] / g["close"].iloc[0] - 1.0) * 100

    order = ["无路由-固定(旧)", "有路由-固定(旧)",
             "有路由-轮次20%-冷却3天", "有路由-轮次20%-冷却7天"]

    # ---- 整体对比 ----
    print()
    print(f"{'腿':<22}{'最终资产':>10}{'总收益%':>9}{'CAGR%':>8}{'最大回撤%':>10}"
          f"{'轮数':>6}{'止损次数':>8}{'极端清仓':>8}{'期末':>10}")
    print("-" * 88)
    for k in order:
        r = legs[k]
        years_n = 2
        cagr = (r["final_equity"] / INITIAL_CASH) ** (1.0 / years_n) - 1.0
        print(f"{k:<22}{r['final_equity']:>10.2f}{r['return_pct']:>9.2f}{cagr * 100:>8.2f}"
              f"{r['max_dd_pct']:>10.2f}{r['rounds']:>6}{r.get('n_stops', 0):>8}"
              f"{r.get('n_extreme', 0):>8}{r['end_state']:>10}")
    print("-" * 88)

    # ---- 分年度 ----
    print()
    print(f"{'腿':<22}{'2021收益%':>10}{'2022收益%':>10}")
    print("-" * 88)
    for k in order:
        y1 = legs[k]["yearly"].get(2021, {"return_pct": 0.0})
        y2 = legs[k]["yearly"].get(2022, {"return_pct": 0.0})
        print(f"{k:<22}{y1['return_pct']:>10.2f}{y2['return_pct']:>10.2f}")
    print(f"{'DOGE 自身涨跌':<22}{doge_y.get(2021, 0):>10.2f}{doge_y.get(2022, 0):>10.2f}")
    print("-" * 88)

    # ---- Q2: 冷却 3 天 vs 7 天 的时间线对比 ----
    l3, l7 = legs["有路由-轮次20%-冷却3天"], legs["有路由-轮次20%-冷却7天"]
    print()
    print(" 冷却机制对比(止损 → 冷却后重武装的时间线):")
    print("-" * 88)
    for tag, leg in [("冷却3天", l3), ("冷却7天", l7)]:
        stops = [t.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
                 for t in leg["stop_times"]]
        rearm = [t.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
                 for t in leg["round_times"]]
        print(f" {tag}: 止损 {len(stops)} 次 @ {stops if stops else '无'}")
        print(f"       开仓 {len(rearm)} 轮")
    # 逐次止损后的下一次重武装时间差
    print()
    print(" 每次止损后的重新武装对比:")
    for i, (s3, s7) in enumerate(zip(l3["stop_times"], l7["stop_times"])):
        n3 = next((t for t in l3["round_times"] if t > s3), None)
        n7 = next((t for t in l7["round_times"] if t > s7), None)
        fmt = lambda t: t.tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M") if t is not None else "此后未再开仓"
        s3s = s3.tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")
        s7s = s7.tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")
        print(f"  止损#{i + 1}: 3天腿 @{s3s} -> 重武装 {fmt(n3)} | "
              f"7天腿 @{s7s} -> 重武装 {fmt(n7)}")

    # ---- 图表 ----
    if plt is not None:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(13, 6))
        t = df["time"]
        ax.plot(t, legs["无路由-固定(旧)"]["equity"], color="#1f77b4",
                linestyle="--", linewidth=0.9, label="无路由-固定止损(旧基线)")
        ax.plot(t, legs["有路由-固定(旧)"]["equity"], color="#d62728",
                linestyle="--", linewidth=0.9, label="有路由-固定止损(旧基线)")
        ax.plot(t, legs["有路由-轮次20%-冷却3天"]["equity"], color="#d62728",
                linewidth=1.1, label="胜出组合: 有路由-轮次20%-冷却3天")
        ax.plot(t, legs["有路由-轮次20%-冷却7天"]["equity"], color="#ff7f0e",
                linewidth=1.1, label="对照: 冷却7天")
        ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8,
                   label="初始资金 50 USDT")
        ax.axvline(pd.Timestamp("2022-01-01", tz="UTC"), color="lightgray",
                   linestyle=":", linewidth=0.8)
        ax.set_title(f"外样本检验: DOGE/USDT DCA 胜出组合 ({START_DATE} ~ {END_DATE})")
        ax.set_xlabel("时间")
        ax.set_ylabel("总资产 (USDT)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PNG_OUT, dpi=150)
        plt.close(fig)
        print(f"\n [图表] 已保存 {PNG_OUT}")

    pd.DataFrame({k: {kk: vv for kk, vv in v.items()
                      if kk not in ("equity", "yearly", "round_times", "stop_times")}
                  for k, v in legs.items()}).T.to_csv(CSV_OUT, encoding="utf-8-sig")
    print(f" [数据] 已保存 {CSV_OUT}")


if __name__ == "__main__":
    main()
