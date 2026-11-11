# -*- coding: utf-8 -*-
"""
adaptive_v3.py —— 止损基因重构 v3: 轮次峰值重置 + 宽回撤 + 冷却重武装
======================================================================
三年(2023-01-01 ~ 2025-12-31)大周期验证。三个新基因:

  A. 回撤宽度扩大  : [20%, 25%, 30%](旧基因 15% 比策略正常心跳还窄)
  B. 轮次峰值重置  : 每开新一轮 DCA, 把"总资产峰值"重置为本轮开仓时的总资产,
                     只测本轮开仓以来的回撤 —— 上一轮攒下的利润只作地板,
                     不再被止损线威胁(旧基因看账户历史峰值, 2023 年的老峰值
                     在 2024 年照样拉响止损, 是"牛市前误杀"的元凶)
  C. 冷却重武装    : 触发止损后冷却 [3天, 7天](1h 粒度 = N*24 根 K 线),
                     冷却结束后行情收复日线 EMA60 上方才允许重新开跑,
                     不再永久停机。

对比对象(同参数: 止盈 3% / 加仓间距 3% / 每轮最多 9 次加仓 / 5U / 0.1% 手续费):
  - 固定 42.5U 止损版(旧): 无路由 86.41 / 有路由 41.71
  - 历史峰值 15% 止损版(旧): 无路由 44.17 / 有路由 44.32
  - 本版 v3: 无路由/有路由 x 宽度[20,25,30] x 冷却[3,7] 共 12 条新腿

语义说明: 轮次止损线 = 本轮开仓以来的总资产峰值 x (1 - 宽度),
总资产 = 现金 + 持仓市值(与用户口径一致)。

输出:
  - 14 条腿整体对比表 + 分年度收益表(回答: 2024 牛市跟上了吗? 2025 熊市躲过了吗?)
  - adaptive_v3_compare.png 资金曲线
  - adaptive_v3_results.csv / adaptive_v3_yearly.csv

运行: python adaptive_v3.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import regime_detect as rd
from adaptive_backtest import (compute_ema4h, backtest_baseline as fixed_baseline,
                               backtest_router as fixed_router)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ===================== 区间与公共参数 =====================
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
CACHE_3Y = "doge_usdt_1h_3y.csv"

INITIAL_CASH = 50.0
BUY_USDT = 5.0
MAX_ADDS = 9
TAKE_PROFIT = 0.03
DROP = 0.03
FEE_RATE = 0.001
RANGE, EXTREME = 0, 3

WIDTHS = [0.20, 0.25, 0.30]      # 轮次回撤宽度
COOLDOWN_DAYS = [3, 7]           # 止损后冷却天数

PNG_OUT = "adaptive_v3_compare.png"
CSV_OUT = "adaptive_v3_results.csv"
CSV_YEAR = "adaptive_v3_yearly.csv"

# ===================== 统计工具 =====================


def _year_breakdown(equity, years, rounds_years, initial):
    out = {}
    for y in sorted(set(int(v) for v in years)):
        idx = np.where(years == y)[0]
        if len(idx) == 0:
            continue
        start_eq = float(equity[idx[0] - 1]) if idx[0] > 0 else initial
        seg = equity[idx]
        peak = np.maximum.accumulate(seg)
        dd = float(np.min((seg - peak) / peak)) if peak.max() > 0 else 0.0
        out[y] = {
            "return_pct": (float(seg[-1]) / start_eq - 1.0) * 100,
            "max_dd_pct": dd * 100,
            "rounds": rounds_years.count(y),
        }
    return out


def _wrap(tag, equity, years, rounds_years, cash, qty, n_buys, n_sells,
          rounds, rounds_tp, n_stops, n_extreme, cooldown_left) -> dict:
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peak) / peak)) if peak.max() > 0 else 0.0
    final = float(equity[-1])
    years_n = len(set(int(v) for v in years))
    cagr = (final / INITIAL_CASH) ** (1.0 / years_n) - 1.0
    if qty > 0:
        state = "持仓中"
    elif cooldown_left > 0:
        state = "冷却中"
    else:
        state = "空仓"
    return {
        "tag": tag,
        "final_equity": round(final, 4),
        "return_pct": round((final / INITIAL_CASH - 1.0) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "rounds": rounds, "rounds_tp": rounds_tp,
        "n_stops": n_stops, "n_extreme": n_extreme,
        "n_trades": n_buys + n_sells,
        "n_buys": n_buys, "n_sells": n_sells,
        "end_state": state,
        "yearly": _year_breakdown(equity, years, rounds_years, INITIAL_CASH),
        "equity": equity,
    }

# ===================== v3 回测腿 =====================


def backtest_baseline_v3(df, ema4h, d60, width, cool_days) -> dict:
    """无路由腿(4h EMA60 开仓) + 轮次峰值止损 + 冷却重武装。"""
    prices = df["close"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy(dtype=int)
    n = len(prices)
    cool_bars = cool_days * 24
    cash, qty = INITIAL_CASH, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    round_peak = INITIAL_CASH
    cooldown_left = 0
    need_rearm = False
    n_buys = n_sells = rounds = rounds_tp = n_stops = 0
    rounds_years = []
    equity = np.empty(n)

    for i in range(n):
        p = prices[i]
        if cooldown_left > 0:                        # 冷却期: 不交易
            cooldown_left -= 1
            equity[i] = cash
            continue
        if qty <= 0:                                 # 空仓
            e = ema4h[i]
            if not np.isnan(e) and p > e:
                if need_rearm and not (not np.isnan(d60[i]) and p > d60[i]):
                    pass                             # 冷却结束但未收复日线EMA60 → 继续等
                else:
                    fee = BUY_USDT * FEE_RATE
                    qty += (BUY_USDT - fee) / p
                    cash -= BUY_USDT
                    cost, buys, last_buy = BUY_USDT, 1, p
                    round_peak = cash + qty * p      # ★ 轮次峰值从本轮重置
                    need_rearm = False
                    n_buys += 1
                    rounds += 1
                    rounds_years.append(int(years[i]))
        else:                                        # 持仓中
            if buys - 1 < MAX_ADDS and p <= last_buy * (1 - DROP):
                fee = BUY_USDT * FEE_RATE
                qty += (BUY_USDT - fee) / p
                cash -= BUY_USDT
                cost += BUY_USDT
                buys += 1
                last_buy = p
                n_buys += 1
            avg = cost / qty
            if p >= avg * (1 + TAKE_PROFIT):         # 止盈优先
                cash += qty * p * (1 - FEE_RATE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                rounds_tp += 1
            else:
                total = cash + qty * p
                if total <= round_peak * (1.0 - width):   # ★ 轮次峰值止损
                    cash += qty * p * (1 - FEE_RATE)
                    qty, cost, buys = 0.0, 0.0, 0
                    n_sells += 1
                    n_stops += 1
                    cooldown_left = cool_bars
                    need_rearm = True
                else:
                    round_peak = max(round_peak, total)
        equity[i] = cash + qty * p

    return _wrap(f"无路由-回撤{int(width * 100)}%-冷却{cool_days}天",
                 equity, years, rounds_years, cash, qty, n_buys, n_sells,
                 rounds, rounds_tp, n_stops, 0, cooldown_left)


def backtest_router_v3(df, router, width, cool_days) -> dict:
    """有路由腿(1h 震荡闸门 + 日线 EMA60) + 轮次峰值止损 + 冷却重武装。
    注: 本腿开仓本就要求现价 > 日线 EMA60, 冷却后的重武装条件自动满足。"""
    prices = df["close"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy(dtype=int)
    reg = router["regime_code"].to_numpy(dtype=float)
    d60 = router["ema60_daily"].to_numpy(dtype=float)
    n = len(prices)
    cool_bars = cool_days * 24
    cash, qty = INITIAL_CASH, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    round_peak = INITIAL_CASH
    cooldown_left = 0
    n_buys = n_sells = rounds = rounds_tp = n_stops = n_extreme = 0
    rounds_years = []
    round_times = []          # 每轮开仓时间(用于分析冷却/重武装节奏)
    stop_times = []           # 每次轮次止损时间
    equity = np.empty(n)

    for i in range(n):
        p = prices[i]
        if cooldown_left > 0:
            cooldown_left -= 1
            equity[i] = cash
            continue
        if qty <= 0:
            if reg[i] == RANGE and not np.isnan(d60[i]) and p > d60[i]:
                fee = BUY_USDT * FEE_RATE
                qty += (BUY_USDT - fee) / p
                cash -= BUY_USDT
                cost, buys, last_buy = BUY_USDT, 1, p
                round_peak = cash + qty * p
                n_buys += 1
                rounds += 1
                rounds_years.append(int(years[i]))
                round_times.append(df["time"].iloc[i])
        else:
            if reg[i] == EXTREME:                    # 极端风险清仓
                cash += qty * p * (1 - FEE_RATE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                n_extreme += 1
            else:
                if (reg[i] == RANGE and buys - 1 < MAX_ADDS
                        and p <= last_buy * (1 - DROP)):
                    fee = BUY_USDT * FEE_RATE
                    qty += (BUY_USDT - fee) / p
                    cash -= BUY_USDT
                    cost += BUY_USDT
                    buys += 1
                    last_buy = p
                    n_buys += 1
                avg = cost / qty
                if p >= avg * (1 + TAKE_PROFIT):
                    cash += qty * p * (1 - FEE_RATE)
                    qty, cost, buys = 0.0, 0.0, 0
                    n_sells += 1
                    rounds_tp += 1
                else:
                    total = cash + qty * p
                    if total <= round_peak * (1.0 - width):
                        cash += qty * p * (1 - FEE_RATE)
                        qty, cost, buys = 0.0, 0.0, 0
                        n_sells += 1
                        n_stops += 1
                        cooldown_left = cool_bars
                        stop_times.append(df["time"].iloc[i])
                    else:
                        round_peak = max(round_peak, total)
        equity[i] = cash + qty * p

    res = _wrap(f"有路由-回撤{int(width * 100)}%-冷却{cool_days}天",
                equity, years, rounds_years, cash, qty, n_buys, n_sells,
                rounds, rounds_tp, n_stops, n_extreme, cooldown_left)
    res["round_times"] = round_times
    res["stop_times"] = stop_times
    return res

# ===================== 主流程 =====================


def main() -> None:
    print("=" * 96)
    print(f" 止损基因 v3: 轮次峰值重置 + 宽回撤 + 冷却重武装 ({START_DATE} ~ {END_DATE})")
    print("=" * 96)

    df = rd.fetch_history(START_DATE, END_DATE, cache_csv=CACHE_3Y)
    router = rd.compute_router(df)
    ema4h = compute_ema4h(df)
    d60 = router["ema60_daily"].to_numpy(dtype=float)

    legs = {}
    bf = fixed_baseline(df, ema4h, trailing=False)
    rf = fixed_router(df, router, trailing=False)
    legs["无路由-固定(旧)"] = {
        **bf, "tag": "无路由-固定(旧)", "n_stops": 1 if bf["stopped"] else 0,
        "rounds": bf["rounds_opened"]}
    legs["有路由-固定(旧)"] = {
        **rf, "tag": "有路由-固定(旧)", "n_stops": 1 if rf["stopped"] else 0,
        "rounds": rf["rounds_opened"]}
    for w in WIDTHS:
        for c in COOLDOWN_DAYS:
            legs[f"无路由-回撤{int(w * 100)}%-冷却{c}天"] = backtest_baseline_v3(df, ema4h, d60, w, c)
            legs[f"有路由-回撤{int(w * 100)}%-冷却{c}天"] = backtest_router_v3(df, router, w, c)

    doge_y = {}
    for y, g in df.groupby(df["time"].dt.year):
        doge_y[y] = (g["close"].iloc[-1] / g["close"].iloc[0] - 1.0) * 100

    order = (["无路由-固定(旧)", "有路由-固定(旧)"]
             + [f"{fam}-回撤{int(w * 100)}%-冷却{c}天"
                for fam in ["无路由", "有路由"] for w in WIDTHS for c in COOLDOWN_DAYS])

    # ---- 整体对比表 ----
    print()
    print(f"{'腿':<22}{'最终资产':>10}{'总收益%':>9}{'CAGR%':>8}{'最大回撤%':>10}"
          f"{'轮数':>6}{'止盈':>6}{'止损次数':>8}{'交易':>7}{'期末':>10}")
    print("-" * 96)
    for k in order:
        r = legs[k]
        print(f"{k:<22}{r['final_equity']:>10.2f}{r['return_pct']:>9.2f}{r['cagr_pct']:>8.2f}"
              f"{r['max_dd_pct']:>10.2f}{r['rounds']:>6}{r['rounds_tp']:>6}"
              f"{r['n_stops']:>8}{r['n_buys'] + r['n_sells']:>7}{r['end_state']:>10}")
    print("-" * 96)

    # ---- 分年度收益表 ----
    print()
    print(f" 分年度收益(DOGE: " + " | ".join(
        f"{y} {doge_y[y]:+.0f}%" for y in sorted(doge_y)) + "):")
    print(f"{'腿':<22}{'2023收益%':>10}{'2024收益%':>10}{'2025收益%':>10}{'三年回撤%':>10}")
    print("-" * 96)
    for k in order:
        y3 = legs[k]["yearly"].get(2023, {"return_pct": 0.0})
        y4 = legs[k]["yearly"].get(2024, {"return_pct": 0.0})
        y5 = legs[k]["yearly"].get(2025, {"return_pct": 0.0})
        print(f"{k:<22}{y3['return_pct']:>10.2f}{y4['return_pct']:>10.2f}"
              f"{y5['return_pct']:>10.2f}{legs[k]['max_dd_pct']:>10.2f}")
    print("-" * 96)

    # ---- 双达标筛选: 2024 跟上牛市 且 2025 躲过熊市 ----
    print()
    print(" 双达标候选(2024 收益 >= +20% 且 2025 损失 <= -10%):")
    hits = [k for k in order
            if legs[k]["yearly"].get(2024, {}).get("return_pct", -99) >= 20
            and legs[k]["yearly"].get(2025, {}).get("return_pct", -99) >= -10]
    if hits:
        for k in hits:
            r = legs[k]
            print(f"   ★ {k:<22} 三年总收益 {r['return_pct']:+.2f}% | "
                  f"2024 {r['yearly'][2024]['return_pct']:+.2f}% | "
                  f"2025 {r['yearly'][2025]['return_pct']:+.2f}% | 回撤 {r['max_dd_pct']:.2f}%")
    else:
        print("   (无任何腿同时达标 —— 如实报告)")

    # ---- 图表 ----
    if plt is not None:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(13, 6))
        t = df["time"]
        styles = [
            ("无路由-固定(旧)", "#1f77b4", "--", 0.9, "无路由-固定止损(旧)"),
            ("无路由-回撤25%-冷却7天", "#1f77b4", "-", 1.1, "无路由-轮次止损25%/冷却7天"),
            ("有路由-回撤25%-冷却7天", "#d62728", "-", 1.1, "有路由-轮次止损25%/冷却7天"),
            ("有路由-固定(旧)", "#d62728", "--", 0.9, "有路由-固定止损(旧)"),
        ]
        for key, color, ls, lw, label in styles:
            ax.plot(t, legs[key]["equity"], color=color, linestyle=ls,
                    linewidth=lw, label=label)
        ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8,
                   label="初始资金 50 USDT")
        for y in sorted(doge_y)[1:]:
            ax.axvline(pd.Timestamp(f"{y}-01-01", tz="UTC"), color="lightgray",
                       linestyle=":", linewidth=0.8)
        ax.set_title(f"DOGE/USDT DCA 止损基因 v3: 轮次峰值重置 + 宽回撤 + 冷却重武装"
                     f" ({START_DATE} ~ {END_DATE})")
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
    year_rows = []
    for y in sorted(doge_y):
        for k in order:
            yy = legs[k]["yearly"].get(y, {})
            year_rows.append({
                "年份": y, "腿": k, "DOGE涨跌%": round(doge_y[y], 2),
                "收益%": yy.get("return_pct", 0.0),
                "回撤%": yy.get("max_dd_pct", 0.0),
                "轮数": yy.get("rounds", 0),
            })
    pd.DataFrame(year_rows).to_csv(CSV_YEAR, index=False, encoding="utf-8-sig")
    print(f" [数据] 已保存 {CSV_OUT} / {CSV_YEAR}")


if __name__ == "__main__":
    main()
