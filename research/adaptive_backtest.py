# -*- coding: utf-8 -*-
"""
adaptive_backtest.py —— 三年大周期: 固定止损 vs 动态峰值止损 闭环对比
======================================================================
数据: 2023-01-01 ~ 2025-12-31, DOGE/USDT 1h(币安现货, UTC 日界)

四条腿(止盈/加仓/路由参数完全一致):
  无路由 x {固定止损(42.5U, 子弹打光+跌破才触发), 动态止损(历史峰值回撤 15%)}
  有路由 x {固定止损, 动态止损}
其中"有路由" = 1h 震荡闸门 + 日线 EMA60 大过滤 + 1h 极端风险清仓(不变)。

动态止损(新基因)语义: 记录总资产历史峰值, 只要总资产(现金+持仓市值)
跌破 峰值 x (1 - 15%), 无条件全部清仓并永久停止 —— 保住牛市攒下的利润。

输出:
  - 4 列整体对比表 + 分年度对比表(固定/动态各一张)
  - adaptive_compare_3y_trailing.png(四条资金曲线, 带年份分隔线)
  - adaptive_compare_3y_trailing.csv / adaptive_yearly_3y_trailing.csv

运行: python adaptive_backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import regime_detect as rd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ===================== 回测区间与公共参数 =====================
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
CACHE_3Y = "doge_usdt_1h_3y.csv"

INITIAL_CASH = 50.0
BUY_USDT = 5.0
MAX_ADDS = 9
TAKE_PROFIT = 0.03
DROP = 0.03
STOP_LOSS_PCT = 0.15            # 固定止损: 跌破初始资金 15%(42.5U) 且子弹打光
TRAILING_PCT = 0.15             # 动态止损: 历史峰值回撤 15%
FEE_RATE = 0.001
STOP_LINE = INITIAL_CASH * (1.0 - STOP_LOSS_PCT)

RANGE, EXTREME = 0, 3
PNG_OUT = "adaptive_compare_3y_trailing.png"
CSV_OUT = "adaptive_compare_3y_trailing.csv"
CSV_YEAR = "adaptive_yearly_3y_trailing.csv"


def compute_ema4h(df: pd.DataFrame, period: int = 60) -> np.ndarray:
    """v2 同款 4h EMA60(仅用于无路由腿的开仓过滤)。"""
    bucket = df["time"].dt.floor("4h")
    close4h = df.assign(bucket=bucket).groupby("bucket")["close"].last()
    ema = close4h.ewm(span=period, adjust=False).mean()
    ema.iloc[:period] = np.nan
    ema = ema.shift(1)
    return bucket.map(ema).to_numpy()

# ===================== 回测腿 =====================


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


def _summarize(equity, years, rounds_years, cash, qty, n_buys, n_sells,
               rounds, rounds_tp, n_extreme, stopped, stop_mode) -> dict:
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peak) / peak)) if peak.max() > 0 else 0.0
    final = float(equity[-1])
    years_n = len(set(int(v) for v in years))
    cagr = (final / INITIAL_CASH) ** (1.0 / years_n) - 1.0
    return {
        "final_equity": round(final, 4),
        "return_pct": round((final / INITIAL_CASH - 1.0) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "n_buys": n_buys, "n_sells": n_sells,
        "rounds_opened": rounds,
        "rounds_tp": rounds_tp,
        "n_extreme": n_extreme,
        "stopped": bool(stopped),
        "stop_mode": stop_mode,
        "end_state": "止损停止" if stopped else ("持仓中" if qty > 0 else "空仓"),
        "yearly": _year_breakdown(equity, years, rounds_years, INITIAL_CASH),
        "equity": equity,
    }


def backtest_baseline(df: pd.DataFrame, ema4h: np.ndarray, trailing: bool) -> dict:
    """无路由基线(4h EMA60 过滤)。trailing=True 用动态峰值止损, 否则固定 42.5U 止损。"""
    prices = df["close"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy(dtype=int)
    n = len(prices)
    cash, qty = INITIAL_CASH, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    stopped = False
    peak = INITIAL_CASH                       # 总资产历史峰值
    n_buys = n_sells = rounds = rounds_tp = 0
    rounds_years = []
    equity = np.empty(n)

    for i in range(n):
        p = prices[i]
        if stopped:
            equity[i] = cash
            continue
        if qty <= 0:
            e = ema4h[i]
            if not np.isnan(e) and p > e:
                fee = BUY_USDT * FEE_RATE
                qty += (BUY_USDT - fee) / p
                cash -= BUY_USDT
                cost, buys, last_buy = BUY_USDT, 1, p
                n_buys += 1
                rounds += 1
                rounds_years.append(int(years[i]))
        else:
            if buys - 1 < MAX_ADDS and p <= last_buy * (1 - DROP):
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
            elif (not trailing and qty > 0 and buys >= MAX_ADDS + 1
                    and cash + qty * p < STOP_LINE):          # 固定止损
                cash += qty * p * (1 - FEE_RATE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                stopped = True

        total = cash + qty * p
        if trailing and total <= peak * (1.0 - TRAILING_PCT):   # 动态峰值止损
            if qty > 0:
                cash += qty * p * (1 - FEE_RATE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
            stopped = True                                      # 无条件清仓停止
            total = cash
        else:
            peak = max(peak, total)
        equity[i] = total

    return _summarize(equity, years, rounds_years, cash, qty, n_buys, n_sells,
                      rounds, rounds_tp, 0, stopped,
                      "动态峰值止损" if trailing else "固定42.5U止损")


def backtest_router(df: pd.DataFrame, router: pd.DataFrame, trailing: bool) -> dict:
    """有路由(1h 震荡闸门 + 日线 EMA60)。trailing=True 用动态峰值止损。"""
    prices = df["close"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy(dtype=int)
    reg = router["regime_code"].to_numpy(dtype=float)
    d60 = router["ema60_daily"].to_numpy(dtype=float)
    n = len(prices)
    cash, qty = INITIAL_CASH, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    stopped = False
    peak = INITIAL_CASH
    n_buys = n_sells = rounds = rounds_tp = n_extreme = 0
    rounds_years = []
    equity = np.empty(n)

    for i in range(n):
        p = prices[i]
        if stopped:
            equity[i] = cash
            continue
        if qty <= 0:
            if reg[i] == RANGE and not np.isnan(d60[i]) and p > d60[i]:
                fee = BUY_USDT * FEE_RATE
                qty += (BUY_USDT - fee) / p
                cash -= BUY_USDT
                cost, buys, last_buy = BUY_USDT, 1, p
                n_buys += 1
                rounds += 1
                rounds_years.append(int(years[i]))
        else:
            if reg[i] == EXTREME:                     # 1) 极端风险 → 无条件清仓避险
                cash += qty * p * (1 - FEE_RATE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                n_extreme += 1
            else:
                if (reg[i] == RANGE and buys - 1 < MAX_ADDS
                        and p <= last_buy * (1 - DROP)):     # 2) 仅震荡时加仓
                    fee = BUY_USDT * FEE_RATE
                    qty += (BUY_USDT - fee) / p
                    cash -= BUY_USDT
                    cost += BUY_USDT
                    buys += 1
                    last_buy = p
                    n_buys += 1
                avg = cost / qty
                if p >= avg * (1 + TAKE_PROFIT):            # 3) 止盈(任何状态)
                    cash += qty * p * (1 - FEE_RATE)
                    qty, cost, buys = 0.0, 0.0, 0
                    n_sells += 1
                    rounds_tp += 1
                elif (not trailing and qty > 0 and buys >= MAX_ADDS + 1
                        and cash + qty * p < STOP_LINE):    # 固定止损
                    cash += qty * p * (1 - FEE_RATE)
                    qty, cost, buys = 0.0, 0.0, 0
                    n_sells += 1
                    stopped = True

        total = cash + qty * p
        if trailing and total <= peak * (1.0 - TRAILING_PCT):   # 动态峰值止损
            if qty > 0:
                cash += qty * p * (1 - FEE_RATE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
            stopped = True
            total = cash
        else:
            peak = max(peak, total)
        equity[i] = total

    return _summarize(equity, years, rounds_years, cash, qty, n_buys, n_sells,
                      rounds, rounds_tp, n_extreme, stopped,
                      "动态峰值止损" if trailing else "固定42.5U止损")

# ===================== 主流程 =====================


def main() -> None:
    print("=" * 88)
    print(f" 三年大周期: 固定止损 vs 动态峰值止损 ({START_DATE} ~ {END_DATE})")
    print("=" * 88)

    df = rd.fetch_history(START_DATE, END_DATE, cache_csv=CACHE_3Y)
    router = rd.compute_router(df)
    ema4h = compute_ema4h(df)

    legs = {
        "无路由-固定": backtest_baseline(df, ema4h, trailing=False),
        "无路由-动态": backtest_baseline(df, ema4h, trailing=True),
        "有路由-固定": backtest_router(df, router, trailing=False),
        "有路由-动态": backtest_router(df, router, trailing=True),
    }

    doge_y = {}
    for y, g in df.groupby(df["time"].dt.year):
        doge_y[y] = (g["close"].iloc[-1] / g["close"].iloc[0] - 1.0) * 100

    # ---- 整体对比(4 列) ----
    print()
    print(f"{'整体指标':<12}{'无路由-固定':>16}{'无路由-动态':>16}{'有路由-固定':>16}{'有路由-动态':>16}")
    print("-" * 88)
    labels = [
        ("最终总资产", lambda r: f"{r['final_equity']:.4f} U"),
        ("总收益率", lambda r: f"{r['return_pct']:.2f} %"),
        ("年化 CAGR", lambda r: f"{r['cagr_pct']:.2f} %"),
        ("最大回撤", lambda r: f"{r['max_dd_pct']:.2f} %"),
        ("总交易次数", lambda r: f"{r['n_buys'] + r['n_sells']}次(买{r['n_buys']}/卖{r['n_sells']})"),
        ("开出轮数/止盈", lambda r: f"{r['rounds_opened']}/{r['rounds_tp']}"),
        ("极端清仓", lambda r: str(r["n_extreme"]) if r["n_extreme"] else "—"),
        ("止损触发", lambda r: "是" if r["stopped"] else "否"),
        ("期末状态", lambda r: r["end_state"]),
    ]
    for label, fn in labels:
        print(f"{label:<12}" + "".join(f"{fn(legs[k]):>16}" for k in legs))
    print("-" * 88)

    # ---- 分年度对比 ----
    for mode, k1, k2 in [("固定 42.5U 止损", "无路由-固定", "有路由-固定"),
                         ("动态峰值止损(15%)", "无路由-动态", "有路由-动态")]:
        print()
        print(f" 分年度对比 —— {mode}:")
        print(f"{'年份':<6}{'DOGE涨跌':>10}{'无路由收益':>12}{'无路由回撤':>12}"
              f"{'有路由收益':>12}{'有路由回撤':>12}{'有路由轮数':>12}")
        print("-" * 88)
        for y in sorted(doge_y):
            a, b = legs[k1]["yearly"][y], legs[k2]["yearly"][y]
            print(f"{y:<8}{doge_y[y]:>9.1f}%{a['return_pct']:>11.2f}%{a['max_dd_pct']:>11.2f}%"
                  f"{b['return_pct']:>11.2f}%{b['max_dd_pct']:>11.2f}%{b['rounds']:>12}")
        print("-" * 88)

    # ---- 结论行 ----
    print()
    for k in legs:
        print(f" {k:<12} 最终 {legs[k]['final_equity']:.2f} USDT"
              f"({legs[k]['return_pct']:+.2f}%), 最大回撤 {legs[k]['max_dd_pct']:.2f}%")
    print()

    # ---- 图表 ----
    if plt is not None:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(13, 6))
        t = df["time"]
        ax.plot(t, legs["无路由-动态"]["equity"], linewidth=1.1, color="#1f77b4",
                label="无路由 + 动态峰值止损")
        ax.plot(t, legs["无路由-固定"]["equity"], linewidth=0.9, color="#1f77b4",
                linestyle="--", alpha=0.7, label="无路由 + 固定止损(旧)")
        ax.plot(t, legs["有路由-动态"]["equity"], linewidth=1.1, color="#d62728",
                label="有路由 + 动态峰值止损")
        ax.plot(t, legs["有路由-固定"]["equity"], linewidth=0.9, color="#d62728",
                linestyle="--", alpha=0.7, label="有路由 + 固定止损(旧)")
        ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8,
                   label="初始资金 50 USDT")
        for y in sorted(doge_y)[1:]:
            ax.axvline(pd.Timestamp(f"{y}-01-01", tz="UTC"), color="lightgray",
                       linestyle=":", linewidth=0.8)
        ax.set_title(f"DOGE/USDT DCA: 固定止损 vs 动态峰值止损 ({START_DATE} ~ {END_DATE})")
        ax.set_xlabel("时间")
        ax.set_ylabel("总资产 (USDT)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PNG_OUT, dpi=150)
        plt.close(fig)
        print(f" [图表] 已保存 {PNG_OUT}")

    pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in ("equity", "yearly")}
                  for k, v in legs.items()}).T.to_csv(CSV_OUT, encoding="utf-8-sig")
    year_rows = []
    for y in sorted(doge_y):
        for k in legs:
            year_rows.append({
                "年份": y, "腿": k,
                "DOGE涨跌%": round(doge_y[y], 2),
                "收益%": legs[k]["yearly"][y]["return_pct"],
                "回撤%": legs[k]["yearly"][y]["max_dd_pct"],
                "轮数": legs[k]["yearly"][y]["rounds"],
            })
    pd.DataFrame(year_rows).to_csv(CSV_YEAR, index=False, encoding="utf-8-sig")
    print(f" [数据] 已保存 {CSV_OUT} / {CSV_YEAR}")


if __name__ == "__main__":
    main()
