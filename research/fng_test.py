# -*- coding: utf-8 -*-
"""
fng_test.py —— 恐惧贪婪指数(FNG)外样本验证: 贪婪闸门有没有用?
================================================================
问题: 把"FNG >= 80 暂停买入"加进无路由-固定止损基线, 在 2021-2025 三年
外样本里到底能不能赢?

三条腿(止盈 3% / 加仓间距 3% / 42.5U 固定止损 / 4h EMA60 过滤全部一致,
只差 FNG 闸门):
  基线          : 4h EMA60 过滤 + 3% 网格 + 固定止损
  FNG≥80停开新仓 : 前一日 FNG >= 80 时暂停开新仓, 加仓照常
  FNG≥80停全部   : 前一日 FNG >= 80 时暂停开新仓和加仓

防未来函数: FNG 是日更且由当日行情计算, 当日值天然包含当日信息,
因此每根 1h K 线只使用"前一天"的 FNG 值(shift 1 天)。

数据: FNG 历史来自 api.alternative.me(免费, 本地缓存 fng_history.csv);
K 线用本地缓存 doge_usdt_1h_3y.csv。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from adaptive_backtest import (compute_ema4h, backtest_baseline,
                               _summarize, INITIAL_CASH)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

FNG_GATE = 80.0          # 贪婪闸门
FNG_FEAR = 15.0          # 极端恐惧(仅统计用)
DATA_PARTS = ["doge_usdt_1h_2021_22.csv", "doge_usdt_1h_3y.csv"]   # 合并成 2021-2025 五年
FNG_CACHE = "fng_history.csv"
PNG_OUT = "fng_test_compare.png"
CSV_OUT = "fng_test_results.csv"

# ===================== FNG 数据 =====================


def fetch_fng() -> pd.Series:
    if os.path.exists(FNG_CACHE):
        print(f"[FNG] 使用本地缓存 {FNG_CACHE}")
        s = pd.read_csv(FNG_CACHE)
        s["ts"] = pd.to_datetime(s["ts"], utc=True)
        return s.set_index("ts")["value"].astype(int)
    print("[FNG] 从 api.alternative.me 下载历史恐惧贪婪指数 ...")
    for attempt in range(3):
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=30)
            data = r.json().get("data") or []
            if not data:
                raise RuntimeError("API 返回空数据")
            df = pd.DataFrame(data)
            df["ts"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
            df["value"] = df["value"].astype(int)
            df = df.sort_values("ts")
            out = df.set_index("ts")["value"]
            out = out.groupby(out.index).last()          # 去重
            out.to_csv(FNG_CACHE, encoding="utf-8-sig")
            print(f"[FNG] 下载成功: {len(out)} 天历史数据")
            return out
        except Exception as exc:
            print(f"[FNG] 下载失败(第{attempt + 1}次): {exc}")
            time.sleep(3)
    sys.exit("[FNG] 下载失败, 请检查网络后重试")

# ===================== FNG 闸门回测腿 =====================


def backtest_fng(df: pd.DataFrame, ema4h: np.ndarray, fng_arr: np.ndarray,
                 gate: float = FNG_GATE, pause_adds: bool = True, tag: str = "") -> dict:
    """基线策略 + FNG 贪婪闸门(前一日 FNG >= gate 暂停买入)。"""
    prices = df["close"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy(dtype=int)
    n = len(prices)
    cash, qty = INITIAL_CASH, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    stopped = False
    n_buys = n_sells = rounds = rounds_tp = 0
    rounds_years = []
    n_greed = n_skip_entry = n_skip_add = 0
    equity = np.empty(n)

    for i in range(n):
        p = prices[i]
        greedy = (not np.isnan(fng_arr[i])) and fng_arr[i] >= gate
        if greedy:
            n_greed += 1
        if stopped:
            equity[i] = cash
            continue
        if qty <= 0:                                  # 空仓
            e = ema4h[i]
            if greedy:
                n_skip_entry += 1                     # 贪婪期: 暂停开新仓
            elif not np.isnan(e) and p > e:
                fee = 5 * 0.001
                qty += (5 - fee) / p
                cash -= 5
                cost, buys, last_buy = 5.0, 1, p
                n_buys += 1
                rounds += 1
                rounds_years.append(int(years[i]))
        else:                                         # 持仓中
            if buys - 1 < 9 and p <= last_buy * 0.97:
                if pause_adds and greedy:
                    n_skip_add += 1                   # 贪婪期: 暂停加仓
                else:
                    fee = 5 * 0.001
                    qty += (5 - fee) / p
                    cash -= 5
                    cost += 5
                    buys += 1
                    last_buy = p
                    n_buys += 1
            avg = cost / qty
            if p >= avg * 1.03:                       # 止盈(任何状态)
                cash += qty * p * 0.999
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                rounds_tp += 1
            elif qty > 0 and buys >= 10 and cash + qty * p < 42.5:
                cash += qty * p * 0.999                # 固定止损
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                stopped = True
        equity[i] = cash + qty * p

    res = _summarize(equity, years, rounds_years, cash, qty, n_buys, n_sells,
                     rounds, rounds_tp, 0, stopped, tag)
    res["n_greed_bars"] = n_greed
    res["n_skip_entry"] = n_skip_entry
    res["n_skip_add"] = n_skip_add
    return res

# ===================== 主流程 =====================


def main() -> None:
    print("=" * 82)
    print(" FNG 外样本验证: 贪婪闸门(FNG>=80 暂停买入)有没有用? (2021-2025 五年)")
    print("=" * 82)

    parts = []
    for f in DATA_PARTS:
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["time"] = pd.to_datetime(d["time"], utc=True)
            parts.append(d)
        else:
            print(f"[警告] 缺少缓存 {f}")
    if not parts:
        sys.exit("[错误] 无本地 K 线缓存, 请先运行 adaptive_oos.py / adaptive_backtest.py")
    df = (pd.concat(parts).drop_duplicates("ts").sort_values("ts")
          .reset_index(drop=True))
    print(f"[K线] 合并缓存: {len(df)} 根 1h K 线 | "
          f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")
    fng = fetch_fng()
    fng_prev = fng.shift(1)                           # ★ 防未来函数: 只用前一天的值
    day = df["time"].dt.normalize()
    fng_arr = day.map(fng_prev).to_numpy(dtype=float)
    print(f"[FNG] 覆盖 K 线 {int((~np.isnan(fng_arr)).sum())}/{len(df)} 根")
    print(f"[FNG] 三年中 FNG>=80 天数: {int((fng >= FNG_GATE).sum())} | "
          f"FNG<=15 天数: {int((fng <= FNG_FEAR).sum())}")

    ema4h = compute_ema4h(df)
    legs = {
        "基线(无FNG)": backtest_baseline(df, ema4h, trailing=False),
        "FNG≥80停开新仓": backtest_fng(df, ema4h, fng_arr, pause_adds=False, tag="FNG-停开仓"),
        "FNG≥80停全部买入": backtest_fng(df, ema4h, fng_arr, pause_adds=True, tag="FNG-停全部"),
    }

    # ---- 整体对比 ----
    print()
    print(f"{'腿':<18}{'最终资产':>10}{'总收益%':>9}{'最大回撤%':>10}"
          f"{'轮数':>6}{'止盈':>6}{'贪婪跳过':>10}")
    print("-" * 82)
    for k, r in legs.items():
        skips = r.get("n_skip_entry", 0) + r.get("n_skip_add", 0)
        print(f"{k:<18}{r['final_equity']:>10.2f}{r['return_pct']:>9.2f}"
              f"{r['max_dd_pct']:>10.2f}{r['rounds_opened']:>6}{r['rounds_tp']:>6}"
              f"{skips:>10}")
    print("-" * 82)

    # ---- 分年度 ----
    print()
    print(f"{'腿':<18}{'2021':>10}{'2022':>10}{'2023':>10}{'2024':>10}{'2025':>10}")
    print("-" * 82)
    for k, r in legs.items():
        vals = [r["yearly"].get(y, {}).get("return_pct", 0.0) for y in (2021, 2022, 2023, 2024, 2025)]
        print(f"{k:<18}" + "".join(f"{v:>10.2f}" for v in vals))
    print("-" * 82)

    # ---- 结论行 ----
    base = legs["基线(无FNG)"]
    print()
    for k in ["FNG≥80停开新仓", "FNG≥80停全部买入"]:
        r = legs[k]
        diff = r["return_pct"] - base["return_pct"]
        dd = r["max_dd_pct"] - base["max_dd_pct"]
        print(f" {k}: 相对基线 {('多赚' if diff >= 0 else '少赚')} {abs(diff):.2f} 个百分点, "
              f"回撤 {'多' if dd >= 0 else '少'} {abs(dd):.2f} 个百分点")

    # ---- 图表 ----
    if plt is not None:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(13, 6))
        t = df["time"]
        ax.plot(t, legs["基线(无FNG)"]["equity"], color="#1f77b4", linewidth=1.1,
                label="基线(无FNG)")
        ax.plot(t, legs["FNG≥80停开新仓"]["equity"], color="#2ca02c", linewidth=0.9,
                label="FNG≥80 暂停开新仓")
        ax.plot(t, legs["FNG≥80停全部买入"]["equity"], color="#d62728", linewidth=0.9,
                label="FNG≥80 暂停全部买入")
        ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8,
                   label="初始资金 50 USDT")
        for y in (2022, 2023, 2024, 2025):
            ax.axvline(pd.Timestamp(f"{y}-01-01", tz="UTC"), color="lightgray",
                       linestyle=":", linewidth=0.8)
        ax.set_title("FNG 贪婪闸门外样本验证: 2021-2025 DOGE/USDT 网格")
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
