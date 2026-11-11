# -*- coding: utf-8 -*-
"""
backtest_2026.py —— 最新外样本: 2026-01-01 至今 基线策略验证(DOGE + ETH)
=========================================================================
策略 = 无路由-固定止损基线(4h EMA60 过滤 + 跌3%加仓 + 涨3%止盈 + 42.5U止损),
与实盘机器人、以及 2021-2025 五年检验的是同一套。
同时统计"空仓等待占比"(价格在 EMA60 下方的时间), 解释为什么经常没单。
"""
import sys
import time

import ccxt
import numpy as np
import pandas as pd

from adaptive_backtest import compute_ema4h, backtest_baseline, INITIAL_CASH

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

START, END = "2026-01-01", "2026-11-11"
SYMBOLS = ["DOGE/USDT", "ETH/USDT"]
# 注: 当前网络下币安主站连不上, 数据用 Bitget 公开行情(价格序列与币安几乎一致)


def fetch_1h(symbol: str, cache: str) -> pd.DataFrame:
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(END, tz="UTC").normalize().timestamp() * 1000) + 86_400_000
    rows, since = [], start_ms
    while since < end_ms:
        batch = ex.fetch_ohlcv(symbol, "1h", since=since, limit=200)
        if not batch:
            break
        rows.extend(batch)
        ns = batch[-1][0] + 3_600_000
        if ns <= since:
            break
        since = ns
        if len(batch) < 200 or batch[-1][0] >= end_ms - 3_600_000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    now_ms = int(time.time() * 1000)
    df = df[df["ts"] + 3_600_000 <= now_ms].reset_index(drop=True)   # 丢弃未收盘K线
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    print("=" * 92)
    print(f" 最新外样本: {START} ~ 至今 基线策略验证")
    print("=" * 92)
    results = {}
    for sym in SYMBOLS:
        cache = f"doge_usdt_1h_2026.csv" if sym.startswith("DOGE") else "eth_usdt_1h_2026.csv"
        df = fetch_1h(sym, cache)
        ema4h = compute_ema4h(df)
        base = backtest_baseline(df, ema4h, trailing=False)
        bh_ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100
        below = (df["close"] < ema4h)
        idle_pct = below.sum() / below.notna().sum() * 100
        results[sym] = {"df": df, "base": base, "bh_ret": bh_ret, "idle_pct": idle_pct}
        print()
        print(f"  [{sym}]  {len(df)} 根 1h K 线 | 区间 {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")
        print(f"   买入持有          : {bh_ret:+.2f} %")
        print(f"   策略最终总资产    : {base['final_equity']:.2f} USDT "
              f"({base['return_pct']:+.2f} %)")
        print(f"   策略最大回撤      : {base['max_dd_pct']:.2f} %")
        print(f"   开出轮数/止盈轮数 : {base['rounds_opened']} / {base['rounds_tp']}")
        print(f"   期末状态          : {base['end_state']}")
        print(f"   空仓等待占比      : {idle_pct:.1f} % 的时间价格在 EMA60 下方")
    print("-" * 92)

    if plt is not None:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for ax, sym in zip(axes, SYMBOLS):
            r = results[sym]
            t = r["df"]["time"]
            ax.plot(t, r["base"]["equity"], color="#1f77b4", linewidth=1.1,
                    label="策略总资产")
            bh = INITIAL_CASH * r["df"]["close"] / r["df"]["close"].iloc[0]
            ax.plot(t, bh, color="#d62728", linewidth=0.9, linestyle="--",
                    label="买入持有对照")
            ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(f"{sym} 基线策略 vs 买入持有 (2026-01-01 ~ 至今)")
            ax.set_ylabel("总资产 (USDT)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig("backtest_2026.png", dpi=150)
        plt.close(fig)
        print(" [图表] 已保存 backtest_2026.png")


if __name__ == "__main__":
    main()
