# -*- coding: utf-8 -*-
"""futures_gates.py —— 合约趋势策略三关检验"""
import itertools
import sys
import time

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
from futures_backtest import to_4h, turtle


def load_doge(use_2026=True) -> pd.DataFrame:
    d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
    d2 = pd.read_csv("doge_usdt_1h_3y.csv")
    parts = [d1, d2]
    if use_2026:
        parts.append(pd.read_csv("doge_usdt_1h_2026.csv"))
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_bitget_1h(symbol, start, end):
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").normalize().timestamp() * 1000) + 86_400_000
    rows, since = [], start_ms
    while since < end_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, "1h", since=since, limit=200)
        except Exception:
            break
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
    if len(df):
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main() -> None:
    df6 = load_doge(use_2026=True)
    d4 = to_4h(df6)
    df5 = df6[df6["time"].dt.year <= 2025].reset_index(drop=True)
    d45 = to_4h(df5)

    print("=" * 88)
    print(" 第一关: 参数敏感性(DOGE 五年 4h, 基线=突破120/离场60/风险1%/杠杆2x/双向)")
    print("=" * 88)
    base = turtle(d45, entry_n=120, exit_n=60, risk_pct=0.01, lev=2.0,
                  funding_8h=0.0, long_short=True)
    print(f"  基线: {base['ret']:+9.2f}% DD {base['dd']:6.2f}% 交易 {base['closes']} "
          f"爆仓 {base['liq']} 胜率 {base['winrate']:.0f}%")
    print("  -- 突破周期 --")
    for en in (60, 90, 180):
        r = turtle(d45, entry_n=en, exit_n=max(30, en // 2), risk_pct=0.01,
                   lev=2.0, long_short=True)
        print(f"  突破{en:>3}: {r['ret']:+9.2f}% DD {r['dd']:6.2f}% 交易 {r['closes']} "
              f"爆仓 {r['liq']} 胜率 {r['winrate']:.0f}%")
    print("  -- 单笔风险 --")
    for rp in (0.005, 0.02, 0.03):
        r = turtle(d45, risk_pct=rp, lev=2.0, long_short=True)
        print(f"  风险{rp * 100:4.1f}%: {r['ret']:+9.2f}% DD {r['dd']:6.2f}% 交易 {r['closes']}")
    print("  -- 杠杆 --")
    for lv in (1.0, 3.0):
        r = turtle(d45, lev=lv, risk_pct=0.01, long_short=True)
        print(f"  杠杆{lv:.0f}x: {r['ret']:+9.2f}% DD {r['dd']:6.2f}% 爆仓 {r['liq']}")
    print("  -- 资金费率情景(每8小时, 基线参数) --")
    for fund in (0.0, 0.0001, 0.0003):
        r = turtle(d45, funding_8h=fund, risk_pct=0.01, lev=2.0, long_short=True)
        print(f"  费率{fund * 10000:.0f}%%(万): {r['ret']:+9.2f}% DD {r['dd']:6.2f}%")

    print()
    print("=" * 88)
    print(" 第二关: 滚动前推(训练12月→检验3月, DOGE 2022-2025)")
    print("=" * 88)
    test_len = pd.Timedelta(days=92)
    train_len = pd.Timedelta(days=365)
    t0 = df5["time"].iloc[0] + train_len
    t_end = df5["time"].iloc[-1]
    starts = []
    s = t0
    while s < t_end - pd.Timedelta(days=30):
        starts.append(s)
        s += test_len
    fixed_w, opt_w = [], []
    grid = list(itertools.product((60, 120, 180), (0.005, 0.01, 0.02), (1.0, 2.0)))
    for ts in starts:
        te = min(ts + test_len, t_end)
        seg = df5[(df5["time"] >= ts - train_len) & (df5["time"] <= te)].reset_index(drop=True)
        tr_seg = seg[seg["time"] < ts].reset_index(drop=True)
        d_seg, d_tr = to_4h(seg), to_4h(tr_seg)
        idx = (d_seg["time"] >= ts).to_numpy()
        best, best_r = None, -1e9
        for en, rp, lv in grid:
            r = turtle(d_tr, entry_n=en, exit_n=max(30, en // 2),
                       risk_pct=rp, lev=lv, long_short=True)
            if r["ret"] > best_r:
                best_r, best = r["ret"], (en, rp, lv)
        rf = turtle(d_seg, entry_n=120, exit_n=60, risk_pct=0.01, lev=2.0,
                    long_short=True)
        ro = turtle(d_seg, entry_n=best[0], exit_n=max(30, best[0] // 2),
                    risk_pct=best[1], lev=best[2], long_short=True)
        f = rf["equity"][idx]
        o = ro["equity"][idx]
        fixed_w.append((f[-1] / f[0] - 1) * 100 if len(f) > 1 else 0)
        opt_w.append((o[-1] / o[0] - 1) * 100 if len(o) > 1 else 0)
    tot = lambda w: (np.prod(np.array(w) / 100 + 1) - 1) * 100
    print(f"  {len(starts)} 窗复利: 固定基线 {tot(fixed_w):+.2f}% | 每窗重选 {tot(opt_w):+.2f}% "
          f"| 正窗占比 {(np.array(fixed_w) > 0).mean() * 100:.0f}%")

    print()
    print("=" * 88)
    print(" 第三关: 2026 多品种(基线参数)")
    print("=" * 88)
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"):
        dfx = fetch_bitget_1h(sym, "2026-01-01", "2026-11-12")
        if len(dfx) < 2000:
            print(f"  {sym}: 数据不足")
            continue
        d4x = to_4h(dfx)
        r = turtle(d4x, entry_n=120, exit_n=60, risk_pct=0.01, lev=2.0,
                   long_short=True)
        bh = (dfx["close"].iloc[-1] / dfx["close"].iloc[0] - 1) * 100
        print(f"  {sym:<10} 趋势 {r['ret']:+8.2f}% DD {r['dd']:6.2f}% "
              f"交易 {r['closes']} | 持有 {bh:+8.2f}%")


if __name__ == "__main__":
    main()
