# -*- coding: utf-8 -*-
"""
compound_gates.py —— 全自动复投版的三关检验
============================================
第一关 比例敏感性: fraction 0.3~1.0 扫 8 档, 看高原/孤峰
第二关 滚动前推:   固定比例 vs 每窗重选比例 vs 固定50U基线, 2022-2025 逐季
第三关 多品种:     2026 六品种, 复投版 vs 固定版
"""
import itertools
import sys
import time

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab
from beauty_filter_audit import run_fixed
from compound_test import run_compound

# ================= 数据 =================


def load_doge_5y() -> pd.DataFrame:
    d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
    d2 = pd.read_csv("doge_usdt_1h_3y.csv")
    df = pd.concat([d1, d2], ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_bitget_1h(symbol: str, start: str, end: str) -> pd.DataFrame:
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


def seg_ret(eq, idx):
    eq = np.asarray(eq, dtype=float)[idx]
    eq = eq[eq > 0]
    return (eq[-1] / eq[0] - 1) * 100 if len(eq) > 1 else 0.0


# ================= 第一关 =================


def gate1(df5y: pd.DataFrame) -> None:
    print("=" * 84)
    print(" 第一关: 复投比例敏感性(DOGE 五年)")
    print("=" * 84)
    ema = ab.compute_ema4h(df5y, period=60)
    rows = []
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r = run_compound(df5y, ema, frac)
        rows.append((frac, r["return_pct"], r["max_dd_pct"], r["rounds"]))
        print(f"  净值x{frac:.1f}: {r['return_pct']:+8.2f}%  DD {r['max_dd_pct']:6.2f}%  "
              f"轮数 {r['rounds']}")
    res = pd.DataFrame(rows, columns=["frac", "ret", "dd", "rounds"])
    res.to_csv("compound_sensitivity.csv", index=False, encoding="utf-8-sig")
    mono = (res.ret.diff().dropna() > 0).all() or (res.ret.diff().dropna() < 0).all()
    print(f"  → 收益随比例单调性: {'单调' if mono else '非单调(有局部最优, 需谨慎)'}")


# ================= 第二关 =================


def gate2(df5y: pd.DataFrame) -> None:
    print()
    print("=" * 84)
    print(" 第二关: 滚动前推(训练12月→检验3月), 对比 固定50U vs 复投x0.5 vs 复投x0.8 vs 每窗重选")
    print("=" * 84)
    test_len = pd.Timedelta(days=92)
    train_len = pd.Timedelta(days=365)
    t0 = df5y["time"].iloc[0] + train_len
    t_end = df5y["time"].iloc[-1]
    starts = []
    s = t0
    while s < t_end - pd.Timedelta(days=30):
        starts.append(s)
        s += test_len
    fixed_w, c05_w, c08_w, opt_w = [], [], [], []
    for ts in starts:
        te = min(ts + test_len, t_end)
        seg = df5y[(df5y["time"] >= ts - train_len) & (df5y["time"] <= te)].reset_index(drop=True)
        train_seg = seg[seg["time"] < ts].reset_index(drop=True)
        idx = (seg["time"] >= ts).to_numpy()
        ema = ab.compute_ema4h(seg, period=60)
        ema_tr = ab.compute_ema4h(train_seg, period=60)
        # 训练期选比例
        best_frac, best_r = None, -1e9
        for frac in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r = run_compound(train_seg, ema_tr, frac)
            if r["return_pct"] > best_r:
                best_r, best_frac = r["return_pct"], frac
        rf = run_fixed(seg, ema, "close")
        r5 = run_compound(seg, ema, 0.5)
        r8 = run_compound(seg, ema, 0.8)
        ro = run_compound(seg, ema, best_frac)
        fixed_w.append(seg_ret(rf["equity"], idx))
        c05_w.append(seg_ret(r5["equity"], idx))
        c08_w.append(seg_ret(r8["equity"], idx))
        opt_w.append(seg_ret(ro["equity"], idx))
        print(f"  {ts:%Y-%m-%d}~{te:%Y-%m-%d} 训练选比例 {best_frac:.1f} | "
              f"固定{fixed_w[-1]:+7.2f}% x0.5{c05_w[-1]:+7.2f}% x0.8{c08_w[-1]:+7.2f}% "
              f"重选{opt_w[-1]:+7.2f}%")
    tot = lambda w: (np.prod(np.array(w) / 100 + 1) - 1) * 100
    print("-" * 84)
    print(f"  {len(starts)} 窗复利: 固定50U {tot(fixed_w):+8.2f}% | 复投x0.5 {tot(c05_w):+8.2f}% "
          f"| 复投x0.8 {tot(c08_w):+8.2f}% | 每窗重选 {tot(opt_w):+8.2f}%")
    pd.DataFrame({"start": starts, "fixed": fixed_w, "x05": c05_w,
                  "x08": c08_w, "opt": opt_w}).to_csv(
        "compound_walkforward.csv", index=False, encoding="utf-8-sig")


# ================= 第三关 =================


def gate3() -> None:
    print()
    print("=" * 84)
    print(" 第三关: 多品种 2026(固定50U vs 复投x0.8)")
    print("=" * 84)
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"):
        dfx = fetch_bitget_1h(sym, "2026-01-01", "2026-11-12")
        if len(dfx) < 2000:
            print(f"  {sym}: 数据不足")
            continue
        emax = ab.compute_ema4h(dfx, period=60)
        rf = run_fixed(dfx, emax, "close")
        rc = run_compound(dfx, emax, 0.8)
        print(f"  {sym:<10} 固定 {rf['return_pct']:+7.2f}% (DD {rf['max_dd_pct']:.1f}) | "
              f"复投x0.8 {rc['return_pct']:+7.2f}% (DD {rc['max_dd_pct']:.1f})")


def main() -> None:
    df5y = load_doge_5y()
    print(f"五年数据: {len(df5y)} 根")
    gate1(df5y)
    gate2(df5y)
    gate3()
    print()
    print("完成: compound_sensitivity.csv / compound_walkforward.csv")


if __name__ == "__main__":
    main()
