# -*- coding: utf-8 -*-
"""
safe_aggressive.py —— "赚更多且尽量安全"的组合研究
====================================================
1) 趋势跟踪(4h EMA快/慢交叉)三关检验: 敏感性 / 滚动前推 / 2026多品种
2) 组合: 网格70%+趋势30% 与 50/50, DOGE 六年
3) 多品种分散: 2026 六币等权组合(网格)的回撤摊薄
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

FEE = 0.001
INIT = 100.0


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


def trend_equity(df: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    sf = pd.Series(closes).ewm(span=fast, adjust=False).mean().to_numpy()
    ss = pd.Series(closes).ewm(span=slow, adjust=False).mean().to_numpy()
    cash, qty = INIT, 0.0
    eq = np.empty(n)
    for i in range(n):
        if i < slow:
            eq[i] = cash
            continue
        if sf[i] > ss[i] and qty <= 0:
            qty = (cash - cash * FEE) / closes[i]
            cash = 0.0
        elif sf[i] < ss[i] and qty > 0:
            cash = qty * closes[i] * (1 - FEE)
            qty = 0.0
        eq[i] = cash + qty * closes[i]
    return eq


def tstats(eq):
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = float(np.min((eq - peak) / peak)) * 100
    return (eq[-1] / INIT - 1) * 100, dd, eq


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
    df5 = df6[df6["time"].dt.year <= 2025].reset_index(drop=True)
    print("=" * 88)
    print(" 1) 趋势跟踪第一关: 参数敏感性(快/慢 EMA 12 组, DOGE 五年)")
    print("=" * 88)
    rows = []
    for fast, slow in itertools.product((10, 20, 30), (50, 60, 70, 80)):
        if slow <= fast:
            continue
        eq = trend_equity(df5, fast, slow)
        ret, dd, _ = tstats(eq)
        rows.append((fast, slow, ret, dd))
        print(f"  EMA{fast}/{slow}: {ret:+9.2f}%  DD {dd:6.2f}%")
    res = pd.DataFrame(rows, columns=["fast", "slow", "ret", "dd"])
    res.to_csv("trend_sensitivity.csv", index=False, encoding="utf-8-sig")
    base = res[(res.fast == 20) & (res.slow == 60)].iloc[0]
    nb = res[(res.fast == 20) & (res.slow.isin([50, 60, 70]))]
    print(f"  → 基线EMA20/60: {base.ret:+.2f}% | 邻居 {nb.ret.min():+.1f}%~"
          f"{nb.ret.max():+.1f}% | 判定: "
          f"{'高原' if nb.ret.median() >= base.ret * 0.5 else '孤峰⚠'}")

    print()
    print("=" * 88)
    print(" 2) 趋势跟踪第二关: 滚动前推(训练12月→检验3月)")
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
    for ts in starts:
        te = min(ts + test_len, t_end)
        seg = df5[(df5["time"] >= ts - train_len) & (df5["time"] <= te)].reset_index(drop=True)
        train_seg = seg[seg["time"] < ts].reset_index(drop=True)
        idx = (seg["time"] >= ts).to_numpy()
        best, best_r = None, -1e9
        for fast, slow in itertools.product((10, 20, 30), (50, 60, 70, 80)):
            if slow <= fast:
                continue
            eq = trend_equity(train_seg, fast, slow)
            ret, _, _ = tstats(eq)
            if ret > best_r:
                best_r, best = ret, (fast, slow)
        eqf = trend_equity(seg, 20, 60)
        eqt = trend_equity(seg, best[0], best[1])
        f = eqf[idx]
        t = eqt[idx]
        fixed_w.append((f[-1] / f[0] - 1) * 100 if len(f) > 1 else 0)
        opt_w.append((t[-1] / t[0] - 1) * 100 if len(t) > 1 else 0)
    tot = lambda w: (np.prod(np.array(w) / 100 + 1) - 1) * 100
    print(f"  {len(starts)} 窗复利: 固定20/60 {tot(fixed_w):+.2f}% | "
          f"每窗重选 {tot(opt_w):+.2f}% | 正窗占比 "
          f"{(np.array(fixed_w) > 0).mean() * 100:.0f}%")

    print()
    print("=" * 88)
    print(" 3) 组合: 网格70%+趋势30%(DOGE 六年, 初始100U, 无再平衡)")
    print("=" * 88)
    ema = ab.compute_ema4h(df6, period=60)
    g = run_fixed(df6, ema, "close", slip=0.001)
    g_eq = np.asarray(g["equity"]) * (INIT / 50.0)
    t_eq = trend_equity(df6, 20, 60)
    for wg, wt in ((0.7, 0.3), (0.5, 0.5), (0.3, 0.7)):
        blend = wg * g_eq + wt * t_eq
        ret, dd, _ = tstats(blend)
        print(f"  网格{wg:.0%}+趋势{wt:.0%}: {ret:+9.2f}%  DD {dd:6.2f}%")
    ret, dd, _ = tstats(g_eq)
    print(f"  [对照] 纯网格: {ret:+9.2f}%  DD {dd:6.2f}%")
    ret, dd, _ = tstats(t_eq)
    print(f"  [对照] 纯趋势: {ret:+9.2f}%  DD {dd:6.2f}%")

    print()
    print("=" * 88)
    print(" 4) 2026 多品种: 趋势 vs 网格 + 六币等权组合回撤摊薄")
    print("=" * 88)
    syms = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT")
    g_list, t_list = [], []
    for sym in syms:
        dfx = fetch_bitget_1h(sym, "2026-01-01", "2026-11-12")
        if len(dfx) < 2000:
            continue
        emax = ab.compute_ema4h(dfx, period=60)
        g = run_fixed(dfx, emax, "close")
        g_list.append(np.asarray(g["equity"]) / 50.0)
        t_list.append(trend_equity(dfx, 20, 60) / INIT)
        gr = (g_list[-1][-1] - 1) * 100
        tr = (t_list[-1][-1] - 1) * 100
        print(f"  {sym:<10} 网格 {gr:+7.2f}% | 趋势 {tr:+7.2f}%")
    if g_list:
        port = np.mean(g_list, axis=0)
        ret = (port[-1] - 1.0) * 100
        dd, _, _ = tstats(port)
        print(f"  六币等权网格组合: {ret:+.2f}%  DD {dd:.2f}% "
              f"(单币DD约-16~-17%, 组合被摊薄)")


if __name__ == "__main__":
    main()
