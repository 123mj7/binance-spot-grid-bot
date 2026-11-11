# -*- coding: utf-8 -*-
"""
anti_overfit.py —— 按"三关五步"框架的完整反过拟合检验
========================================================
第一关 参数敏感性: 75 组邻居参数, 检验基线(3%/3%/EMA60)是"高原"还是"孤峰"
第二关 滚动前推:   训练12个月 → 检验3个月 → 逐季滚动, 对比"固定基线 vs 每窗重新选参"
第三关 多品种:     同一套参数(零重调)跑 BTC/ETH/SOL/BNB/XRP/DOGE 的 2026 至今
"""
import itertools
import os
import sys
import time

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab

BASE = 0.03
EMA_PERIOD = 60
INIT = 50.0

# ================= 数据 =================


def load_doge_5y() -> pd.DataFrame:
    d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
    d2 = pd.read_csv("doge_usdt_1h_3y.csv")
    df = pd.concat([d1, d2], ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def run_baseline(dfx: pd.DataFrame, drop: float, tp: float, ema_period: int) -> dict:
    ab.DROP = drop
    ab.TAKE_PROFIT = tp
    ema = ab.compute_ema4h(dfx, period=ema_period)
    return ab.backtest_baseline(dfx, ema, trailing=False)


def seg_ret_dd(res: dict, idx) -> tuple:
    eq = np.asarray(res["equity"], dtype=float)
    if len(eq) != idx.sum() and len(eq) == len(idx):
        eq = eq[idx]
    eq = eq[eq > 0]
    if len(eq) < 2:
        return 0.0, 0.0
    ret = (eq[-1] / eq[0] - 1.0) * 100
    dd = float(((eq / np.maximum.accumulate(eq)) - 1.0).min()) * 100
    return ret, dd


# ================= 第一关: 参数敏感性 =================


def gate1_sensitivity(df5y: pd.DataFrame) -> pd.DataFrame:
    print("=" * 84)
    print(" 第一关: 参数敏感性(75 组 = 3 EMA周期 x 5 加仓 x 5 止盈, 五年数据)")
    print("=" * 84)
    csv_path = "anti_overfit_sensitivity.csv"
    if not os.path.exists(csv_path):
        rows = []
        for ema_p in (40, 60, 80):
            for drop, tp in itertools.product([0.02, 0.025, 0.03, 0.035, 0.04],
                                              repeat=2):
                r = run_baseline(df5y, drop, tp, ema_p)
                rows.append((ema_p, drop, tp, r["return_pct"], r["max_dd_pct"],
                             r["rounds_opened"]))
                print(f"  ema{ema_p:<2} drop{drop:.3f} tp{tp:.3f} → "
                      f"{r['return_pct']:+7.2f}%  DD {r['max_dd_pct']:.1f}%  "
                      f"轮数 {r['rounds_opened']}")
        res = pd.DataFrame(rows, columns=["ema", "drop", "tp", "ret", "dd", "rounds"])
        res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        res = pd.read_csv(csv_path)
        print("  (复用已完成的 75 组结果)")
    best = res.loc[res["ret"].idxmax()]
    base = res[(res["drop"] == BASE) & (res["tp"] == BASE)
               & (res["ema"] == EMA_PERIOD)].iloc[0]
    nb = res[(res["drop"].isin([0.025, 0.03, 0.035]))
             & (res["tp"].isin([0.025, 0.03, 0.035])) & (res["ema"] == 60)]
    print("-" * 84)
    print(f"  最优组合        : ema{int(best['ema'])} drop{best['drop']:.3f} "
          f"tp{best['tp']:.3f} → {best['ret']:+.2f}%")
    print(f"  基线(3/3/60)    : {base['ret']:+.2f}% | 75 组中排名 "
          f"{int((res['ret'] >= base['ret']).sum())}")
    print(f"  基线 8 邻居(±0.5%, ema60): 收益 {nb['ret'].min():+.2f}% ~ "
          f"{nb['ret'].max():+.2f}% | 中位 {nb['ret'].median():+.2f}%")
    print(f"  全部 75 组      : 中位 {res['ret'].median():+.2f}% | "
          f"正收益占比 {(res['ret'] > 0).mean() * 100:.0f}%")
    verdict = ("高原型 ✓ 稳健(邻居与基线接近)" if nb["ret"].median() >= base["ret"] * 0.5
               else "孤峰型 ✗ 背历史嫌疑(邻居大幅劣于基线)")
    print(f"  → 第一关判定: {verdict}")
    return res


# ================= 第二关: 滚动前推 =================


def gate2_walkforward(df5y: pd.DataFrame) -> None:
    print()
    print("=" * 84)
    print(" 第二关: 滚动前推(训练12月 → 检验3月 → 逐季滚动, DOGE 五年)")
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

    fixed_win, opt_win = [], []
    for ts in starts:
        te = min(ts + test_len, t_end)
        seg = df5y[(df5y["time"] >= ts - train_len) & (df5y["time"] <= te)].reset_index(drop=True)
        train_seg = seg[seg["time"] < ts].reset_index(drop=True)
        idx_test = (seg["time"] >= ts).to_numpy()
        ema60 = ab.compute_ema4h(seg, period=60)

        # 训练期选参(9 组合, 只看训练段收益)
        best_pair, best_r = None, -1e9
        for drop, tp in itertools.product([0.025, 0.03, 0.035], repeat=2):
            r = run_baseline(train_seg, drop, tp, 60)
            if r["return_pct"] > best_r:
                best_r, best_pair = r["return_pct"], (drop, tp)

        # 固定基线在检验期
        rf = run_baseline(seg, BASE, BASE, 60)
        ret_f, dd_f = seg_ret_dd(rf, idx_test)
        # 每窗选参在检验期
        ro = run_baseline(seg, best_pair[0], best_pair[1], 60)
        ret_o, dd_o = seg_ret_dd(ro, idx_test)

        fixed_win.append(ret_f)
        opt_win.append(ret_o)
        print(f"  {ts:%Y-%m-%d} ~ {te:%Y-%m-%d} | 训练选参 drop{best_pair[0]:.3f}/"
              f"tp{best_pair[1]:.3f}(训练{best_r:+.1f}%) | 检验: 固定{ret_f:+6.2f}% "
              f"vs 选参{ret_o:+6.2f}%")

    wf = pd.DataFrame({"window_start": starts, "fixed": fixed_win, "opt": opt_win})
    wf.to_csv("anti_overfit_walkforward.csv", index=False, encoding="utf-8-sig")
    tot_f = (np.prod(np.array(fixed_win) / 100 + 1) - 1) * 100
    tot_o = (np.prod(np.array(opt_win) / 100 + 1) - 1) * 100
    print("-" * 84)
    print(f"  {len(starts)} 个检验窗 | 固定基线复利 {tot_f:+.2f}% "
          f"({(np.array(fixed_win) > 0).mean() * 100:.0f}% 窗口为正)")
    print(f"  每窗重新选参复利 {tot_o:+.2f}% ({'优于' if tot_o > tot_f else '劣于'}固定基线)")
    print("  → 第二关判定: 若逐窗选参显著优于固定参数, 说明参数在适应噪音(过拟合迹象);")
    print("     若两者接近, 说明固定参数足够稳健。")


# ================= 第三关: 多品种 =================


def fetch_bitget_1h(symbol: str, start: str, end: str) -> pd.DataFrame:
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").normalize().timestamp() * 1000) + 86_400_000
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
    df = df[df["ts"] + 3_600_000 <= now_ms].reset_index(drop=True)
    return df


def gate3_multi_asset() -> None:
    print()
    print("=" * 84)
    print(" 第三关: 多品种验证(同一套参数零重调, 2026-01-01 至今)")
    print("=" * 84)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
    rows = []
    for sym in symbols:
        dfx = fetch_bitget_1h(sym, "2026-01-01", "2026-11-12")
        if len(dfx) < 2000:
            print(f"  {sym}: 数据不足({len(dfx)})")
            continue
        r = run_baseline(dfx, BASE, BASE, 60)
        bh = (dfx["close"].iloc[-1] / dfx["close"].iloc[0] - 1) * 100
        rows.append((sym, len(dfx), r["return_pct"], r["max_dd_pct"], bh,
                     r["rounds_opened"], r["rounds_tp"]))
        print(f"  {sym:<10} 策略 {r['return_pct']:+7.2f}% | 最大回撤 {r['max_dd_pct']:6.2f}% "
              f"| 持有 {bh:+7.2f}% | 轮数 {r['rounds_opened']}/{r['rounds_tp']}")
    res = pd.DataFrame(rows, columns=["symbol", "bars", "strategy_ret", "max_dd",
                                      "buy_hold", "rounds", "rounds_tp"])
    res.to_csv("anti_overfit_multiasset.csv", index=False, encoding="utf-8-sig")
    if len(res):
        pos = (res["strategy_ret"] > 0).sum()
        beat = (res["strategy_ret"] > res["buy_hold"]).sum()
        print("-" * 84)
        print(f"  {pos}/{len(res)} 个品种正收益 | {beat}/{len(res)} 个品种跑赢买入持有")
        print(f"  策略均值 {res.strategy_ret.mean():+.2f}% vs 持有均值 {res.buy_hold.mean():+.2f}%")
        print("  → 第三关判定: 多品种同参下若普遍正收益/跑赢持有 = 可迁移规律;")
        print("     若只有 DOGE 好 = 拟合了单一品种的特殊噪音(视频说的环境依赖)。")


def main() -> None:
    df5y = load_doge_5y()
    print(f"五年数据: {len(df5y)} 根 | {df5y['time'].iloc[0]} ~ "
          f"{df5y['time'].iloc[-1]} (UTC)")
    gate1_sensitivity(df5y)
    gate2_walkforward(df5y)
    gate3_multi_asset()
    print()
    print("全部检验完成: anti_overfit_sensitivity.csv / "
          "anti_overfit_walkforward.csv / anti_overfit_multiasset.csv")


if __name__ == "__main__":
    main()
