# -*- coding: utf-8 -*-
"""
grid_trailing.py —— 动态止损版 DCA 多进程参数网格搜索(144 组合)
================================================================
对近 180 天 DOGE/USDT 1h 数据穷举参数:

  1. 趋势过滤 EMA 周期(日线) : [50, 60, 100, 120]
  2. 进场门槛 1h ADX 阈值    : [20, 22, 25]   (ADX 低于阈值 = 震荡, 允许开仓)
  3. 止盈比例               : [1.5%, 2%, 3%, 4%]
  4. 止损回撤线(历史峰值回撤): [10%, 15%, 20%]

固定参数(与自适应回测一致): 加仓间距 3% / 每轮最多 9 次加仓 / 首单与加仓各 5U /
手续费 0.1% 单边 / 1h ATR>=2倍 极端清仓。开仓需 1h 震荡 且 现价 > 日线 EMA(N);
动态止损 = 总资产跌破历史峰值 x (1-回撤线) 时无条件清仓并永久停止。

用 multiprocessing.Pool 并行(Windows spawn 模式, 数据经 initializer 注入)。

输出:
  - 桌面 dca_grid_trailing_results.csv(含 EMA周期/ADX阈值/止盈/止损/总收益/最大回撤)
  - 控制台打印按"最大回撤最小且总收益为正"排序的 Top 5 神仙参数

运行: python grid_trailing.py
"""

from __future__ import annotations

import os
import sys
import time
from itertools import product

import multiprocessing as mp
import numpy as np
import pandas as pd

import regime_detect as rd

# ===================== 搜索空间 =====================
EMA_GRID = [50, 60, 100, 120]        # 日线 EMA 周期(趋势过滤)
ADX_GRID = [20.0, 22.0, 25.0]        # 1h ADX 震荡阈值
TP_GRID = [0.015, 0.02, 0.03, 0.04]  # 止盈比例
TRAIL_GRID = [0.10, 0.15, 0.20]      # 动态止损回撤线

# ---- 固定参数 ----
INITIAL_CASH = 50.0
BUY_USDT = 5.0
MAX_ADDS = 9
DROP = 0.03
FEE_RATE = 0.001
ATR_SPIKE = 2.0

DESKTOP_CSV = os.path.join(os.path.expanduser("~"), "Desktop",
                           "dca_grid_trailing_results.csv")

_G: dict = {}


def _init_worker(close, adx, atr_r, lower, upper, emas) -> None:
    _G["close"] = close
    _G["adx"] = adx
    _G["atr_r"] = atr_r
    _G["lower"] = lower
    _G["upper"] = upper
    _G["emas"] = emas


def run_combo(task: dict) -> dict:
    """回测单个参数组合(1h 路由 + 动态峰值止损)。"""
    try:
        close = _G["close"]
        adx = _G["adx"]
        atr_r = _G["atr_r"]
        lower = _G["lower"]
        upper = _G["upper"]
        ema = _G["emas"][task["ema"]]
        thr, tp, trail = task["adx_thr"], task["tp"], task["trail"]
        n = len(close)

        # ---- 按 ADX 阈值生成 1h 状态码: 0=震荡 3=极端风险 4=其他 ----
        ok = ~(np.isnan(adx) | np.isnan(atr_r) | np.isnan(lower) | np.isnan(upper))
        reg = np.full(n, 4.0)
        reg[ok & (atr_r >= ATR_SPIKE)] = 3.0
        reg[ok & (atr_r < ATR_SPIKE) & (adx < thr)
            & (close > lower) & (close < upper)] = 0.0

        cash, qty = INITIAL_CASH, 0.0
        last_buy, buys, cost = 0.0, 0, 0.0
        stopped = False
        peak = INITIAL_CASH
        n_buys = n_sells = rounds = 0
        equity = np.empty(n)

        for i in range(n):
            p = close[i]
            if stopped:
                equity[i] = cash
                continue
            if qty <= 0:                          # 空仓: 1h震荡 且 高于日线EMA(N) 才开仓
                e = ema[i]
                if reg[i] == 0 and not np.isnan(e) and p > e:
                    fee = BUY_USDT * FEE_RATE
                    qty += (BUY_USDT - fee) / p
                    cash -= BUY_USDT
                    cost, buys, last_buy = BUY_USDT, 1, p
                    n_buys += 1
                    rounds += 1
            else:                                 # 持仓中
                if reg[i] == 3:                   # 1) 极端风险清仓
                    cash += qty * p * (1 - FEE_RATE)
                    qty, cost, buys = 0.0, 0.0, 0
                    n_sells += 1
                else:
                    if (reg[i] == 0 and buys - 1 < MAX_ADDS   # 2) 仅震荡加仓
                            and p <= last_buy * (1 - DROP)):
                        fee = BUY_USDT * FEE_RATE
                        qty += (BUY_USDT - fee) / p
                        cash -= BUY_USDT
                        cost += BUY_USDT
                        buys += 1
                        last_buy = p
                        n_buys += 1
                    avg = cost / qty
                    if p >= avg * (1 + tp):       # 3) 止盈
                        cash += qty * p * (1 - FEE_RATE)
                        qty, cost, buys = 0.0, 0.0, 0
                        n_sells += 1

            total = cash + qty * p
            if total <= peak * (1.0 - trail):     # 4) 动态峰值止损
                if qty > 0:
                    cash += qty * p * (1 - FEE_RATE)
                    qty, cost, buys = 0.0, 0.0, 0
                    n_sells += 1
                stopped = True
                total = cash
            else:
                peak = max(peak, total)
            equity[i] = total

        pk = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - pk) / pk)) if pk.max() > 0 else 0.0
        final = float(equity[-1])
        return {
            "ema_period": task["ema"],
            "adx_thr": task["adx_thr"],
            "tp_pct": round(tp * 100, 1),
            "trail_pct": round(trail * 100, 1),
            "final_equity": round(final, 4),
            "return_pct": round((final / INITIAL_CASH - 1.0) * 100, 2),
            "max_dd_pct": round(max_dd * 100, 2),
            "rounds": rounds,
            "n_buys": n_buys,
            "n_sells": n_sells,
            "end_state": "止损停止" if stopped else ("持仓中" if qty > 0 else "空仓"),
        }
    except Exception as exc:
        return {**task, "error": repr(exc)}


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print(" 动态止损版 DCA 多进程网格搜索(144 组合, 近 180 天)")
    print("=" * 72)

    df = rd.fetch_history()
    router = rd.compute_router(df)
    close = router["close"].to_numpy(dtype=float)
    adx = router["adx_1h"].to_numpy(dtype=float)
    atr_r = router["atr_ratio_1h"].to_numpy(dtype=float)
    lower = router["boll_lower"].to_numpy(dtype=float)
    upper = router["boll_upper"].to_numpy(dtype=float)
    emas = {p: rd.compute_daily_ema60(df, p).to_numpy(dtype=float) for p in EMA_GRID}
    print(f"[指标] 日线 EMA{EMA_GRID} 与 1h ADX/ATR/BOLL 已预计算")

    tasks = [{"ema": p, "adx_thr": a, "tp": t, "trail": tr}
             for p, a, t, tr in product(EMA_GRID, ADX_GRID, TP_GRID, TRAIL_GRID)]
    n_workers = min(os.cpu_count() or 4, 8)
    print(f"[并行] {len(tasks)} 组参数, {n_workers} 个子进程并行回测 ...")

    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(close, adx, atr_r, lower, upper, emas)) as pool:
        for i, res in enumerate(pool.imap_unordered(run_combo, tasks, chunksize=1), 1):
            results.append(res)
            if i % 24 == 0 or i == len(tasks):
                print(f"[进度] {i}/{len(tasks)}")

    for r in results:
        if "error" in r:
            print(f"[警告] 组合失败: {r}")
    rows = [r for r in results if "error" not in r]
    if not rows:
        sys.exit("[错误] 所有组合都失败了。")

    table = pd.DataFrame(rows)
    table["profit"] = table["return_pct"] > 0
    # 排序: 盈利的组合在前, 且按最大回撤从小到大(数值从大到小)
    by_dd = table.sort_values(["profit", "max_dd_pct"],
                              ascending=[False, False]).reset_index(drop=True)

    out = by_dd.rename(columns={
        "ema_period": "EMA周期", "adx_thr": "ADX阈值", "tp_pct": "止盈%",
        "trail_pct": "止损回撤%", "return_pct": "总收益%", "max_dd_pct": "最大回撤%",
        "rounds": "开出轮数", "end_state": "期末状态", "profit": "是否盈利",
    })
    cols = ["EMA周期", "ADX阈值", "止盈%", "止损回撤%", "总收益%", "最大回撤%",
            "开出轮数", "期末状态", "是否盈利"]
    out[cols].to_csv(DESKTOP_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[输出] 已保存桌面 CSV: {DESKTOP_CSV}")

    top5 = by_dd[by_dd["profit"]].head(5)
    n_profit = int(table["profit"].sum())
    print()
    print("-" * 72)
    print(f" 144 组合中总收益为正的有 {n_profit} 组")
    print(" 按『最大回撤最小且总收益为正』排序的 Top 5 神仙参数:")
    print("-" * 72)
    for i, r in top5.iterrows():
        print(f"  Top{i + 1}: EMA{r['ema_period']:<4} | ADX阈值 {r['adx_thr']:<4.0f}"
              f"| 止盈 {r['tp_pct']:<4.1f}% | 止损回撤 {r['trail_pct']:<4.0f}%"
              f"| 收益 {r['return_pct']:>6.2f}% | 回撤 {r['max_dd_pct']:>6.2f}%"
              f"| 轮数 {r['rounds']} | {r['end_state']}")
    print("-" * 72)
    print(f" 最差 3 组(参考):")
    for i, r in table.sort_values("max_dd_pct").head(3).iterrows():
        print(f"   EMA{r['ema_period']} | ADX {r['adx_thr']} | 止盈 {r['tp_pct']}%"
              f"| 止损 {r['trail_pct']}% | 收益 {r['return_pct']:.2f}%"
              f"| 回撤 {r['max_dd_pct']:.2f}%")
    print(f" 耗时 {time.time() - t0:.1f} 秒(含下载)")


if __name__ == "__main__":
    main()
