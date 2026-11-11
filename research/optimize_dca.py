# -*- coding: utf-8 -*-
"""
optimize_dca.py —— DCA 网格策略参数优化器(多进程网格搜索)
==========================================================
在近 180 天 DOGE/USDT 1h 数据上穷举 64 组参数:

  EMA 周期(4h 趋势过滤) : [30, 50, 60, 80]
  止盈比例              : [1.5%, 2%, 3%, 5%]
  加仓间距              : [2%, 3%, 4%, 5%]

使用 multiprocessing.Pool 并行回测。Windows 下为 spawn 模式:
  - K 线数据和预计算的 4 组 EMA 通过 Pool 的 initializer 一次性注入每个子进程,
    避免每个任务重复序列化传输;
  - 每个子进程只跑属于自己的参数组合, 互不干扰。

固定参数(与 v2 回测一致):
  - 首单 5 USDT, 每次加仓 5 USDT, 每轮最多加仓 9 次(共 50U)
  - 手续费 0.1% 单边
  - 子弹打光 且 总资产 < 初始资金 85%(42.5U) 时无条件清仓止损并停止
  - EMA 过滤只限制"开第一单"; 已持仓后的加仓/止盈/止损不受过滤影响
  - EMA 用"上一根已收盘 4h K 线"计算(无未来函数)

输出:
  dca_grid_results.csv  —— 按最终总资产降序的全部 64 组
  dca_grid_results.xlsx —— 工作表1: 按最终总资产排序; 工作表2: 按最大回撤排序

依赖: pip install ccxt pandas numpy openpyxl
运行: python optimize_dca.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import product

import multiprocessing as mp
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    sys.exit("未安装 ccxt, 请先执行: pip install ccxt")

# ===================== 参数空间 =====================
SYMBOL = "DOGE/USDT"
TIMEFRAME = "1h"
DAYS = 180
EMA_GRID = [30, 50, 60, 80]                    # 4h EMA 周期
TAKE_PROFIT_GRID = [0.015, 0.02, 0.03, 0.05]   # 止盈比例
DROP_GRID = [0.02, 0.03, 0.04, 0.05]           # 加仓间距

# ---- 固定参数(与 v2 一致, 不在搜索范围内) ----
INITIAL_CASH = 50.0
BUY_USDT = 5.0          # 首单/每次加仓金额
MAX_ADDS = 9            # 每轮最多加仓 9 次(共 10 次买入 = 50U)
STOP_LOSS_PCT = 0.15    # 总资产跌破 42.5U 止损
FEE_RATE = 0.001        # 0.1% 单边

CSV_OUT = "dca_grid_results.csv"
XLSX_OUT = "dca_grid_results.xlsx"
CACHE_CSV = "doge_usdt_1h.csv"

# ===================== 数据下载 =====================


def fetch_history() -> pd.DataFrame:
    """下载 180 天 1h K 线; 网络失败时回退到本地缓存 CSV。"""
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)
        print(f"[下载] 币安 {SYMBOL} {TIMEFRAME}, 最近 {DAYS} 天 ...")
        rows, since = [], start_ms
        now_ms = exchange.milliseconds()
        while since < now_ms:
            batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            new_since = batch[-1][0] + 1
            if new_since <= since:
                break
            since = new_since
            if len(batch) < 1000:
                break
            time.sleep(max(exchange.rateLimit or 100, 200) / 1000.0)
        if not rows:
            raise RuntimeError("交易所返回空数据")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        # 剔除尚未收盘的最后一根 K 线(close 仍在变动)
        df = df[df["ts"] + 3_600_000 <= now_ms].reset_index(drop=True)
        df.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
        print(f"[下载] 成功: {len(df)} 根 K 线 | "
              f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")
        return df
    except Exception as exc:
        print(f"[下载] 在线获取失败: {exc}")
        if os.path.exists(CACHE_CSV):
            print(f"[下载] 使用本地缓存 {CACHE_CSV}")
            df = pd.read_csv(CACHE_CSV)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df
        sys.exit("[下载] 无可用数据, 请检查网络后重试。")

# ===================== EMA(与 v2 同口径) =====================


def compute_ema4h(df: pd.DataFrame, period: int) -> np.ndarray:
    """1h 收盘价聚合为 4h, 算 EMA(period), 回填到每根 1h K 线。
    前 period 根 4h K 线为预热期(NaN), 且只用上一根已收盘 4h K 线(shift(1))。"""
    bucket = df["time"].dt.floor("4h")
    close4h = df.assign(bucket=bucket).groupby("bucket")["close"].last()
    ema4h = close4h.ewm(span=period, adjust=False).mean()
    ema4h.iloc[:period] = np.nan
    ema4h = ema4h.shift(1)
    return bucket.map(ema4h).to_numpy()

# ===================== 单组合回测(在子进程中运行) =====================

_G: dict = {}          # 子进程全局: {"df": DataFrame, "emas": {period: ndarray}}


def _init_worker(df: pd.DataFrame, emas: dict) -> None:
    """Pool initializer: 把数据和 EMA 缓存一次性注入子进程。"""
    _G["df"] = df
    _G["emas"] = emas


def backtest_one(task: dict) -> dict:
    """回测单个参数组合(与 v2 策略逻辑完全一致)。"""
    try:
        df = _G["df"]
        ema = _G["emas"][task["ema"]]
        prices = df["close"].to_numpy(dtype=float)
        n = len(prices)

        stop_line = INITIAL_CASH * (1.0 - STOP_LOSS_PCT)
        max_buys = MAX_ADDS + 1
        tp = task["tp"]
        drop = task["drop"]

        cash, qty = INITIAL_CASH, 0.0
        last_buy_price = 0.0
        buys_in_cycle = 0
        cycle_cost = 0.0
        stopped = False
        n_buys = n_sells = 0
        equity = np.empty(n)

        for i in range(n):
            price = prices[i]

            if stopped:                       # 止损后: 资产不再变化
                equity[i] = cash
                continue

            if qty <= 0:                      # 空仓 → 趋势过滤后开第一单
                e = ema[i]
                if not np.isnan(e) and price > e:
                    fee = BUY_USDT * FEE_RATE
                    bought = (BUY_USDT - fee) / price
                    qty += bought
                    cash -= BUY_USDT
                    cycle_cost += BUY_USDT
                    buys_in_cycle = 1
                    last_buy_price = price
                    n_buys += 1
            else:                             # 持仓中
                # 1) 加仓
                if (buys_in_cycle - 1 < MAX_ADDS
                        and price <= last_buy_price * (1.0 - drop)):
                    fee = BUY_USDT * FEE_RATE
                    bought = (BUY_USDT - fee) / price
                    qty += bought
                    cash -= BUY_USDT
                    cycle_cost += BUY_USDT
                    buys_in_cycle += 1
                    last_buy_price = price
                    n_buys += 1
                # 2) 止盈
                avg_cost = cycle_cost / qty
                if price >= avg_cost * (1.0 + tp):
                    cash += qty * price * (1.0 - FEE_RATE)
                    qty = 0.0
                    cycle_cost = 0.0
                    buys_in_cycle = 0
                    last_buy_price = 0.0
                    n_sells += 1
                # 3) 止损
                elif (qty > 0 and buys_in_cycle >= max_buys
                        and cash + qty * price < stop_line):
                    cash += qty * price * (1.0 - FEE_RATE)
                    qty = 0.0
                    cycle_cost = 0.0
                    buys_in_cycle = 0
                    n_sells += 1
                    stopped = True

            equity[i] = cash + qty * price

        final_equity = float(equity[-1])
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / peak)) if peak.max() > 0 else 0.0
        end_state = "止损停止" if stopped else ("持仓中" if qty > 0 else "空仓")

        return {
            "ema_period": task["ema"],
            "take_profit_pct": round(tp * 100, 1),
            "drop_pct": round(drop * 100, 1),
            "final_equity": round(final_equity, 4),
            "return_pct": round((final_equity / INITIAL_CASH - 1.0) * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_buys": n_buys,
            "n_sells": n_sells,
            "stop_loss": bool(stopped),
            "end_state": end_state,
        }
    except Exception as exc:                  # 单组合失败不影响整个网格
        return {**task, "error": repr(exc)}

# ===================== 主流程 =====================


def main() -> None:
    t0 = time.time()
    print("=" * 64)
    print(" DCA 网格策略 多进程参数网格搜索 (64 组合)")
    print("=" * 64)

    df = fetch_history()
    emas = {p: compute_ema4h(df, p) for p in EMA_GRID}
    print(f"[指标] 4h EMA{EMA_GRID} 已预计算")

    tasks = [{"ema": p, "tp": tp, "drop": d}
             for p, tp, d in product(EMA_GRID, TAKE_PROFIT_GRID, DROP_GRID)]
    n_workers = min(os.cpu_count() or 4, 8)
    print(f"[并行] {len(tasks)} 组参数, {n_workers} 个子进程并行回测 ...")

    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(df, emas)) as pool:
        for i, res in enumerate(pool.imap_unordered(backtest_one, tasks, chunksize=1), 1):
            results.append(res)
            if i % 16 == 0 or i == len(tasks):
                print(f"[进度] {i}/{len(tasks)}")

    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"[警告] 组合 {r.get('ema')}/{r.get('tp')}/{r.get('drop')} 失败: {r['error']}")
    rows = [r for r in results if "error" not in r]
    if not rows:
        sys.exit("[错误] 所有组合都失败了, 请检查环境。")

    table = pd.DataFrame(rows)
    by_equity = table.sort_values("final_equity", ascending=False).reset_index(drop=True)
    by_dd = table.sort_values("max_drawdown_pct", ascending=False).reset_index(drop=True)

    by_equity.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[输出] 已保存 {CSV_OUT}")
    try:
        with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
            by_equity.to_excel(writer, sheet_name="按最终总资产排序", index=False)
            by_dd.to_excel(writer, sheet_name="按最大回撤排序", index=False)
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
        print(f"[输出] 已保存 {XLSX_OUT}(两个工作表)")
    except Exception as exc:
        print(f"[输出] Excel 保存失败({exc}), 仅输出 CSV")

    # ---- 摘要 ----
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    bh = (INITIAL_CASH * (1 - FEE_RATE) / first_close) * last_close * (1 - FEE_RATE)
    print()
    print("-" * 64)
    print(f" 数据范围: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC), {len(df)} 根 K 线")
    print(f" 买入持有对照: 最终 {bh:.4f} USDT ({(bh / INITIAL_CASH - 1) * 100:.2f} %)")
    print("-" * 64)
    print(" 按最终总资产 Top 5:")
    print(by_equity.head(5).to_string(index=False))
    print("-" * 64)
    print(" 按最大回撤最小(最稳) Top 5:")
    print(by_dd.head(5).to_string(index=False))
    print("-" * 64)
    print(f" 耗时 {time.time() - t0:.1f} 秒(含下载)")


if __name__ == "__main__":
    main()
