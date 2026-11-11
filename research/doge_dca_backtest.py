# -*- coding: utf-8 -*-
"""
DOGE/USDT 现货 DCA 网格回测 (币安 1 小时 K 线, 近 180 天)
==========================================================

策略规则:
  1. 空仓时: 市价买入 5 USDT 的 DOGE(初始买入, 开启新一轮)
  2. 持仓中: 现价相对"上次买入价"下跌 >= 3% 时, 加仓 5 USDT
  3. 每轮最多加仓 9 次(每轮最多买入 10 次, 合计最多投入 50 USDT);
     加仓次数用完后若价格继续下跌, 不再买入, 持有等待反弹("死扛")
  4. 止盈: 现价 >= 本轮平均成本价 * (1 + 3%) 时, 全部市价卖出;
     本轮结束, 重置状态, 等待下一轮初始买入
  5. 手续费: 每次买/卖按成交额的 0.1% 扣除

回测假设:
  - 信号在 K 线收盘时判定, 以该根 K 线收盘价作为市价成交(无未来函数)
  - 平均成本价 = 本轮实际支出 USDT / 当前持仓数量(支出已含手续费)
  - 回测结束若仍持仓, 总资产按最后一根 K 线收盘价折算市值(不强制平仓)

依赖安装:
  pip install ccxt pandas numpy matplotlib

运行:
  python doge_dca_backtest.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import matplotlib
    matplotlib.use("Agg")          # 无图形界面环境下也能保存图片
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ===================== 可调参数 =====================
SYMBOL = "DOGE/USDT"
TIMEFRAME = "1h"
DAYS = 180                  # 回看天数
INITIAL_BUY_USDT = 5.0      # 初始买入金额(USDT)
ADD_BUY_USDT = 5.0          # 每次加仓金额(USDT)
DROP_TRIGGER = 0.03         # 相对上次买入价下跌 3% 触发加仓
MAX_ADDS = 9                # 每轮最多加仓次数(不含初始买入)
TAKE_PROFIT = 0.03          # 高于平均成本 3% 触发止盈
COOLDOWN_BARS = 0           # 止盈卖出后等待几根 K 线再开下一轮(0 = 下一根立即开始)
INITIAL_CASH = 50.0         # 初始资金(每轮最多投入 50 USDT)
FEE_RATE = 0.001            # 手续费 0.1% (单边)
CSV_FILE = "doge_usdt_1h.csv"       # K 线缓存
TRADE_FILE = "doge_dca_trades.csv"  # 成交明细
PNG_FILE = "doge_dca_equity.png"    # 资金曲线图
TZ_DISPLAY = "Asia/Shanghai"        # 输出用本地时区

# ===================== 数据下载 =====================


def fetch_history() -> pd.DataFrame:
    """从币安下载最近 DAYS 天的 1 小时 K 线(每页 1000 根, 自动翻页)。"""
    if ccxt is None:
        sys.exit("未安装 ccxt, 请先执行: pip install ccxt")

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
        if new_since <= since:          # 无进展, 防止死循环
            break
        since = new_since
        if len(batch) < 1000:           # 已拿到最新一根
            break
        time.sleep(max(exchange.rateLimit or 100, 200) / 1000.0)

    if not rows:
        sys.exit("[下载] 未获取到数据, 请检查网络/代理后重试。")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"[下载] 成功: {len(df)} 根 K 线 | "
          f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC) | 已保存 {CSV_FILE}")
    return df

# ===================== 回测 =====================


def run_backtest(df: pd.DataFrame) -> dict:
    cash = INITIAL_CASH          # 可用资金(USDT)
    qty = 0.0                    # 当前持仓(DOGE 数量)
    last_buy_price = None        # 本轮上一次买入价
    buys_in_cycle = 0            # 本轮已买入次数(含初始买入)
    cycle_cost = 0.0             # 本轮累计支出(USDT, 含手续费)
    cooldown = 0                 # 止盈卖出后的等待 K 线数
    trades = []                  # 成交记录
    equity = []                  # 逐 K 线总资产(现金 + 持仓市值)

    for row in df.itertuples(index=False):
        price = float(row.close)                 # 以收盘价作为"市价"

        if cooldown > 0:                         # 冷却期: 不交易
            cooldown -= 1
        elif qty == 0.0:                         # 空仓 → 初始买入, 开启新一轮
            spend = INITIAL_BUY_USDT
            if cash < spend:                     # 资金不足(正常情况下不会发生)
                break
            fee = spend * FEE_RATE
            bought = (spend - fee) / price
            qty += bought
            cash -= spend
            cycle_cost += spend
            buys_in_cycle = 1
            last_buy_price = price
            trades.append([row.time, "BUY", "初始买入", price, bought, cash])
        else:                                    # 持仓中
            # 1) 加仓: 现价比上次买入价下跌 >= 3%, 且加仓次数未用完
            if (buys_in_cycle - 1 < MAX_ADDS
                    and price <= last_buy_price * (1.0 - DROP_TRIGGER)):
                fee = ADD_BUY_USDT * FEE_RATE
                bought = (ADD_BUY_USDT - fee) / price
                qty += bought
                cash -= ADD_BUY_USDT
                cycle_cost += ADD_BUY_USDT
                buys_in_cycle += 1
                last_buy_price = price           # 加仓后更新"上次买入价"
                trades.append([row.time, "BUY", "加仓", price, bought, cash])

            # 2) 止盈: 现价 >= 平均成本 * (1 + 3%)
            avg_cost = cycle_cost / qty          # 平均成本(含手续费)
            if price >= avg_cost * (1.0 + TAKE_PROFIT):
                gross = qty * price
                fee = gross * FEE_RATE
                proceeds = gross - fee
                cash += proceeds
                trades.append([row.time, "SELL", "止盈平仓", price, qty, cash])
                qty = 0.0                        # 重置本轮状态
                cycle_cost = 0.0
                buys_in_cycle = 0
                last_buy_price = None
                cooldown = COOLDOWN_BARS

        equity.append(cash + qty * price)        # 每根 K 线结束后的总资产

    # ---- 统计 ----
    final_price = float(df["close"].iloc[-1])
    final_equity = cash + qty * final_price      # 期末总资产(按市值折算)

    equity_s = pd.Series(equity, index=df.index, dtype=float)
    max_drawdown = float((equity_s / equity_s.cummax() - 1.0).min())

    trades_df = pd.DataFrame(
        trades, columns=["time", "side", "type", "price", "qty", "cash_after"])
    trades_df["time"] = trades_df["time"].dt.tz_convert(TZ_DISPLAY)
    trades_df.to_csv(TRADE_FILE, index=False, encoding="utf-8-sig")

    return {
        "final_equity": final_equity,
        "total_return": final_equity / INITIAL_CASH - 1.0,
        "max_drawdown": max_drawdown,
        "n_buys": int((trades_df["side"] == "BUY").sum()),
        "n_sells": int((trades_df["side"] == "SELL").sum()),
        "qty": qty,                              # 期末剩余持仓
        "avg_cost": cycle_cost / qty if qty > 0 else 0.0,
        "final_price": final_price,
        "equity": equity,
        "trades": trades_df,
    }

# ===================== 绘图 =====================


def plot_equity(df: pd.DataFrame, equity: list) -> None:
    """画出逐 K 线资金曲线并保存为 PNG。"""
    if plt is None:
        print("[图表] 未安装 matplotlib, 跳过绘图。安装命令: pip install matplotlib")
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df["time"], equity, linewidth=1.2, color="#1f77b4", label="策略总资产")
    ax.axhline(INITIAL_CASH, color="gray", linestyle="--", linewidth=0.9,
               label=f"初始资金 {INITIAL_CASH:.0f} USDT")
    ax.set_title(f"{SYMBOL} 现货 DCA 网格策略 资金曲线"
                 f"(近 {DAYS} 天, {TIMEFRAME} K 线)")
    ax.set_xlabel("时间")
    ax.set_ylabel("总资产 (USDT)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=150)
    plt.close(fig)
    print(f"[图表] 已保存 {PNG_FILE}")

# ===================== 主流程 =====================


def main() -> None:
    print("=" * 62)
    print(" DOGE/USDT 现货 DCA 网格回测")
    print("=" * 62)

    df = fetch_history()
    result = run_backtest(df)
    plot_equity(df, result["equity"])

    print()
    print("-" * 62)
    print(f"{'回测结果':^60}")
    print("-" * 62)
    print(f"  数据范围       : {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")
    print(f"  K 线数量       : {len(df)} 根")
    print(f"  总交易次数     : {result['n_buys'] + result['n_sells']} 次"
          f"(买入 {result['n_buys']} / 卖出 {result['n_sells']})")
    print(f"  完成止盈轮数   : {result['n_sells']} 轮")
    print(f"  最终总资产     : {result['final_equity']:.4f} USDT")
    print(f"  总收益率       : {result['total_return'] * 100:.2f} %")
    print(f"  最大回撤       : {result['max_drawdown'] * 100:.2f} %")
    if result["qty"] > 0:                        # 期末仍持仓("死扛"中)
        print(f"  期末状态       : 持仓 {result['qty']:.2f} DOGE"
              f"(均价 {result['avg_cost']:.6f}, 现价 {result['final_price']:.6f})")
    else:
        print(f"  期末状态       : 空仓")
    print("-" * 62)

    trades = result["trades"]
    if not trades.empty:
        print(f"\n成交明细(前 10 笔 + 后 5 笔, 完整列表见 {TRADE_FILE}):")
        show = pd.concat([trades.head(10), trades.tail(5)])
        print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
