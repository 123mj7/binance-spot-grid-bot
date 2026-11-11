# -*- coding: utf-8 -*-
"""
DOGE/USDT 现货 DCA 网格回测 v2.1 —— 大趋势过滤 + 严格止损(审计加固版)
======================================================================

在 v1 策略基础上新增两个风控条件:
  A. 大趋势过滤: 只有当现价 > 4小时 EMA60 时, 才允许开启新一轮的第一单
     (5 USDT 初始买入); 现价在 EMA60 下方时机器人休息, 不开新仓。
     已经持仓后的加仓与止盈不受过滤影响。
     EMA60 用"上一根已收盘"的 4h K 线计算(再滞后一根 4h K 线),
     完全无未来函数。
  B. 严格止损: 当本轮 50 USDT 子弹全部打光(初始 + 9 次加仓 = 10 次买入),
     且当前总资产(现金 + DOGE 持仓市值)跌破初始资金的 15% (即 < 42.5 USDT)
     时, 无条件全部市价清仓止损:
       - 清空持仓, 并把本轮所有状态变量重置为初始值;
       - 机器人按 RESUME_AFTER_STOP 决定: False = 永久停止(原始需求),
         True = 重置后继续等待下一次开仓条件。
     止损判定每根 K 线独立进行, 用的是总资产口径, 不是只看 USDT 余额。

本脚本同时回测两个版本并对比:
  - 基线策略(无过滤、无止损)  = 上一版 v1 逻辑
  - 新策略(趋势过滤 + 严格止损)
输出:
  - 对比表格 + 审计信息(总资产峰值/谷值、子弹打光期间最低总资产)
  - 两条资金曲线叠加图(doge_dca_compare.png)
  - 逐 K 线状态审计表与成交明细(含成交后持仓/资产列)写入 CSV

依赖安装: pip install ccxt pandas numpy matplotlib
运行: python doge_dca_v2.py
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
TIMEFRAME_MS = 3_600_000    # 1h K 线的毫秒数(用于剔除未收盘的最后一根)
DAYS = 180                  # 回看天数
INITIAL_BUY_USDT = 5.0      # 初始买入金额(USDT)
ADD_BUY_USDT = 5.0          # 每次加仓金额(USDT)
DROP_TRIGGER = 0.03         # 相对上次买入价下跌 3% 触发加仓
MAX_ADDS = 9                # 每轮最多加仓次数(不含初始买入)
TAKE_PROFIT = 0.03          # 高于平均成本 3% 触发止盈
COOLDOWN_BARS = 0           # 止盈卖出后等待几根 K 线再开下一轮(0 = 下一根立即开始)
INITIAL_CASH = 50.0         # 初始资金(每轮最多投入 50 USDT)
FEE_RATE = 0.001            # 手续费 0.1% (单边)

EMA4_TIMEFRAME = "4h"
EMA4_PERIOD = 60            # 4 小时 EMA 周期(60 根 4h K 线 ≈ 10 天)
STOP_LOSS_PCT = 0.15        # 总资产跌破初始资金 15% (即 42.5 USDT) 止损
RESUME_AFTER_STOP = False   # False = 止损后永久停止(原始需求); True = 重置后继续等开仓

CSV_FILE = "doge_usdt_1h.csv"          # K 线缓存
TRADE_BASE_FILE = "doge_dca_trades.csv"       # 基线策略成交明细
TRADE_NEW_FILE = "doge_dca_v2_trades.csv"     # 新策略成交明细
STATE_NEW_FILE = "doge_dca_v2_state.csv"      # 新策略逐 K 线状态审计表
PNG_FILE = "doge_dca_compare.png"             # 对比资金曲线图
TZ_DISPLAY = "Asia/Shanghai"                  # 输出用本地时区

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
    # 剔除尚未收盘的最后一根 K 线(其 close 仍在实时变动), 回测只用已确认的数据
    df = df[df["ts"] + TIMEFRAME_MS <= now_ms].reset_index(drop=True)
    if df.empty:
        sys.exit("[下载] 剔除未收盘 K 线后没有数据, 请稍后重试。")

    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"[下载] 成功: {len(df)} 根 K 线 | "
          f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC) | 已保存 {CSV_FILE}")
    return df

# ===================== 4 小时 EMA60 =====================


def compute_ema4h(df: pd.DataFrame, period: int = EMA4_PERIOD) -> np.ndarray:
    """把 1h 收盘价聚合成 4h 收盘价, 计算 EMA(period) 后回填到每根 1h K 线。

    关键防未来函数处理:
      每根 1h K 线只使用"上一根已收盘 4h K 线"算出的 EMA 值(shift(1)),
      即该 4h K 线的收盘价对当前 1h K 线而言已经完全确认。
    """
    bucket = df["time"].dt.floor("4h")          # 该 1h K 线所属的 4h K 线(开盘时间)
    close4h = df.assign(bucket=bucket).groupby("bucket")["close"].last()
    ema4h = close4h.ewm(span=period, adjust=False).mean()
    ema4h.iloc[:period] = np.nan                # 预热期: 数据不足 period 根时强制为 NaN,
                                                # 保证过滤用的是真正的 60 周期 EMA
    ema4h = ema4h.shift(1)                      # 只使用上一根已收盘 4h K 线的 EMA
    return bucket.map(ema4h).to_numpy()         # 与 df 等长的数组, 预热期为 NaN

# ===================== 回测引擎 =====================


def run_backtest(df: pd.DataFrame,
                 ema4h: np.ndarray | None = None,
                 use_trend_filter: bool = False,
                 use_stop_loss: bool = False) -> dict:
    max_buys = MAX_ADDS + 1                    # 每轮最多买入 10 次 = 50 USDT
    stop_loss_equity = INITIAL_CASH * (1.0 - STOP_LOSS_PCT)   # 42.5 USDT

    cash = INITIAL_CASH          # 可用资金(USDT)
    qty = 0.0                    # 当前持仓(DOGE 数量)
    last_buy_price = None        # 本轮上一次买入价
    buys_in_cycle = 0            # 本轮已买入次数(含初始买入)
    cycle_cost = 0.0             # 本轮累计支出(USDT, 含手续费)
    cooldown = 0                 # 止盈卖出后的等待 K 线数
    stopped = False              # 止损触发后机器人永久停止
    stop_loss_time = None        # 止损触发时间
    trades = []                  # 成交记录
    equity = []                  # 逐 K 线总资产(现金 + 持仓市值)
    state = []                   # 逐 K 线内部状态(审计用)
    min_equity_exhausted = float("inf")   # 子弹打光且仍持仓期间的最低总资产

    ema_vals = ema4h if ema4h is not None else [None] * len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        price = float(row.close)                 # 以收盘价作为"市价"

        if stopped:                              # 已止损: 资产不再变化
            equity.append(cash)
            state.append([row.time, cash, 0.0, 0, cash, True])
            continue

        if cooldown > 0:                         # 冷却期: 不交易
            cooldown -= 1
        elif qty == 0.0:                         # 空仓 → 判断是否开启新一轮
            # --- 大趋势过滤: 仅当现价 > 4h EMA60 时才允许开第一单 ---
            ema_val = ema_vals[i]
            above_ema = (ema_val is not None and not np.isnan(ema_val)
                         and price > float(ema_val))
            if use_trend_filter and not above_ema:
                pass                             # 在 EMA60 下方: 机器人休息
            else:
                spend = INITIAL_BUY_USDT
                if cash < spend:                 # 资金不足(正常情况下不会发生)
                    break
                fee = spend * FEE_RATE
                bought = (spend - fee) / price
                qty += bought
                cash -= spend
                cycle_cost += spend
                buys_in_cycle = 1
                last_buy_price = price
                trades.append([row.time, "BUY", "初始买入", price, bought,
                               qty, cash, cash + qty * price])
        else:                                    # 持仓中
            # --- 1) 加仓: 现价比上次买入价下跌 >= 3%, 且加仓次数未用完 ---
            if (buys_in_cycle - 1 < MAX_ADDS
                    and price <= last_buy_price * (1.0 - DROP_TRIGGER)):
                fee = ADD_BUY_USDT * FEE_RATE
                bought = (ADD_BUY_USDT - fee) / price
                qty += bought
                cash -= ADD_BUY_USDT
                cycle_cost += ADD_BUY_USDT
                buys_in_cycle += 1
                last_buy_price = price           # 加仓后更新"上次买入价"
                trades.append([row.time, "BUY", "加仓", price, bought,
                               qty, cash, cash + qty * price])

            # --- 2) 止盈: 现价 >= 平均成本 * (1 + 3%) ---
            avg_cost = cycle_cost / qty          # 平均成本(含手续费)
            if price >= avg_cost * (1.0 + TAKE_PROFIT):
                gross = qty * price
                fee = gross * FEE_RATE
                cash += gross - fee
                trades.append([row.time, "SELL", "止盈平仓", price, qty,
                               0.0, cash, cash])
                qty = 0.0                        # 重置本轮状态
                cycle_cost = 0.0
                buys_in_cycle = 0
                last_buy_price = None
                cooldown = COOLDOWN_BARS

        # --- 3) 严格止损: 每根 K 线独立判定, 口径 = 总资产(现金 + 持仓市值) ---
        total_assets = cash + qty * price
        if qty > 0 and buys_in_cycle >= max_buys:
            min_equity_exhausted = min(min_equity_exhausted, total_assets)
            if use_stop_loss and total_assets < stop_loss_equity:
                gross = qty * price
                fee = gross * FEE_RATE
                cash += gross - fee
                trades.append([row.time, "SELL", "止损清仓", price, qty,
                               0.0, cash, cash])
                stop_loss_time = row.time
                # 清空持仓, 并把本轮所有状态变量重置为初始值
                qty = 0.0
                cycle_cost = 0.0
                buys_in_cycle = 0
                last_buy_price = None
                cooldown = 0
                stopped = not RESUME_AFTER_STOP
                total_assets = cash

        equity.append(total_assets)              # 每根 K 线结束后的总资产
        state.append([row.time, cash, qty, buys_in_cycle, total_assets, stopped])

    # ---- 统计 ----
    final_price = float(df["close"].iloc[-1])
    final_equity = cash + qty * final_price      # 期末总资产(按市值折算)

    equity_s = pd.Series(equity, index=df.index, dtype=float)
    max_drawdown = float((equity_s / equity_s.cummax() - 1.0).min())

    trades_df = pd.DataFrame(trades, columns=[
        "time", "side", "type", "price", "qty", "pos_after", "cash_after", "equity_after"])
    if not trades_df.empty:
        trades_df["time"] = trades_df["time"].dt.tz_convert(TZ_DISPLAY)

    state_df = pd.DataFrame(state, columns=[
        "time", "cash", "qty", "buys_in_cycle", "equity", "stopped"])
    state_df["time"] = state_df["time"].dt.tz_convert(TZ_DISPLAY)

    n_buys = int((trades_df["side"] == "BUY").sum()) if not trades_df.empty else 0
    n_sells = int((trades_df["side"] == "SELL").sum()) if not trades_df.empty else 0
    return {
        "final_equity": final_equity,
        "total_return": final_equity / INITIAL_CASH - 1.0,
        "max_drawdown": max_drawdown,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "qty": qty,                              # 期末剩余持仓
        "avg_cost": cycle_cost / qty if qty > 0 else 0.0,
        "final_price": final_price,
        "stopped": stopped,
        "stop_loss_time": stop_loss_time,
        "min_equity_exhausted": min_equity_exhausted,
        "equity": equity,
        "state": state_df,
        "trades": trades_df,
    }

# ===================== 绘图 =====================


def plot_compare(df: pd.DataFrame, base: dict, new: dict) -> None:
    """基线 vs 新策略 资金曲线叠加图。"""
    if plt is None:
        print("[图表] 未安装 matplotlib, 跳过绘图。安装命令: pip install matplotlib")
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 6))
    t = df["time"]
    ax.plot(t, base["equity"], linewidth=1.2, color="#1f77b4",
            label="基线策略(无过滤 / 无止损)")
    ax.plot(t, new["equity"], linewidth=1.2, color="#d62728",
            label="新策略(4h EMA60 过滤 + 15% 止损)")
    ax.axhline(INITIAL_CASH, color="gray", linestyle="--", linewidth=0.9,
               label="初始资金 50 USDT")
    ax.axhline(INITIAL_CASH * (1.0 - STOP_LOSS_PCT), color="orange",
               linestyle=":", linewidth=0.9, label="止损线 42.5 USDT")
    ax.set_title(f"{SYMBOL} 现货 DCA 网格策略对比 资金曲线"
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
    print(" DOGE/USDT 现货 DCA 网格回测 v2.1: 基线 vs 新策略")
    print("=" * 62)

    df = fetch_history()
    ema = compute_ema4h(df)
    print(f"[指标] 4h EMA{EMA4_PERIOD} 计算完成(前 {EMA4_PERIOD + 1} 根 4h K 线为预热期)")

    base = run_backtest(df)                     # 基线: 无过滤、无止损
    new = run_backtest(df, ema4h=ema,
                       use_trend_filter=True, use_stop_loss=True)   # 新策略

    base["trades"].to_csv(TRADE_BASE_FILE, index=False, encoding="utf-8-sig")
    new["trades"].to_csv(TRADE_NEW_FILE, index=False, encoding="utf-8-sig")
    new["state"].to_csv(STATE_NEW_FILE, index=False, encoding="utf-8-sig")
    plot_compare(df, base, new)

    def trades_str(r: dict) -> str:
        return f"{r['n_buys'] + r['n_sells']} 次(买 {r['n_buys']} / 卖 {r['n_sells']})"

    def status_str(r: dict) -> str:
        if r["stopped"]:
            return f"已止损停止({r['stop_loss_time'].strftime('%m-%d %H:%M')})"
        if r["qty"] > 0:
            return f"持仓死扛 {r['qty']:.1f} 枚"
        return "空仓"

    stop_line = INITIAL_CASH * (1.0 - STOP_LOSS_PCT)
    sl_new = (f"{new['stop_loss_time'].strftime('%m-%d %H:%M')} 触发"
              if new["stop_loss_time"] is not None else "未触发")

    print()
    print("-" * 62)
    print(f"{'指标':<10}{'基线策略':>22}{'新策略':>28}")
    print("-" * 62)
    print(f"{'总交易次数':<12}{trades_str(base):>22}{trades_str(new):>28}")
    print(f"{'最终总资产':<12}{base['final_equity']:>21.4f}{new['final_equity']:>27.4f} USDT")
    print(f"{'总收益率':<12}{base['total_return'] * 100:>21.2f}{new['total_return'] * 100:>27.2f} %")
    print(f"{'最大回撤':<12}{base['max_drawdown'] * 100:>21.2f}{new['max_drawdown'] * 100:>27.2f} %")
    print(f"{'止损触发':<12}{'不启用':>22}{sl_new:>28}")
    print(f"{'期末状态':<12}{status_str(base):>22}{status_str(new):>28}")
    print("-" * 62)

    # ---- 审计信息: 回撤锚点与止损判定的逐 K 线核对 ----
    st = new["state"]
    peak = st.loc[st["equity"].idxmax()]
    trough = st.loc[st["equity"].idxmin()]
    print()
    print("-" * 62)
    print(" 审计信息(新策略, 来自逐 K 线状态表):")
    print(f"  总资产峰值   : {peak['equity']:.4f} USDT @ {peak['time'].strftime('%m-%d %H:%M')}")
    print(f"  总资产谷值   : {trough['equity']:.4f} USDT @ {trough['time'].strftime('%m-%d %H:%M')}")
    print(f"  最大回撤核对 : 谷值/峰值-1 = {(trough['equity'] / peak['equity'] - 1.0) * 100:.2f} %"
          f"  (回撤的分母是曲线峰值 {peak['equity']:.2f} USDT, 不是初始资金 50)")
    for label, r in [("基线策略", base), ("新策略", new)]:
        if r["min_equity_exhausted"] < float("inf"):
            print(f"  {label}: 子弹打光且持仓期间最低总资产 = {r['min_equity_exhausted']:.4f} USDT")
        else:
            print(f"  {label}: 期间未出现子弹打光")
    print(f"  止损判定口径 : 子弹打光 且 总资产(现金+DOGE市值) < {stop_line:.1f} USDT,"
          f" 每根 K 线独立检查")
    print(f"  止损后处理   : 清空持仓 + 全部状态重置; RESUME_AFTER_STOP = {RESUME_AFTER_STOP}")
    print("-" * 62)

    print(f"\n新策略逐 K 线状态表已保存: {STATE_NEW_FILE}")
    print(f"新策略成交明细已保存: {TRADE_NEW_FILE} (含成交后持仓/资产列)")
    print(f"对比图已保存: {PNG_FILE}")

    if not new["trades"].empty:
        print("\n新策略成交明细(前 8 笔 + 后 4 笔):")
        show = pd.concat([new["trades"].head(8), new["trades"].tail(4)])
        print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
