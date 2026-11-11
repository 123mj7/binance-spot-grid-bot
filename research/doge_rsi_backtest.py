# -*- coding: utf-8 -*-
"""
DOGE/USDT 1 小时 K 线 RSI(14) 超买超卖回测
============================================

数据源: CCXT 下载最近 180 天 DOGE/USDT 1h K 线
        (默认 Binance, 失败自动依次尝试 OKX / Bybit / Gate.io)
策略  : RSI(14) < 30 → 全仓买入 DOGE;  RSI(14) > 70 → 全部卖出
假设  : 信号以 K 线收盘时的 RSI 判定, 并以该根 K 线收盘价成交(无未来函数)
费用  : 买入、卖出各收取成交额 0.1% 的手续费
平仓  : 回测结束时若仍持仓, 按最后一根 K 线收盘价强制平仓(同样计费)

依赖: pip install ccxt pandas numpy
运行: python doge_rsi_backtest.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:  # 允许在没有 ccxt 的环境里先看代码, 运行时给出提示
    ccxt = None

# ===================== 可调参数 =====================
SYMBOL = "DOGE/USDT"
TIMEFRAME = "1h"
DAYS = 180                  # 回看天数
RSI_PERIOD = 14             # RSI 周期
RSI_OVERSOLD = 30.0         # 买入阈值
RSI_OVERBOUGHT = 70.0       # 卖出阈值
INITIAL_USDT = 50.0         # 初始资金
FEE_RATE = 0.001            # 手续费 0.1% (单边)
CSV_FILE = "doge_usdt_1h.csv"    # K 线缓存
TRADE_FILE = "doge_trades.csv"   # 成交明细
EXCHANGE_IDS = ["binance", "okx", "bybit", "gateio"]
TZ_DISPLAY = "Asia/Shanghai"     # 输出用本地时区

# ===================== 数据下载 =====================


def fetch_history() -> pd.DataFrame:
    """依次尝试多个交易所, 下载最近 DAYS 天的 1h K 线。"""
    if ccxt is None:
        sys.exit("未安装 ccxt, 请先执行: pip install ccxt")

    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)

    for ex_id in EXCHANGE_IDS:
        try:
            exchange = getattr(ccxt, ex_id)({"enableRateLimit": True})
            print(f"[下载] 尝试 {ex_id}: {SYMBOL} {TIMEFRAME}, 最近 {DAYS} 天 ...")
            now_ms = exchange.milliseconds()
            rows, since = [], start_ms
            while since < now_ms:
                batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                new_since = batch[-1][0] + 1
                if new_since <= since:      # 无进展, 防止死循环
                    break
                since = new_since
                if len(batch) < 1000:       # 已拿到最新一根
                    break
                time.sleep(max(exchange.rateLimit or 100, 200) / 1000.0)

            if not rows:
                raise RuntimeError("交易所返回空数据")

            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
            print(f"[下载] 成功: {len(df)} 根 K 线 | "
                  f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC) | 已保存 {CSV_FILE}")
            return df
        except Exception as exc:
            print(f"[下载] {ex_id} 失败: {exc}")

    sys.exit("[下载] 所有交易所均下载失败, 请检查网络/代理后重试。")

# ===================== 指标与回测 =====================


def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder 平滑 RSI(与主流行情软件一致)。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # 无下跌但有上涨时 RSI = 100; 完全走平时 RSI 无定义, 保持 NaN
    rsi = rsi.where((avg_loss != 0.0) | (avg_gain == 0.0), 100.0)
    return rsi


def run_backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"], RSI_PERIOD)

    cash = INITIAL_USDT        # 现金(USDT)
    qty = 0.0                  # 持仓数量(DOGE)
    trades = []                # 成交记录
    equity = []                # 逐 K 线市值
    held_bars = 0              # 持仓的 K 线数

    for row in df.itertuples(index=False):
        price, rsi = float(row.close), row.rsi
        if not np.isnan(rsi):
            if qty == 0.0 and rsi < RSI_OVERSOLD:
                fee = cash * FEE_RATE
                qty = (cash - fee) / price          # 全仓买入(扣手续费)
                trades.append([row.time, "BUY", price, qty, 0.0, fee])
                cash = 0.0
            elif qty > 0.0 and rsi > RSI_OVERBOUGHT:
                gross = qty * price
                fee = gross * FEE_RATE
                cash = gross - fee                  # 全部卖出(扣手续费)
                trades.append([row.time, "SELL", price, qty, cash, fee])
                qty = 0.0
        if qty > 0.0:
            held_bars += 1
        equity.append(cash + qty * price)

    # 期末仍持仓 → 按最后一根 K 线收盘价强平
    if qty > 0.0:
        last_price = float(df["close"].iloc[-1])
        gross = qty * last_price
        fee = gross * FEE_RATE
        cash = gross - fee
        trades.append([df["time"].iloc[-1], "SELL(期末强平)", last_price, qty, cash, fee])
        qty = 0.0
        equity[-1] = cash

    equity = pd.Series(equity, index=df.index, dtype=float)
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    # 对照: 从第 RSI_PERIOD 根 K 线起买入并持有
    first_close = float(df["close"].iloc[RSI_PERIOD])
    last_close = float(df["close"].iloc[-1])
    bh_final = (INITIAL_USDT * (1.0 - FEE_RATE) / first_close) * last_close * (1.0 - FEE_RATE)

    trades_df = pd.DataFrame(trades, columns=["time", "side", "price", "qty", "cash_after", "fee"])
    trades_df["time"] = trades_df["time"].dt.tz_convert(TZ_DISPLAY)
    trades_df.to_csv(TRADE_FILE, index=False, encoding="utf-8-sig")

    n_buys = int(trades_df["side"].str.startswith("BUY").sum())
    return {
        "final_equity": cash,
        "total_return": cash / INITIAL_USDT - 1.0,
        "bh_return": bh_final / INITIAL_USDT - 1.0,
        "max_drawdown": max_drawdown,
        "n_buys": n_buys,
        "n_sells": len(trades_df) - n_buys,
        "hold_ratio": held_bars / len(df),
        "trades": trades_df,
    }


def main() -> None:
    print("=" * 62)
    print(" DOGE/USDT 1h K 线 · RSI(14) 超买超卖回测")
    print("=" * 62)

    df = fetch_history()
    result = run_backtest(df)

    print()
    print("-" * 62)
    print(f"{'回测结果':^60}")
    print("-" * 62)
    print(f"  初始资金       : {INITIAL_USDT:>12.2f} USDT")
    print(f"  最终净值       : {result['final_equity']:>12.4f} USDT")
    print(f"  总收益率       : {result['total_return'] * 100:>11.2f} %")
    print(f"  买入持有收益率 : {result['bh_return'] * 100:>11.2f} %")
    print(f"  最大回撤       : {result['max_drawdown'] * 100:>11.2f} %")
    print(f"  成交笔数       : 买入 {result['n_buys']} 次, 卖出 {result['n_sells']} 次")
    print(f"  持仓时间占比   : {result['hold_ratio'] * 100:>11.2f} %")
    print("-" * 62)

    if result["trades"].empty:
        print("\n近 180 天内未触发任何交易。")
    else:
        print(f"\n成交明细(完整列表见 {TRADE_FILE}):")
        print(result["trades"].to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
