# -*- coding: utf-8 -*-
"""
regime_detect.py —— 多时间框架行情路由器(v2, 1h 判定)
=======================================================
重构说明(任务1):
  - 大趋势过滤: 日线 EMA60(由 1h 聚合的日线收盘计算, 只用已收盘日线, 无未来函数)
    现价在日线 EMA60 下方 = 大趋势偏空 -> 绝对不开新仓。
  - 状态判定: ADX / ATR / BOLL 全部切到 1 小时级别:
      趋势行情 : 1h ADX(14) > 25, 方向由 1h EMA20 与 1h EMA60 判定
      震荡行情 : 1h ADX(14) < 22(门槛放宽) 且收盘价在 1h BOLL(20,2) 上下轨之间
      极端风险 : 1h ATR(14) 放大到其 20 期均值的 2 倍以上
      其余     : 中性
    优先级: 极端风险 > 趋势 > 震荡 > 中性 > 预热期
  - 路由动作:
      震荡 + 现价 > 日线 EMA60 -> 开仓窗口(激活网格/DCA)
      趋势(上/下) / 中性      -> 暂停买入
      极端风险                 -> 清仓避险
      现价 < 日线 EMA60        -> 大趋势偏空, 绝对不买

输出:
  - 控制台: 1h 状态分布 + 每日汇总(主状态/震荡小时数/开仓窗口小时数)
  - regime_1h.csv           : 每根 1h K 线的状态明细
  - regime_daily_summary.csv: 每日汇总

依赖: pip install ccxt pandas numpy
运行: python regime_detect.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    sys.exit("未安装 ccxt, 请先执行: pip install ccxt")

# ===================== 参数 =====================
SYMBOL = "DOGE/USDT"
TIMEFRAME = "1h"
DAYS = 180
TZ = "Asia/Shanghai"

ADX_TREND_1H = 25.0       # 1h ADX > 25 = 趋势
ADX_RANGE_1H = 22.0       # 1h ADX < 22 = 震荡(放宽)
ATR_SPIKE_1H = 2.0        # 1h ATR 放大到均值的 2 倍 = 极端风险
ADX_PERIOD = 14
ATR_PERIOD, ATR_SMA = 14, 20
BOLL_PERIOD, BOLL_STD = 20, 2.0
EMA_FAST_1H, EMA_SLOW_1H = 20, 60
EMA_DAILY = 60            # 日线 EMA60 = 大趋势过滤

CSV_1H = "regime_1h.csv"
CSV_DAILY = "regime_daily_summary.csv"
CACHE_CSV = "doge_usdt_1h.csv"

REGIME_CODE = {"震荡": 0, "趋势上涨": 1, "趋势下跌": 2,
               "极端风险": 3, "中性": 4, "预热期": 5}

# ===================== 数据下载 =====================


def fetch_history(start_date: str | None = None, end_date: str | None = None,
                  days: int = DAYS, cache_csv: str = CACHE_CSV) -> pd.DataFrame:
    """下载 1h K 线。
    start_date/end_date 传 "YYYY-MM-DD"(UTC, end_date 含当日)时下载固定历史区间;
    都不传时下载最近 days 天。网络失败时回退到本地缓存 CSV。"""
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        if start_date:
            start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        else:
            start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        if end_date:
            # end_date 含当日 -> 用次日零点作为排他边界
            end_ms = int(pd.Timestamp(end_date, tz="UTC").normalize().timestamp() * 1000) \
                + 86_400_000
            params = {"endTime": end_ms}
        else:
            end_ms = exchange.milliseconds()
            params = {}
        print(f"[下载] 币安 {SYMBOL} {TIMEFRAME}, "
              f"{'最近 ' + str(days) + ' 天' if not start_date else start_date + ' ~ ' + end_date} ...")
        rows, since = [], start_ms
        while since < end_ms:
            batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000,
                                         params=params)
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
        if end_date:
            df = df[df["ts"] < end_ms]                        # 精确截断到指定区间
        else:
            df = df[df["ts"] + 3_600_000 <= end_ms]           # 去掉未收盘K线
        df = df.reset_index(drop=True)
        df.to_csv(cache_csv, index=False, encoding="utf-8-sig")
        print(f"[下载] 成功: {len(df)} 根 K 线 | "
              f"{df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")
        return df
    except Exception as exc:
        print(f"[下载] 在线获取失败: {exc}")
        if os.path.exists(cache_csv):
            print(f"[下载] 使用本地缓存 {cache_csv}")
            df = pd.read_csv(cache_csv)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df
        sys.exit("[下载] 无可用数据, 请检查网络后重试。")

# ===================== 日线 EMA60(大趋势过滤) =====================


def compute_daily_ema60(df: pd.DataFrame, period: int = EMA_DAILY) -> pd.Series:
    """把日线 EMA60 回填到每根 1h K 线。
    只使用"上一根已收盘日线"(shift(1))计算, 无未来函数;
    前 period 天为预热期(NaN), 预热期内不允许开仓。"""
    day = df["time"].dt.tz_convert(TZ).dt.normalize()
    daily_close = pd.Series(df["close"].to_numpy(), index=day).resample("1D").last()
    ema = daily_close.ewm(span=period, adjust=False).mean()
    ema.iloc[:period] = np.nan
    ema = ema.shift(1)
    return day.map(ema)

# ===================== 1h 级别状态识别 =====================


def compute_router(df: pd.DataFrame) -> pd.DataFrame:
    """计算 1h 级别路由所需全部指标与状态, 返回带新列的 df 副本。"""
    out = df.copy()
    close = out["close"]

    out["ema20_1h"] = close.ewm(span=EMA_FAST_1H, adjust=False).mean()
    out["ema60_1h"] = close.ewm(span=EMA_SLOW_1H, adjust=False).mean()
    out.iloc[:EMA_FAST_1H, out.columns.get_loc("ema20_1h")] = np.nan
    out.iloc[:EMA_SLOW_1H, out.columns.get_loc("ema60_1h")] = np.nan

    mid = close.rolling(BOLL_PERIOD, min_periods=BOLL_PERIOD).mean()
    std = close.rolling(BOLL_PERIOD, min_periods=BOLL_PERIOD).std(ddof=0)
    out["boll_lower"] = mid - BOLL_STD * std
    out["boll_upper"] = mid + BOLL_STD * std

    prev_close = close.shift(1)
    tr = pd.concat([out["high"] - out["low"],
                    (out["high"] - prev_close).abs(),
                    (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
    out["atr_ratio_1h"] = atr / atr.rolling(ATR_SMA, min_periods=ATR_SMA).mean()

    up = out["high"].diff()
    down = -out["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=out.index)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD,
                                  adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD,
                                    adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out["adx_1h"] = dx.ewm(alpha=1.0 / ADX_PERIOD, min_periods=ADX_PERIOD,
                           adjust=False).mean()

    out["ema60_daily"] = compute_daily_ema60(df)
    out["regime_1h"] = out.apply(classify_1h, axis=1)
    out["regime_code"] = out["regime_1h"].map(REGIME_CODE)
    out["open_ok"] = ((out["regime_1h"] == "震荡")
                      & out["ema60_daily"].notna()
                      & (out["close"] > out["ema60_daily"]))
    return out


def classify_1h(row) -> str:
    if (np.isnan(row["adx_1h"]) or np.isnan(row["ema20_1h"]) or np.isnan(row["ema60_1h"])
            or np.isnan(row["boll_lower"]) or np.isnan(row["atr_ratio_1h"])):
        return "预热期"
    if row["atr_ratio_1h"] >= ATR_SPIKE_1H:                       # 3) 极端风险优先
        return "极端风险"
    if row["adx_1h"] > ADX_TREND_1H:                              # 1) 趋势
        return "趋势上涨" if row["ema20_1h"] > row["ema60_1h"] else "趋势下跌"
    if (row["adx_1h"] < ADX_RANGE_1H
            and row["boll_lower"] < row["close"] < row["boll_upper"]):   # 2) 震荡
        return "震荡"
    return "中性"

# ===================== 主流程 =====================


def main() -> None:
    print("=" * 84)
    print(" 多时间框架路由器: 日线 EMA60 大趋势过滤 + 1h ADX/ATR/BOLL 状态判定")
    print("=" * 84)

    df = fetch_history()
    router = compute_router(df)

    # ---- 1h 状态分布 ----
    counts = router["regime_1h"].value_counts()
    print()
    print(" 1h 状态分布(共 %d 根 K 线):" % len(router))
    for s in ["震荡", "趋势上涨", "趋势下跌", "极端风险", "中性", "预热期"]:
        if s in counts:
            print(f"   {s:<8} {counts[s]:>5} 根 ({counts[s] / len(router) * 100:5.1f}%)")
    win = router["open_ok"].sum()
    print(f"   开仓窗口(震荡 且 现价 > 日线EMA60): {win} 根 ({win / len(router) * 100:.1f}%)")

    # ---- 每日汇总 ----
    d = router.copy()
    d["day"] = d["time"].dt.tz_convert(TZ).dt.normalize()
    daily = d.groupby("day").agg(
        震荡小时=("regime_1h", lambda s: int((s == "震荡").sum())),
        趋势小时=("regime_1h", lambda s: int(s.str.startswith("趋势").sum())),
        极端小时=("regime_1h", lambda s: int((s == "极端风险").sum())),
        开仓窗小时=("open_ok", lambda s: int(s.sum())),
        主状态=("regime_1h", lambda s: s.mode().iloc[0]),
        收盘=("close", "last"),
        日线EMA60=("ema60_daily", "last"),
    )
    daily["高于EMA60"] = daily["收盘"] > daily["日线EMA60"]
    print()
    print(f"{'日期':<12}{'主状态':<10}{'震荡h':>5}{'趋势h':>5}{'极端h':>5}{'开窗h':>5}"
          f"{'收盘':>10}{'EMA60':>10}  高于EMA")
    print("-" * 84)
    prev = None
    for date, row in daily.iterrows():
        if row["主状态"] != prev and prev is not None:
            print(f"  >>> 主状态切换: {prev} -> {row['主状态']} ({date:%Y-%m-%d})")
        prev = row["主状态"]
        ema = f"{row['日线EMA60']:>10.6f}" if not np.isnan(row["日线EMA60"]) else "      --  "
        print(f"{date:%Y-%m-%d}   {row['主状态']:<10}{row['震荡小时']:>5}{row['趋势小时']:>5}"
              f"{row['极端小时']:>5}{row['开仓窗小时']:>5}{row['收盘']:>10.6f}{ema}"
              f"{'  是' if row['高于EMA60'] else '  否'}")
    print("-" * 84)

    router.to_csv(CSV_1H, index=False, encoding="utf-8-sig")
    daily.reset_index().to_csv(CSV_DAILY, index=False, encoding="utf-8-sig")
    print(f" 明细已保存: {CSV_1H} / {CSV_DAILY}")


if __name__ == "__main__":
    main()
