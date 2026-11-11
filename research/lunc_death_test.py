# -*- coding: utf-8 -*-
"""LUNC/USDT 2022-05 死亡螺旋(-99.99%)的同一策略表现, 数据源: 币安 data-api。"""
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab
from beauty_filter_audit import run_fixed

start_ms = int(pd.Timestamp("2022-04-01", tz="UTC").timestamp() * 1000)
end_ms = int(pd.Timestamp("2022-08-01", tz="UTC").normalize().timestamp() * 1000) + 86_400_000
rows, since = [], start_ms
while since < end_ms:
    for attempt in range(3):
        try:
            r = requests.get("https://data-api.binance.vision/api/v3/klines",
                             params={"symbol": "LUNCUSDT", "interval": "1h",
                                     "startTime": since, "limit": 1000}, timeout=20)
            data = r.json()
            break
        except Exception:
            data = None
            time.sleep(2)
    if not isinstance(data, list) or not data:
        break
    rows.extend(data)
    ns = data[-1][0] + 3_600_000
    if ns <= since:
        break
    since = ns
    if len(data) < 1000:
        break
    time.sleep(0.4)

print("拿到", len(rows), "根")
if len(rows) > 800:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]
                      + ["x"] * 6)
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(float)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    emax = ab.compute_ema4h(df, period=60)
    r = run_fixed(df, emax, "close")
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"区间 {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (UTC)")
    print(f"买入持有 {bh:+9.2f}% | 策略 {r['return_pct']:+9.2f}% "
          f"DD {r['max_dd_pct']:.1f}% | 轮数 {r['rounds']}/{r['rounds_tp']} | {r['end_state']}")
