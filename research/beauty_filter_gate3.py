# -*- coding: utf-8 -*-
"""滤镜三第三轮: 找真实可用的"死币/崩溃币"数据源。"""
import sys
import time

import ccxt
import pandas as pd
import requests

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab
from beauty_filter_audit import run_fixed

# 1) 先探测哪个源有死亡币历史
print("--- 数据源探测 ---")
for name, fn in (
        ("binance data-api", lambda: requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": "LUNCUSDT", "interval": "1h",
                    "startTime": 1651363200000, "limit": 5}, timeout=15)),
        ("bitget LUNC", lambda: ccxt.bitget({"enableRateLimit": True,
            "options": {"defaultType": "spot"}}).fetch_ohlcv(
            "LUNC/USDT", "1h", since=1651363200000, limit=5)),
        ("bitget LUNA", lambda: ccxt.bitget({"enableRateLimit": True,
            "options": {"defaultType": "spot"}}).fetch_ohlcv(
            "LUNA/USDT", "1h", since=1654099200000, limit=5)),
):
    try:
        data = fn()
        if isinstance(data, requests.Response):
            j = data.json()
            print(f"  {name}: {'OK, ' + str(len(j)) + ' 根' if isinstance(j, list) and j else '空: ' + str(j)[:60]}")
        else:
            print(f"  {name}: OK, {len(data)} 根, 首根 "
                  f"{pd.to_datetime(data[0][0], unit='ms', utc=True)}")
    except Exception as exc:
        print(f"  {name}: 失败 {str(exc)[:70]}")

# 2) 用可用源跑死亡币检验
def fetch_binance_klines(sym_id, start_iso, end_iso):
    start_ms = int(pd.Timestamp(start_iso, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_iso, tz="UTC").normalize().timestamp() * 1000) + 86_400_000
    rows, since = [], start_ms
    while since < end_ms:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": sym_id, "interval": "1h",
                                 "startTime": since, "limit": 1000}, timeout=20)
        data = r.json()
        if not isinstance(data, list) or not data:
            break
        rows.extend(data)
        ns = data[-1][0] + 3_600_000
        if ns <= since:
            break
        since = ns
        if len(data) < 1000:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]
                      + ["x"] * 6)
    if len(df):
        df = df[["ts", "open", "high", "low", "close", "volume"]].astype(float)
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def fetch_bitget_1h(symbol, start_iso, end_iso):
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    start_ms = int(pd.Timestamp(start_iso, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_iso, tz="UTC").normalize().timestamp() * 1000) + 86_400_000
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
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


print("\n--- 死亡币策略检验 ---")
tests = []
try:
    df = fetch_binance_klines("LUNCUSDT", "2022-05-01", "2022-08-01")
    if len(df) > 1000:
        tests.append(("LUNC/USDT(币安, 2022死亡螺旋)", df))
except Exception:
    pass
try:
    df = fetch_bitget_1h("LUNA/USDT", "2022-06-01", "2022-10-01")
    if len(df) > 1000:
        tests.append(("LUNA/USDT(bitget, 2022 -93%崩盘)", df))
except Exception:
    pass

for name, dfx in tests:
    emax = ab.compute_ema4h(dfx, period=60)
    r = run_fixed(dfx, emax, "close")
    bh = (dfx["close"].iloc[-1] / dfx["close"].iloc[0] - 1) * 100
    print(f"  {name}: {len(dfx)} 根")
    print(f"    买入持有 {bh:+9.2f}% | 策略 {r['return_pct']:+9.2f}% "
          f"DD {r['max_dd_pct']:.1f}% | 轮数 {r['rounds']}/{r['rounds_tp']} | "
          f"{r['end_state']}")
if not tests:
    print("  无可用的死亡币数据源(网络受限), 该关保持开放, 留给贡献者补测")
