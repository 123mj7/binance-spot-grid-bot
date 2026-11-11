# -*- coding: utf-8 -*-
"""verify_blackswan.py —— 黑天鹅检测器(含成交量异动)验证"""
import statistics
import sys
from datetime import datetime, timezone

import ccxt

import binance_live_bot as b

ex = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
cfg = b.load_config()
plan = cfg["blackswan"]
CRASH, SPIKE, VOL = plan["btc_crash_pct"], plan["vol_spike_x"], plan["volume_spike_x"]


def metrics(candles, lookback=24):
    closes = [float(c[4]) for c in candles]
    last_ret = (closes[-1] / closes[-2] - 1.0) * 100
    hourly = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    recent, base = hourly[-24:], hourly[-480:]
    r_std = statistics.pstdev(recent) if len(recent) > 2 else 0.0
    b_std = statistics.pstdev(base) if len(base) > 30 else 0.0
    spike = (r_std / b_std) if b_std > 1e-9 else 0.0
    vol_ratio = b._volume_spike(candles, lookback)
    return last_ret, spike, vol_ratio


def window_check(name, as_of_iso, pre_days=14):
    """逐小时回放最后 72 小时, 报告首次触发时刻(与实盘逐周期检查一致)。"""
    as_of_ms = int(datetime.fromisoformat(as_of_iso)
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    start_ms = as_of_ms - pre_days * 24 * 3600 * 1000
    try:
        rows = ex.fetch_ohlcv("BTC/USDT", "1h", since=start_ms,
                              limit=pre_days * 24 + 26)
        rows = [r for r in rows if r[0] <= as_of_ms]
        rows = sorted(rows, key=lambda r: r[0])
    except Exception as exc:
        print(f"[{name}] 数据获取失败: {exc}")
        return
    if len(rows) < 100:
        print(f"[{name}] 数据不足")
        return
    fired = []
    for i in range(max(60, len(rows) - 72), len(rows)):
        lr, sp, vr = metrics(rows[:i + 1])
        if lr <= -CRASH or sp >= SPIKE or vr >= VOL:
            fired.append((rows[i][0], lr, sp, vr))
            if len(fired) >= 3:
                break
    if fired:
        for t, lr, sp, vr in fired[:3]:
            ts = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
            print(f"[{name}] ✓ 触发 @ {ts} | 1h涨跌 {lr:+7.2f}% | 波动率比 {sp:5.2f} "
                  f"| 量能比 {vr:5.2f}")
    else:
        lr, sp, vr = metrics(rows)
        print(f"[{name}] ✗ 最后72小时未触发 | 末值: 1h涨跌 {lr:+.2f}% | "
              f"波动率比 {sp:.2f} | 量能比 {vr:.2f}")


print("=" * 78)
print(f" 黑天鹅检测器验证(阈值: 暴跌{CRASH}% / 波动{SPIKE}倍 / 量能{VOL}倍)")
print("=" * 78)

# 1) 当前实时信号
ohlcv = sorted(ex.fetch_ohlcv("BTC/USDT", "1h", limit=520), key=lambda r: r[0])[:-1]
lr, sp, vr = metrics(ohlcv)
print(f"[实时] BTC   1h涨跌 {lr:+.2f}% | 波动率比 {sp:.2f} | 量能比 {vr:.2f}")
sym = cfg["symbols"][0]
sc = sorted(ex.fetch_ohlcv(sym, "1h", limit=60), key=lambda r: r[0])[:-1]
print(f"[实时] {sym} 量能比 {b._volume_spike(sc, 24):.2f}")
risk = b.blackswan_check(ex, cfg, {"active": False, "reason": "", "until_ms": 0})
print(f"[实时] 检测器状态: {risk}")

# 2) 历史灾变窗口回放
print()
print(" 历史窗口回放(逐小时检查, 与实盘同逻辑):")
window_check("LUNA 暴跌(2022-05-11~13)", "2022-05-13T23:59:00")
window_check("FTX 暴雷(2022-11-08~10)", "2022-11-10T23:59:00")
window_check("平静期(2023-06-01~03)", "2023-06-03T23:59:00")
