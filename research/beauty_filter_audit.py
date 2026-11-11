# -*- coding: utf-8 -*-
"""
beauty_filter_audit.py —— 四个美颜滤镜审计
============================================
滤镜一 前视偏差: 对比"收盘价成交"(有嫌疑) vs "信号后下一根开盘价成交"(严格无前视)
滤镜二 过拟合  : 见 anti_overfit.py 三关(已完成, 此处不重复)
滤镜三 幸存者偏差: 用已死的币(LUNC 2022 死亡螺旋 / FTT 2022 崩盘)跑同一策略
滤镜四 偷价假设: 加滑点 0/0.1%/0.3%/0.5% 重跑五年基线
"""
import sys
import time

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Jeff Qu\Documents\deepseek-harness\default-workspace")
import adaptive_backtest as ab

INIT = 50.0
BUY = 5.0
MAX_ADDS = 9
TP, DROP = 0.03, 0.03
STOP_LINE = INIT * 0.85
FEE = 0.001


def run_fixed(df: pd.DataFrame, ema4h: np.ndarray, fill: str, slip: float = 0.0) -> dict:
    """fill='close' 收盘价成交(原回测口径); fill='open' 下一根开盘价成交(严格无前视)。"""
    prices = df["close"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    n = len(prices)
    cash, qty = INIT, 0.0
    last_buy, buys, cost = 0.0, 0, 0.0
    stopped = False
    rounds = rounds_tp = n_buys = n_sells = 0
    equity = np.empty(n)
    exec_px = 0.0
    act = None                     # ('buy','add','tp','stop') 待执行的信号
    for i in range(n):
        p = prices[i]
        # ---- 1) 执行上一根定下的信号: 成交价 = 本根开盘(或最后用收盘) ----
        if act is not None and not stopped:
            exec_px = opens[i] if i > 0 else p
            if act == "buy":
                px = exec_px * (1 + slip)
                qty += (BUY - BUY * FEE) / px
                cash -= BUY
                cost, buys, last_buy = BUY, 1, px
                n_buys += 1
                rounds += 1
            elif act == "add":
                px = exec_px * (1 + slip)
                qty += (BUY - BUY * FEE) / px
                cash -= BUY
                cost += BUY
                buys += 1
                last_buy = px
                n_buys += 1
            elif act == "tp":
                px = exec_px * (1 - slip)
                cash += qty * px * (1 - FEE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                rounds_tp += 1
            elif act == "stop":
                px = exec_px * (1 - slip)
                cash += qty * px * (1 - FEE)
                qty, cost, buys = 0.0, 0.0, 0
                n_sells += 1
                stopped = True
            act = None
        # ---- 2) 用本根收盘价做决策, 信号下一根执行 ----
        if stopped:
            act = None
        elif qty <= 0:
            e = ema4h[i]
            if not np.isnan(e) and p > e:
                act = "buy"
        else:
            if buys - 1 < MAX_ADDS and p <= last_buy * (1 - DROP):
                act = "add"
            elif p >= (cost / qty) * (1 + TP):
                act = "tp"
            elif buys >= MAX_ADDS + 1 and cash + qty * p < STOP_LINE:
                act = "stop"
        equity[i] = cash + qty * p
    # 末根信号直接按收盘成交
    if act is not None and not stopped:
        px = prices[-1]
        if act in ("buy", "add"):
            px *= (1 + slip)
            qty += (BUY - BUY * FEE) / px
            cash -= BUY
            if act == "buy":
                cost, buys, rounds = BUY, 1, rounds + 1
            else:
                cost, buys = cost + BUY, buys + 1
            n_buys += 1
        else:
            px *= (1 - slip)
            cash += qty * px * (1 - FEE)
            qty = 0.0
            n_sells += 1
            if act == "tp":
                rounds_tp += 1
            else:
                stopped = True
        equity[-1] = cash
    peak = np.maximum.accumulate(equity)
    dd = float(np.min((equity - peak) / peak)) * 100 if peak.max() > 0 else 0.0
    return {"return_pct": (equity[-1] / INIT - 1) * 100, "max_dd_pct": dd,
            "rounds": rounds, "rounds_tp": rounds_tp,
            "n_buys": n_buys, "n_sells": n_sells, "equity": equity,
            "end_state": "止损停止" if stopped else ("持仓中" if qty > 0 else "空仓")}


def load_doge_5y() -> pd.DataFrame:
    d1 = pd.read_csv("doge_usdt_1h_2021_22.csv")
    d2 = pd.read_csv("doge_usdt_1h_3y.csv")
    df = pd.concat([d1, d2], ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_bitget_1h(symbol: str, start: str, end: str) -> pd.DataFrame:
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").normalize().timestamp() * 1000) + 86_400_000
    rows, since = [], start_ms
    while since < end_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, "1h", since=since, limit=200)
        except Exception as exc:
            print(f"    {symbol} 数据获取失败: {str(exc)[:80]}")
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
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main() -> None:
    df5y = load_doge_5y()
    ema = ab.compute_ema4h(df5y, period=60)
    print("=" * 84)
    print(" 滤镜一: 前视偏差审计(五年 DOGE, EMA 信号已 shift(1), 只对比成交时点)")
    print("=" * 84)
    r_close = run_fixed(df5y, ema, "close")
    r_open = run_fixed(df5y, ema, "open")
    print(f"  收盘价成交(原口径)  : {r_close['return_pct']:+8.2f}%  DD {r_close['max_dd_pct']:.1f}%"
          f"  轮数 {r_close['rounds']}")
    print(f"  下一根开盘成交(严格) : {r_open['return_pct']:+8.2f}%  DD {r_open['max_dd_pct']:.1f}%"
          f"  轮数 {r_open['rounds']}")
    gap = r_close["return_pct"] - r_open["return_pct"]
    print(f"  → 前视偏差贡献约 {gap:+.2f} 个百分点"
          f"{' (明显, 原口径高估)' if gap > 8 else ' (温和, 策略主体不受影响)'}")

    print()
    print("=" * 84)
    print(" 滤镜四: 偷价/滑点审计(五年 DOGE, 收盘口径 + 双边滑点)")
    print("=" * 84)
    for slip in (0.0, 0.001, 0.003, 0.005):
        r = run_fixed(df5y, ema, "close", slip=slip)
        print(f"  滑点 {slip * 100:.1f}% : {r['return_pct']:+8.2f}%  DD {r['max_dd_pct']:.1f}%")

    print()
    print("=" * 84)
    print(" 滤镜三: 幸存者偏差审计(用已死/崩盘的币跑同一策略)")
    print("=" * 84)
    for sym, start, end in (("LUNC/USDT", "2022-01-01", "2023-01-01"),
                            ("FTT/USDT", "2022-06-01", "2023-06-01")):
        dfx = fetch_bitget_1h(sym, start, end)
        if len(dfx) < 2000:
            print(f"  {sym}: 数据不足({len(dfx)}), 跳过")
            continue
        emax = ab.compute_ema4h(dfx, period=60)
        r = run_fixed(dfx, emax, "close")
        bh = (dfx["close"].iloc[-1] / dfx["close"].iloc[0] - 1) * 100
        print(f"  {sym} {start}~{end} ({len(dfx)}根):")
        print(f"    买入持有 {bh:+9.2f}% | 策略 {r['return_pct']:+9.2f}% "
              f"DD {r['max_dd_pct']:.1f}% | 轮数 {r['rounds']} | {r['end_state']}")
    print()
    print("完成: 四滤镜审计(滤镜二见 anti_overfit.py 三关)")


if __name__ == "__main__":
    main()
