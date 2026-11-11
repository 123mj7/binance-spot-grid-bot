# -*- coding: utf-8 -*-
"""
binance_live_bot.py —— 币安现货 DCA 实盘机器人(策略模块)
==========================================================
(纯实盘, 无干跑模式; 多币种可选但默认单币种; 断电自检; 黑天鹅避险)

启动流程:
  1. 读取 API Key(config_real.json 或环境变量 BINANCE_API_KEY/SECRET)
  2. API 连接自检: 真实"挂单 → 立即撤单"一笔远低于市价的限价单,
     验证密钥具备现货交易权限(不会真的成交)
  3. 断电恢复自检: 撤残留挂单 + 以交易所真实持仓为准 + 成交记录重建本轮
  4. 进入主循环: 每 60 秒读实时余额与行情, 执行 DCA 策略

策略(无路由-固定止损基线): 4h EMA60 过滤开仓, 跌3%加仓5U(每轮最多9次),
涨3%止盈, 子弹打光且总资产<预算85%止损; 实时可用USDT动态预算; 滑点监控;
BNB抵扣(可选); 黑天鹅避险(暂停买入)。

⚠️ 真金白银, 使用前必读《实盘使用说明.txt》。
"""

from __future__ import annotations

import csv
import json
import msvcrt
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    sys.exit("未安装 ccxt: pip install ccxt")

TZ = timezone(timedelta(hours=8))

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "live_trade_real.txt")
CONFIG_FILE = os.path.join(BASE_DIR, "config_real.json")
STOP_FILE = os.path.join(BASE_DIR, "STOP_REAL.txt")
LOCK_FILE = os.path.join(BASE_DIR, "live_real.lock")
_lock_handle = None

DEFAULTS = {
    "binance": {"api_key": "", "api_secret": "", "testnet": False},
    "api_key": "", "api_secret": "",   # 兼容旧版配置
    "symbols": ["DOGE/USDT"],
    "alloc_per_symbol": 50.0,
    "initial_buy_usdt": 5.0,
    "add_buy_usdt": 5.0,
    "max_adds": 9,
    "take_profit": 0.03,
    "drop_trigger": 0.03,
    "stop_loss_pct": 0.15,
    "min_notional": 5.0,
    "slippage_warn_pct": 0.5,
    "bnb_fee_plan": {
        "enabled": False,
        "buy_once_usdt": 5.0,
        "buy_threshold_usdt": 0.5,
        "min_free_usdt": 60.0,
    },
    "blackswan": {
        "enabled": True,
        "btc_crash_pct": 3.0,
        "vol_spike_x": 2.5,
        "volume_spike_x": 3.0,      # 成交量异动: 最后1小时成交量 > 前24小时均量的 3 倍
        "volume_lookback": 24,      # 成交量基线窗口(小时)
        "pause_hours": 4.0,
    },
    "check_interval_sec": 60,
}

# ===================== 基础工具 =====================


def log(msg: str) -> None:
    line = f"[{datetime.now(TZ):%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        need_bom = (not os.path.exists(LOG_FILE)) or os.path.getsize(LOG_FILE) == 0
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            if need_bom:
                f.write("\ufeff")
            f.write(line + "\n")
            f.flush()
    except OSError:
        pass
    try:
        print(line)
    except Exception:
        pass


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as exc:
            log(f"配置文件读取失败: {exc}, 使用默认配置")
    # 兼容旧版配置: 顶层 api_key/api_secret 迁移到 binance 段
    legacy_key = str(cfg.get("api_key") or "").strip()
    legacy_sec = str(cfg.get("api_secret") or "").strip()
    if legacy_key:
        cfg.setdefault("binance", {})["api_key"] = legacy_key
    if legacy_sec:
        cfg.setdefault("binance", {})["api_secret"] = legacy_sec
    # 环境变量
    env_key = os.environ.get("BINANCE_API_KEY", "").strip()
    env_sec = os.environ.get("BINANCE_API_SECRET", "").strip()
    if env_key:
        cfg.setdefault("binance", {})["api_key"] = env_key
        cfg.setdefault("binance", {})["api_secret"] = env_sec
    return cfg


def make_exchange(cfg: dict):
    sec = cfg.get("binance") or {}
    key = (sec.get("api_key") or cfg.get("api_key") or "").strip()
    secret = (sec.get("api_secret") or cfg.get("api_secret") or "").strip()
    if not key or not secret:
        sys.exit("[错误] 缺少币安 API Key: 请在【API设置】里填写")
    ex = ccxt.binance({"apiKey": key, "secret": secret,
                       "enableRateLimit": True,
                       "options": {"defaultType": "spot"}})
    testnet = bool(sec.get("testnet"))
    if testnet:
        ex.set_sandbox_mode(True)          # 连币安官方测试网(假钱练手)
    tag = "测试网" if testnet else ""
    try:
        ex.fetch_balance()
        log(f"[密钥] BINANCE{tag} API 验证通过(现货)")
    except Exception as exc:
        log(f"[错误] BINANCE{tag} API 验证失败: {exc}")
        sys.exit(f"[错误] BINANCE{tag} API 验证失败(密钥错/IP白名单?): {exc}")
    return ex


def api_connection_test(exchange, cfg: dict) -> str:
    """启动自检: 验证 API 具备现货交易权限, 不产生真实成交。
    优先用币安官方"测试下单"接口(test=True, 不产生真实订单);
    兜底方案: 真实挂一笔贴近市价的限价单并立即撤单。"""
    symbol = (cfg.get("symbols") or ["DOGE/USDT"])[0]
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker.get("last") or 0)
        if price <= 0:
            return f"✗ {symbol} 行情异常, 无法自检"
        limit_price = price * 0.98                    # 贴近市价(避免价格过滤器拒绝)
        qty = 6.0 / limit_price                       # 保证名义额 >= 5U
        amount = exchange.amount_to_precision(symbol, qty)
        limit_price = exchange.price_to_precision(symbol, limit_price)
    except Exception as exc:
        return f"✗ 行情获取失败: {exc}"
    try:
        exchange.create_order(symbol, "limit", "buy", amount, limit_price,
                              params={"test": True})
        return (f"✓ API 连接成功: {symbol} 已通过订单权限测试(测试单, 未真实挂单), "
                f"具备现货交易权限")
    except Exception:
        pass
    try:
        order = exchange.create_limit_buy_order(symbol, amount, limit_price)
        exchange.cancel_order(order["id"], symbol)
        return (f"✓ API 连接成功: {symbol} 测试挂单(限价 {float(limit_price):.6f}) "
                f"已下单并立即撤单(订单 {order['id']}), 具备现货交易权限")
    except Exception as exc:
        return f"✗ API 交易权限测试失败: {exc}"

# ===================== 状态持久化 =====================


def state_file(symbol: str) -> str:
    return os.path.join(BASE_DIR, f"live_state_{symbol.replace('/', '_')}.json")


def load_state(symbol: str, cfg: dict) -> dict:
    path = state_file(symbol)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
                st.setdefault("realized_pnl", 0.0)
                return st
        except Exception:
            log(f"[{symbol}] 状态文件损坏, 重建")
    return {
        "cash": cfg["alloc_per_symbol"], "qty": 0.0,
        "last_buy_price": None, "buys_in_cycle": 0, "cycle_cost": 0.0,
        "cycle_start_ts": None, "stop_loss_triggered": False,
        "realized_pnl": 0.0, "trades": [],
    }


def save_state(symbol: str, st: dict) -> None:
    st["updated_at"] = datetime.now(TZ).isoformat()
    try:
        with open(state_file(symbol), "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2, default=str)
    except OSError as exc:
        log(f"[{symbol}] 保存状态失败: {exc}")

# ===================== 断电/关机恢复自检 =====================


def reconcile(exchange, symbol: str, st: dict, cfg: dict) -> None:
    log(f"[自检] {symbol} 开始断电恢复自检 ...")
    try:
        for o in exchange.fetch_open_orders(symbol):
            exchange.cancel_order(o["id"], symbol)
            log(f"[自检] {symbol} 撤掉残留挂单 {o['id']} {o['side']} "
                f"{o['amount']} @ {o['price'] or '市价'}")
    except Exception as exc:
        log(f"[自检] {symbol} 查询/撤销挂单失败: {exc}")

    base = symbol.split("/")[0]
    try:
        bal = exchange.fetch_balance()
        qty_ex = float((bal.get(base) or {}).get("free") or 0) \
            + float((bal.get(base) or {}).get("used") or 0)
    except Exception as exc:
        log(f"[自检] {symbol} 读取余额失败: {exc} → 保守模式, 本轮只等止盈/止损")
        if st["qty"] > 0:
            st["buys_in_cycle"] = cfg["max_adds"] + 1
        return

    qty_st = float(st.get("qty") or 0)
    if abs(qty_ex - qty_st) <= 1e-8:
        log(f"[自检] {symbol} 持仓一致: {qty_ex:.6f}, 无需修复")
        return
    log(f"[自检] {symbol} 持仓不一致! 本地 {qty_st:.6f} vs 交易所 {qty_ex:.6f}"
        f" → 以交易所为准")
    old_cost = float(st.get("cycle_cost") or 0)
    st["qty"] = qty_ex
    if qty_ex <= 0:
        log(f"[自检] {symbol} 交易所已空仓 → 重置本轮")
        st["cycle_cost"] = 0.0
        st["buys_in_cycle"] = 0
        st["last_buy_price"] = None
        st["cycle_start_ts"] = None
        st["cash"] += old_cost
        return

    since = int(st.get("cycle_start_ts") or 0) or None
    try:
        trades = exchange.fetch_my_trades(symbol, since=since, limit=500)
        buys = []
        for t in sorted(trades, key=lambda x: x.get("timestamp") or 0):
            if t.get("side") == "sell":
                buys = []
            else:
                buys.append(t)
        if buys:
            cost = sum(float(t.get("cost") or 0) for t in buys)
            st["cycle_cost"] = cost
            st["buys_in_cycle"] = len(buys)
            st["last_buy_price"] = float(buys[-1]["price"])
            st["qty"] = sum(float(t.get("amount") or 0) for t in buys)
            st["cash"] += old_cost - cost
            log(f"[自检] {symbol} 本轮重建: {len(buys)} 次买入, 成本 {cost:.2f}U, "
                f"最新买价 {st['last_buy_price']:.6f}")
        else:
            log(f"[自检] {symbol} 未找到买入记录 → 保守模式")
            st["buys_in_cycle"] = cfg["max_adds"] + 1
    except Exception as exc:
        log(f"[自检] {symbol} 拉取成交记录失败: {exc} → 保守模式, 本轮禁止加仓")
        st["buys_in_cycle"] = cfg["max_adds"] + 1
    log(f"[自检] {symbol} 恢复完成: 持仓 {st['qty']:.6f}, "
        f"本轮已买 {st['buys_in_cycle']} 次, 成本 {st['cycle_cost']:.2f}U")

# ===================== 行情 =====================


def fetch_price(exchange, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    price = ticker.get("last") or ticker.get("close")
    if not price:
        raise RuntimeError("行情缺少最新价")
    return float(price)


def fetch_ema60(exchange, symbol: str):
    ohlcv = sorted(exchange.fetch_ohlcv(symbol, "4h", limit=91),
                   key=lambda r: r[0])[:-1]
    closes = [float(c[4]) for c in ohlcv]
    if len(closes) < 60:
        return None
    alpha = 2.0 / 61.0
    ema = closes[0]
    for c in closes[1:]:
        ema = alpha * c + (1.0 - alpha) * ema
    return ema

# ===================== 智能推荐(白名单 + 余额分档) =====================

# 稳定币种白名单: 市值前10大主流现货(已剔除稳定币USDT/USDC与meme币,
# 市值排名随时浮动, 此为快照精选)
STABLE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT",
]


def recommend_setup(exchange, cfg: dict, balance) -> dict:
    """智能推荐: 按账户余额自动分档 + 在白名单里选最优币种。

    分档规则: 每轮预算 = 余额 x 30%(50U 起, 500U 封顶, 取 50 的整数倍),
    每单 = 每轮预算 / 10; 余额 < 60U 判定为不足(建议测试网练手)。
    选币规则: 只考虑 BTC/ETH/BNB(永不主动推荐山寨)。
      三者中已站上 4h EMA60 的选离得最近的;
      都未站上则选距 EMA60 最近的, 并注明机器人会等其站上后自动开仓。"""
    MAJORS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")
    rec = {"enough": True, "round_budget": 50.0, "buy_size": 5.0,
           "symbol": "BTC/USDT", "reason": ""}
    if balance is not None:
        if balance < 60.0:
            rec["enough"] = False
            rec["reason"] = (f"账户 USDT 余额仅 {balance:.2f}, "
                             f"不足支撑完整一轮(50U), 建议先勾选测试网用假钱练手")
            return rec
        r = max(50.0, min(500.0, (balance * 0.3) // 50 * 50))
        rec["round_budget"] = r
        rec["buy_size"] = round(r / 10.0, 2)

    gaps = []
    for sym in MAJORS:
        try:
            price = fetch_price(exchange, sym)
            ema = fetch_ema60(exchange, sym)
            if price is None or ema is None:
                continue
            gaps.append((sym, price / ema - 1.0))
        except Exception:
            continue
    if gaps:
        above = [(s, g) for s, g in gaps if g >= 0]
        if above:
            above.sort(key=lambda x: -abs(x[1]))
            sym = above[0][0]
            rec["symbol"] = sym
            rec["reason"] = f"{sym} 现价已站上 4h EMA60, 随时可开第一单"
        else:
            gaps.sort(key=lambda x: -x[1])
            sym, g = gaps[0]
            rec["symbol"] = sym
            rec["reason"] = (f"BTC/ETH/BNB 目前均未站上 4h EMA60; {sym} 距最近"
                             f"({g * 100:+.1f}%), 机器人会等其站上 EMA60 后自动开仓")
    else:
        rec["reason"] = "行情扫描失败, 默认推荐 BTC/USDT"
    return rec

# ===================== 下单(实盘) =====================


def market_buy(exchange, symbol: str, usdt: float, ref_price=None, warn_pct=0.5):
    """跨交易所市价买入: 按参考价换算基础币数量下单(留 0.1% 余量防余额不足)。"""
    ref = float(ref_price or 0)
    if ref <= 0:
        ticker = exchange.fetch_ticker(symbol)
        ref = float(ticker.get("last") or ticker.get("close") or 0)
    qty = (float(usdt) * 0.999) / ref
    amount = exchange.amount_to_precision(symbol, qty)
    order = exchange.create_market_buy_order(symbol, amount)
    res = {"id": order["id"],
           "qty": float(order.get("filled") or order.get("amount") or 0),
           "cost": float(order.get("cost") or usdt),
           "avg": float(order.get("average") or 0)}
    if ref and res["avg"]:
        slip = (res["avg"] / ref - 1.0) * 100
        tag = " ⚠ 滑点偏大!" if abs(slip) > warn_pct else ""
        log(f"[{symbol}] 滑点: 成交价 {res['avg']:.6f} vs 参考价 {ref:.6f} "
            f"({slip:+.2f}%){tag}")
    return res


def market_sell(exchange, symbol: str, qty: float, ref_price=None, warn_pct=0.5):
    amount = exchange.amount_to_precision(symbol, float(qty))
    order = exchange.create_market_sell_order(symbol, amount)
    res = {"id": order["id"],
           "qty": float(order.get("filled") or order.get("amount") or 0),
           "cost": float(order.get("cost") or 0),
           "avg": float(order.get("average") or 0)}
    if ref_price and res["avg"]:
        slip = (res["avg"] / ref_price - 1.0) * 100
        tag = " ⚠ 滑点偏大!" if abs(slip) > warn_pct else ""
        log(f"[{symbol}] 滑点: 成交价 {res['avg']:.6f} vs 参考价 {ref_price:.6f} "
            f"({slip:+.2f}%){tag}")
    return res

# ===================== BNB 手续费抵扣管理 =====================

BNB_STATE_FILE = os.path.join(BASE_DIR, "bnb_fee_state.json")


def _load_bnb_state() -> dict:
    try:
        with open(BNB_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_check_ms": 0}


def _save_bnb_state(st_bnb: dict) -> None:
    try:
        with open(BNB_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st_bnb, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def bnb_fee_setup(exchange, cfg: dict, force: bool = False) -> None:
    """BNB 抵扣: 账户几乎没有 BNB 且 USDT 充裕时买一次最小额 BNB 并开启抵扣。
    仅币安适用; Bitget 请手动开 BGB 抵扣。"""
    if bool((cfg.get("binance") or {}).get("testnet")):
        if force:
            log("[BNB] 币安测试网模式, 跳过 BNB 抵扣(测试网无此功能)")
        return
    plan = cfg.get("bnb_fee_plan") or {}
    if not plan.get("enabled"):
        return
    st_bnb = _load_bnb_state()
    now_ms = int(time.time() * 1000)
    if not force and now_ms - int(st_bnb.get("last_check_ms") or 0) < 24 * 3600 * 1000:
        return
    st_bnb["last_check_ms"] = now_ms
    _save_bnb_state(st_bnb)

    try:
        burn = exchange.sapi_get_bnb_burn_status() or {}
        burn_on = bool(burn.get("spotBNBBurn"))
    except Exception as exc:
        log(f"[BNB] 查询手续费抵扣状态失败: {exc}")
        return
    try:
        bal = exchange.fetch_balance()
        usdt_free = float((bal.get("USDT") or {}).get("free") or 0)
        bnb_free = float((bal.get("BNB") or {}).get("free") or 0)
    except Exception as exc:
        log(f"[BNB] 读取余额失败: {exc}")
        return
    try:
        bnb_price = fetch_price(exchange, "BNB/USDT")
    except Exception:
        bnb_price = 0.0
    bnb_value = bnb_free * bnb_price
    buy_threshold = float(plan.get("buy_threshold_usdt", 0.5))
    min_free = float(plan.get("min_free_usdt", 60.0))
    buy_usdt = float(plan.get("buy_once_usdt", 5.0))
    log(f"[BNB] 检查: BNB余额 {bnb_free:.4f}(≈{bnb_value:.2f}U) | "
        f"USDT可用 {usdt_free:.2f} | 抵扣开关 {'开' if burn_on else '关'}")

    bought = False
    if bnb_value < buy_threshold and usdt_free >= min_free:
        try:
            res = market_buy(exchange, "BNB/USDT", buy_usdt,
                             ref_price=bnb_price or None,
                             warn_pct=cfg.get("slippage_warn_pct", 0.5))
            log(f"[BNB] 账户无 BNB(≈{bnb_value:.2f}U), 已买入 {buy_usdt:.2f}U 的 BNB "
                f"用于抵扣手续费(仅此一次, 订单 {res['id']})")
            bought = True
        except Exception as exc:
            log(f"[BNB] 买入 BNB 失败: {exc}")
            return
    else:
        if bnb_value >= buy_threshold:
            log("[BNB] 账户已有 BNB, 不补买(不触碰你的持仓)")
        else:
            log(f"[BNB] USDT 余额不足 {min_free:.0f}U, 本次不买 BNB")

    if not burn_on and (bought or bnb_value > 0.001):
        try:
            params = {"spotBNBBurn": "true"}
            if "interestBNBBurn" in burn:
                params["interestBNBBurn"] = str(burn["interestBNBBurn"]).lower()
            exchange.sapi_post_bnb_burn(params)
            log("[BNB] 已开启 BNB 手续费抵扣: 现货费率 0.1% → 0.075%(省 25%)")
        except Exception as exc:
            log(f"[BNB] 开启抵扣失败: {exc}")

# ===================== 黑天鹅检测 =====================


def _volume_spike(candles, lookback: int = 24) -> float:
    """最后一根已收盘 1h K 线的成交量 / 前 lookback 根均量, 数据不足返回 0。"""
    if len(candles) < lookback + 2:
        return 0.0
    vols = [float(c[5]) for c in candles]
    avg = sum(vols[-1 - lookback:-1]) / lookback
    return (vols[-1] / avg) if avg > 0 else 0.0


def blackswan_check(exchange, cfg: dict, risk: dict) -> dict:
    """黑天鹅检测(市场自身信号, 三路并查):
      1) BTC 1小时闪崩(跌超阈值)  2) BTC 波动率骤升  3) 成交量异动
      (BTC 或交易币种的最后1小时成交量 > 前24小时均量的 3 倍 = 大资金进场)
      触发后暂停买入(开新仓+加仓) N 小时, 止盈/止损照常。"""
    plan = cfg.get("blackswan") or {}
    if not plan.get("enabled", True):
        return risk
    symbol = (cfg.get("symbols") or ["DOGE/USDT"])[0]
    try:
        ohlcv = sorted(exchange.fetch_ohlcv("BTC/USDT", "1h", limit=520),
                       key=lambda r: r[0])[:-1]
        closes = [float(c[4]) for c in ohlcv]
        if len(closes) < 60:
            return risk
        crash_pct = float(plan.get("btc_crash_pct", 3.0))
        spike_x = float(plan.get("vol_spike_x", 2.5))
        vol_x = float(plan.get("volume_spike_x", 3.0))
        lookback = int(plan.get("volume_lookback", 24))
        last_ret = (closes[-1] / closes[-2] - 1.0) * 100
        hourly = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        recent, base = hourly[-24:], hourly[-480:]
        recent_std = statistics.pstdev(recent) if len(recent) > 2 else 0.0
        base_std = statistics.pstdev(base) if len(base) > 30 else 0.0
        spike = (recent_std / base_std) if base_std > 1e-9 else 0.0
        btc_vol_ratio = _volume_spike(ohlcv, lookback)
        sym_vol_ratio = 0.0
        try:
            sym_candles = sorted(exchange.fetch_ohlcv(symbol, "1h", limit=60),
                                 key=lambda r: r[0])[:-1]
            sym_vol_ratio = _volume_spike(sym_candles, lookback)
        except Exception:
            pass
    except Exception as exc:
        log(f"[黑天鹅] 检测失败(跳过本轮): {exc}")
        return risk

    now_ms = int(time.time() * 1000)
    triggered = (last_ret <= -crash_pct or spike >= spike_x
                 or btc_vol_ratio >= vol_x or sym_vol_ratio >= vol_x)
    if triggered:
        if not risk.get("active"):
            if last_ret <= -crash_pct:
                reason = f"BTC 1小时暴跌 {last_ret:+.2f}%"
            elif spike >= spike_x:
                reason = f"BTC 波动率骤升 {spike:.1f} 倍"
            elif btc_vol_ratio >= vol_x:
                reason = f"BTC 成交量异动 {btc_vol_ratio:.1f} 倍"
            else:
                reason = f"{symbol} 成交量异动 {sym_vol_ratio:.1f} 倍"
            pause_hours = float(plan.get("pause_hours", 4.0))
            risk.update({"active": True, "reason": reason,
                         "until_ms": now_ms + int(pause_hours * 3600 * 1000)})
            log(f"[黑天鹅] ⚠ 检测到风险信号({reason}) → "
                f"暂停买入(开新仓+加仓) {pause_hours:.0f} 小时, 止盈/止损照常")
    elif risk.get("active") and now_ms >= risk.get("until_ms"):
        log("[黑天鹅] 避险期结束, 恢复买入")
        risk.update({"active": False, "reason": "", "until_ms": 0})
    return risk

# ===================== 成交记录 =====================


def record_trade(symbol: str, st: dict, side: str, ttype: str,
                 price: float, trade_qty: float) -> None:
    row = {
        "time": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "side": side, "type": ttype,
        "price": round(price, 8), "qty": round(trade_qty, 8),
        "pos_after": round(st["qty"], 8), "cash_after": round(st["cash"], 4),
        "equity_after": round(st["cash"] + st["qty"] * price, 4),
    }
    st["trades"].append(row)
    try:
        path = os.path.join(BASE_DIR, f"live_trades_{symbol.replace('/', '_')}.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        pass

# ===================== 一键平仓 =====================


def flatten_all(exchange, cfg: dict, states: dict) -> list:
    """一键平仓: 取消所有挂单 + 全部现单市价清仓 + 状态重置。"""
    msgs = []
    for s in cfg["symbols"]:
        st = states.get(s) or {}
        try:
            for o in exchange.fetch_open_orders(s):
                exchange.cancel_order(o["id"], s)
                msgs.append(f"[平仓] {s} 撤掉挂单 {o['id']} {o['side']}")
        except Exception as exc:
            msgs.append(f"[平仓] {s} 撤单失败: {exc}")
        qty_held = float(st.get("qty") or 0)
        try:
            bal = exchange.fetch_balance()
            base = s.split("/")[0]
            qty_held = float((bal.get(base) or {}).get("free") or 0) \
                + float((bal.get(base) or {}).get("used") or 0)
        except Exception:
            pass
        if qty_held > 0:
            price = 0.0
            try:
                price = fetch_price(exchange, s)
            except Exception:
                pass
            proceeds = 0.0
            try:
                if qty_held * price >= cfg.get("min_notional", 5.0):
                    res = market_sell(exchange, s, qty_held, ref_price=price or None,
                                      warn_pct=cfg.get("slippage_warn_pct", 0.5))
                    proceeds = res["cost"]
                    msgs.append(f"[平仓] {s} 现单已清仓 {res['qty']:.6f}, "
                                f"得 {proceeds:.4f}U (订单 {res['id']})")
                else:
                    msgs.append(f"[平仓] {s} 持仓市值低于 5U, 留作灰尘({qty_held:.8f})")
            except Exception as exc:
                msgs.append(f"[平仓] {s} 清仓失败(请手动处理!): {exc}")
                continue
            round_cost = float(st.get("cycle_cost") or 0)
            st["realized_pnl"] = float(st.get("realized_pnl") or 0) + proceeds - round_cost
            st["cash"] = float(st.get("cash") or 0) + proceeds
            st["qty"] = 0.0
            st["cycle_cost"] = 0.0
            st["buys_in_cycle"] = 0
            st["last_buy_price"] = None
            st["cycle_start_ts"] = None
            st["stop_loss_triggered"] = False
            try:
                record_trade(s, st, "SELL", "一键平仓", price, qty_held)
            except Exception:
                pass
            save_state(s, st)
        else:
            msgs.append(f"[平仓] {s} 无持仓")
    return msgs

# ===================== 策略步进(纯实盘) =====================


def step_symbol(exchange, symbol: str, st: dict, cfg: dict,
                available_usdt: float, flat_count: int,
                risk_paused: bool = False):
    """单币种一个周期检查; 返回最新价(供账户概览)。"""
    stop_line = cfg["alloc_per_symbol"] * (1.0 - cfg["stop_loss_pct"])
    price = fetch_price(exchange, symbol)
    ema = fetch_ema60(exchange, symbol)
    qty, cash = float(st["qty"]), float(st["cash"])

    if st.get("stop_loss_triggered"):
        return price

    if qty <= 0:
        if ema is None:
            log(f"[{symbol}] 价格 {price:.6f} | EMA60(4h) 预热中 | 空仓等待")
            return price
        if price > ema:
            if risk_paused:
                log(f"[{symbol}] 黑天鹅避险期, 暂停开新仓(等待风险解除)")
                return price
            budget = min(cfg["alloc_per_symbol"],
                         available_usdt / max(1, flat_count))
            spend = min(cfg["initial_buy_usdt"], cash, budget)
            if spend < cfg["min_notional"] or available_usdt < cfg["min_notional"]:
                log(f"[{symbol}] 价格 {price:.6f} | EMA60(4h) {ema:.6f} | "
                    f"可用USDT不足(可用 {available_usdt:.2f}, 该轮预算 {budget:.2f}), "
                    f"暂停开仓")
                return price
            try:
                res = market_buy(exchange, symbol, spend, ref_price=price,
                                 warn_pct=cfg.get("slippage_warn_pct", 0.5))
                log(f"[{symbol}] 价格 {price:.6f} | 实盘买入 {spend:.2f}U "
                    f"成交 {res['qty']:.6f} @ {res['avg']:.6f} 订单 {res['id']}")
                st["qty"] += res["qty"]
                st["cash"] -= res["cost"]
                st["cycle_cost"] = res["cost"]
                st["buys_in_cycle"] = 1
                st["last_buy_price"] = res["avg"] or price
                st["cycle_start_ts"] = int(time.time() * 1000)
                record_trade(symbol, st, "BUY", "初始买入", price, res["qty"])
                save_state(symbol, st)
            except Exception as exc:
                log(f"[{symbol}] 买入失败(稍后重试): {exc}")
                return price
        else:
            log(f"[{symbol}] 价格 {price:.6f} | EMA60(4h) {ema:.6f} | 空仓, "
                f"等待价格 > EMA60")
        return price

    # ---- 持仓中 ----
    avg_cost = float(st["cycle_cost"]) / qty
    tp_price = avg_cost * (1.0 + cfg["take_profit"])
    add_price = (float(st["last_buy_price"]) or price) * (1.0 - cfg["drop_trigger"])
    bullets_done = st["buys_in_cycle"] >= cfg["max_adds"] + 1

    if not bullets_done and price <= add_price:
        if risk_paused:
            log(f"[{symbol}] 黑天鹅避险期, 暂停加仓(等待风险解除)")
            return price
        spend = min(cfg["add_buy_usdt"], available_usdt, max(cash, 0.0))
        if spend < cfg["min_notional"]:
            log(f"[{symbol}] 可用USDT不足(可用 {available_usdt:.2f}), 跳过加仓")
            return price
        try:
            res = market_buy(exchange, symbol, spend, ref_price=price,
                             warn_pct=cfg.get("slippage_warn_pct", 0.5))
            log(f"[{symbol}] 价格 {price:.6f} | 实盘加仓 {spend:.2f}U "
                f"成交 {res['qty']:.6f} @ {res['avg']:.6f} 订单 {res['id']} "
                f"(本轮第 {st['buys_in_cycle'] + 1}/10 次)")
            st["qty"] += res["qty"]
            st["cash"] -= res["cost"]
            st["cycle_cost"] += res["cost"]
            st["buys_in_cycle"] += 1
            st["last_buy_price"] = res["avg"] or price
            record_trade(symbol, st, "BUY", "加仓", price, res["qty"])
            save_state(symbol, st)
        except Exception as exc:
            log(f"[{symbol}] 加仓失败(稍后重试): {exc}")
            return price
    elif price >= tp_price:
        sold = qty
        round_cost = float(st["cycle_cost"])
        try:
            res = market_sell(exchange, symbol, sold, ref_price=price,
                              warn_pct=cfg.get("slippage_warn_pct", 0.5))
            proceeds = res["cost"]
            log(f"[{symbol}] 价格 {price:.6f} | 实盘止盈卖出 {res['qty']:.6f} "
                f"得 {proceeds:.4f}U 订单 {res['id']}")
            profit = proceeds - round_cost
            st["realized_pnl"] = float(st.get("realized_pnl") or 0) + profit
            st["cash"] += proceeds
            st["qty"] = 0.0
            st["cycle_cost"] = 0.0
            st["buys_in_cycle"] = 0
            st["last_buy_price"] = None
            st["cycle_start_ts"] = None
            record_trade(symbol, st, "SELL", "止盈平仓", price, sold)
            save_state(symbol, st)
            log(f"[{symbol}] 本轮止盈完成: 本轮盈亏 {profit:+.2f}U, "
                f"累计已实现盈亏 {st['realized_pnl']:+.2f}U")
        except Exception as exc:
            log(f"[{symbol}] 止盈卖出失败(稍后重试): {exc}")
            return price
    elif bullets_done and cash + qty * price < stop_line:
        sold = qty
        round_cost = float(st["cycle_cost"])
        try:
            res = market_sell(exchange, symbol, sold, ref_price=price,
                              warn_pct=cfg.get("slippage_warn_pct", 0.5))
            proceeds = res["cost"]
            log(f"[{symbol}] 价格 {price:.6f} | 实盘止损清仓 {res['qty']:.6f} "
                f"得 {proceeds:.4f}U 订单 {res['id']}")
            profit = proceeds - round_cost
            st["realized_pnl"] = float(st.get("realized_pnl") or 0) + profit
            st["cash"] += proceeds
            st["qty"] = 0.0
            st["cycle_cost"] = 0.0
            st["buys_in_cycle"] = 0
            st["last_buy_price"] = None
            st["stop_loss_triggered"] = True
            record_trade(symbol, st, "SELL", "止损清仓", price, sold)
            save_state(symbol, st)
            log(f"[{symbol}] 本轮止损完成: 本轮盈亏 {profit:+.2f}U, "
                f"该币种永久停止(可删状态文件重新开始)")
        except Exception as exc:
            log(f"[{symbol}] 止损卖出失败(稍后重试): {exc}")
            return price
    else:
        if bullets_done:
            wait = (f"等待: 涨至 {tp_price:.6f} 止盈, 或总资产跌破 {stop_line:.1f} "
                    f"止损(子弹已打光)")
        else:
            wait = (f"等待: 跌至 {add_price:.6f} 以下加仓(本轮已买 "
                    f"{st['buys_in_cycle']}/10 次), 或涨至 {tp_price:.6f} 止盈")
        log(f"[{symbol}] 价格 {price:.6f} | EMA60(4h) "
            f"{ema if ema is not None else float('nan'):.6f} | 持仓 {qty:.2f}, "
            f"均价 {avg_cost:.6f}, 总资产 {cash + qty * price:.2f}U | {wait}")
    return price

# ===================== 实时账户概览 =====================


def account_overview(cfg: dict, states: dict, prices: dict, bal=None) -> None:
    """实时打印: USDT 余额 / 总资产 / 已实现盈亏 / 未实现盈亏。"""
    if bal is None:
        log("[账户] 余额读取失败, 跳过概览")
        return
    realized = sum(float(st.get("realized_pnl") or 0) for st in states.values())
    unreal = sum(float(st.get("qty") or 0) * float(prices.get(s) or 0)
                 - float(st.get("cycle_cost") or 0)
                 for s, st in states.items() if float(st.get("qty") or 0) > 0)
    usdt_free = float((bal.get("USDT") or {}).get("free") or 0)
    usdt_locked = float((bal.get("USDT") or {}).get("used") or 0)
    base_value = 0.0
    for s in cfg["symbols"]:
        base = s.split("/")[0]
        q = float((bal.get(base) or {}).get("free") or 0) \
            + float((bal.get(base) or {}).get("used") or 0)
        base_value += q * float(prices.get(s) or 0)
    total = usdt_free + usdt_locked + base_value
    log(f"[账户] USDT可用 {usdt_free:.2f} | 冻结 {usdt_locked:.2f} | "
        f"持仓市值 {base_value:.2f} | 总资产 {total:.2f} | "
        f"已实现盈亏 {realized:+.2f} | 未实现盈亏 {unreal:+.2f}")

# ===================== 主流程 =====================


def main() -> None:
    cfg = load_config()
    log("=" * 72)
    log(" 币安现货 DCA 实盘机器人启动(纯实盘)")
    log(f" 币种: {cfg['symbols']} | "
        f"每币预算 {cfg['alloc_per_symbol']}U | 检查间隔 {cfg['check_interval_sec']}s")
    log(" ⚠️ 真实下单, 请确认已阅读《实盘使用说明.txt》")

    if os.path.exists(STOP_FILE):
        log("检测到 STOP_REAL.txt 紧急刹车文件, 拒绝启动。")
        sys.exit(0)

    if not _acquire_lock():
        log("检测到另一个机器人实例正在运行, 本次启动退出。")
        sys.exit(0)

    exchange = make_exchange(cfg)
    test_msg = api_connection_test(exchange, cfg)
    log(test_msg)
    if test_msg.startswith("✗"):
        _release_lock()
        sys.exit(test_msg)

    states = {s: load_state(s, cfg) for s in cfg["symbols"]}
    for s in cfg["symbols"]:
        try:
            reconcile(exchange, s, states[s], cfg)
        except Exception as exc:
            log(f"[自检] {s} 失败: {exc}")
        save_state(s, states[s])
    bnb_fee_setup(exchange, cfg, force=True)

    risk = {"active": False, "reason": "", "until_ms": 0}
    try:
        while True:
            if os.path.exists(STOP_FILE):
                log("检测到 STOP_REAL.txt, 保存状态并停止(不再开新单)。")
                break
            try:
                bal = exchange.fetch_balance()
                available = float((bal.get("USDT") or {}).get("free") or 0)
            except Exception as exc:
                bal = None
                available = 0.0
                log(f"[账户] 实时余额读取失败: {exc}")
            flat_count = sum(1 for s in cfg["symbols"]
                             if not states[s].get("stop_loss_triggered")
                             and float(states[s].get("qty") or 0) <= 0)
            bnb_fee_setup(exchange, cfg)
            risk = blackswan_check(exchange, cfg, risk)

            prices = {}
            for s in cfg["symbols"]:
                try:
                    p = step_symbol(exchange, s, states[s], cfg, available, flat_count,
                                    risk_paused=risk.get("active", False))
                    if p:
                        prices[s] = p
                except Exception as exc:
                    log(f"[{s}] 本轮检查异常: {exc}")
            account_overview(cfg, states, prices, bal)
            for s in cfg["symbols"]:
                save_state(s, states[s])
            time.sleep(cfg["check_interval_sec"])
    except KeyboardInterrupt:
        log("收到中断信号, 保存状态退出。")
    finally:
        for s in cfg["symbols"]:
            save_state(s, states[s])
        _release_lock()


def _acquire_lock() -> bool:
    """单实例锁: Windows 文件锁(进程退出时内核自动释放, 不会因 PID 复用误判)。"""
    global _lock_handle
    f = None
    try:
        f = open(LOCK_FILE, "a+")
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        _lock_handle = f
        return True
    except (OSError, IOError):
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        return False


def _release_lock() -> None:
    global _lock_handle
    if _lock_handle is not None:
        try:
            _lock_handle.seek(0)
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except (OSError, IOError):
            pass
        try:
            _lock_handle.close()
        except (OSError, IOError):
            pass
        _lock_handle = None


if __name__ == "__main__":
    main()
