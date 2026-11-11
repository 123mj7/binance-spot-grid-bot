# -*- coding: utf-8 -*-
"""
live_suite.py —— 币安现货 DCA 实盘机器人(一体机, 仅币安)
==========================================================
监控台 + 策略引擎合一:
  - 启动流程: 读币安密钥 → 真实"挂单→立即撤单"自检 API 交易权限(不成交)
    → 断电恢复自检(撤残留挂单/对齐真实持仓) → 进入策略循环
  - 支持币安【真实盘】与币安【测试网】(测试网 = 官方模拟环境, 假钱练手)
  - 实时账户概览: USDT余额/总资产/已实现盈亏/未实现盈亏 + 币种明细 + 日志
  - 按钮:
      [API设置]    填 Key/Secret + 选币种 + 预算; 保存即自动启动
      [启动·恢复]  暂停后恢复自动校准挂单与现单; 首次启动即自检
      [暂停策略]   只停交易决策, 不平仓不撤单
      [一键平仓]   终止策略 + 撤全部挂单 + 现单全部市价清仓
  - 关闭窗口 = 缩到系统托盘, 策略继续; 托盘菜单: 打开/暂停/一键平仓/退出

依赖: pip install ccxt pystray pillow
打包: pyinstaller --onefile --windowed --hidden-import pystray._win32 live_suite.py
"""

from __future__ import annotations

import json
import msvcrt
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog

import ccxt                      # noqa: F401
import binance_live_bot as botlib

try:
    from PIL import Image, ImageDraw
    import pystray
    TRAY_OK = True
except ImportError:
    TRAY_OK = False

TZ = timezone(timedelta(hours=8))

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_LOCK = os.path.join(BASE_DIR, "live_suite.lock")

TESTNET_URL = "https://testnet.binance.vision"

# ===================== 单实例(文件锁, 进程退出自动释放) =====================

_suite_lock_handle = None


def acquire_suite_lock() -> bool:
    """文件锁: 进程退出时内核自动释放, 不会因 PID 复用误判。"""
    global _suite_lock_handle
    f = None
    try:
        f = open(SUITE_LOCK, "a+")
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        _suite_lock_handle = f
        return True
    except (OSError, IOError):
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        return False


def release_suite_lock() -> None:
    global _suite_lock_handle
    if _suite_lock_handle is not None:
        try:
            _suite_lock_handle.seek(0)
            msvcrt.locking(_suite_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except (OSError, IOError):
            pass
        try:
            _suite_lock_handle.close()
        except (OSError, IOError):
            pass
        _suite_lock_handle = None

# ===================== 策略线程 =====================


class BotLoop(threading.Thread):
    def __init__(self, cfg: dict):
        super().__init__(daemon=True, name="bot-loop")
        self.cfg = cfg
        self.stop_ev = threading.Event()
        self.pause_ev = threading.Event()
        self.pause_ev.set()
        self.recalibrate = False
        self.last_prices = {}
        self.last_bal = None
        self.error = None

    def run(self) -> None:
        try:
            if not botlib._acquire_lock():
                self.error = "另一个机器人实例正在运行(可能开着实盘启动.bat), 本程序仅监控"
                botlib.log(self.error)
                return
            try:
                self.exchange = botlib.make_exchange(self.cfg)
            except SystemExit as exc:
                self.error = f"启动失败: {exc}"
                botlib._release_lock()
                return
            test_msg = botlib.api_connection_test(self.exchange, self.cfg)
            botlib.log(test_msg)
            if test_msg.startswith("✗"):
                self.error = test_msg
                botlib._release_lock()
                return
            self.states = {s: botlib.load_state(s, self.cfg) for s in self.cfg["symbols"]}
            self.risk = {"active": False, "reason": "", "until_ms": 0}
            for s in self.cfg["symbols"]:
                try:
                    botlib.reconcile(self.exchange, s, self.states[s], self.cfg)
                except Exception as exc:
                    botlib.log(f"[自检] {s} 失败: {exc}")
                botlib.save_state(s, self.states[s])
            botlib.bnb_fee_setup(self.exchange, self.cfg, force=True)

            while not self.stop_ev.is_set():
                self.pause_ev.wait()          # 暂停时挂起: 不产生任何交易动作
                if self.stop_ev.is_set():
                    break
                if self.recalibrate:
                    self.recalibrate = False
                    botlib.log("[校准] 恢复启动: 自动校准挂单与持仓 ...")
                    for s in self.cfg["symbols"]:
                        try:
                            botlib.reconcile(self.exchange, s, self.states[s], self.cfg)
                            botlib.save_state(s, self.states[s])
                        except Exception as exc:
                            botlib.log(f"[校准] {s} 失败: {exc}")
                    botlib.bnb_fee_setup(self.exchange, self.cfg, force=True)
                if os.path.exists(botlib.STOP_FILE):
                    botlib.log("检测到 STOP_REAL.txt, 保存状态并停止。")
                    break
                try:
                    bal = self.exchange.fetch_balance()
                    available = float((bal.get("USDT") or {}).get("free") or 0)
                except Exception as exc:
                    bal = None
                    available = 0.0
                    botlib.log(f"[账户] 实时余额读取失败: {exc}")
                flat_count = sum(1 for s in self.cfg["symbols"]
                                 if not self.states[s].get("stop_loss_triggered")
                                 and float(self.states[s].get("qty") or 0) <= 0)
                botlib.bnb_fee_setup(self.exchange, self.cfg)
                self.risk = botlib.blackswan_check(self.exchange, self.cfg, self.risk)

                prices = {}
                for s in self.cfg["symbols"]:
                    try:
                        p = botlib.step_symbol(self.exchange, s, self.states[s],
                                               self.cfg, available, flat_count,
                                               risk_paused=self.risk.get("active", False))
                        if p:
                            prices[s] = p
                    except Exception as exc:
                        botlib.log(f"[{s}] 本轮检查异常: {exc}")
                botlib.account_overview(self.cfg, self.states, prices, bal)
                self.last_prices = prices
                self.last_bal = bal
                for s in self.cfg["symbols"]:
                    botlib.save_state(s, self.states[s])
                self.stop_ev.wait(self.cfg["check_interval_sec"])
        finally:
            try:
                for s in self.cfg["symbols"]:
                    botlib.save_state(s, self.states[s])
            except Exception:
                pass
            botlib._release_lock()

    def stop(self) -> None:
        self.pause_ev.set()
        self.stop_ev.set()

# ===================== GUI =====================


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = botlib.load_config()
        self.loop = None
        self.icon = None
        root.title("币安现货 DCA 实盘机器人(一体机)")
        root.geometry("680x760")
        root.resizable(True, True)

        self.banner = tk.Label(root, text="检查中...", font=("Microsoft YaHei", 13, "bold"),
                               bg="#eeeeee", pady=6)
        self.banner.pack(fill="x")

        ov = tk.LabelFrame(root, text=" 实时账户概览 ", font=("Microsoft YaHei", 10))
        ov.pack(fill="x", padx=10, pady=6)
        self.ov_labels = {}
        grid = [("usdt_free", "USDT可用"), ("usdt_lock", "USDT冻结"),
                ("mv", "持仓市值"), ("total", "总资产"),
                ("realized", "已实现盈亏"), ("unreal", "未实现盈亏")]
        for i, (key, text) in enumerate(grid):
            r, c = divmod(i, 3)
            frame = tk.Frame(ov)
            frame.grid(row=r, column=c, sticky="w", padx=10, pady=3)
            tk.Label(frame, text=text + ":", font=("Microsoft YaHei", 10)).pack(side="left")
            lab = tk.Label(frame, text="--", font=("Consolas", 12, "bold"))
            lab.pack(side="left")
            self.ov_labels[key] = lab
        self.quote_lab = tk.Label(ov, text="", fg="#888888", anchor="w",
                                  font=("Microsoft YaHei", 9))
        self.quote_lab.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 4))

        symf = tk.LabelFrame(root, text=" 币种明细 ", font=("Microsoft YaHei", 10))
        symf.pack(fill="x", padx=10, pady=6)
        self.symf = symf
        self.sym_rows = {}
        self.rebuild_symbol_rows()

        logf = tk.LabelFrame(root, text=" 机器人日志(live_trade_real.txt 最新15行) ",
                             font=("Microsoft YaHei", 10))
        logf.pack(fill="both", expand=True, padx=10, pady=6)
        self.log_box = scrolledtext.ScrolledText(logf, height=14, wrap="none",
                                                 font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, padx=6, pady=4)

        btns = tk.Frame(root)
        btns.pack(fill="x", padx=10, pady=8)
        tk.Button(btns, text="API设置", command=self.open_api_dialog,
                  font=("Microsoft YaHei", 10), bg="#e3f2fd").pack(side="left", padx=4)
        tk.Button(btns, text="启动·恢复", command=self.on_start,
                  font=("Microsoft YaHei", 10), bg="#e8f5e9").pack(side="left", padx=4)
        tk.Button(btns, text="暂停策略", command=self.on_pause,
                  font=("Microsoft YaHei", 10), bg="#fff3e0").pack(side="left", padx=4)
        tk.Button(btns, text="一键平仓", command=self.on_flatten,
                  font=("Microsoft YaHei", 10), bg="#ffcdd2").pack(side="left", padx=4)
        tk.Button(btns, text="刷新显示", command=self._refresh_all,
                  font=("Microsoft YaHei", 10)).pack(side="left", padx=4)
        tk.Button(btns, text="最小化到托盘", command=self.hide_to_tray,
                  font=("Microsoft YaHei", 10)).pack(side="right", padx=4)

        self._tray_first_hint = True
        self._setup_tray()
        self._refresh_all()
        self.auto_refresh()
        root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)

    # ---------- 托盘 ----------
    def _setup_tray(self):
        if not TRAY_OK:
            return
        try:
            img = Image.new("RGB", (64, 64), "#1b5e20")
            d = ImageDraw.Draw(img)
            d.ellipse((8, 8, 56, 56), fill="#c8e6c9")
            d.ellipse((20, 20, 44, 44), fill="#1b5e20")
            self.icon = pystray.Icon(
                "doge_live", img, "DOGE 实盘机器人",
                menu=pystray.Menu(
                    pystray.MenuItem("打开监控台", self.show_window, default=True),
                    pystray.MenuItem("暂停策略", lambda: self.root.after(0, self.on_pause)),
                    pystray.MenuItem("一键平仓", lambda: self.root.after(0, self.on_flatten)),
                    pystray.MenuItem("彻底退出", self.quit_app)))
            threading.Thread(target=self.icon.run, daemon=True).start()
        except Exception:
            self.icon = None

    def hide_to_tray(self):
        if self.icon is not None:
            if self._tray_first_hint:
                self._tray_first_hint = False
                if not messagebox.askyesno("提示", "窗口将最小化到系统托盘, 机器人继续运行。\n"
                                            "托盘图标右键可重新打开/停止/退出。\n确定继续吗?"):
                    return
            self.root.withdraw()
        else:
            self.quit_app()

    def show_window(self):
        self.root.after(0, self._do_show)

    def _do_show(self):
        self.root.deiconify()
        self.root.lift()

    def quit_app(self):
        if self.loop and self.loop.is_alive():
            self.loop.stop()
            self.loop.join(timeout=5)
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:
                pass
        release_suite_lock()
        self.root.after(0, self.root.destroy)

    # ---------- 启动/暂停/一键平仓 ----------
    def on_start(self):
        if self.loop and self.loop.is_alive():
            if not self.loop.pause_ev.is_set():
                self.loop.recalibrate = True
                self.loop.pause_ev.set()
                messagebox.showinfo("启动", "策略已恢复!\n恢复前自动校准: 检查挂单与现单, "
                                            "有问题自动对齐交易所真实状态(见日志[校准])。")
            else:
                messagebox.showinfo("启动", "策略已在运行中")
            self._refresh_all()
            return
        self.loop = BotLoop(self.cfg)
        self.loop.start()
        messagebox.showinfo("启动", "策略已启动:\n1) 真实挂单→撤单自检 API 交易权限\n"
                                    "2) 断电恢复自检(对齐挂单/持仓)\n3) 进入策略循环")
        self._refresh_all()

    def on_pause(self):
        if not (self.loop and self.loop.is_alive()):
            messagebox.showinfo("暂停", "策略未在运行")
            return
        self.loop.pause_ev.clear()
        botlib.log("策略已暂停(不撤单、不平仓, 已有持仓与挂单保持原样)。")
        messagebox.showinfo("暂停", "策略已暂停:\n"
                                    "- 不再开新仓/加仓/止盈/止损\n"
                                    "- 已有持仓与挂单保持原样\n"
                                    "- 点【启动·恢复】自动校准后继续")
        self._refresh_all()

    def on_flatten(self):
        if not messagebox.askyesno("一键平仓", "确定执行一键平仓吗?\n\n"
                                              "将: 终止策略 + 取消所有挂单 + 全部现单市价清仓。\n"
                                              "⚠ 立即真实卖出所有持仓!"):
            return
        if self.loop and self.loop.is_alive():
            self.loop.stop()
            self.loop.join(timeout=10)
        botlib.log("用户触发一键平仓: 终止策略 + 撤挂单 + 清现单 ...")

        def worker():
            try:
                ex = getattr(self.loop, "exchange", None)
                if ex is None:
                    ex = botlib.make_exchange(self.cfg)
                states = {s: botlib.load_state(s, self.cfg) for s in self.cfg["symbols"]}
                msgs = botlib.flatten_all(ex, self.cfg, states)
                for m in msgs:
                    botlib.log(m)
                self.root.after(0, lambda: (messagebox.showinfo(
                    "一键平仓完成", "\n".join(msgs)), self._refresh_all()))
            except SystemExit as exc:
                self.root.after(0, lambda: messagebox.showerror("平仓失败", str(exc)))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("平仓失败", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- 币种明细行 ----------
    def rebuild_symbol_rows(self):
        for child in self.symf.winfo_children():
            child.destroy()
        self.sym_rows = {}
        for i, sym in enumerate(self.cfg["symbols"]):
            tk.Label(self.symf, text=sym, font=("Microsoft YaHei", 10, "bold"),
                     width=12, anchor="w").grid(row=i, column=0, padx=6, pady=2)
            lab = tk.Label(self.symf, text="--", anchor="w", font=("Consolas", 9))
            lab.grid(row=i, column=1, sticky="w", padx=4)
            self.sym_rows[sym] = lab

    # ---------- API 设置窗口(仅币安, 智能推荐) ----------
    def open_api_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("币安 API 设置(真实盘 / 测试网 · 智能推荐)")
        dlg.geometry("560x520")
        dlg.resizable(False, False)
        dlg.grab_set()

        bcfg = self.cfg.get("binance", {}) or {}
        rec_alloc = [float(self.cfg.get("alloc_per_symbol", 50))]

        bin_frame = tk.LabelFrame(dlg, text=" 币安 Binance API ", font=("Microsoft YaHei", 9))
        bin_frame.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(12, 2))
        bin_frame.columnconfigure(1, weight=1)
        tk.Label(bin_frame, text="API Key:", font=("Microsoft YaHei", 9)).grid(
            row=0, column=0, sticky="w", padx=8, pady=2)
        bin_key = tk.Entry(bin_frame, show="*", width=46)
        bin_key.grid(row=0, column=1, padx=6, pady=2)
        bin_key.insert(0, bcfg.get("api_key", ""))
        tk.Label(bin_frame, text="API Secret:", font=("Microsoft YaHei", 9)).grid(
            row=1, column=0, sticky="w", padx=8, pady=2)
        bin_sec = tk.Entry(bin_frame, show="*", width=46)
        bin_sec.grid(row=1, column=1, padx=6, pady=2)
        bin_sec.insert(0, bcfg.get("api_secret", ""))
        bin_testnet = tk.BooleanVar(value=bool(bcfg.get("testnet")))
        tk.Checkbutton(bin_frame, text="这是币安测试网(Testnet)密钥 — 官方模拟环境, 假钱练手",
                       variable=bin_testnet, font=("Microsoft YaHei", 10)).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        tk.Label(bin_frame, text="测试网注册网址(可选中复制):", font=("Microsoft YaHei", 9)).grid(
            row=3, column=0, sticky="w", padx=8, pady=2)
        url_var = tk.StringVar(value=TESTNET_URL)
        tk.Entry(bin_frame, textvariable=url_var, state="readonly",
                 readonlybackground="white", width=46).grid(row=3, column=1, padx=6, pady=2)
        tk.Label(bin_frame, text="测试网方法: 打开上面网址 → 用 GitHub 登录 → 生成测试网密钥 → 填上方并勾选『测试网』。\n"
                                 "真实盘: 币安官网创建 API(只勾现货交易、绝不勾提现) → 填上方, 不勾『测试网』。",
                 fg="#666666", justify="left", font=("Microsoft YaHei", 8)).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4))

        bnb_var = tk.BooleanVar(
            value=bool((self.cfg.get("bnb_fee_plan") or {}).get("enabled")))
        tk.Checkbutton(dlg, text="余额充裕时自动补买 BNB 抵扣手续费(仅真实盘, 费率0.1%→0.075%)",
                       variable=bnb_var, font=("Microsoft YaHei", 10)).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(6, 2))

        tk.Label(dlg, text="交易币种(稳定币白名单):", font=("Microsoft YaHei", 10)).grid(
            row=2, column=0, sticky="w", padx=12, pady=2)
        sym_var = tk.StringVar(value=(self.cfg.get("symbols") or ["BTC/USDT"])[0])
        sym_combo = ttk.Combobox(dlg, textvariable=sym_var, state="readonly",
                                 values=botlib.STABLE_SYMBOLS, width=47)
        sym_combo.grid(row=2, column=1, padx=6, pady=2, sticky="w")
        tk.Label(dlg, text=f"已内置 {len(botlib.STABLE_SYMBOLS)} 个主流稳定币种(已剔除野鸡币), "
                           f"点【智能推荐】自动帮你选",
                 fg="#888888", font=("Microsoft YaHei", 8)).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=12)

        tip = ("⚠ 密钥明文保存在本机 config_real.json。请务必:\n"
               "  1. 只勾『现货交易』权限, 绝不勾『提现』\n"
               "  2. 建议开启 IP 白名单\n  3. 不把密钥发给任何人\n"
               "  [测试连接] 会真实挂一笔低价限价单并立即撤单(不会成交), 验证交易权限")
        tk.Label(dlg, text=tip, justify="left", fg="#b71c1c",
                 font=("Microsoft YaHei", 9)).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 2))

        status_lab = tk.Label(dlg, text="", font=("Microsoft YaHei", 9))
        status_lab.grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=4)

        def do_test():
            key = bin_key.get().strip()
            sec = bin_sec.get().strip()
            if not key or not sec:
                status_lab.config(text="✗ 请先填写币安 API Key 和 Secret", fg="#b71c1c")
                return
            status_lab.config(text="正在币安真实挂单→撤单测试交易权限...", fg="black")
            result = {}

            def worker():
                try:
                    ex = ccxt.binance({"apiKey": key, "secret": sec,
                                       "enableRateLimit": True,
                                       "options": {"defaultType": "spot"}})
                    if bin_testnet.get():
                        ex.set_sandbox_mode(True)
                    ex.fetch_balance()
                    cfg2 = dict(self.cfg)
                    cfg2["symbols"] = [sym_var.get().strip().upper() or "BTC/USDT"]
                    result["msg"] = botlib.api_connection_test(ex, cfg2)
                except SystemExit as exc:
                    result["msg"] = f"✗ {exc}"
                except Exception as exc:
                    result["msg"] = f"✗ 连接失败: {exc}"

            def done():
                try:
                    if dlg.winfo_exists():
                        msg = result.get("msg")
                        if not msg:
                            dlg.after(200, done)
                            return
                        status_lab.config(text=msg,
                                          fg="#1b5e20" if msg.startswith("✓")
                                          else "#b71c1c")
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()
            dlg.after(200, done)

        def do_save():
            sym = sym_var.get().strip().upper()
            if not sym.endswith("/USDT"):
                messagebox.showerror("错误", "请选择白名单中的币种")
                return
            alloc = float(rec_alloc[0])
            buy = round(alloc / 10.0, 2)
            cfg2 = {}
            if os.path.exists(botlib.CONFIG_FILE):
                try:
                    with open(botlib.CONFIG_FILE, "r", encoding="utf-8") as f:
                        cfg2 = json.load(f)
                except Exception:
                    cfg2 = {}
            cfg2.update({
                "binance": {"api_key": bin_key.get().strip(),
                            "api_secret": bin_sec.get().strip(),
                            "testnet": bool(bin_testnet.get())},
                "symbols": [sym],
                "alloc_per_symbol": alloc,
                "initial_buy_usdt": buy,
                "add_buy_usdt": buy,
                "bnb_fee_plan": {**cfg2.get("bnb_fee_plan", {}),
                                 "enabled": bool(bnb_var.get())},
            })
            try:
                with open(botlib.CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(cfg2, f, ensure_ascii=False, indent=2)
            except OSError as exc:
                messagebox.showerror("错误", f"保存失败: {exc}")
                return
            self.cfg = botlib.load_config()
            self.rebuild_symbol_rows()
            if self.loop and self.loop.is_alive():
                self.loop.stop()
                self.loop.join(timeout=10)
                self.loop = None
            bsec = self.cfg.get("binance") or {}
            k = (bsec.get("api_key") or "").strip()
            s = (bsec.get("api_secret") or "").strip()
            if k and s:
                tag = "测试网(假钱)" if bsec.get("testnet") else "真实盘"
                self.loop = BotLoop(self.cfg)
                self.loop.start()
                msg = (f"配置已保存, 已自动启动币安{tag}!\n"
                       f"币种 {sym} | 每轮预算 {alloc:.0f}U | 每单 {buy:.2f}U | 止损线 "
                       f"{alloc * 0.85:.0f}U。\n启动流程: 订单权限自检 → 断电恢复自检 → 策略循环。")
            else:
                msg = "配置已保存。请填写币安 API Key 后点【启动·恢复】。"
            self._refresh_all()
            messagebox.showinfo("已保存", msg)
            dlg.destroy()

        def do_recommend():
            key = bin_key.get().strip()
            sec = bin_sec.get().strip()
            if not bin_testnet.get() and (not key or not sec):
                status_lab.config(text="✗ 智能推荐需要密钥: 请先填 API Key(或勾选测试网)",
                                  fg="#b71c1c")
                return
            sim_bal = None
            sim_label = "测试网(假钱)"
            if bin_testnet.get():
                # 测试网的钱是假的, 不参与分档; 让用户输入想模拟的真实余额
                ans = simpledialog.askstring(
                    "测试网模拟余额",
                    "测试网的钱是假的, 不参与余额分档。\n"
                    "输入你想模拟的真实余额(USDT), 按它来分档:\n"
                    "例如 500 → 每轮150U; 800 → 每轮200U\n"
                    "(留空或点取消 = 使用标准 50U 档)",
                    initialvalue="500", parent=dlg)
                try:
                    sim_bal = float(ans) if ans else None
                except (TypeError, ValueError):
                    sim_bal = None
                sim_label = f"模拟余额 {sim_bal:.2f}(测试网假钱)" if sim_bal else sim_label
            status_lab.config(text="正在检测余额并扫描白名单币种行情...", fg="black")
            result = {}

            def worker():
                try:
                    balance = None
                    if not bin_testnet.get():
                        ex = ccxt.binance({"apiKey": key, "secret": sec,
                                           "enableRateLimit": True,
                                           "options": {"defaultType": "spot"}})
                        bal = ex.fetch_balance()
                        balance = float((bal.get("USDT") or {}).get("free") or 0)
                    else:
                        balance = sim_bal          # 测试网按模拟余额分档
                    pub = ccxt.binance({"enableRateLimit": True, "timeout": 10000})
                    result["balance"] = balance
                    result["sim_label"] = sim_label
                    result["rec"] = botlib.recommend_setup(pub, self.cfg, balance)
                except Exception as exc:
                    result["err"] = str(exc)

            def done():
                try:
                    if not dlg.winfo_exists():
                        return
                    if "err" in result:
                        status_lab.config(text=f"✗ 推荐失败: {result['err']}", fg="#b71c1c")
                        return
                    if "rec" not in result:
                        dlg.after(200, done)
                        return
                    rec = result["rec"]
                    bal = result.get("balance")
                    if not rec["enough"]:
                        messagebox.showwarning("资金不足", rec["reason"]
                                               + "\n\n可勾选『测试网』先用假钱练手。")
                        status_lab.config(text="资金不足, 未启动", fg="#b71c1c")
                        return
                    if bin_testnet.get():
                        bal_txt = result.get("sim_label", "测试网(假钱)")
                    else:
                        bal_txt = f"{bal:.2f}"
                    msg = (f"检测到 USDT 余额: {bal_txt}\n"
                           f"推荐每轮预算: {rec['round_budget']:.0f} U "
                           f"(每单 {rec['buy_size']:.2f} U, 止损线 {rec['round_budget'] * 0.85:.0f} U)\n"
                           f"推荐币种: {rec['symbol']}\n"
                           f"理由: {rec['reason']}\n\n是否按推荐配置启动?")
                    if messagebox.askyesno("智能推荐", msg):
                        sym_var.set(rec["symbol"])
                        rec_alloc[0] = rec["round_budget"]
                        do_save()
                    else:
                        sym_var.set(rec["symbol"])
                        rec_alloc[0] = rec["round_budget"]
                        status_lab.config(text=f"已预选 {rec['symbol']}(可手动改币种后点保存)",
                                          fg="#1b5e20")
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()
            dlg.after(200, done)

        tk.Button(dlg, text="智能推荐", command=do_recommend, bg="#e8f5e9",
                  font=("Microsoft YaHei", 10)).grid(
            row=6, column=0, sticky="w", padx=12, pady=12)
        tk.Button(dlg, text="测试连接", command=do_test,
                  font=("Microsoft YaHei", 10)).grid(row=6, column=0, padx=140, pady=12)
        tk.Button(dlg, text="保存", command=do_save, bg="#e3f2fd",
                  font=("Microsoft YaHei", 10)).grid(row=6, column=1, sticky="w", padx=6, pady=12)

    # ---------- 刷新 ----------
    def auto_refresh(self):
        self._refresh_all()
        self.root.after(5000, self.auto_refresh)

    def _refresh_all(self):
        loop_alive = bool(self.loop and self.loop.is_alive())
        ex_name = "币安·测试网" if bool((self.cfg.get("binance") or {}).get("testnet")) \
            else "币安"
        risk = getattr(self.loop, "risk", None)
        risk_txt = ""
        if risk and risk.get("active"):
            until = datetime.fromtimestamp(risk["until_ms"] / 1000, TZ).strftime("%H:%M")
            risk_txt = f" | ⚠ 黑天鹅避险: 暂停买入(至 {until})"
        if loop_alive and not self.loop.pause_ev.is_set():
            self.banner.config(text=f"策略已暂停({ex_name}): 持仓/挂单保持{risk_txt}",
                               bg="#fff3e0", fg="#e65100")
        elif loop_alive:
            self.banner.config(text=f"实盘机器人({ex_name}): 运行中{risk_txt}",
                               bg="#c8e6c9", fg="#1b5e20")
        elif getattr(self.loop, "error", None):
            self.banner.config(text=getattr(self.loop, "error", ""),
                               bg="#fff3e0", fg="#e65100")
        else:
            self.banner.config(text=f"实盘机器人({ex_name}): 未启动(请先填 API Key 并【启动·恢复】)",
                               bg="#ffcdd2", fg="#b71c1c")
        t = datetime.now(TZ).strftime("%H:%M:%S")
        self.quote_lab.config(text=f"显示更新于 {t}(策略每 60 秒检查一次)")

        states = {s: self._load_state(s) for s in self.cfg["symbols"]}
        prices = dict(getattr(self.loop, "last_prices", {}) or {})
        bal = getattr(self.loop, "last_bal", None)

        realized = sum(float(st.get("realized_pnl") or 0) for st in states.values() if st)
        unreal = sum(float(st.get("qty") or 0) * float(prices.get(s) or 0)
                     - float(st.get("cycle_cost") or 0)
                     for s, st in states.items()
                     if st and float(st.get("qty") or 0) > 0)

        if bal is not None:
            usdt_free = float((bal.get("USDT") or {}).get("free") or 0)
            usdt_lock = float((bal.get("USDT") or {}).get("used") or 0)
            mv = 0.0
            for s in self.cfg["symbols"]:
                base = s.split("/")[0]
                q = float((bal.get(base) or {}).get("free") or 0) \
                    + float((bal.get(base) or {}).get("used") or 0)
                mv += q * float(prices.get(s) or 0)
            total = usdt_free + usdt_lock + mv
        else:
            usdt_free = sum(float(st.get("cash") or 0) for st in states.values() if st)
            usdt_lock = 0.0
            mv = sum(float(st.get("qty") or 0) * float(prices.get(s) or 0)
                     for s, st in states.items() if st)
            total = usdt_free + mv

        self.ov_labels["usdt_free"].config(text=f"{usdt_free:.2f}")
        self.ov_labels["usdt_lock"].config(text=f"{usdt_lock:.2f}")
        self.ov_labels["mv"].config(text=f"{mv:.2f}")
        self.ov_labels["total"].config(text=f"{total:.2f}")
        for key, val in (("realized", realized), ("unreal", unreal)):
            color = "#b71c1c" if val < 0 else ("#1b5e20" if val > 0 else "black")
            self.ov_labels[key].config(text=f"{val:+.2f}", fg=color)

        for sym, lab in self.sym_rows.items():
            st = states.get(sym)
            price = prices.get(sym)
            if not st or price is None:
                lab.config(text="--(启动机器人后显示实时行情)")
                continue
            qty = float(st.get("qty") or 0)
            if st.get("stop_loss_triggered"):
                txt = f"价格 {price:.6f} | 已止损停止"
            elif qty <= 0:
                txt = f"价格 {price:.6f} | 空仓, 现金 {st.get('cash', 0):.2f}U"
            else:
                avg = float(st.get("cycle_cost") or 0) / qty if qty > 0 else 0
                txt = (f"价格 {price:.6f} | 持仓 {qty:.2f} | 均价 {avg:.6f} | "
                       f"本轮已买 {st.get('buys_in_cycle', 0)}/10 次 | "
                       f"浮盈亏 {qty * price - float(st.get('cycle_cost') or 0):+.2f}U")
            lab.config(text=txt)

        log = self._log_tail(15)
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", log)
        self.log_box.config(state="disabled")

    def _load_state(self, sym: str):
        return botlib.load_state(sym, self.cfg)

    def _log_tail(self, n=15):
        try:
            with open(botlib.LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return "(暂无日志)"


def main() -> None:
    if "--smoke" in sys.argv:
        cfg = botlib.load_config()
        print("config:", {k: cfg[k] for k in ("symbols", "alloc_per_symbol")})
        ex = ccxt.binance({"enableRateLimit": True, "timeout": 8000})
        bsec = cfg.get("binance") or {}
        if bsec.get("api_key") and bsec.get("api_secret"):
            try:
                ex2 = botlib.make_exchange(cfg)
                print(botlib.api_connection_test(ex2, cfg))
            except SystemExit as exc:
                print("API自检:", exc)
        else:
            print("(未配置API密钥: 仅做公开行情自检; 连接/挂单撤单测试需密钥)")
            for s in cfg["symbols"]:
                try:
                    print(s, "价格", botlib.fetch_price(ex, s),
                          "EMA60(4h)", botlib.fetch_ema60(ex, s))
                except Exception as exc:
                    print(s, "行情失败:", exc)
        print("一体机冒烟自检完成")
        sys.exit(0)

    if not acquire_suite_lock():
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("提示", "实盘机器人已在运行(另一个窗口/托盘图标)。\n"
                                  "请从系统托盘打开, 本实例将退出。")
            root.destroy()
        except Exception:
            pass
        sys.exit(0)

    root = tk.Tk()
    App(root)
    try:
        root.mainloop()
    finally:
        release_suite_lock()


if __name__ == "__main__":
    main()
