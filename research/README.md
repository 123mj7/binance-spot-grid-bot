# research —— 策略回测证据链

本目录是从几十个策略变体里做外样本检验的完整研究脚本,
最终幸存的"朴素基线"已实现在上一层的 binance_live_bot.py。

## 核心文件

| 文件 | 作用 |
|---|---|
| regime_detect.py | 下载历史 1h K 线(币安公开数据) |
| adaptive_backtest.py | 基线策略 vs ATH 动态止损 四腿对比 |
| adaptive_v3.py | v3 逐轮止损 + 冷却期变体 |
| adaptive_oos.py | 2021-2022 / 2023-2025 / 五年连续外样本 |
| strategy_engine.py | 无路由策略引擎(基线) |
| optimize_dca.py | 参数网格搜索(多进程) |
| fng_test.py / fng_fear_test.py | 恐惧贪婪指数闸门检验(被淘汰) |
| backtest_2026.py | 2026-01 至今最新外样本 |
| verify_blackswan.py | 黑天鹅检测器历史灾变窗口验证 |

## 使用方法

1. 先下载数据:

```python
import regime_detect as rd
df = rd.fetch_history("2021-01-01", "2025-12-31", cache_csv="doge_usdt_1h_3y.csv")
```

2. 再跑回测(以五年外样本为例):

```python
from adaptive_backtest import compute_ema4h, backtest_baseline
ema4h = compute_ema4h(df)
result = backtest_baseline(df, ema4h, trailing=False)
```

3. 数据源默认币安公开接口; 网络受限时可在脚本内换成其他交易所的
   公开 K 线(价格序列因套利几乎一致)。

## 关键结论(勿重复造轮子)

- 4h EMA60 入场过滤: 保留(唯一通过的过滤)
- 1h 路由 / ATH 动态止损 / v3 止损 / FNG 闸门: 全部被外样本淘汰
- 黑天鹅检测: 保留(缩手 4h, 不接飞刀; LUNA/FTX 窗口实测触发, 平静期零误报)
- 复投 / 组合级熔断 / 死拿无止损版: 已检验, 未获稳定增益, 未上线
  (复投在 50U 下未激活、500U 下全部档位输给固定等比; 相关脚本已移除,
  见历史提交; 结论: 该策略的利润最好的归宿是闲置现金)
- **合约(永续)方向: 已研究并否决(2026-11)** —— 曾用含资金费/爆仓/杠杆的
  引擎检验海龟式(4h)与快突破(1h)两版趋势策略, DOGE 五年胜率仅 5~8%,
  全部亏损, 未通过任何一关; 相关脚本已从仓库移除(见历史提交),
  结论: 本策略基因不适合合约, 勿重蹈覆辙。
