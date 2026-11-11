# -*- coding: utf-8 -*-
"""审计脚本: 核对"两版本最后一轮持仓都是 585.2 枚 DOGE"这一疑点。

直接读取两份成交 CSV, 提取各自最后一轮的买入序列并逐项对比:
  - 若首笔时间/价格相同且买入价格序列完全一致 → 说明两版本进入了同一轮,
    之后的 3% 加仓链由价格路径唯一决定, 持仓数量相同是必然结果, 不是 bug。
"""
import pandas as pd

print("=" * 74)
print(" 审计: 两个版本最后一轮的买入序列是否相同?")
print("=" * 74)
for name, f in [("基线策略(25笔)", "doge_dca_trades.csv"),
                ("新策略(47笔)", "doge_dca_v2_trades.csv")]:
    t = pd.read_csv(f)
    t["time"] = pd.to_datetime(t["time"])
    last_sell = t.index[t["side"] == "SELL"].max() if (t["side"] == "SELL").any() else -1
    cycle = t[t.index > last_sell]
    print(f"\n{name}: 总成交 {len(t)} 笔(买 {(t['side'] == 'BUY').sum()} / "
          f"卖 {(t['side'] == 'SELL').sum()})")
    print(f"  最后一轮买入 {len(cycle)} 笔:")
    print(f"  首笔: {cycle.iloc[0]['time']} @ {cycle.iloc[0]['price']:.6f} "
          f"({cycle.iloc[0]['type']})")
    print(f"  买入价格序列: {['%.6f' % p for p in cycle['price']]}")
    print(f"  累计持仓数量: {cycle['qty'].sum():.6f} DOGE")
print()
