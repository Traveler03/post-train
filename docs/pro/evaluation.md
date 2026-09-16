# Pro 评测与结果

## 当前主结论：去 Spike 后的前期增益

v2 / pc7 与去 Spike 的 v2.1 / pc8 按相近 epoch、相同题目直接比较。仅报告约 **0.2–0.8 epoch 前期**，不把它写成最终模型效果，也不与未训练 baseline 混淆。

四档综合通过率变化依次为 **+5.76、+3.24、+5.93、+2.44pp**，算术平均 **+4.34pp**。综合采用 Drive、Docs、Slides 共同已判题的微平均，不是三域宏平均。Drive 四档都正向，Docs 与 Slides 有升有降。

完整数据分布、92 条删除证据、实际训练曲线及评测方法见 [pc7 → pc8 实验报告](spike-study/README.md)。

## 实验身份

| 对照 | 数据与训练 | 本次评测 checkpoint |
|---|---|---|
| 原版 pc7 / v2 | 源记录 1,149；Train 1,102 / Val 32；276 步 | s27 / s54 / s81 / s108 |
| 去 Spike pc8 / v2.1 | 源记录 1,057；Train 1,010 / Val 32；252 步 | s25 / s50 / s75 / s100 |
| Milly 0827 复跑 | 仓库原有的 v7 工程复现日志 | 不用于本次过滤效果对照 |

两版预处理各丢弃 15 条超长记录；Val 数量相同不代表成员相同。历史 pc7/pc8 日志和此前保存的 Milly 复跑日志必须区分。

## 怎样计算，哪些话不能说

- 每档对双方均有 PASS/FAIL 的同一 `item_id` 配对，不把双方各自分数简单相减；共同题数为 243 / 247 / 253 / 246。
- 仅读取既有 run 和判官缓存，不重跑或重新判分；固定原版首轮，pc8 轮次及源文件指纹全部记录在[聚合结果](spike-study/artifacts/early_evaluation.json)。
- 没有判决或无法配对的题不在这个分母内；因此不是全部提交题通过率，缺失题可能造成偏差。
- 四档来自训练沿途 checkpoint，不是四个独立重复实验。其 +4.34pp 均值是描述性统计。
- 精确 McNemar 检验对四组综合比较做 Holm 校正后均未达 0.05。应说“前期观察到提升”，不能说“显著、稳定提升”或“已证明删除坏数据带来确定因果增益”。
- 不报告后期/最终效果；不再以 pc8 s75 对 pc7 s162 三次均值的跨 epoch 比较概括过滤收益。

## 评测完整性与复测

区分正常任务失败、执行异常、缺 trace 和判官未完成。不能把缺失当作已成功，也不能对正常错题无限补跑到正确。关键结论仍需多次独立训练与评测，检查执行模型、adapter、思考档位、工具环境和判官版本。

相同训练配方并不意味着隔离了全部变量：过滤改变数据顺序、batch 构成及每个 epoch 的更新次数，真实工具环境也可能变化。不能仅靠训练 loss 判断模型收益。

## 历史材料的用途

[Pro 流程原文](../source-materials/notion/pro-flow.md)和[项目汇总原文](../source-materials/notion/pro-results.md)保留历史记录，其其他版本、baseline 和后期读数不纳入这次前期 Spike 结论。旧截图按各自完成题数显示，不能直接当作本次同题配对后的分数。

[原有启动脚本](../../source-snapshot/milly/training/scripts/dsv4/launch_prochain_v7_milly_0827.sh)及[Milly 复跑日志说明](../source-materials/artifacts/pro-v7/training-run/README.md)用于工程复现；不将其 step 276 与没有对应 run ID 的项目汇总表强行绑定。
