# 原始材料与证据

本页链接的原文、截图和源码都保存在本仓库内，可直接在 GitHub 阅读，不再依赖 Notion 或 GitLab 权限。

## 使用范围与结论分层

- **个人职责：** 用户明确确认，整个数据合成及 LoRA SFT 过程均由本人搭建。
- **实现与配置：** 依据用户提供的 GitLab `main`／`milly` ZIP 快照；相关文件已纳入本仓库。
- **项目过程与效果：** 依据 Notion 导出原文、文字表格和实验截图；相关材料已纳入本仓库。
- **本次验证：** 现有离线测试，以及 pc7/pc8 源文件全量差分、训练日志指标提取和前期同题评测核验；没有重跑 GPU 训练、线上评测或判官调用。
- **待补关联：** v7 数据已绑定到训练 run 和 step 276 checkpoint；该 checkpoint 与汇总评测结果尚未绑定。

代码快照缺少 Git 作者历史；这不推翻本人已经确认的职责，只限制本次用提交记录进一步核验。

## Notion 导出原文

| 原始材料 | 支持内容 |
|---|---|
| [Pro 流程梳理和训练记录](../source-materials/notion/pro-flow.md) | 工具空间、历史结构、请求边界、数据质量、版本迭代；明确记录 v2.1 删除 v2 的 grad norm 尖峰数据并训练 pc8，内含 27 张本地实验截图 |
| [Pro 评测结果汇总](../source-materials/notion/pro-benchmark-summary.md) | P0、邮件、日历、Drive、Docs、Sheets、Slides 等汇总表 |
| [Pro 后训练效果与方法](../source-materials/notion/pro-results.md) | Drive／Docs／Slides 分数、目标工具覆盖、缺口合成、授权情境 |

## Pro 本地源码

以下文件来自上传的 `milly` 快照，目录结构保持原样：

| 文件 | 支持内容 |
|---|---|
| [Pro 原始流量分析说明](../../source-snapshot/milly/docs/pro/08_线上数据深描_0826.md) | 统计分母、Agent 归属、任务族、读写与上下文 |
| [raw_deep.py](../../source-snapshot/milly/tools/pro_trace_analysis/raw_deep.py) | 原始轨迹分析 |
| [deep_report.py](../../source-snapshot/milly/tools/pro_trace_analysis/deep_report.py) | 分布报告 |
| [prepare_dsv4_data.py](../../source-snapshot/milly/training/scripts/dsv4/prepare_dsv4_data.py) | 消息转换、mask、整轨迹渲染、截断、分组 |
| [dsv4_sft_data.py](../../source-snapshot/milly/training/scripts/dsv4/dsv4_sft_data.py) | Packed 数据、label／mask 移位、分段边界 |
| [train_dsv4.py](../../source-snapshot/milly/training/scripts/dsv4/train_dsv4.py) | LoRA 挂点、训练配置、有效 token 归一 |
| [Pro v7 启动脚本](../../source-snapshot/milly/training/scripts/dsv4/launch_prochain_v7_milly_0827.sh) | 98k、r16、约 2 epoch 的具体配方 |
| [test_render_modes.py](../../source-snapshot/milly/training/scripts/dsv4/tests/test_render_modes.py) | 渲染和监督边界的 19 项离线测试 |
| [plan_checkpoint_eval.py](../../source-snapshot/milly/eval/scripts/plan_checkpoint_eval.py) | checkpoint 评测计划 |
| [run_checkpoint_eval.py](../../source-snapshot/milly/eval/scripts/run_checkpoint_eval.py) | adapter 加载及执行 |
| [retrying_eval_runner.py](../../source-snapshot/milly/eval/scripts/retrying_eval_runner.py) | 执行失败补测、状态恢复 |
| [评测读数陷阱](../../source-snapshot/milly/eval/docs/06_读数陷阱.md) | 波动、缺失、环境变化和错误比较 |

原始记录提到的 `gen_gap_queries.py` 未在上传快照中找到。相关流程以项目记录为证据，未声称代码已被逐行检查。

## Pro v7 数据产物

| 文件 | 支持内容 |
|---|---|
| [产物说明](../source-materials/artifacts/pro-v7/README.md) | 规模、切分、token、监督段、窗口结果与证据边界 |
| [meta.json](../source-materials/artifacts/pro-v7/meta.json) | 上游文件指纹、转换参数、过滤原因和 split 统计 |
| [train.windows_meta.json](../source-materials/artifacts/pro-v7/train.windows_meta.json) | 训练窗口数、样本数、填充率和打包算法 |
| [val.windows_meta.json](../source-materials/artifacts/pro-v7/val.windows_meta.json) | 验证窗口数、样本数、填充率和打包算法 |

下载的 `val.jsonl` 仅用于本地结构抽查，未提交：它包含完整内部 system prompt、工具 schema 和测试账号信息。抽查结论已写入产物说明。

## Pro v7 Milly 0827 复跑（不是本次历史 pc7 对照）

| 文件 | 支持内容 |
|---|---|
| [训练运行说明](../source-materials/artifacts/pro-v7/training-run/README.md) | 数据到训练、实际配置、训练动态、checkpoint 和证据边界 |
| [原始 train.log](../source-materials/artifacts/pro-v7/training-run/train.log) | 276 步日志、LoRA 可训练参数、checkpoint 成功记录 |
| [逐步 metrics.csv](../source-materials/artifacts/pro-v7/training-run/metrics.csv) | loss、学习率、grad norm、耗时与吞吐 |
| [训练曲线](../source-materials/artifacts/pro-v7/training-run/sft_metrics.png) | 六类指标可视化 |
| [结构统计](../source-materials/artifacts/pro-v7/training-run/train-structure.json) | 1,102 条 train JSONL 的消息、监督、工具和 meta 聚合统计 |
| [运行摘要](../source-materials/artifacts/pro-v7/training-run/run-summary.json) | 可机器读取的配置、性能和 checkpoint 摘要 |
| [checkpoint 元数据](../source-materials/artifacts/pro-v7/training-run/checkpoint.metadata) | 43 层四类 LoRA target 的 344 个 adapter tensor key |

下载的 `train.jsonl` 同样未提交；其原文约 304 MiB，且启发式扫描命中潜在私密标识。step 276 的本地 checkpoint 权重 shard 为 0 字节，也未提交；元数据和日志能证明服务器端产出，但不能用于本地恢复权重。

## Spike 数据过滤证据边界

- [Notion 原文](../source-materials/notion/pro-flow.md#v2)记录：“v2.1 去掉 v2 里面的 grad norm 尖峰数据（pc8）”。
- [pc7 checkpoint 总表](../source-materials/notion/pro-flow-assets/image%208.png)记录过滤前版本各 checkpoint；[pc7 三次复测](../source-materials/notion/pro-flow-assets/image%209.png)显示 `s162` 存在明显波动。
- [pc8 checkpoint 评测图](../source-materials/notion/pro-flow-assets/image%2014.png)记录过滤后版本多个 checkpoint；其中 `pc8_s75` 为 Drive 83.8%、Docs 83.8%、Slides 78.7%。
- 本次全量核验补齐：源数据 1,149 → 1,057，删除 92 条，无新增或改写；Train 1,102 → 1,010。已找到 pc7/pc8 对应 meta、启动脚本和实际训练日志，聚合证据见[实验报告](spike-study/README.md)。
- 新报告使用约 0.2–0.8 epoch 同题配对，不再以 pc8 s75 对比 pc7 s162 三次均值。四档综合增益 +2.44～+5.93pp，平均 +4.34pp；单轮、非独立重复，Holm 校正后未达显著。
- [删除指纹清单](spike-study/artifacts/removed_fingerprints.csv)和[核验脚本](../../scripts/spike/README.md)是事后重建，不是历史选择脚本；完整 spike 到样本映射及多随机种子因果消融仍缺失。

本次新增材料按公开仓库范围处理，只公开聚合统计、单向指纹、曲线和离线脚本，不新增原始轨迹、完整内部日志、账号或服务端点。旧源码快照的适用范围见[快照说明](../../source-snapshot/README.md)，本次不对旧材料作全仓安全审计保证。

## 输入快照

| 输入 | SHA-256 |
|---|---|
| post-train-main.zip | `9643e61a6aabb0ccefdd1e229d66e086ebc9381a0ffb89ce69f28e2df03ccd2e` |
| post-train-milly.zip | `a440efc542d218dbd0b0c421eda4a8c2fecfbf6c1609edd19e76d6c34e67e593` |

两个快照的同路径文件中，590 个内容相同、15 个不同；另有 101 个 `milly` 独有文件、17 个 `main` 独有文件。这些数字不表示提交顺序。

## 后续最有价值的补充

1. Pro 最终数据集版本、组成与来源清单。
2. 结果表对应的 checkpoint、对照 run、微调 run 及每轮分数。
3. 本人最熟悉的失败复盘，以及相关修改位置。
