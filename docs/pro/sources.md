# 原始材料与证据

本页链接的原文、截图和源码都保存在本仓库内，可直接在 GitHub 阅读，不再依赖 Notion 或 GitLab 权限。

## 使用范围与结论分层

- **个人职责：** 用户明确确认，整个数据合成及 LoRA SFT 过程均由本人搭建。
- **实现与配置：** 依据用户提供的 GitLab `main`／`milly` ZIP 快照；相关文件已纳入本仓库。
- **项目过程与效果：** 依据 Notion 导出原文、文字表格和实验截图；相关材料已纳入本仓库。
- **本次验证：** 有限的现有离线测试，没有 GPU 训练或线上评测复跑。
- **待补关联：** 部分数据版本、checkpoint 和评测结果尚未绑定。

代码快照缺少 Git 作者历史；这不推翻本人已经确认的职责，只限制本次用提交记录进一步核验。

## Notion 导出原文

| 原始材料 | 支持内容 |
|---|---|
| [Pro 流程梳理和训练记录](../source-materials/notion/pro-flow.md) | 工具空间、历史结构、请求边界、数据质量、版本迭代；内含 27 张本地实验截图 |
| [Pro 评测结果汇总](../source-materials/notion/pro-benchmark-summary.md) | P0、邮件、日历、Drive、Docs、Sheets、Slides 等汇总表 |
| [Pro 后训练效果与方法](../source-materials/notion/pro-results.md) | Drive／Docs／Slides 分数、目标工具覆盖、缺口合成、授权情境 |
| [AutoTask 效果报告](../source-materials/notion/autotask-results.md) | User／Official 评测效果、数据版本与清洗记录 |

AutoTask Pipeline 的早期 Notion 页没有稳定导出文件；仓库内保存了同项目更完整的后续源码文档 [worldgen 全流程详解](../../source-snapshot/milly/docs/rq3w/07_worldgen全流程详解_0820.md)。其中“每条规则最多 30 道”和 2,025→1,637 属于后续更新，不能与早期 2,025 个环境口径混写。

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

## AutoTask 本地源码

- [训练数据结构与优化记录](../../source-snapshot/milly/docs/rq3w/04_训练数据长什么样_0819.md)
- [worldgen 全流程详解](../../source-snapshot/milly/docs/rq3w/07_worldgen全流程详解_0820.md)
- [流水线说明及全部纳入脚本](../../source-snapshot/milly/tools/autotask_pipeline/README.md)
- [标准答案替换](../../source-snapshot/milly/projects/autotask_testset_aligned/scripts/regold_batch.py)

源码快照中的两处 Langfuse 明文凭据已改成环境变量读取，个人评测邮箱默认值已删除；具体见[快照说明](../../source-snapshot/README.md)。内部路径和服务地址按 private 仓库用途保留。

## 输入快照

| 输入 | SHA-256 |
|---|---|
| post-train-main.zip | `9643e61a6aabb0ccefdd1e229d66e086ebc9381a0ffb89ce69f28e2df03ccd2e` |
| post-train-milly.zip | `a440efc542d218dbd0b0c421eda4a8c2fecfbf6c1609edd19e76d6c34e67e593` |

两个快照的同路径文件中，590 个内容相同、15 个不同；另有 101 个 `milly` 独有文件、17 个 `main` 独有文件。这些数字不表示提交顺序。

## 后续最有价值的补充

1. Pro 最终数据集版本、组成与来源清单。
2. 对应实验的实际可训练参数和运行配置。
3. 结果表对应的 checkpoint、对照 run、微调 run 及每轮分数。
4. Pro 自身的打包和 GPU 耗时统计。
5. 本人最熟悉的失败复盘，以及相关修改位置。
