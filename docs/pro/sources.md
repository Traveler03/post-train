# 证据索引

## 使用范围与结论分层

- **个人职责：** 用户明确确认，整个数据合成及 LoRA SFT 过程均由本人搭建。
- **实现与配置：** 依据用户提供的 GitLab main／milly ZIP 快照。
- **项目过程与效果：** 依据本会话读取的 Notion 文本与表格。
- **本次验证：** 有限的现有离线测试，没有 GPU 训练或线上评测复跑。
- **待补关联：** 部分数据版本、checkpoint 和评测结果尚未绑定。

代码快照缺少 Git 作者历史；这不推翻本人已经确认的职责，只限制本次用提交记录进一步核验。

## Notion 来源

| 来源 | 支持内容 |
|---|---|
| [Pro 流程梳理和训练记录](https://app.notion.com/p/3c2b2f39aba580229728d9c26a5839d3) | 工具空间、历史结构、当前请求边界、数据质量、版本迭代 |
| [Pro 后训练效果与方法](https://app.notion.com/p/3c9b2f39aba580f5b97ec99c24de8e95) | Drive／Docs／Slides 分数、目标工具覆盖、缺口合成、授权情境 |
| [AutoTask 数据合成 Pipeline](https://app.notion.com/p/3c2b2f39aba580919766c116e0089bed) | 补充资料中的环境合成、规模和任务极性 |
| [AutoTask 效果报告](https://app.notion.com/p/3bbb2f39aba580df8958fd2f9cb3934d) | 补充资料中的训练效果与数据版本 |

Pro 流程页包含多张实验截图，本次没有转录其中的数值。效果引用采用可读取的文字表格。

## GitLab 原始实现

[源仓库 milly 分支](https://git.garena.com/beeai/algo/post-train/-/tree/milly)。

以下为根据上传快照定位的源码链接；远端分支以后可能变化，不代表本次读取了最新提交。

| 文件 | 支持内容 |
|---|---|
| [Pro 原始流量分析说明](https://git.garena.com/beeai/algo/post-train/-/blob/milly/docs/pro/08_线上数据深描_0826.md) | 统计分母、Agent 归属、任务族、读写与上下文 |
| [raw_deep.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/pro_trace_analysis/raw_deep.py) | 原始轨迹分析 |
| [deep_report.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/pro_trace_analysis/deep_report.py) | 分布报告 |
| [prepare_dsv4_data.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/training/scripts/dsv4/prepare_dsv4_data.py) | 消息转换、mask、整轨迹渲染、截断、分组 |
| [dsv4_sft_data.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/training/scripts/dsv4/dsv4_sft_data.py) | Packed 数据、label／mask 移位、分段边界 |
| [train_dsv4.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/training/scripts/dsv4/train_dsv4.py) | LoRA 挂点、训练配置、有效 token 归一 |
| [Pro v7 启动脚本](https://git.garena.com/beeai/algo/post-train/-/blob/milly/training/scripts/dsv4/launch_prochain_v7_milly_0827.sh) | 98k、r16、约 2 epoch 的具体配方 |
| [test_render_modes.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/training/scripts/dsv4/tests/test_render_modes.py) | 渲染和监督边界的 19 项离线测试 |
| [plan_checkpoint_eval.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/eval/scripts/plan_checkpoint_eval.py) | checkpoint 评测计划 |
| [run_checkpoint_eval.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/eval/scripts/run_checkpoint_eval.py) | adapter 加载及执行 |
| [retrying_eval_runner.py](https://git.garena.com/beeai/algo/post-train/-/blob/milly/eval/scripts/retrying_eval_runner.py) | 执行失败补测、状态恢复 |
| [评测读数陷阱](https://git.garena.com/beeai/algo/post-train/-/blob/milly/eval/docs/06_读数陷阱.md) | 波动、缺失、环境变化和错误比较 |

Notion 提到的 `gen_gap_queries.py` 未在上传快照中找到。相关流程以项目记录为证据，未声称代码已被逐行检查。

## 输入快照

| 输入 | SHA-256 |
|---|---|
| post-train-main.zip | `9643e61a6aabb0ccefdd1e229d66e086ebc9381a0ffb89ce69f28e2df03ccd2e` |
| post-train-milly.zip | `a440efc542d218dbd0b0c421eda4a8c2fecfbf6c1609edd19e76d6c34e67e593` |

两个快照的同路径文件中，590 个内容相同、15 个不同；另有 101 个 milly 独有文件、17 个 main 独有文件。这些数字不表示提交顺序。

## 后续最有价值的补充

1. Pro 最终数据集版本、组成与来源清单。
2. 对应实验的实际可训练参数和运行配置。
3. 结果表对应的 checkpoint、对照 run、微调 run 及每轮分数。
4. Pro 自身的打包和 GPU 耗时统计。
5. 本人最熟悉的失败复盘，以及相关修改位置。
