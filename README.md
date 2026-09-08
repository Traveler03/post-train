# Pro 主链路 Agent：数据合成与 LoRA SFT

面向邮件、日历及 Drive／Docs／Slides 等个人办公场景，整理 Pro 主链路的数据建设、LoRA SFT、实验结果、简历表述和面试准备。

**职责已由项目本人确认：独立负责数据合成与 LoRA SFT 全流程建设。** 当前简历与项目介绍聚焦 Pro；AutoTask 的相关经验和指标单独保存在参考资料中。

内容依据仓库内保存的 Notion 导出原文、`main`／`milly` 代码快照及本人对职责的补充。更新日期：2026-09-08。

## 阅读导航

| 想了解什么 | 文档 |
|---|---|
| 项目背景、个人职责、完整讲述 | [项目说明](docs/pro/project.md) |
| 线上轨迹、能力缺口、任务合成、拒绝采样 | [数据流水线](docs/pro/data-pipeline.md) |
| 多轮监督、LoRA 挂点、长上下文和训练配置 | [LoRA SFT](docs/pro/lora-sft.md) |
| 效果数字、比较条件、退化和实验绑定 | [评测与结果](docs/pro/evaluation.md) |
| 可直接修改的简历条目、面试开场 | [简历与介绍](docs/pro/resume.md) |
| 高频追问、回答要点、具体例子 | [面试准备](docs/pro/interview.md) |
| Notion 原文、截图、源码与证据强度 | [原始材料与证据](docs/pro/sources.md) |
| AutoTask 的环境合成和训练优化记录 | [AutoTask 补充资料](docs/reference/autotask.md) |
| 本次分析范围、分支比较与已运行验证 | [分析记录](docs/reference/review-record.md) |

## 项目主线

1. **发现问题：** 分析 Pro 线上任务与动作分布，定位目标能力缺口和失败模式。
2. **建设数据：** 清洗真实轨迹，基于可执行环境补充任务，通过 Rollout 和判官筛选形成训练样本。
3. **实现训练：** 对齐实际工具和上下文，明确多轮监督边界，实现 DeepSeek MoE 的 LoRA SFT。
4. **评估迭代：** 固定比较条件，分析端到端任务效果、子类退化和运行波动。

## 已记录的 Pro 结果

同 Low Thinking 档位下，项目原始报告中的 Drive／Docs／Slides 通过率分别提升 **4.93／5.03／3.22 个百分点**。它们是项目记录结果；数据版本、checkpoint 与评测 run 的完整对应关系仍需补入台账，见[评测文档](docs/pro/evaluation.md)。

AutoTask 的 662 条规则、2,025 个环境、1,281 条训练样本，以及约 76% 打包窗口减少，均保留在 AutoTask 参考文档中，未当作 Pro 的规模或效率指标。

## 文档边界

本仓库同时保存整理后的项目材料、Notion 导出原文、实验截图和相关源码快照，可以直接在 GitHub 内阅读。入口见[原始材料与证据](docs/pro/sources.md)。源码快照能支持实现与配置说明；实验完成情况和模型效果按对应的项目记录描述。
