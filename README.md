# Pro 主链路 Agent：数据合成与 LoRA SFT

面向邮件、日历及 Drive／Docs／Slides 等个人办公场景，整理 Pro 主链路的数据建设、LoRA SFT、实验结果、简历表述和面试准备。

**职责已由项目本人确认：独立负责数据合成与 LoRA SFT 全流程建设。** 当前简历与项目介绍聚焦 Pro；AutoTask 的相关经验和指标单独保存在参考资料中。

内容依据仓库内保存的 Notion 导出原文、`main`／`milly` 代码快照及本人对职责的补充。更新日期：2026-09-11。

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

## 六步项目主线

1. **定义业务问题：** 分析 Pro 线上任务、工具覆盖和失败类型，定位复杂编辑、权限处理和多步执行等能力缺口。
2. **合成与筛选数据：** 清洗真实轨迹，在可执行环境中构造缺口任务，通过 Rollout、规则检查和判官筛选得到可用轨迹。
3. **建立多轮监督：** 区分历史、当前请求、assistant 动作和工具返回，只在目标模型行为上计算 loss。
4. **完成长上下文 SFT：** 适配 DeepSeek MoE 的 LoRA 与 98k 上下文训练，使用 1,102 条样本运行 276 步、约 2 个 epoch 并保存 checkpoint。
5. **用训练信号反查数据：** 在确认训练未发散后，结合 grad norm spike 的跨 epoch 复现位置排查可疑 batch／轨迹。
6. **构建过滤版本并验证：** 形成 v2.1／pc8 并进行 checkpoint 评测，完成“训练反馈—数据选择—重新训练—业务评测”闭环。

## 六步主线的结果

以 `pc7_s162` 三次复测均值为参照，去 spike 数据的 `pc8_s75` 在 Drive／Docs／Slides 上分别变化 **+0.1／+2.3／-0.3pp**，三项宏平均约提升 **0.7pp**。该结果用于说明训练反馈驱动的数据选择闭环；完整比较边界见[评测文档](docs/pro/evaluation.md)。

AutoTask 的 662 条规则、2,025 个环境、1,281 条训练样本，以及约 76% 打包窗口减少，均保留在 AutoTask 参考文档中，未当作 Pro 的规模或效率指标。

## 文档边界

本仓库同时保存整理后的项目材料、Notion 导出原文、实验截图和相关源码快照，可以直接在 GitHub 内阅读。入口见[原始材料与证据](docs/pro/sources.md)。源码快照能支持实现与配置说明；实验完成情况和模型效果按对应的项目记录描述。
