# Agent Post-Training 项目档案

本仓库整理两个相互独立的 Agent 后训练项目：

- **Docs/Drive Agent RL：** 真实工具环境下的端到端 RL 体系、过程奖励、信用分配和 Skill Verifier 演化。
- **Pro 主链路 Agent：** 数据合成、长上下文 LoRA SFT、训练反馈驱动的数据选择。

职责口径由项目本人确认：Docs/Drive 项目主导完整 RL 体系及核心机制设计；Pro 项目独立负责数据合成与 LoRA SFT 全流程。两个项目的数据规模和效果指标分别记录，不交叉归因。更新日期：2026-09-17。

## 项目导航

| 想了解什么 | 文档 |
|---|---|
| Check 有哪些维度、如何评分、不同 Rollout 的流程与轮次 | [Check 与 Rollout 代码核查](docs/reference/check-and-rollout.md) |
| 最终 SFT 有多少轮交互、一条真实样本长什么样 | [SFT 交互统计与逐消息展开](docs/pro/sft-interactions.md) |
| Docs/Drive RL 总入口 | [Docs/Drive Agent RL](docs/docs-drive-rl/README.md) |
| Docs/Drive 端到端架构与职责 | [完整项目说明](docs/docs-drive-rl/project.md) |
| Process Reward、Boundary Advantage、授权与 Skill Evolution | [核心设计](docs/docs-drive-rl/design.md) |
| Docs/Drive 数据、离线回放与 Live 评测 | [实验与结果](docs/docs-drive-rl/evaluation.md) |
| Docs/Drive 简历与面试 | [简历表述](docs/docs-drive-rl/resume.md) · [面试准备](docs/docs-drive-rl/interview.md) |
| 项目背景、个人职责、完整讲述 | [项目说明](docs/pro/project.md) |
| 线上轨迹、能力缺口、任务合成、拒绝采样 | [数据流水线](docs/pro/data-pipeline.md) |
| 线上日志怎样变成 query、文件环境和 SFT 轨迹 | [完整数据构造流程](docs/pro/data-construction.md) |
| 多轮监督、LoRA 挂点、长上下文和训练配置 | [LoRA SFT](docs/pro/lora-sft.md) |
| 效果数字、比较条件、退化和实验绑定 | [评测与结果](docs/pro/evaluation.md) |
| 可直接修改的简历条目、面试开场 | [简历与介绍](docs/pro/resume.md) |
| 高频追问、回答要点、具体例子 | [面试准备](docs/pro/interview.md) |
| Notion 原文、截图、源码与证据强度 | [原始材料与证据](docs/pro/sources.md) |
| 去 Spike 前后分布、删除方法、训练曲线与前期增益 | [pc7 → pc8 实验报告](docs/pro/spike-study/README.md) |
| 本次分析范围、分支比较与已运行验证 | [分析记录](docs/reference/review-record.md) |

## Docs/Drive Agent RL 主线

1. **构造任务与账号世界：** 将 50 个 Docs Case、85 个 Drive Case、Seed 和 Reward 统一成可复现训练契约。
2. **接入真实 Agent Rollout：** 通过账号池和 Ray AgentLoop 并行执行真实文档与文件任务。
3. **回收 On-policy 多轮轨迹：** 使用 Agent–vLLM Relay 捕获每轮 Prompt、Token 和 Logprob，并展开成 VERL 训练样本。
4. **构建 Outcome 与 Process Reward：** 终局 Judge 判断任务结果，状态 DAG Verifier 判断有证据的中间进度和授权分支。
5. **完成 GRPO 更新：** 将全局 Outcome Advantage 与首次完成边界的局部 Advantage 组合，支持多卡 Full/LoRA 训练与恢复。
6. **评测和演化：** 对 Checkpoint 运行三轮 Live 全量评测，并用失败轨迹离线演化 Skill/Verifier。

完整架构、公式、1,080 条 Rollout 回放和结果边界见 [Docs/Drive Agent RL](docs/docs-drive-rl/README.md)。

## Pro 六步项目主线

1. **定义业务问题：** 分析 Pro 线上任务、工具覆盖和失败类型，定位复杂编辑、权限处理和多步执行等能力缺口。
2. **合成与筛选数据：** 从线上日志提取需求、补齐上下文并改写为独立 query，构造可执行环境与缺口任务，通过新的 Rollout、判官和清理得到可用轨迹。
3. **建立多轮监督：** 区分历史、当前请求、assistant 动作和工具返回，只在目标模型行为上计算 loss。
4. **完成长上下文 SFT：** 适配 DeepSeek MoE 的 LoRA 与 98k 上下文训练，使用 1,102 条样本运行 276 步、约 2 个 epoch 并保存 checkpoint。
5. **用训练信号反查数据：** 在确认训练未发散后，结合 grad norm spike 的跨 epoch 复现位置排查可疑 batch／轨迹。
6. **构建过滤版本并验证：** 形成 v2.1／pc8 并进行 checkpoint 评测，完成“训练反馈—数据选择—重新训练—业务评测”闭环。

## Pro 六步主线的结果

v2 → v2.1 删除 **92 / 1,149 条（8.01%）**，主要工具分布基本不变；完整训练的梯度尖峰次数由 16 降到 11，峰值降低约 40%，但 loss 整体接近、尖峰仍存在。

仅看约 **0.2–0.8 epoch 前期**，去 Spike 的 pc8 相对原版训练 pc7，四档同题配对的综合通过率变化为 **+2.44～+5.93pp，四档平均 +4.34pp**。这是单轮描述性结果，尚未证明统计显著或稳定提升，不代表最终效果。数据分布、过滤证据、三张曲线图和复现材料见[完整报告](docs/pro/spike-study/README.md)。

## 文档边界

Pro 项目保存了整理后的材料、Notion 导出原文、实验截图和相关源码快照，入口见[原始材料与证据](docs/pro/sources.md)。Docs/Drive RL 公开脱敏架构、公式、聚合实验事实，以及 [SkillBank 与 Milestone DAG 核心源码](source-snapshot/docs-drive-rl/README.md)；不包含内部账号、服务地址、原始轨迹、权重或完整业务服务实现，证据映射见 [Docs/Drive 来源与证据](docs/docs-drive-rl/sources.md)。
