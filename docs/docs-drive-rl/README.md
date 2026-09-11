# Docs/Drive Agent RL

本目录总结 Google Docs 与 Google Drive 真实工具环境下的 Agent RL 后训练体系，以及本人在 Rollout、过程奖励、信用分配和 Skill Verifier 演化方面的设计。

## 阅读导航

| 内容 | 文档 |
|---|---|
| 项目范围、端到端架构、各层实现与个人职责 | [完整项目说明](project.md) |
| 原创机制、设计动机、公式和取舍 | [核心设计](design.md) |
| 数据规模、离线验证、Live 评测与证据边界 | [实验与结果](evaluation.md) |
| 可直接使用的简历条目和口头介绍 | [简历表述](resume.md) |
| 系统与算法高频追问 | [面试准备](interview.md) |
| 实现来源、证据映射和公开边界 | [来源与证据](sources.md) |

## 一句话概括

> 主导搭建面向 Docs/Drive 真实工具环境的端到端 Agent RL 后训练体系，并设计基于任务状态图的可验证过程奖励、边界信用分配以及可演化的 Skill Verifier。

## 系统闭环

```text
Docs/Drive 任务与账号环境
  → 并行真实 Agent Rollout
  → 每轮 Token/Logprob 轨迹回收
  → Outcome Reward + Milestone Verifier
  → Global GRPO + Boundary Local Advantage
  → 多卡训练与 Checkpoint
  → Live 环境评测与失败分析
  → Skill/Verifier 离线演化
```

## 范围

正文只讨论 Docs/Drive RL，不混入 Pro SFT、AutoTask Morning Brief 或其他业务线的规模和结果。公开材料不包含账号、密钥、内部服务地址、原始用户内容、模型权重和可还原隐私的轨迹。
