# Docs/Drive Agent RL 简历表述

## 1. 推荐项目名称

### 面向真实工具环境的 Docs/Drive Agent RL 后训练体系

技术栈：Python、PyTorch、VERL、GRPO、Ray、vLLM、FSDP、LoRA、Hydra、LLM-as-Judge

## 2. 推荐投递版

面向 Google Docs/Drive 的真实文档与文件操作场景，主导搭建从任务数据、并行 Rollout、Reward/Verifier、GRPO 优化到 Checkpoint Live 评测的端到端 Agent RL 后训练体系，并负责过程奖励与信用分配机制设计。

- **端到端 RL 体系：** 统一 Docs/Drive 任务、账号 Seed 与 Reward 数据契约，打通真实 Agent 环境准备、多轮工具执行、On-policy 轨迹采集、GRPO 更新、Checkpoint 恢复及三轮 Live 全量评测；支持 Full/LoRA 与多 GPU 分布式训练。
- **真实环境 Rollout Runtime：** 设计 ADK–vLLM Relay，将外部 Agent 的每次 Policy 请求路由至本地训练模型，精确回收 Prompt/Completion Token ID 与 Logprob，并将多轮调用展开为 VERL 训练样本；通过 Group 账号世界复用、账号/推理解耦并发、单样本重试和有效 Batch Mask 提升长链路训练稳定性。
- **过程奖励与信用分配：** 设计 Outcome-first 任务状态 DAG，由语义 Checklist Judge、确定性工具证据、Hard Failure、依赖关系和授权上下文共同验证状态完成；提出 Boundary-only Local Advantage，仅在状态首次完成 Turn 注入组内相对信号，且关闭局部权重时严格退化为标准 GRPO。
- **Skill/Verifier 合成演化：** 从 Reward-backed 成功/失败轨迹中完成分类、独立验真、去噪、Experience Card 提取、类别级 Skill/Verifier 生成及 Case 实例化；建立 Proposal/Dev/Sealed Test 的 Case 级切分和 Precision/Recall 发布门禁，避免同 Case 轨迹泄漏与训练期 Reward 漂移。
- **验证与可观测：** 完成 Docs 50 Case、Drive 85 Case、每 Case 8 条共 1,080 条 Rollout 的离线回放，覆盖 4,127 个 Policy Turn，状态相关工具映射率超过 91%、Verifier 异常为 0；建设脱敏监控，覆盖账号、Agent、Relay、GPU、Judge、Reward 和训练阶段。

## 3. 四条紧凑版

### Docs/Drive Agent RL 后训练｜VERL、GRPO、Ray、vLLM

- 主导搭建 Docs/Drive 真实工具环境下的端到端 Agent RL 体系，覆盖任务与账号环境构造、并行 Rollout、Reward、GRPO 多卡训练、Checkpoint 恢复及 Live 评测，支持 Full/LoRA 训练。
- 自研 Agent–vLLM Relay，回收每轮精确 Token/Logprob 并将多轮工具轨迹转换为可训练样本；设计 Group 账号世界复用、账号/推理解耦并发和动态 Padding Mask，保障外部环境长链路训练稳定运行。
- 设计状态 DAG 驱动的 Process Reward，以 Checklist Judge、确定性工具证据、依赖关系和授权分支验证任务进度；提出 Boundary-only Advantage，在首次完成 Turn 注入局部组内相对信号并保持标准 GRPO 兼容。
- 构建轨迹驱动的 Skill/Verifier 合成与演化闭环，完成 135 个 Case、1,080 条 Rollout、4,127 个 Policy Turn 的回放，工具映射率超过 91%、Verifier 异常为 0，并通过分层切分和 P/R 门禁控制 Reward 版本质量。

## 4. 三条极简版

- 主导搭建 Google Docs/Drive 真实工具环境的 Agent RL 后训练体系，打通任务构造、并行 Rollout、On-policy 轨迹采集、GRPO 多卡训练、Checkpoint 恢复与 Live 评测。
- 设计 Agent–vLLM Relay、账号世界复用和多轮轨迹展开，精确回收每次模型调用的 Token/Logprob；支持 Full/LoRA、并发限流、故障重试和脱敏监控。
- 设计状态 DAG + Checklist Judge + 工具证据的 Process Reward 及 Boundary-only Advantage，并构建可版本化演化的 Skill Verifier；完成 1,080 条 Rollout、4,127 个 Turn 的离线验证。

## 5. 30 秒介绍

> 我主导搭建了 Docs/Drive 场景的完整 Agent RL 后训练系统。它让模型在真实账号和工具环境里执行文档读取、编辑、评论、文件检索和分享等任务，通过 Relay 把每次模型调用接到本地 vLLM，回收精确 Token 和 Logprob，再用 VERL 做 GRPO 更新。除了训练工程，我重点设计了基于任务状态图的过程奖励：工具调用只能作为证据，必须同时通过语义 Checklist、确定性检查、状态依赖和授权判断；局部 Advantage 只加在状态首次完成的 Turn。最后再通过 Live 评测和历史失败轨迹驱动 Verifier 演化，形成完整闭环。

## 6. 90 秒介绍

> 这个项目的目标是在 Docs/Drive 真实工具环境中训练 Agent，而不是只优化单轮文本答案。任务会涉及账号里的具体文档和文件，也可能产生编辑、评论和分享等外部副作用，所以首先需要解决环境复现、并行 Rollout 和多轮轨迹采集。
>
> 系统层面，我把 Docs/Drive Case、Seed 和 Reward 统一成训练数据契约；在真实 Agent 的模型调用边界设计 Relay，将请求转发到本地 vLLM，并回收每一轮 Prompt Token、Completion Token 和 Logprob。一个任务中的多个模型调用会展开成训练 Row，同时保留原始 Query Group 和 Turn 边界。账号侧以一个 Query Group 为单位完成清理与 Seed，账号容量和模型请求并发分别控制，从而可以区分环境瓶颈和 GPU 推理瓶颈。
>
> 算法上，单纯把终局奖励广播给所有 Turn 很难解释哪一步有效。我设计了状态 DAG 驱动的 Milestone Verifier：状态描述真实任务事实，工具调用只是证据；状态关闭需要语义 Checklist Judge、确定性结果、依赖关系和授权状态全部一致。局部 Advantage 只加到状态首次完成边界，并与全局 GRPO Outcome Advantage 叠加，局部权重为 0 时严格回到原始 GRPO。
>
> 数据侧，我又从历史成功和失败轨迹中合成类别级 Skill 与 Verifier，按 Case 做 Proposal、Dev 和 Sealed Test 切分，使用 Precision、Recall 和 Terminal Recall 门禁控制版本发布。目前完成 Docs 50 个 Case、Drive 85 个 Case、共 1,080 条 Rollout 和 4,127 个 Policy Turn 的离线回放，状态相关工具映射率超过 91%，并打通了 Checkpoint 的三轮 Live 全量评测。

## 7. 简历数字如何使用

### 推荐使用

- 50 Docs Cases + 85 Drive Cases。
- 每 Case 8 Rollouts，共 1,080 条 Reward-backed Rollout。
- 4,127 个 Policy Turn、2,086 个完成 Segment。
- 状态相关工具映射率 Docs 93.15%、Drive 91.67%。
- 400 条 Docs 轨迹信用审计中，79.29% 为被终局分数覆盖的非终局 Row。
- 三轮 Live 评测 Judge Error 为 0。

### 需要带限定

- Full-training Checkpoint 的 Live Overall Pass 为 Docs 51.33%、Drive 50.98%，但缺少同协议 Base 对照。
- Milestone v1 的已归档结果没有超过此前 Full-training Run，不能作为算法提升数字。
- 小样本候选 Verifier 出现过 100% P/R，但尚未通过 Sealed Test，不能作为最终准确率。

## 8. 避免的表述

- 不写“调用某个工具就获得过程奖励”；实际设计恰恰避免这种做法。
- 不写“完全解决稀疏奖励”；当前方案提供有证据的边界局部信号。
- 不写“独立实现了 VERL/vLLM”；本人负责的是基于它们搭建业务 Agent RL 系统和扩展模块。
- 不写“Milestone 已显著提升通过率”；当前材料不支持。
- 不将 Docs/Drive RL 与 Pro SFT、AutoTask 或其他项目的规模和结果相加。

## 9. 不同岗位的强调顺序

### RL / Post-training 算法岗

1. 状态 Verifier。
2. Boundary-only Advantage。
3. 信用分配审计。
4. Verifier Evolution 和实验门禁。

### LLM Infra / 训练系统岗

1. Agent–vLLM Relay。
2. 账号世界与并发调度。
3. 多轮 Token/Logprob 轨迹转换。
4. 分布式训练、恢复和监控。

### Agent 算法岗

1. 真实 Docs/Drive 环境。
2. 工具证据与任务状态解耦。
3. 授权感知状态图。
4. Skill/Verifier 数据闭环。
