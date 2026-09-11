# Docs/Drive Agent RL 来源与证据

本页记录公开总结所依据的实现和实验材料。原始工程位于内部训练仓库；本公开仓库只保存脱敏总结，不复制账号、密钥、服务地址、原始用户数据、完整轨迹、模型权重或内部业务源码。

## 1. 个人职责口径

根据本人说明，本项目职责包括：

- Docs/Drive Agent RL 端到端体系搭建。
- 真实环境 Rollout、训练与评测链路接入。
- Milestone 状态 Verifier 与 Process Reward 设计。
- Boundary-only Segment Advantage 设计。
- Offline SkillBank/Verifier 合成及 Evolution 设计。
- 训练监控、信用分配审计和实验分析。

基础 VERL、Ray、vLLM、外部 Agent 框架和 Docs/Drive 工具服务不表述为本人从零实现；个人贡献是系统集成、训练扩展、算法机制与完整闭环。

## 2. Rollout 与训练实现

| 内部工程路径 | 支持内容 |
|---|---|
| `algo/grpo_adk/agent_loop.py` | 单样本与 Group Rollout、账号租约、环境准备、真实 Agent 执行、重试和 Trace 回收 |
| `algo/grpo_adk/runtime.py` | Account Pool、Relay Bridge、并发控制、上游请求与安全摘要 |
| `algo/grpo_adk/trajectory.py` | Prompt/Completion Token 与 Logprob 提取、多次模型调用展开 |
| `algo/grpo_adk/manager.py` | 在训练生命周期内创建账号池与 Relay |
| `algo/grpo_adk/batching.py` | 多轮展开后的 Data Parallel Padding 和有效样本 Mask |
| `verl/trainer/ppo/ray_trainer.py` | Rollout Extra Info、Reward、Advantage 与训练主链路接入 |
| `verl/trainer/config/grpo_adk.yaml` | AgentLoop 与 GRPO 配置 Overlay |
| `my_scripts/train_google_docs_single_qwen35_4b_*.sh` | Docs Full/LoRA 训练入口 |
| `my_scripts/train_google_drive_single_qwen35_4b_*.sh` | Drive Full/LoRA 训练入口 |

## 3. Reward、Milestone 与 Advantage

| 内部工程路径 | 支持内容 |
|---|---|
| `scripts/benchmark_single_reward.py` | Docs/Drive Benchmark Judge、Reward 解析、缓存、重试和错误记录 |
| `algo/grpo_adk/milestones.py` | 状态实例加载、工具证据到 Turn 的映射、状态验证与轨迹标注 |
| `algo/grpo_adk/milestone_judge.py` | Checklist Judge、结构化输出和一致性校验 |
| `algo/grpo_adk/milestone_policy.py` | Verifier 策略与运行时约束 |
| `algo/grpo_adk/milestone_advantage.py` | Boundary Reward、组内局部 Advantage 与全局信号组合 |
| `algo/offline_skills/authorization.py` | 授权策略与状态图分支物化 |
| `docs/milestone_segment_advantage.md` | 当前 Advantage 定义、配置、验证数据和已知限制 |

## 4. SkillBank 合成与演化

| 内部工程路径 | 支持内容 |
|---|---|
| `algo/offline_skills/io.py` | Dataset、Rollout 和 Reward 关联 |
| `algo/offline_skills/classifier.py` | Domain、Operation、Outcome、Cardinality、Complexity 分类 |
| `algo/offline_skills/evidence.py` | 工具事件抽取、独立验真、去噪与 Experience Card |
| `algo/offline_skills/blueprints.py` | Family 状态模板和确定性检查 |
| `algo/offline_skills/instances.py` | Case 级状态图实例化、复合任务和 DAG 校验 |
| `algo/offline_skills/replay.py` | 历史 Rollout 的确定性回放 |
| `algo/offline_skills/audit.py` | 状态覆盖和适配审计 |
| `algo/offline_skills/evolution.py` | Case Split、候选 Proposal、P/R 评估与发布门禁 |
| `scripts/generate_offline_skillbank.py` | SkillBank 生成入口 |
| `scripts/evolve_milestone_skillbank.py` | Verifier Evolution 入口与断点续跑 |
| `docs/offline_skill_generation.md` | 生成、实例化、回放和演化流程说明 |

## 5. 数据与实验产物

| 内部工程路径 | 支持内容 |
|---|---|
| `data/benchmark_single/docs/manifest.json` | Docs 50 Case、数据指纹和训练步数口径 |
| `data/benchmark_single/drive/manifest.json` | Drive 85 Case、数据指纹和训练步数口径 |
| `artifacts/offline_skillbank_v5/validation_report.json` | 1,080 条 Rollout 的 Domain/Family 回放汇总 |
| `artifacts/turn_reward_misattribution/summary.json` | 400 条 Docs 轨迹的终局分数 Fan-out 与 Baseline 审计 |
| `docs/milestone_segment_advantage.md` | 4,127 个 Turn、2,086 个 Segment 和工具映射率 |
| `eval/runs/benchmark_docs_single_qwen35_4b/*/summary.json` | Docs 三轮 Live Checkpoint 评测 |
| `eval/runs/benchmark_drive_single_qwen35_4b/*/summary.json` | Drive 三轮 Live Checkpoint 评测 |
| `artifacts/offline_skillbank_evolution_v*/evolution_report.json` | Proposal/Dev/Test Split、候选指标和门禁状态 |

详细转录见[实验与结果](evaluation.md)。

## 6. 测试覆盖

相关自动化测试覆盖：

- Agent Rollout 超时、重试、账号池与并发隔离。
- Relay Correlation、并发 Trace、精确 Token/Logprob 和多调用轨迹。
- 多轮展开后的 Padding、Mask、Reward Extra Info 对齐。
- Benchmark Reward 解析、Judge Retry 与训练异常策略。
- 状态工具映射、Checklist Judge、DAG 依赖和 Boundary Reward。
- 未确认/已确认/预授权/拒绝等 Authorization Branch。
- Skill 分类、资源绑定、复合任务实例化和全量 8 Rollout 回放。
- Evolution 稳定切分、Family/Subtype 指标、候选校验和 Sealed Test Gate。
- 标量监控脱敏、Dashboard 与训练事件聚合。

代表性测试文件：

```text
tests/test_grpo_adk.py
tests/test_grpo_adk_batching.py
tests/test_benchmark_single_training.py
tests/test_benchmark_single_evaluation.py
tests/test_reward_compute.py
tests/test_milestone_advantage.py
tests/test_milestone_judge.py
tests/test_milestone_authorization.py
tests/test_milestone_evolution.py
tests/test_offline_skill_generation.py
tests/test_offline_skill_audit.py
tests/test_grpo_monitoring.py
tests/test_turn_reward_report.py
```

测试存在说明相关行为有自动化约束；除非保存了对应完整测试运行记录，不将其表述为所有测试在当前公开环境重新通过。

## 7. 公开内容边界

### 已公开

- 端到端系统架构。
- 核心模块职责和数据流。
- Process Reward、Authorization 和 Advantage 公式。
- 聚合数据规模、离线回放和 Live 评测结果。
- 设计取舍、已知限制和面试表述。

### 未公开

- SSH Private Key、GitHub Token 或其他凭据。
- Docs/Drive 测试账号和身份信息。
- 内部域名、服务发现、Webhook、鉴权方式和网络拓扑。
- Seed 原文、用户文档、文件名、邮件地址和完整 Tool Arguments。
- 原始 Prompt、System Prompt、Token ID、模型回复和逐条轨迹。
- Model Checkpoint、Adapter 权重和大体积训练数据。
- 可直接复用内部服务的业务源码。

## 8. 结论强度

| 结论 | 证据 |
|---|---|
| RL 主链路已实现 | 代码、配置、Checkpoint 和 Live 评测汇总 |
| 多轮轨迹可回收并训练 | Relay/Trajectory 实现与对应测试 |
| SkillBank 可在历史数据上执行 | 1,080 条全量确定性回放 |
| 状态工具映射覆盖较高 | Docs 93.15%、Drive 91.67% |
| Boundary-only 机制按定义实现 | Advantage 代码、等价性测试与离线回放 |
| Milestone 已提升最终任务效果 | 当前证据不支持 |
| 最新 Evolution 已通过独立测试 | 当前 Sealed Test Gate 未通过 |

新增实验时，应先更新[实验与结果](evaluation.md)的证据表，再更新[简历表述](resume.md)，避免先写结论后补对应关系。
