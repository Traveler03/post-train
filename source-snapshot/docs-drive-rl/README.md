# Docs/Drive RL：SkillBank 与 Milestone DAG 源码快照

这是 Docs/Drive Agent RL 中与离线 SkillBank、任务状态 DAG、授权分支、Milestone Judge、Boundary Advantage 和离线审计直接相关的核心源码快照。

它对应[Docs/Drive Agent RL 项目说明](../../docs/docs-drive-rl/README.md)，用于展示实际实现而非仅展示架构图或伪代码。

## 包含内容

```text
algo/offline_skills/
  离线语料读取、任务分类、资源解析、工具证据去噪
  Experience Card、状态模板、Case 实例化、回放、审计和 Evolution

algo/grpo_adk/
  milestones.py              轨迹工具证据到 Turn 的映射与状态标注
  milestone_judge.py         Checklist LLM Judge 与 Fail-closed 校验
  milestone_policy.py        Authorization-aware 状态图分支
  milestone_advantage.py     Boundary Reward 与 Global/Local Advantage 组合

scripts/
  generate_offline_skillbank.py
  validate_offline_skillbank.py
  evolve_milestone_skillbank.py
  materialize_evolved_skillbank.py
  audit_milestone_fit.py
  render_segment_decay_report.py
  render_turn_reward_misattribution.py

tests/
  Skill 生成、DAG 实例化、授权、Judge、Advantage、Evolution、审计测试

artifacts/offline_skillbank_v5/skills/
  9 个类别级 Skill 的策略、State Template 与 Judge Rubric
```

## 推荐阅读顺序

1. [离线总流程](algo/offline_skills/pipeline.py)：语料到 SkillBank 的编排。
2. [任务分类](algo/offline_skills/classifier.py) 与 [资源绑定](algo/offline_skills/resources.py)：如何从 Query/Metadata 识别任务族和具体对象。
3. [状态实例化](algo/offline_skills/instances.py)：将类别模板绑定为 Case 级 DAG。
4. [授权分支](algo/offline_skills/authorization.py) 与 [Milestone Policy](algo/grpo_adk/milestone_policy.py)：未确认、已确认、预授权和拒绝的状态图差异。
5. [轨迹状态标注](algo/grpo_adk/milestones.py) 与 [Checklist Judge](algo/grpo_adk/milestone_judge.py)：如何从真实工具轨迹得到首次完成 Turn。
6. [Boundary Advantage](algo/grpo_adk/milestone_advantage.py)：如何把状态完成边界转化为局部 GRPO 信号。
7. [演化机制](algo/offline_skills/evolution.py) 与 [演化脚本](scripts/evolve_milestone_skillbank.py)：如何分层切分、评估候选和控制发布。
8. [Skill 模板](artifacts/offline_skillbank_v5/skills/)：查看 Docs/Drive 任务族的实际策略与验收契约。

## 设计约束

- 工具调用是状态证据，不是 Reward 本身。
- 状态 PASS 同时受语义 Checklist、确定性证据、Hard Failure 与 DAG 依赖约束。
- 写操作必须走 Authorization-aware 分支；未确认的外部变更不能关闭执行状态。
- 局部 Advantage 只落在首次完成边界；`local_weight=0` 时严格退化为基线 GRPO。
- 同一 Case 的多个 Rollout 在 Evolution 中不跨 Split，防止轨迹泄漏。

## 依赖与运行边界

该快照保留项目自研核心模块，不尝试复制完整训练平台、真实 Agent 服务、Docs/Drive Connector 或模型权重。

- Python 3.10+
- 离线 Skill 生成/审计需要 `openai`、`pydantic` 等运行依赖；API Key 通过运行时环境变量或本地配置提供，仓库不保存任何凭据。
- Advantage 模块需要 `numpy`、`torch`，并复用上游 VERL 的 GRPO 基础函数。
- 测试中的输入均为合成 Fixture；完整 Benchmark Case、Seed、原始工具轨迹、Prompt 和账号数据未上传。

示例命令仅在具备依赖和自有脱敏输入时使用：

```bash
python scripts/generate_offline_skillbank.py --no-api --output-dir /tmp/skillbank
python scripts/validate_offline_skillbank.py --help
python scripts/evolve_milestone_skillbank.py --help
pytest -q tests/test_milestone_advantage.py tests/test_milestone_authorization.py
```

## 公开边界

已保留：核心算法与工程实现、类别级策略、状态模板、Judge Rubric 和合成测试。

未上传：Case Instance、Experience Card、原始 Prompt、完整生成 Prompt、原始 Rollout/Reward、账号 Seed、内部网络配置、真实凭据、模型权重及完整外部环境实现。
