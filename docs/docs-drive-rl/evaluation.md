# Docs/Drive Agent RL 实验与结果

本页区分四类证据：数据规模、离线回放、信用分配审计和 Live Checkpoint 评测。不同口径不混写。

## 1. 数据规模

| Domain | Case | Reward-backed Rollout | 每 Case Rollout | 训练步数口径 |
|---|---:|---:|---:|---:|
| Docs | 50 | 400 | 8 | Batch Size 8 时 7 Steps |
| Drive | 85 | 680 | 8 | Batch Size 8 时 11 Steps |
| 合计 | 135 | 1,080 | 8 | 分 Domain 训练 |

Case 来源为 Docs/Drive 单轮 benchmark。单轮指用户入口，实际 Agent Rollout 可以包含多个 Policy Call 和工具调用。

## 2. SkillBank v5 离线回放

SkillBank v5 对全部 1,080 条保存轨迹执行了确定性候选回放：

| Domain | Rollout | Benchmark PASS | Deterministic PASS | Benchmark PASS Coverage | Mean Required-state Coverage | Label Agreement |
|---|---:|---:|---:|---:|---:|---:|
| Docs | 400 | 167 | 228 | 72.46% | 70.03% | 61.75% |
| Drive | 680 | 273 | 410 | 69.96% | 68.77% | 55.74% |
| 合计 | 1,080 | 440 | 638 | 70.91% | 69.23% | 57.96% |

解释：

- **Benchmark PASS Coverage：** 已被 Benchmark 判为成功的轨迹中，确定性状态检查能够覆盖的比例。
- **Mean Required-state Coverage：** 每条轨迹满足多少必要状态的平均值。
- **Label Agreement：** 确定性回放与 Benchmark 终局标签的一致程度。

该回放没有重新调用语义 LLM Judge，因此不能把 Label Agreement 写成最终 Verifier 准确率。其用途是验证状态实例、工具证据和 DAG 在全量历史语料上可以执行。

回放完整性：

- 135 个 Case 均有 8 条 Rollout，没有缺组。
- Docs 400 条、Drive 680 条均完成回放。
- Semantic Judge 在离线默认回放中未运行，训练时才使用固定版本 Checklist Judge。

## 3. Turn 与状态映射验证

一次联合历史回放覆盖：

| Domain | Query | Rollout | Policy Turn | 状态相关工具映射率 | 候选状态完成率 | Verifier 异常 |
|---|---:|---:|---:|---:|---:|---:|
| Docs | 50 | 400 | 1,612 | 93.15% | 58.82% | 0 |
| Drive | 85 | 680 | 2,515 | 91.67% | 58.66% | 0 |
| 合计 | 135 | 1,080 | 4,127 | — | — | 0 |

在每组 8 条 Rollout 的联合 Advantage 回放中，4,127 个 Turn 共形成 2,086 个完成 Segment，所有 Advantage 为有限值。

这里的“候选状态完成率”表示历史策略实际走到了多少可程序化检查的状态，不是 LLM Judge 的语义 Precision/Recall。

## 4. 多轮信用分配审计

Docs 审计数据：

| 指标 | 数值 |
|---|---:|
| Query Group | 50 |
| Trajectory | 400 |
| Multi-turn Trajectory | 349 |
| Model-call Row | 1,931 |
| Non-final Row | 1,531 |
| 终局分数 Fan-out 到非终局 Row 的比例 | 79.29% |
| 存在 Row-weighted Baseline 偏差的 Group | 49 / 50 |
| Advantage 符号发生变化的 Row | 94 |
| 受符号变化影响的 Trajectory | 19 |
| 最大 Group Baseline 差值 | 0.1584 |

旧口径将终局分数复制到每个 Model-call Row 后，累计实现分数是原始终局分数和的约 4.05 倍。这个数字用于说明“Row 数量改变了奖励归因”，不表示模型真实获得了 4.05 倍业务收益。

审计推动当前 Milestone 设计做出两项约束：

- 取消 Segment 内反向衰减和 Process Reward 回填。
- 局部 Advantage 只写到状态首次完成边界。

当前版本仍保留基线 GRPO 的全局 Outcome Advantage 行为，因此不能表述为已经完全解决长程信用分配。

## 5. Live Checkpoint 评测

协议：

- 真实 Agent 与 Live 工具环境。
- Docs 每轮 50 个 Case，Drive 每轮 85 个 Case。
- 每个 Checkpoint 独立运行 3 轮。
- 每 Case 每轮采样 1 条轨迹。
- 记录 Mean Reward、Overall Pass、Dimension Pass、各维通过率和 Judge Error。

### 5.1 已归档 Full-training Checkpoint

| Domain | Runs × Cases | Mean Reward | Overall Pass | Dimension Pass | Judge Error |
|---|---:|---:|---:|---:|---:|
| Docs | 3 × 50 | 0.6025 ± 0.0422 | 51.33% ± 5.03pp | 81.07% ± 2.57pp | 0 |
| Drive | 3 × 85 | 0.5973 ± 0.0259 | 50.98% ± 2.96pp | 80.16% ± 1.97pp | 0 |

逐维结果：

| Domain | D1 | D2 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|---:|
| Docs | 74.00% | 61.33% | 74.67% | 98.67% | 96.67% |
| Drive | 73.73% | 65.88% | 64.31% | 98.82% | 98.04% |

这些结果证明 Full-training Checkpoint 已经完成真实环境三轮评测，并且 Judge 链路无错误。由于当前材料没有与底座模型绑定的同协议对照，不写成“相对底座提升多少”。

### 5.2 Milestone v1 Checkpoint

| Domain | Runs × Cases | Mean Reward | Overall Pass | Dimension Pass | Judge Error |
|---|---:|---:|---:|---:|---:|
| Docs | 3 × 50 | 0.5011 ± 0.0134 | 40.67% ± 1.15pp | 72.13% ± 2.01pp | 0 |
| Drive | 3 × 85 | 0.5218 ± 0.0766 | 42.75% ± 9.51pp | 74.20% ± 3.57pp | 0 |

与上表已归档 Full-training Run 相比，Milestone v1 的现有读数更低：

- Docs Overall Pass 低约 10.67pp。
- Drive Overall Pass 低约 8.24pp。

这两组 Run 不是严格同条件 A/B，不能直接归因于某个单一算法修改；但它们明确说明当前证据不支持“Milestone 已提升最终效果”的表述。

## 6. Skill Evolution 门禁

候选 Verifier 的目标门槛：

| 指标 | 门槛 |
|---|---:|
| Micro Precision | ≥ 90% |
| Micro Recall | ≥ 90% |
| Family Precision | ≥ 85% |
| Family Recall | ≥ 85% |
| Terminal Recall | ≥ 90% |
| Dependent Same-turn Rate | ≤ 5% |

部分小样本 Proposal/Dev 回放达到 100% Precision/Recall，但样本只有 1～5 条，且最新候选仍未通过 Sealed Test Gate。因此：

- 可以写“设计了分层切分、P/R 门禁和一致性检查”。
- 不可以写“新 Verifier 已在独立测试集达到 100%”。
- 未通过门禁的候选不会覆盖当前训练使用的固定版本。

## 7. 当前可以支持的成果结论

### 可以写

- 完整搭建 Docs/Drive 真实环境 Agent RL 训练闭环。
- 支持每 Case 8 路 Rollout、多轮 Token/Logprob 回收和多卡 GRPO 更新。
- 完成 1,080 条历史轨迹、4,127 个 Turn 的离线回放。
- 状态相关工具映射率超过 91%，回放 Verifier 异常为 0。
- 完成 Docs/Drive Checkpoint 的三轮 Live 全量评测，Judge Error 为 0。
- 通过 400 条 Docs 轨迹审计量化多轮信用错配，并据此设计 Boundary-only 局部 Advantage。

### 暂时不要写

- “Milestone Process Reward 显著提升最终通过率。”
- “完全解决长程信用分配。”
- “Verifier 在独立测试集达到 100% 准确率。”
- “Live Overall Pass 相对 Base 提升 X%”，除非补齐同协议底座对照。
- 将 Docs、Drive 与其他业务线的数据规模或结果合并。

## 8. 下一步最有价值的实验

1. 固定数据、Seed、Checkpoint Step、推理参数和 Judge 版本，完成标准 GRPO 与 Boundary Advantage 的配对 A/B。
2. 在完整 Dev Gate 通过后只评一次 Sealed Test，避免反复查看测试集。
3. 分别报告 Outcome-only、Verifier-only Audit 与 Combined Advantage，区分奖励质量和训练效果。
4. 增加按任务族的通过率与退化分析，重点观察写操作、评论、分享和边界任务。
5. 对失败轨迹研究前缀保护或轨迹级等权 Baseline，但保持 `local_weight=0` 的基线兼容测试。
