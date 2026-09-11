# Docs/Drive Agent RL 面试准备

## 1. 你在项目里具体负责什么？

推荐回答：

> 我负责 Docs/Drive Agent RL 后训练体系的整体搭建，不只是调 GRPO 参数。数据侧要把 Case、Seed、账号世界和 Reward 统一起来；Rollout 侧要让真实 Agent 使用本地 Policy，并回收每轮 Token/Logprob；训练侧要处理多轮展开、组内 Advantage、Padding Mask、分布式更新和 Checkpoint；评测侧要跑真实环境并把失败轨迹回流。除此之外，我重点设计了状态 DAG Process Reward、授权感知 Verifier、Boundary-only Advantage 和 Skill/Verifier 演化机制。

若不是所有外部平台组件都由本人实现，应主动说明：Docs/Drive 工具服务和基础 VERL/vLLM 是已有能力，本人负责把它们组织成可训练、可验证的 Agent RL 系统并完成关键扩展。

## 2. 为什么需要 RL，而不是继续做 SFT？

Docs/Drive 任务有以下特点：

- 任务成功依赖完整执行结果，不只是某一句回复。
- 同一目标存在多种合法工具路径。
- 模型需要根据工具返回继续决策。
- 失败轨迹也包含有价值的相对信息。
- 最终质量来自策略与环境交互，静态示范难覆盖所有状态。

SFT 适合建立基础工具协议和行为格式；GRPO 可以在同一任务的多条 Rollout 之间比较最终结果，并直接优化实际策略分布。难点是环境成本、Judge 噪声和长程信用分配，这正是本项目重点解决的问题。

## 3. 如何让真实 Agent 产生 On-policy 训练轨迹？

核心是 Relay，而不是事后对文本重新打概率：

1. Agent 发起模型请求时附加唯一 Correlation ID。
2. Relay 将原始请求转发到当前训练 Policy 的本地 vLLM。
3. vLLM 返回 Completion Token 和对应 Logprob。
4. Relay 按 Correlation ID 保存每次调用的 Prompt/Token/Logprob。
5. Agent 继续根据返回结果调用真实工具。
6. 任务结束后将多次模型调用展开为训练 Row。

因为动作本身由当前 Policy 采样并同步保存行为概率，所以轨迹可以用于 On-policy 更新。

## 4. 为什么不直接把整个多轮对话拼成一个样本？

每次模型调用的可见上下文和行为概率不同。简单拼接可能：

- 把工具返回之后的信息泄漏给之前的动作。
- 无法正确绑定每段 Completion 的行为 Logprob。
- 混淆终局 Reward 与具体 Model Call。
- 让不同 Turn 的长度和 Mask 难以验证。

当前做法按 Policy Call 展开，同时保留 UID、Turn Index 和原始 Group，使训练框架能够正确计算 Reward、Advantage 和 Loss。

## 5. 账号世界为什么按 Group 复用？

GRPO 对同一 Query 采样多条 Rollout。若每条都重复清理和 Seed，环境准备成本很高；若完全共享又可能污染状态。

当前折中是：

- 同一 Query Group 清理和 Seed 一次。
- Group 内多条 Rollout 从同一初始世界启动。
- Group 完成后统一清理。
- 明确承认并行期间世界可能被写操作改变。

这提高吞吐，但不是严格隔离。面试时不要把它描述成独立沙箱快照；更准确的说法是“同初始状态的共享账号世界”。

## 6. 为什么账号并发与 Relay 并发要分开？

两个并发控制的是不同瓶颈：

- 账号并发过高会增加环境清理、Seed 和工具服务压力。
- Relay 并发过高会增加 vLLM KV Cache、排队和 Decode 延迟。

如果只用一个总并发，无法判断慢是因为没有账号、Agent/Tool 阻塞，还是 GPU 推理拥塞。拆分后可以分别根据账号等待、Relay 活跃数和 Token 吞吐调参。

## 7. Process Reward 为什么不能按工具调用给分？

工具调用只是动作，不是任务事实。以下轨迹都可能“调用正确工具但任务失败”：

- 选错目标文件。
- 编辑范围或内容错误。
- 工具返回失败。
- 未确认就执行外部写入。
- 写入成功但没有验证或报告。

因此本人把 Milestone 定义成状态，例如“已建立权威内容”“已获得必要确认”“目标修改已验证”。工具结果只是状态 Judge 的证据之一。

## 8. LLM Judge 如何防止误判？

使用多层约束，而不是无条件相信 Judge：

- Judge 只能根据当前 Case 的实例化 Checklist 和脱敏轨迹证据判断。
- 确定性检查先识别工具成功、失败、目标和 Turn 范围。
- PASS 必须满足 DAG 前置依赖。
- 与确定性证据冲突的 PASS 被拒绝。
- 结构化输出非法、规则缺失或 Judge 不可用时 Fail-closed。
- 固定 Verifier、Judge Model 和 Schema Version，保存 Response ID 与耗时供审计。

仍需承认 LLM Judge 有噪声，所以候选 Verifier 要经过历史回放、重复一致性检查和独立 Split 门禁。

## 9. 授权感知状态图解决什么问题？

例子：用户说“把这份文档修改成下面的版本”。

若产品策略要求先确认外部变更，正确轨迹可能是：

```text
定位文档
  → 读取当前内容
  → 生成具体修改预览
  → 请求用户明确确认
  → 当前任务结束，不执行写入
```

后续用户确认后，正确轨迹才是：

```text
识别已有确认
  → 执行修改
  → 重新读取或检查结果
  → 报告完成
```

若 Verifier 不理解授权，两条轨迹会共用一个“修改成功”终点，从而奖励第一次请求中的越权写入。授权分支把确认本身变成合法终态，并对未经确认的写操作做 Veto。

## 10. Boundary-only Advantage 是什么？

状态首次完成的 Turn 是唯一局部奖励边界。设同组中某状态完成比例为 `p`，完成该状态的轨迹在边界获得 `1-p` 的局部 Advantage，其余 Turn 为 0。

直觉：

- 越少 Rollout 能完成的状态，成功边界越有区分度。
- 全组都完成时没有额外信号。
- 不把后来的成功反向归因给所有前置 Turn。
- 不因 Verifier 未完成就额外惩罚整条轨迹。

它与全局 Outcome Advantage 相加，而不是替代终局目标。

## 11. 为什么没有给未完成 Milestone 的轨迹负局部分？

“未完成”不一定等于当前 Turn 的策略错误，也可能是：

- 外部工具失败。
- 证据缺失。
- 状态不适用。
- 轨迹选择了另一条合法路径。

在 Verifier 尚有噪声时施加负局部分风险更大。因此第一版采用正向、边界、组内相对的保守设计，失败结果仍由全局 Outcome Reward 处理。

## 12. 信用分配审计发现了什么？

Docs 历史数据有 400 条轨迹和 1,931 个 Model-call Row，79.29% 是非终局 Row。若把每条轨迹的终局分数复制到所有 Row，长轨迹会累计更多分数；49/50 个 Query Group 的 Row-weighted Baseline 与每轨迹等权 Baseline 不同，94 个 Row 的 Advantage 符号发生变化。

这说明“轨迹展开”不只是数据格式问题，它会改变优化统计。当前设计先取消局部 Process Reward 的反向回填，只在首次完成边界加局部信号；全局 Baseline 的进一步改进作为独立问题保留。

## 13. SkillBank 是训练 Prompt 吗？

不完全是。SkillBank 包含两个部分：

- Skill：从历史轨迹提炼的类别级策略和失败经验。
- Verifier：可实例化的状态、Checklist、证据门控、Hard Failure 和依赖 DAG。

本项目最重要的是 Verifier 作为 Reward Contract 的作用。即使 Skill 文本不注入 Policy Prompt，状态图仍可用于过程标注和 Advantage。

## 14. 如何避免 Skill Evolution 过拟合历史轨迹？

- 同一 Case 的 8 条 Rollout 不跨 Split。
- 按任务 Family 和 Operation Subtype 分层。
- Proposal、Dev、Sealed Test 三段式流程。
- 候选先通过完整 Dev Gate，才允许打开 Sealed Test。
- 同时检查 Micro、Family、Subtype、Terminal Recall 和依赖共触发。
- 未过门禁不覆盖当前训练版本。

需要坦诚：目前最新候选仍未通过 Sealed Test Gate，因此只能讲演化框架和小样本回放，不能宣称独立测试集已验证最终精度。

## 15. 当前效果如何？

可以分层回答：

1. **系统打通：** Docs 50 Case、Drive 85 Case 均支持每 Case 8 路 Rollout、训练和 Checkpoint 评测。
2. **离线验证：** 1,080 条 Rollout、4,127 个 Policy Turn 完成回放，状态相关工具映射率超过 91%，Verifier 异常为 0。
3. **Live 评测：** 已归档 Full-training Checkpoint 三轮 Overall Pass 为 Docs 51.33%、Drive 50.98%，Judge Error 为 0。
4. **算法增益：** Milestone v1 的现有结果低于此前 Full-training Run，且不是严格 A/B，所以目前不声称 Process Reward 已提升最终通过率。

最后一点反而能体现实验判断：区分“机制实现正确”和“最终效果已经提高”。

## 16. 训练慢时如何定位？

按阶段判断：

| 现象 | 更可能的瓶颈 |
|---|---|
| Account Wait 上升 | 账号世界容量 |
| Cleanup/Seed 高 | 环境准备服务 |
| Agent Wait 高、Relay 正常 | Router、工具或外部 Agent |
| Relay 活跃数接近上限 | 本地模型请求排队 |
| Relay P95 上升、Tokens/s 下降 | vLLM/KV Cache/Decode |
| Judge Duration 或 Attempts 上升 | Checklist Judge |
| Rollout 后显存接近总量 | KV Cache 与 Backward 显存冲突 |
| Reward 全零或组内方差消失 | 信号质量，而非单纯吞吐问题 |

## 17. 最难的工程问题是什么？

推荐从“真实轨迹可训练化”回答：

> 最难的不是启动 GRPO，而是保证真实 Agent 的每次模型调用都能由当前 Policy 生成、拿到精确行为概率、在并发环境中正确归属、再按 Turn 展开而不破坏 Group 和 Mask。这里同时涉及外部服务、账号状态、Ray 并发、vLLM 请求、超时和训练统计。我通过 Correlation ID Relay、Group 生命周期、两级并发和 Fail-closed Trace 回收把这条链路闭合。

## 18. 如果重新设计会改什么？

- 为写操作提供真正隔离的账号快照或串行 Group，消除共享状态干扰。
- 对全局 Outcome Advantage 评估轨迹级等权 Baseline。
- 增加失败轨迹的正确前缀保护，但避免 Judge 噪声产生错误正奖励。
- 完成标准 GRPO 与 Boundary Advantage 的同配置配对 A/B。
- 在完整 Dev Gate 通过后只评一次 Sealed Test。
- 将工具执行的确定性 Post-condition 覆盖扩展到更多 Docs/Drive 操作。

## 19. 面试展示顺序

建议按下面的顺序讲，不要从公式开始：

1. Docs/Drive 真实任务为什么难。
2. 完整 RL 系统的数据流。
3. Relay 如何获得 On-policy 多轮轨迹。
4. 为什么终局 Reward 不够。
5. 状态 Verifier 与授权分支。
6. Boundary-only Advantage。
7. 离线规模、Live 结果和当前限制。

准备一张系统图、一条脱敏轨迹和一张实验边界表，足以覆盖大多数追问。
