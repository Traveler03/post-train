# 最终 SFT 有多少轮交互，一条轨迹长什么样

核验日期：2026-09-17。直接读取 Pro v7 数据目录的最终 `train.jsonl`，共 **1,102 条**，SHA-256 为 `81584e95e3159c700bded3029c8e38f34e1e09999cfe831dd9c3d58ebfd4110a`，与仓库原有 [训练结构记录](../source-materials/artifacts/pro-v7/training-run/train-structure.json)一致。该统计针对原版 v2 对应的 v7/pc7 数据，不能套到过滤后的 pc8。

**按发起一批工具调用算一轮，每条中位数 4 轮；按实际监督的 assistant 消息算、包含答复，中位数 5 轮。** 98.64% 的样本含至少两轮 assistant 监督，但这批任务入口是一个用户请求，不是多次真人追问。

## 1. 全量统计

| 每条最终 SFT 样本 | 平均 | 中位数 | P90 | 最少–最多 | 全集总数 |
|---|---:|---:|---:|---:|---:|
| 发起工具调用的 assistant 轮数 | 4.63 | 4 | 8 | 0–15 | 5,097 |
| 实际监督的 assistant 轮数，包含答复 | 5.57 | 5 | 9 | 1–16 | 6,134 |
| 工具调用次数 | 7.39 | 6 | 15 | 0–36 | 8,140 |
| 工具结果消息数 | 7.33 | 6 | 15 | 0–36 | 8,075 |
| messages 总条数，包含所有上下文 | 17.04 | 15 | 28 | 5–52 | 18,782 |

1,087 / 1,102 条有至少两轮 assistant 监督，比例 98.64%。完整直方图、精确均值和原文件指纹见 [机器可读统计](../source-materials/artifacts/pro-v7/interaction-stats.json)。分位数采用排序后的 `floor((n-1)×p)` 下标，不作线性插值。

实际监督轮数的分布：

| assistant 监督轮数 | 样本数 |
|---|---:|
| 1 | 15 |
| 2–3 | 201 |
| 4–5 | 435 |
| 6–9 | 342 |
| 10–16 | 109 |

### 统计时不能混用的单位

- 一次 assistant 同时调用两个工具，是一轮输出、两次工具调用；工具返回一般各自占一条消息。
- `sup=true` 的 assistant 消息才计入直接监督轮数。全量 assistant 消息是 7,236 条，其中 1,102 条不监督；检查的结构中包含每样本一条平台预置初始化应答。
- 937 条样本有两个 `user` 消息，165 条有三个。平台 preamble、时间解析等也可用 user 角色，因此不能据角色数量推断真人交互轮数。
- “发起工具调用的轮数”不保证每次都有一条普通 tool 返回。例如转交是不同的终止形式；全集 8,140 次调用与 8,075 条结果消息不一一相等。本统计没有据此完成全量调用链完整性审计。
- 工具 schema 数表示可用动作空间，不是实际调用次数。示例有 32 个 schema，但仅调用 11 次工具。

### 上游 Rollout 与最终 SFT 分母不同

| 数据 | 条数 | 模型轮数/监督轮数平均 | 中位数 | P90 | 范围 |
|---|---:|---:|---:|---:|---:|
| 上游保存的 v2 Rollout，n_call_llm_this_turn | 1,149 | 5.65 | 5 | 10 | 1–17 |
| 最终 train，sup=true 的 assistant 消息 | 1,102 | 5.57 | 5 | 9 | 1–16 |

两行分别是原轨迹记录的模型调用数和最终监督消息数，不能直接视为逐条一一相等。转换元信息记录 15 条超长被剔除，其余切为 train 1,102 / val 32。Val 的历史结构统计为每条 2–12 个监督段、中位数 5，见 [产物说明](../source-materials/artifacts/pro-v7/README.md)。

## 2. 一条真实训练样本的逐消息展开

取最终 `train.jsonl` 的**第一行**。下文保留实际角色、消息顺序、工具调用类型和监督位置；请求与答复作脱敏转述，文件名统一称“客户拜访模板 A”和“干扰模板 B”，不公开账号、资源 ID、工具参数、内部提示词或思考正文。它是实际样本的可读摘要，不是可直接用于训练的 JSON。

用户请求的大意：

> 复制客户拜访模板 A，把标题中的客户名称和日期分别改成这次客户、本周五，其余占位符保留。

关联的合成世界有模板 A 和干扰模板 B 两个文件；要复制正确模板、保留原模板和其他占位符。用户未给出具体客户名。

这条样本共 **22 条 messages、6 轮有效 assistant 输出、5 轮工具交互、11 次工具调用**：

| 下标 | role | 内容/动作摘要 | sup |
|---:|---|---|---|
| 0 | system | 系统提示词 | false |
| 1 | user | 平台注入的技能、连接器和用户上下文 | false |
| 2 | assistant | 平台预置初始化应答，不是本次模型生成 | false |
| 3 | user | 当前用户的复制与修改请求 | false |
| 4 | assistant | 第 1 轮：本轮思考 + memory_search × 2 | true |
| 5 | tool | 第一次记忆查询：未找到相关记忆 | false |
| 6 | tool | 第二次记忆查询：未找到相关记忆 | false |
| 7 | assistant | 第 2 轮：思考 + google_drive_search、memory_search | true |
| 8 | tool | Drive 搜索结果 | false |
| 9 | tool | 记忆查询结果 | false |
| 10 | assistant | 第 3 轮：思考 + tool_search | true |
| 11 | tool | 返回 Slides 工具定义 | false |
| 12 | assistant | 第 4 轮：思考 + proxy_tool 调用 list-slides、info | true |
| 13 | tool | 幻灯片列表 | false |
| 14 | tool | 演示文稿信息 | false |
| 15 | assistant | 第 5 轮：思考 + proxy_tool 调用 read-slide × 4 | true |
| 16 | tool | 第 1 页内容 | false |
| 17 | tool | 第 2 页内容 | false |
| 18 | tool | 第 3 页内容 | false |
| 19 | tool | 第 4 页内容 | false |
| 20 | user | 平台注入的 temporal_context，解析“这周五”的具体日期 | false |
| 21 | assistant | 第 6 轮：思考 + 最终回复，询问缺失客户名 | true |

最终回复的大意是：已找到四页的客户拜访模板；开始复制前，需要用户提供具体客户名称，日期将改为解析出的本周五，其余占位符保持不动。

**这条没有后续用户补充客户名，也没有复制或编辑调用。** 它结束于查证和澄清，被保存在合成 PASS 池且进入最终 train。v61 允许合理推进和澄清通过；不能据 PASS 把它描述为已完成复制编辑，也不能据最终通过声称每一步都最优。本次没有重新运行判官。

顶层字段与监督位置：

```text
messages：上述 22 条消息
tools：32 个工具 schema
sup：与 messages 等长；true 下标为 [4, 7, 10, 12, 15, 21]
meta：来源字段，本文件中这些字段均为空
```

上述六条 assistant 含独立 `reasoning_content`；调用轮使用 `tool_calls` 表示动作，最终答复使用 `content`。工具结果用 `tool_call_id` 关联调用。训练监督包括选中的思考、工具选择、调用参数和回复；system/user/tool 只提供上下文。一条完整轨迹保存为一条 SFT 样本，并没有因为六轮监督就拆成六条训练记录。

为了便于逐项核对，仓库另存 [脱敏结构摘要](../source-materials/artifacts/pro-v7/trajectory-example-outline.json)。它只有角色、工具名称、内容摘要、消息下标和监督标记，不能替代原始训练样本。第一条结构可与保存轨迹核对，不表示已恢复所有 1,102 条样本的完整来源链。

## 3. 如何复算

使用 Python 3.10+ 标准库，在仓库根目录运行；路径替换为有权限读取的本地原件：

```bash
python3 scripts/audit_sft_interactions.py \
  --train /path/to/data_prochain_v7/train.jsonl \
  --rollouts /path/to/v2_pass_trajectories.jsonl
```

`--rollouts` 可省略。脚本只读 JSONL，向标准输出打印聚合统计、文件大小和 SHA-256，不输出用户内容、工具参数、账号、完整路径或思考正文，不联网或启动训练。它检查 messages/sup 长度一致、sup 为布尔值、直接监督位置为 assistant。

无需私有数据即可运行[离线测试](../../scripts/test_audit_sft_interactions.py)，检查计数口径、无效输入及已发布产物的一致性：

```bash
python3 -B scripts/test_audit_sft_interactions.py
```

本次脚本结果与已有 train-structure.json 的条数、消息数、监督总数、角色总数、分位数和文件指纹核对一致。上述统计不包含 pc8 过滤后数据，也不代表后续三用户轮分支；后者及判官口径见 [Check 与 Rollout 核查](../reference/check-and-rollout.md)。
