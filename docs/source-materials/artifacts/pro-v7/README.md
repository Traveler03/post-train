# Pro v7 数据产物说明

本目录保存从训练服务器下载的 Pro v7 数据转换元信息。二进制 token、mask 和 NumPy 索引未纳入仓库；它们体积较大，对面试证据的增益有限。

## 可核验结论

- 上游源文件：`pro_chain_sft_v2/v2.jsonl.gz`，SHA-256 为 `7e5143a7d5d3aba1b7e6fa4ec549810f0749fa17c682c267fceabce907097fc3`。
- 转换读入 1,149 条，保留 1,134 条；15 条因长度超过 98,304 token 丢弃。
- 按 `rule_head` 整组切分：train 1,102 条，val 32 条，共享任务组数为 0。
- train 共 75,370,567 token，长度 p50／p90／max 为 67,541／80,830／98,297；直接监督 token 占 5.02%。
- val 共 2,202,436 token，长度 p50／p90／max 为 65,880／83,431／95,510；直接监督 token 占 6.31%。
- 98,304 token 窗口使用 best-fit-decreasing，但 train 和 val 都是一条样本占一个窗口，`max_segments=1`；填充率分别约 69.57% 和 70.01%。

## 验证集结构抽查

下载的 `val.jsonl` 有 32 条。只做结构统计，未把正文提交到仓库，因为它包含完整内部 system prompt、工具 schema 和测试账号信息：

- 每条 7～34 条消息，中位数 16。
- 每条 2～12 个直接监督的 assistant 段，中位数 5；合计 178 个监督段。
- `sup` 与 `messages` 等长，所有 `sup=true` 位置均为 assistant 消息。
- 每条携带 32 或 35 个工具 schema。
- 所有样本的 `meta` 字段值均为空，无法仅靠该文件追溯到原始 trace、模型或规则。

## 证据边界

`meta.json` 同时记录了三项待核问题：源文件“不在已知清单”、转换脚本 Git commit 为 `?`、实际渲染档位分布为空。因此这些文件能证明最终数据形态与规模，但还不能把每条样本绑定到合成 seed、Rollout、判官结果或最终评测 run。
