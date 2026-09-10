# Pro v7 训练运行证据

本目录保存 Pro v7 在 2026-08-27 的实际训练日志、逐步指标、可视化和 checkpoint 元数据。原始训练样本含完整内部 Prompt、工具 schema 与测试账号信息，未提交；这里只保留结构统计和 SHA-256。

## 训练闭环

```text
pro_chain_sft_v2/v2.jsonl.gz
  -> 读入 1,149，过滤超长 15，train 1,102 / val 32
  -> DeepSeek-V4-Flash-0731 + LoRA SFT
  -> 276 / 276 steps，累计消费 2,208 个样本（2.0036 epoch）
  -> step 276 checkpoint 在训练服务器保存成功
```

实际运行从 2026-08-27 09:49:27 启动；首条训练指标为 10:31:19，最后一条为 13:00:22，训练完成标记为 13:00:49。日志记录 11 次 checkpoint 成功保存：27、54、81、108、135、162、189、216、243、270、276。

## 实际配置与挂载

| 项目 | 实际值 |
|---|---|
| 模型目录标签 | DeepSeek-V4-Flash-0731 |
| 框架 | Megatron-Bridge |
| GPU 进程 | 8 |
| 并行 | TP=1，PP=1，EP=8，CP=4 |
| 序列长度 | 98,304 |
| Global / micro batch | 8 / 1 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target | `linear_q_down_proj`、`linear_q_up_proj`、`linear_kv_proj`、`linear_proj` |
| 学习率 | 1e-4，最小 1e-5 |
| 激活重计算 / Packed | full / 开启 |
| 随机种子 | 42 |

日志报告 38,395,904 个可训练参数，占当前模型参数的 0.09%。checkpoint 元数据中可提取出第 0～42 层的四类 attention target，每类 86 个 `linear_in` / `linear_out` 权重，共 344 个 adapter tensor key；这比只看启动参数更能证明 LoRA 实际挂载。

## 训练动态

- 276 步全部完成，skipped iteration 和 NaN iteration 均为 0。
- loss 首步 0.4368、末步 0.3475、全程最小 0.2717、均值 0.4608；10 步滚动均值末尾为 0.4428。
- loss 随样本波动明显，前 25 步均值 0.4273、后 25 步均值 0.4775，因此只能说训练数值稳定完成，不能仅凭这条训练曲线声称 loss 单调收敛或模型效果提升。
- grad norm 中位数 0.166；按分析脚本的 1.66 启发式阈值标记到 29 次尖峰，最大值 886.158。优化器启用了 `clip_grad=1.0`，日志值是裁剪前 norm；尖峰没有产生 NaN 或跳步。首个 epoch 的 8 个尖峰位置中有 5 个在约 138 步后再次出现尖峰，支持少数 batch／样本关联假设，但不是严格的样本级因果证明。
- 稳态 step time p50 / p95 为 23.61 / 27.585 秒；约 0.331 samples/s、4,066 padded tokens/s/GPU、463 model TFLOP/s/GPU。token 吞吐按固定窗口估算，不是直接监督 token 的吞吐。

## 数据结构抽查

本地 `train.jsonl` 共 1,102 条、311,534,235 bytes，SHA-256 为 `81584e95e3159c700bded3029c8e38f34e1e09999cfe831dd9c3d58ebfd4110a`：

- 1,102 条均可解析，顶层字段均为 `messages`、`sup`、`tools`、`meta`。
- 每条 5～52 条消息，中位数 15；每条 1～16 个直接监督段，中位数 5，合计 6,134 个。
- `sup` 与 `messages` 全部等长，监督标记均为布尔值，6,134 个 `sup=true` 全部落在 assistant 消息上。
- 另有 1,102 个 assistant 消息不参与直接监督；这与保留平台初始化/历史 assistant 上下文而不计算 loss 的配方一致。
- 每条带 32 或 35 个工具 schema，共出现 41 个不同工具名。
- `meta` 虽包含 `trace_id`、`model`、`rule_lang`、`n_triggers`、`rule_head`，但 1,102 条中这些值全部为空，无法从最终 JSONL 逐条反查合成来源。

敏感信息启发式扫描在原始训练 JSONL 中命中邮箱、电话号码样式和 credential-like 文本；这不等于每次命中都是真实凭据，但足以说明不能把原文直接提交。未发现 Bearer token 或私钥头。完整扫描不替代人工隐私审查。

## 文件与完整性

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `train.log` | 10,462 行原始训练日志 | `104500b48f7c878bfe9d708c54c6485576f8fd2b729371c90614304aff010b5d` |
| `metrics.csv` | 276 步逐步指标 | `d704aed12908383013c5dcfbd12ef387186514cf410217bb34ca88f276bbb2c9` |
| `sft_metrics.png` | loss、学习率、grad norm、耗时与吞吐图 | `629000821eaeb0888df9472c8f2c7af0d26de271a74834d7186c9273a1bd465c` |
| `checkpoint.metadata` | step 276 torch distributed checkpoint 元数据 | `f52cdfa26f02658e38d352be42c525273bf9a87c31d1d763f327addca9549a16` |

`latest_checkpointed_iteration.txt` 的值为 276，日志也记录 step 276 保存成功。但下载到本地的 `__0_0.distcp` 是 0 字节，因此当前材料能证明服务器端 checkpoint 产出，不能在本地恢复 adapter 权重。`latest_wandb_artifact_path.txt` 仅为 `dummy/dummy`，不提供可用 W&B 追溯。

当前目标是面试复盘而非本地恢复或续训，所以无需继续下载权重分片；原始日志、逐步指标和 checkpoint 元数据已经足以证明本次训练实际运行及产出。
