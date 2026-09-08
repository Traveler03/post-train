# AutoTask 补充资料

当前简历主线聚焦 Pro。本页保留此前核实的 AutoTask 环境合成、训练优化和效果口径，便于后续需要时引用。

本人已确认数据合成和 LoRA SFT 全流程由其搭建。AutoTask 与 Pro 的指标按场景和实验分开使用。

## 1. 场景

AutoTask 根据用户预设规则，在定时或事件触发后检查环境并执行任务。例如晨报、重要邮件提醒、会议准备和条件监控。

主要行为问题包括：需要交付时遗漏内容、承诺操作但未执行，以及条件不满足时仍打扰用户。

正确静默是一种成功行为。对外不推送，不等于模型生成零 token；仍可包含工具动作和结构化决策。协议里的 pass=false 与判官的 FAIL 不是同一概念。

## 2. 环境合成

历史轨迹只保存局部观测，无法直接恢复完整账号环境。因此按真实规则和历史情境构造可执行环境，包含邮件、日历、联系人及必要文件，并明确任务条件与评分要求。

关键约束：

- 规则中的实体、环境对象和评分依据一致。
- 时间、时区和触发时刻一致。
- 任务证据足以确定正确行为。
- 存在合理干扰信息，模型需要依据条件判断。
- 要求执行的操作必须有实际工具结果，不能只看承诺性回复。

来源：[AutoTask Pipeline](https://app.notion.com/p/3c2b2f39aba580919766c116e0089bed)。

## 3. 已记录批次

| 指标 | 数值 |
|---|---:|
| 规则 | 662 |
| 环境／任务 | 2,025 |
| 其中要求静默的环境 | 485 |
| 执行后判官通过 | 1,443 |
| 进一步渲染筛选后的整轨迹样本 | 1,281 |

485 是任务环境数量，不是最终训练集中的静默样本数量。1,281 也不是 Pro 的数据规模。

## 4. 整轨迹优化

仓库的 AutoTask 记录比较逐轮展开和整轨迹渲染：

| 表示 | 样本 | 打包窗口 |
|---|---:|---:|
| 逐轮展开 | 6,257 | 3,749 |
| 整轨迹 | 1,281 | 896 |

同一窗口长度下，窗口数量及 token 槽位减少约 76.1%。这不是 GPU 时间或吞吐的直接测量。

对应实现检查 token 前缀、监督位置和结构假设。数据表示对应不保证完整训练动态相同；每步监督 token、batch 和更新次数仍需重新核算。

来源：[训练数据结构与优化记录](https://git.garena.com/beeai/algo/post-train/-/blob/milly/docs/rq3w/04_训练数据长什么样_0819.md)。

## 5. AutoTask 效果候选

| 评测集 | Low Thinking 底座 | 后训练 | 差值 |
|---|---:|---:|---:|
| User | 62.60% | 73.60% | +11.00pp |
| Official | 71.00% | 75.48% | +4.48pp |

来源：[AutoTask 效果报告](https://app.notion.com/p/3bbb2f39aba580df8958fd2f9cb3934d)。

该表属于页面标注的 1,425 条 Low Thinking 数据对比，与前述 1,281 条整轨迹批次尚未建立对应关系。页面段落中的“3.5pp”与表内差值不一致，此处采用表内数值相减。

## 6. 可迁移的经验

- 用可执行环境生成能够核验的训练反馈。
- 通过噪声与边界情境训练条件化行为。
- 保留工具和模型配置，防止混入过期协议。
- 按实际来源检查训练／评测隔离。
- 将“环境是否有效”“任务是否完成”“轨迹是否值得模仿”分开判断。

## 7. 实现索引

- [流水线说明](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/autotask_pipeline/README.md)
- [规则与来源抽取](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/autotask_pipeline/extract_source.py)
- [时间审计](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/autotask_pipeline/audit_seed_time_contract.py)
- [文件身份审计](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/autotask_pipeline/audit_drive_identity.py)
- [轨迹收割](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/autotask_pipeline/harvest_select.py)
- [SFT 渲染](https://git.garena.com/beeai/algo/post-train/-/blob/milly/tools/autotask_pipeline/render_sft.py)

自定义规则的 8-gram Jaccard 检查阈值为 0.70；这是一种文本近似检查，不保证语义层面完全无重合。官方模板共享题面时，需要独立环境及相应的泛化定义。
