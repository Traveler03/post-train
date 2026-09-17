# Pro 数据合成 Pipeline：从线上需求到整轨迹 SFT

核验日期：2026-09-17。本文详细解释 Pro v2 的任务、环境、执行轨迹和最终 SFT 数据怎样产生。最终轮次统计针对原版 v7 / pc7 对应的 1,102 条 train；pc8 使用之后过滤出的版本，不能共用这套统计。本文不混入独立的 Docs/Drive RL 项目，也不把后续三用户轮分支当作历史 v2 的默认流程。

这是一组脚本与批次操作组成的产线，不是已经验证可以一条命令从头跑到底的公开产品。本次只做代码、历史记录和产物核查，没有重新造数据、上传 seed、清理账号、运行 Agent、执行判官或训练。

## 先说结论

**线上日志提供真实需求；补齐上下文后，把需求改成独立 query，再配一套合成文件环境，让模型重新执行。通过判官、完成清理的执行轨迹，才成为 SFT 数据。**

不是直接拿线上失败回答训练，也不是让大模型把旧回答改成正确答案。最容易漏掉的中间环节，是“回查完整对话、补全指代”和“为任务准备可执行的文件环境”。

```text
线上日志
  → 提取用户原话 query，保留 trace_id
  → 按 trace_id 找回上下文，改写为独立 query_zh
  → 构造文件环境 seed + 验收目标 expected_output
  → 模型在新会话里实际执行，产生新轨迹
  → 判官 PASS + 数据清理，组成 v2
  → 去 Spike 得到 v2.1，再预处理、切分并训练 pc8
```

### 谁在生成什么

| 角色 | 本链路中的职责 | 不是在做什么 |
|---|---|---|
| 线上日志 | 提供真实需求及理解需求所需的历史 | 不是把线上旧回答直接当本批训练答案 |
| Claude Opus 5 | 改写独立 query，构造环境与验收目标；缺口分支生成新题 | 不提前编好整段 assistant/tool 对话 |
| 环境准备与工具服务 | 根据 seed 准备测试账号里的对象，执行真实工具请求并返回结果 | 不是仅向模型塞一段“假设存在这些文件”的背景文本 |
| DeepSeek-V4-Flash-0731 在线执行模型 | 根据 query、工具定义及逐步观察，生成思考、工具选择、参数和回复 | 不是照着预先指定的固定步骤播放 |
| GPT-5.5 + 综合评估 Prompt v61 | 根据任务、约束与执行证据评价本轮行为 | 不负责生成这条执行轨迹 |
| 转换与训练程序 | 清理和编码轨迹，确定监督位置，产出 SFT 数据 | 不把 tool 返回本身当作 assistant 的生成目标 |

### 三种数据不要混为一谈

| 阶段 | 主要字段 / 内容 | 是否已经是最终 SFT 样本 |
|---|---|---|
| 任务包 | query_zh、world.seed、world.expected_output；另有 world_story 设计说明 | 否：只是题目、初始条件和验收要求 |
| 保存的 Rollout | system_instruction、tools、contents、output、n_call_llm_this_turn，以及运行关联字段 | 否：是这次真实执行的记录，还要清理、转换 |
| 最终训练记录 | messages、tools、sup、meta | 是：一行 JSONL 保存一条整轨迹样本 |

### 阶段与产物总览

以下脚本名用于定位内部实现，公开仓库不包含这些服务脚本的完整副本。

| 阶段 | 实现定位 | 输入 | 主要产物 |
|---|---|---|---|
| 挖需求 | mine_synthesis_queries.py | 线上轨迹与候选元信息、评测 query 参照 | query、trace_id、day、领域 / 工具标签 |
| 回查历史 | materialize_v1_full_traces.py | 候选请求及来源键 | full_traces.jsonl，含原始上下文与 v1_query |
| 独立化改写 | rewrite_queries.py | 当前请求、有限历史文本、领域标签 | query_zh、intent_summary、intent_source、rewrite_ok |
| 造世界与目标 | gen_worlds_prochain.py | 改写后的需求与有限原始参照；也支持手写场景 | worlds.jsonl，逐 case 的 world、ok、warnings |
| 准备可注入文件 | materialize_slides_binaries() 等 | 例如 Slides 页面结构 | 必要的二进制文件与 content_url |
| 装配任务 | build_dataset_csv.py | world 记录 | 扁平 seed JSON + 数据集 CSV，初始含上传占位符 |
| 上传与登记 | upload_seeds.py、upload_csv_to_langfuse.py；历史也有批次上传脚本 | seed JSON、CSV | 回填后的 seed 地址、评测平台 dataset item |
| 实际执行与判分 | fire_batch.py + e2e Runner | dataset item、执行与判官配置 | 新执行 trace、工具证据、判定结果 |
| 收割与转换 | harvest_pass.py、to_sft_reasoning.py | 入选 PASS 列表、执行日志 | 保存轨迹与清理后的中间 SFT 格式 |
| 训练预处理 | prep_prochain.py | 上游 v2.jsonl.gz、编码器和长度配置 | train / val JSONL、token、mask、索引与转换元信息 |

## 1. 从线上日志摘出用户原话：query 不是凭空生成的

`mine_synthesis_queries.py` 先筛选候选日志，再在原始 `contents` 中定位当前用户轮，从文本部分去掉平台注入块，写入 `query`。

输出先是一条轻量记录：`trace_id`、`day`、`query`、办公域和工具标签等，**不是完整训练轨迹**。

当前脚本筛选 `claude-` 来源记录，要求 system / tools 等输入可用，排除依赖附件和识别出的测试账号，要求涉及目标办公工具；随后做 query 去重和与评测 query 的相似性排除。**没有“原轨迹必须 FAIL”的条件**，也不能据此声称所有形式的训练／评测泄漏都已排除。

下面用一个虚构例子贯穿各步，不是线上原始样本：

> 历史里讨论了 A 项目的预算；当前用户说：“把刚才说的那个项目预算汇总成文档。”

这句话就是提取出的原始 `query`，暂时仍然依赖“刚才说的”上下文。

## 2. 回查完整对话，改成能单独执行的 query

这一步实际上有两个脚本，不能跳过前一个。

1. `materialize_v1_full_traces.py` 按 `day` 和 `trace_id` 回查原始日志，保留完整 `contents / output / tool_spans`，把候选 `query` 存为 `v1_query`，得到 `full_traces.jsonl`。
2. `rewrite_queries.py` 将 `v1_query`、当前轮之前最多 12 条历史消息的文本，以及办公域／工具标签交给 **Claude Opus 5**，先判断真实意图，再改写为中文独立请求。

这里的 12 条是历史消息条目，不是 12 个完整用户轮；工具返回不展开给意图改写模型。要求保留任务范围与限定条件，把人名等实体通用化，不凭空增加需求。

示意中的句子会被整理成：

> “请汇总 A 项目的预算明细，并新建一份预算汇总文档。”

关键输出分别是：

- `query_zh`：之后真正发给执行模型的用户请求，必须不依赖旧对话。
- `intent_summary`：供构造环境和人工核对的意图说明，不是执行模型的额外提示。
- `intent_source`：意图来自当前轮、近期历史还是多句合并。
- `rewrite_ok`：意图找不到或改写失败就标为失败，后续应跳过，不能强行编任务。

**原始上下文在这里用来还原需求，不会作为旧会话直接重放给执行模型。**

这一步补的是已有证据支持的意图、对象和约束，不应把未知信息随意补成“用户已提供”，也不能把未授权的请求改成已授权。独立 query 不代表所有业务参数必然齐全；缺信息时，执行模型仍可能需要澄清。

## 3. 给 query 配一套文件环境：这时得到的还只是题目

`gen_worlds_prochain.py` 用 Opus 5 根据 `query_zh`、`intent_summary`、工具标签和有限原始参照，生成一个可执行场景。

主要产物是：

- `seed`：需要预先存在的文件、正文、权限、联系人等；按任务需要也可含邮件、日历。
- `expected_output.goal`：验收目标，说明做到什么才算完成，不是让执行模型照抄的答案。
- `world_story`：内部场景说明，不下发给执行模型。

预算例子中，需要造出真正包含预算明细的 A 项目文件，以及必要的权限；可以加入其他项目文件作为干扰。验收目标可以是“准确汇总这些明细，并成功创建预算汇总文档”。

**不是复制用户线上 Drive 的完整快照，而是保留任务意图、重新配场景。** 对象必须存在，内容应足够支撑任务，文件类型必须能被环境实际使用；输出还要经过 `validate_world()` 的结构和可机械检查的约束校验。

提示词要求替换真实姓名、邮箱等内容，但这不等于已经完成可靠的全量去标识审计。记录中过去出现过失败改写被回退使用的问题；当前代码已加入 `rewrite_ok=False` 跳过逻辑，旧批次仍需要单独核对。

### 3.1 任务包长什么样

下面按代码字段手工写一个**虚构的最小示意**，不是原始 world，也不是已经生成或执行过的训练样本。实际文件名需要满足批次的唯一化约束；字段和值仍要经过环境验证。

```json
{
  "case_id": "example_budget_001",
  "query_zh": "请汇总 A 项目的预算明细，并新建一份预算汇总文档，保留原始文件。",
  "world": {
    "world_story": "用 A 项目明细支持汇总，B 项目文件作为干扰。仅为字段示意。",
    "seed": {
      "drive_files": [
        {
          "id": "f01",
          "filename": "A项目预算_example_budget_001",
          "mime": "application/vnd.google-apps.document",
          "content": "A 项目预算：研发 10 万元，测试 3 万元，设计 2 万元。"
        },
        {
          "id": "f02",
          "filename": "B项目预算_example_budget_001",
          "mime": "application/vnd.google-apps.document",
          "content": "B 项目预算：研发 9 万元。"
        }
      ],
      "contacts": [],
      "emails": [],
      "calendar_events": []
    },
    "expected_output": {
      "goal": "根据 A 项目的预算明细，新建汇总文档，合计 15 万元。",
      "describe": "",
      "key_constraints": "不得混入 B 项目数据，不修改原始预算文件。"
    }
  }
}
```

执行模型得到的是 query 和正常的 Agent 上下文。seed 导入环境后，预算内容需要通过搜索、读取等工具获取；expected_output 留给评估。不能为了方便执行而直接把目标总额作为标准答案塞进模型请求。

seed 还可以包含文件权限与评论；联系人、邮件和日历按任务需要构造，不是每道题都必须有四类资源。此生成器同时支持没有线上参照的手写场景，因此不能仅凭代码主线声称每条历史样本都能对应到唯一线上请求。

### 3.2 Slides 不能停在“有页面描述”

普通文档可使用 content 正文；原生 Slides 使用 slides_content 的标题和要点结构。当前代码提供可选的 --build-pptx 路径：

```text
slides_content
  → python-pptx 创建实际 PPTX
  → 上传文件
  → 回填 content_url 和 source_binary_mime
  → 删除这份记录里临时的 slides_content
```

这个开关默认关闭。缺依赖或上传失败时，代码可能保留页面结构并记录 warning；不能把“world JSON 已生成”理解为“Slides 文件已经可注入”。flatten_seed() 对未转换的 slides_content 只给 warning，仍把该条目写出，不会替调用者完成二进制准备。

### 3.3 世界生成的校验与失败处理

| 情况 | 当前代码行为 |
|---|---|
| rewrite_ok=False | gen_one() 直接跳过，不再用原始线上文本硬造世界 |
| 非法 JSON、goal 为空、文件 / 邮件 / 日历全空、包含不支持的 folder MIME | 解析或 validate_world() 判失败；gen_one() 最多追加一次修复请求 |
| 中文比例低、文件名重复、正文或 Slides 内容不足、疑似缺干扰项 | 记录 warning，不保证自动剔除 |
| 重试后仍失败，或单条生成异常 | 保存失败状态 / 诊断信息，不等于这条已可执行 |

每条生成结束会保存逐 case JSON 和增量文件；正常结束后写最终 worlds.jsonl，二进制补全结果以最终版本为准。**worlds.jsonl 中也保留失败记录，而当前 build_dataset_csv.py 没有统一按 ok 自动过滤输入行。** 因此校验返回失败与“后续所有入口都会硬拦截”不是一回事；发车前需要检查入选文件、ok / warnings、query 和 seed 的实际可用性。本次不据代码存在推断历史批次已经执行了所有检查。

## 4. 导入环境，让模型重新完成任务

`build_dataset_csv.py` 把 world 拆成扁平 seed JSON 和数据集 CSV：

- `query_zh` 对应任务输入 `input.question`。
- 文件等初始化数据通过 `metadata.seed_data_file` 交给环境注入器。
- 验收目标进入 `expected_output.describe`，供评估使用。

### 4.1 从 world 到能运行的 dataset item

1. flatten_seed() 去掉 world / seed 外层包装，输出顶层 drive_files、contacts、emails、calendar_events；后几类资源还做注入器所需的字段转换。
2. build_row() 写入 CSV。seed_data_file 最初是 PENDING_S3_UPLOAD:: 占位，clean_account_data=True 表示请求 Runner 在准备测试账号时清理已有数据。
3. 单独上传 seed JSON 并回填真实地址。当前通用 upload_seeds.py 若有上传失败会停止，不继续写新 CSV；历史批次也使用过单独的上传脚本。
4. upload_csv_to_langfuse.py 将 CSV 转为 dataset item；发现 seed 地址仍是占位符时拒绝继续。CSV 的 expected_output.describe 对应平台对象的 expectedOutput.describe。
5. 在授权的测试账号和运行环境上执行清理、seed 注入及 Agent 运行。文件实际可访问、权限可用，才算完成环境准备。

build_dataset_csv.py 本身不会替调用者上传；它的 --upload 分支当前直接退出。数据集上传与 fire_batch.py 默认 dry-run，真实提交需要对应的显式选项。这里描述的是历史产线的操作边界，不是在公开仓库提供可直接运行的线上命令。

还有一个配置传递细节：build_row() 支持把 pin_account 写成 metadata.user_email，但本次看到的通用上传器 row_to_item() 没有复制该字段。**CSV 写过固定账号，不等于平台最终 item 必然保留它**；若要确认执行账号和工具灰度分支，应核对实际上传脚本、最终 item 和运行日志。该发现不用于反推历史批次一定配置错误。

### 4.2 执行模型看到什么，轨迹怎样增长

随后将 seed 导入测试账号环境，发起新会话。执行模型是 **DeepSeek-V4-Flash-0731 在线版**；`query_zh` 是第一句用户请求，模型看不到此前用于改写的线上对话。

预算例子中，模型需要实际搜索文件、读取明细、完成汇总、创建文档并回复用户。保存的是这次新的 assistant 输出、工具调用及工具返回，**不是线上旧回答，也不是 Opus 写好的标准轨迹**。

这批任务从一个真实用户轮起步；训练所说的“多轮”主要指 assistant 与工具之间的多步交互，不应理解成直接训练完整的线上多用户轮会话。

```text
初始上下文：system + 工具定义 + 平台上下文 + query
  → 模型输出 reasoning_content、可选正文、tool_calls
  → 环境执行这些调用，追加 tool 结果
  → 模型读取增长后的上下文，选择下一步
  → 重复，直到回复 / 澄清 / 转交，或因运行限制与异常结束
```

上图用最终 SFT 字段描述行为；原始执行日志使用 thought 文本、function_call、function_response 等 parts，并不是每个接口都直接输出同一种 messages JSON。字段归一化发生在后面的收割与转换阶段。

不是 Opus 为执行模型预先规定“先搜索两次，再读四页”。执行顺序、参数、是否重试和实际轮数由模型与环境交互产生。一次输出可以包含多个调用；tool 返回来自执行环境，不是合成模型假扮的 tool 消息。超时或触及上限也不保证留下可训练轨迹，仍要看后续判分与清理。

2026-09-17 全量复核最终 1,102 条 train：工具调用轮数中位 4，包含答复的 assistant 监督轮数中位 5。详细分布、统计口径与一条真实轨迹见 [SFT 交互统计](sft-interactions.md)。三批拒绝采样、单轨迹工具交互和后续三用户轮分支的区别见 [Rollout 核查](../reference/check-and-rollout.md)。

### 4.3 实测轮次与轨迹形态

| 最终 train，1,102 条 | 每条平均 | 中位数 | 范围 |
|---|---:|---:|---:|
| sup=true 的 assistant 输出，工具调用和回复都算 | 5.57 轮 | 5 | 1–16 |
| 发起工具调用的 assistant 输出 | 4.63 轮 | 4 | 0–15 |
| 实际工具调用条目 | 7.39 次 | 6 | 0–36 |

只看参与监督的 assistant 消息：1,029 条是前面连续调工具、最后文字回复；8 条直接回复；65 条以 transfer_to_agent 结束。5,097 个工具调用轮中，2,299 个同时有非空 content。因此“前面调工具”不等于“前面完全没有文字”，最后一轮也可能是澄清而非业务完成。

前面讨论的真实模板样本是：搜索记忆 → 搜索文件 → 查找 Slides 工具 → 获取列表与信息 → 读取四页 → 询问缺失客户名。共 5 轮工具调用加 1 轮回复，实际调用工具 11 次；没有继续复制编辑。完整脱敏展开与 assistant 字段示例见 [SFT 交互统计与轨迹](sft-interactions.md)。

## 5. 收集 PASS，清理并转换为 SFT

构建记录中的判官为 **GPT-5.5（`gpt-5.5-2026-04-23`）+ 综合评估 Prompt v61**。判官检查新执行是否满足任务与验收要求，收集 PASS；失败任务可以补跑再筛选，不把失败轨迹直接混入这个 PASS 池。

固定 v61 原文的口径是“当前用户轮是否合理推进任务”：合理澄清或等待确认也可 PASS，不能自动等同于最终业务动作已完成。D1–D5、severity、数据质量闸与 RL 数值奖励的区别见 [Check 评分标准](../reference/check-and-rollout.md)。

### 5.1 判分与拒绝采样

五维分别检查意图与推进、工具使用、事实证据、安全、回复可用性；每维 PASS / FAIL，五维全部 PASS 才综合 PASS。判官可以利用 query、历史、验收目标、seed 摘要和实际执行证据；模型选择不同但有效的工具路径，不应仅因路线不同被判失败。

拒绝采样是在任务与环境基础上重新采执行轨迹、按判定挑选结果，不是把失败回答交给 Opus 改写成一个成功回答。应区分：

- 同一题重新执行：是另一条候选 Rollout，不是原会话中新增一个用户轮。
- 修改 query 或验收目标后执行：题目版本已经变化，应保留版本关联，不能当成完全相同条件的复测。
- 环境不可用、日志缺失、模型执行失误：是不同失败来源，不能全部归为模型能力问题。

### 5.2 PASS 名单怎样变成保存轨迹

harvest_pass.py 使用事先选出的 PASS case / run 关联，并沿评测 trace 找到实际执行 trace；它不是在收割时再调用一次判官。

harvest_one() 的核心操作是：

1. 找目标执行模型的 call_llm 观测，而不是把所有辅助模型与子代理输出混在一起。
2. 按时间排序，读取最后一次目标模型调用。
3. 取该次调用输入中的累计 contents，其中已经包含前面的工具交互；再保留该次 output。
4. 一并保存 system_instruction、tools、thinking_config、运行关联字段，以及 n_call_llm_this_turn。

```text
最后一次调用的 input.contents = 用户请求 + 前面各次模型输出及工具结果
最后一次调用的 output         = 本轮终点输出
两者合起来                   = 这一执行单元的完整轨迹材料
```

因此“收最后一次调用”不等于“只收最后一句回答”。收割仍可能因找不到 trace 链接、没有目标模型调用或读取失败而少于判官 PASS 数；当前批次入口逐条捕获异常继续处理，并将各批写入不同文件，之后再合并去重。单条 API 读取成功也不等于完成全链路完整性审计。

### 5.3 清理与中间格式转换

PASS 之后还需要清理与格式转换：

- 去重，核对执行模型来源，排除非目标模型的辅助轨迹污染。
- 按终局类型处理答复标签，剔除无有效终局、协议 token 泄漏或答复混进思考通道等异常。
- `to_sft_reasoning.py` 按原始 `thought` 标记分开 `reasoning_content` 与 `content`，避免把思考混入正文。
- 训练预处理 `prep_prochain.py` 再转换消息、生成监督标记；这次整轨迹训练的产物包含 `messages / tools / sup`。用户输入和工具返回只作上下文，选定的 assistant 行为才进入直接监督。

因此，**“query + seed + 验收目标”是题目；“通过判定并清理后的实际执行过程”才是训练数据。**

PASS 不代表每一步都最优，仍可能有冗余调用、错误恢复或转交。对需要先确认的任务，正确停下来询问也可以是成功，不是所有成功样本都必须完成写入／删除动作。

注意当前 to_sft_reasoning.py 中有些检查仅记录信号，例如最终答复形态、答复后调工具、工具错误或空结果，丢弃决策留给下游。**v1/v2 构建记录明确未运行通用 merge_filter.py**；本批确有的专项清理与“通用转换器能记录的全部规则”不能画等号。保存 Rollout 与通用转换器要求的输入字段也不完全相同，历史批次有适配与合并步骤，不能把所有脚本不加适配地用管道串起来。

### 5.4 最终 SFT 格式与监督边界

prep_prochain.py:to_openai() 将中间格式转换为最终消息结构：

| 中间内容 | 最终结构 / 检查 |
|---|---|
| assistant 文字后紧跟同一次输出的 tool_call 条目 | 合并成一条 assistant，避免多造一个结束符 |
| 一组没有前置文字的 tool_call | 一条 content 为空、含 tool_calls 的 assistant |
| tool_call 的 name / arguments | function.name / function.arguments；arguments 保存为 JSON 字符串 |
| tool_response | role=tool，通过 tool_call_id 对应调用 |
| reasoning_content | 独立保留，不混进给用户看的 content |
| 历史 / 初始化与当前监督行为 | 转为与 messages 等长的布尔 sup |

当前转换器会拒绝无法解析的工具参数、同一次输出中不一致的监督标记、无法配对的工具结果等结构错误；末尾恰好一个 transfer_to_agent 无普通返回是专门允许的终局，不应当成一般日志截断。

最终样本的训练含义：

```text
messages[i] 是 system / user / tool      → sup[i] = false
messages[i] 是平台初始化 assistant       → sup[i] = false
messages[i] 是选中的模型 assistant 行为  → sup[i] = true
```

被选中的思考、正文、工具名称和参数参与监督，工具返回只提供下一步决策的上下文。本批按 whole 方式保存并渲染整轨迹，含多个监督段；不能把 6 个 assistant 监督轮说成 6 条最终 train 记录，也不能说只有末尾回复进 loss。

预处理还进行动作数 / 内容 round-trip 检查、token 编码、长度控制、切分及 token-level mask 构造。已保存 v7 元信息报告读入 1,149、仅因超过 98,304 token 丢弃 15、留下 1,134；不是每个代码中存在的异常分支都在这次输入上发生过。

切分元信息称按 rule_head 整组切，任务组交集为 0；实现的 group key 还会回退到 trace_id 或行序号。最终 train 的五个 meta 字段全部为空，不能把这个结构性“组交集为 0”扩大成已经完成语义任务、用户或线上来源层面的全部去重。分组与文件指纹见 [转换元信息](../source-materials/artifacts/pro-v7/meta.json)。

## 6. v2 怎样补缺口：同一个环境，可以再出不同的题

前面是“从线上需求构造新场景”的主线。v2 还增加了一条分支：**复用已经检查过的世界，seed 不改，只改 query 和验收目标，再重新执行、收 PASS。**

`gen_gap_queries.py` 针对缺口生成 Docs 写入、Slides 创建、Drive 移动／删除等任务。它深拷贝原 world，替换 `query_zh` 和 `expected_output.goal`，保留环境内容。

例如，同一套已有文件可以分别构造两类题，以下仍是示意：

- 明确授权：“把 A 项目已确认作废的旧预算文件移到回收站。”正确行为是核实对象后执行。
- 试探询问：“A 项目这份旧预算是不是可以删了？”在这类未明确授权的高风险场景中，应先确认，不直接删除。

不是给相同 query 配相反标签，而是通过不同请求表达不同授权范围，再配相应的验收目标。**因此训练样本与线上请求不是一对一关系。**

根据 v2 构建记录：新增 792 个 query，三轮累计判定 PASS 525 个；收割去重、排除污染后得到 519 条，进一步清理后保留 468 条。最终 **v2 = 老 PASS 池选出的 681 条 + 新缺口轨迹 468 条 = 1,149 条**。

```text
已有合成世界 ──选可支持目标动作的环境──→ 新 query + 新 goal，seed 不变
                                            ↓
                                      重新执行与判分
                                            ↓
792 个新增 query → 累计 PASS 525 → 收割 / 去重 519 → 清理后 468
                                                           ↓
                         老合成池选出 681 ───────────→ v2 共 1,149
```

gen_gap_queries.py 以 base_case_id 记录复用关系，深拷贝 world 后替换目标。这里可能一套世界对应多个 query，因此数世界、数 query、数执行尝试和数最终轨迹会得到不同数字。

三批通过率记录为 55.3% → 30.2% → 22.5%，分母不同；不是每批都对完整 792 题执行，也不是用户连聊三轮。现有材料支持累计漏斗，不能用这三个百分比反推未经核验的每批精确候选数或完整尝试总量。

老池也来自上述合成与重新执行流程；“老池”不是“原始线上轨迹直接入训”。v2 还对老池只读样本做了工具覆盖保留、按目标评测域分配配额和检索密集轨迹优先选择。这里是构建记录的版本级说明，不等于已把最终每条数据重新串回线上 trace、seed 和判官结果。

## 7. 去 Spike 在最后哪一步

前述流程先产出 v2；**去 Spike 是对已构造好的 v2 再做训练反馈驱动的删除，不是最初挖 query，也不是只删除判官 FAIL。**

- 原版：v2 共 1,149 条；预处理丢弃 15 条超长记录后，Train / Val 为 1,102 / 32，用于 pc7。
- 过滤版：从 v2 删除 92 条，得到 v2.1 共 1,057 条；预处理同样丢弃 15 条超长记录后，Train / Val 为 1,010 / 32，用于 pc8。

源文件差分确认这 92 条是纯删除，没有新增或改写保留记录。历史按 assistant 内容指纹定位删除，但候选选择脚本和完整 spike → 样本映射仍未找全；不能说已复原全部筛选规则。两版 Val 数量相同，成员是否相同尚未核对。分布、训练曲线及前期对比见 [Spike 报告](spike-study/README.md)。

| 名称 | 在链路中的含义 | 原始条数 | 最终 Train / Val |
|---|---|---:|---:|
| v2 | 合成、实际执行、PASS 收割及清理后的上游数据版本 | 1,149 | 1,102 / 32 |
| v7 | 已保存训练数据目录 / 配方名称，其元信息指向 v2 上游 | 同上 | 1,102 / 32 |
| pc7 | 去 Spike 对照中的原版训练臂 | 使用 v2 | 1,102 / 32 |
| v2.1 / pc8 | 先从 v2 删除 92 条，再预处理的过滤版 / 训练臂 | 1,057 | 1,010 / 32 |

相同数据规模不代表不同训练运行是同一个实验。仓库中的 Milly v7 复跑日志、历史 pc7 对照及 checkpoint 评测的绑定范围仍应分别看各自证据，不能只凭“v7 / pc7”标签互相替代。

## 8. 真正的多用户轮分支在什么位置

上述主线是一个 query 起步；线上历史在改写阶段被用来理解需求，不要求真人在 Rollout 中继续回复。如果模型提出澄清，这条单用户轮样本可以到此结束。

若要训练“收到信息或确认后继续执行”，后续 gen_multiturn_queries.py 另生成：

```text
T1 fixed：固定的用户请求
  → 模型与工具多步交互，可能提出澄清或请求确认
T2 simulated：模拟用户根据历史补信息、确认或要求继续
  → 模型继续执行
T3 simulated：继续追问或表示结束
```

这是最多三个配置用户轮，不是固定三个模型输出。模拟用户的指令给模拟器，不是额外提示给执行模型；每个用户轮内部都可以有多次工具调用。hard severity 或执行异常可提前停止。当前 mixed 模式对 [END] 跳过该模拟轮，auto 模式才终止整个会话。

这条分支将 multi_turn 写入任务输入；后续收割可保存累计整段会话，转换器按 is_multiturn 调整监督边界，并利用上游 no_loss_turns 排除问题轮。独立会话 Judge 需要额外配置，不能只凭一个 PASS 标签断言已做完整会话级验收。详见 [多用户轮与 Judge 的版本边界](../reference/check-and-rollout.md)。

**不能用这段后续代码解释原版 1,102 条 train 的用户轮次，也不能把“请求确认→用户授权→继续执行”压成一条初始已授权 query 后，声称训练到了同样的交互能力。**

## 9. 怎样复核这条 Pipeline

### 历史批次应逐项核对的材料

| 检查点 | 应核对的证据 | 单看什么还不够 |
|---|---|---|
| 需求来源与独立化 | 候选 trace、当前请求、改写结果、rewrite_ok | 只有一段看起来合理的 query |
| 环境可执行 | 入选 world、ok / warnings、文件正文 / 二进制、可用 seed 地址、注入结果 | 只有 worlds.jsonl 或 CSV 已生成 |
| 运行条件 | 最终 dataset item、账号与工具分支、执行模型、判官版本、run 配置 | 只有脚本默认参数或 CSV 中的账号字段 |
| 执行证据 | 实际 trace、模型调用与工具返回、终局和异常信息 | 只有模型说“完成了” |
| PASS 与收割 | 判定结果、PASS 名单、实际执行关联、收割异常和去重结果 | 只有文件名里带 pass |
| 训练输入 | 源文件 SHA、转换参数与丢弃原因、messages / sup、split 指纹 | 只有样本总数相同 |

以上是复核清单，不表示已经恢复了每条最终样本的全部 lineage。当前已核对版本级流程、保存产物指纹、全量最终 train 结构及第一条样本的结构对应；仍缺完整线上请求 → seed → 执行 → 判官 → 最终行的逐条关联。

### 公开仓库能安全复算什么

不需要私有输入的离线测试：

```bash
python3 -B scripts/test_audit_sft_interactions.py
```

有本地原件读取权限时，可以只读复算聚合统计：

```bash
python3 -B scripts/audit_sft_interactions.py \
  --train /path/to/data_prochain_v7/train.jsonl \
  --rollouts /path/to/v2_pass_trajectories.jsonl
```

该脚本不执行数据生成或工具任务，只输出聚合数与源文件指纹，不输出原始请求、参数、账号或思考正文。已发布 [统计 JSON](../source-materials/artifacts/pro-v7/interaction-stats.json)、[脱敏轨迹摘要](../source-materials/artifacts/pro-v7/trajectory-example-outline.json)与[SFT 逐消息说明](sft-interactions.md)。

### 可以对外准确概括成什么

> 我们先从线上日志提取真实需求，利用历史还原意图并改成独立 query，再用合成模型构造可执行文件环境和验收目标。随后让目标模型在测试账号中真实调用工具，收集模型输出与工具反馈，用判官和专项清理筛选轨迹，最后转换成整轨迹、多监督段的 SFT 数据。为补齐能力缺口，还会复用已有环境，重新生成不同操作或授权条件的题目。这里的多轮主要指一个用户请求内的模型—工具交互；真正的多用户轮属于另一个后续分支。

## 实现依据与边界

本次依据本地脚本和 v1/v2 构建记录补齐以上链路，具体文件、函数和证据范围见[数据构造来源索引](sources.md#数据构造链路补充核查)。本次扩充流程说明和聚合统计，不上传这些内部脚本的完整副本、线上原文、合成环境原文或服务配置；预算例子为示意，模板轨迹为明确标注的脱敏结构摘要。

- 构建记录明确 v1/v2 **没有运行通用 `merge_filter.py`**，不能把生产轨迹清洗流程里的全部规则算作这批数据已经执行过的步骤。
- 历史记录对执行时 High/Low 档位存在冲突，这里只报告已确认的模型版本，不根据端点名称推断档位。
- 本次是流程与代码核查，没有重新调用模型、造数据、执行判官或训练，也没有重新逐条认定全部样本都应为 PASS。
