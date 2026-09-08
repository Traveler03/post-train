# scripts 目录说明

> 📌 **正本** —— 照着做的操作文档。发现不对请就地改,别另开一篇。

> ## 📍 0818 迁移:`data/autotask_rq2/scripts/` → `tools/autotask_pipeline/`
>
> **为什么搬**:它是 rq1/rq2/rq3w **三条线共用的工具链**,挂在 `data/autotask_rq2/` 下
> 是历史包袱 —— 它当初是为 rq2 建的,后来 rq3w(最新那批合成数据)的造世界、装配、
> 渲染、收割**全靠这一套**,没有另起一份。而且 `data/` 放的是**产物**,工具不该在里面。
>
> **搬迁做了什么**:`git mv`(75 个文件,历史保留)→ 修 3 处自指路径
> (`fire_high.py` 里 `ROOT/'scripts'` 改成 `HERE`)→ 改 8 个调用方 + `MANIFEST.json`
> 的 `canonical` → 44 个模块逐个真 import 验过 → 83 个测试 + manifest clean。
>
> ⚠️ **还剩 10 处 `parent.parent/'out'` 的历史默认值**指向不存在的 `tools/out/`。
> 它们是死代码(`data/autotask_rq2/out/` 本来就是空的,7/9 的脚本有必填 CLI 参数
> 盖掉它们),而且现在会**响亮地失败**而不是静悄悄读到旧批次的残留 —— 比原来安全。
>
> 已查实的共用证据:
>
> - **8 个脚本里直接硬写了 rq3w 分支**:`assemble_items.py`、`audit_item_language.py`、
>   `extract_source.py`、`preflight_anchor.py`、`verify_effort.py`、`run_batch_to_sft.sh`、
>   以及两个测试 `tests/test_seed_source_pipeline.py`、`tests/test_pipeline_contracts.py`。
> - rq3w 的运行日志里调过 `verify_model.py`(12 处)、`upload_all.py`(4 处)。
>
> **⇒ 改这里的任何脚本,要按 rq1 / rq2 / rq3w 三条线一起验**,别只想着 rq2。
> 改完必须跑 `python3 -m pytest -q tests` 再 `verify_manifest.py`。
>
> **为什么不把目录搬走**:搬 `tools/autotask_pipeline/` 会打断所有脚本里的相对路径
> (以及 `MANIFEST.json` 里写死的 `canonical` 绝对路径),风险大于收益。
> 0818 的决定是**只理归属、不动物理位置**。
>
> 三条线各自的产物在哪:
>
> | 线 | 产物位置 | 入 git 吗 |
> |---|---|---|
> | rq1(教师采样版,0803) | `data/autotask_rq1/` | 部分 |
> | rq2 | `data/autotask_rq2/`(大数据已 ignore) | 部分 |
> | **rq3w(最新)** | `data/autotask_rq3w/v7_20260812/` | **整个在 .gitignore** —— 文档另放 `docs/rq3w/` |
>
> rq3w 的整理计划与**可复现必留文件清单**见 `docs/rq3w/00_整理计划与可复现清单_0818.md`;
> 它前面三代(v2/v3/v4)为什么整批作废见 `docs/rq3w/01_世代作废记录_v2v3v4.md`。

根目录只保留当前 rq2 主链与仍可复用的诊断/修复工具；硬编码 rq1 数据集、固定数量或 0801–0802 一次性事故修复的脚本已移到 `archive/rq1/`。

当前主链：

1. `extract_source.py` → `build_user_mailboxes.py` → `gen_worlds.py` → `lint_cases.py`
2. `assemble_items.py` → `audit_drive_identity.py`（隔离注入后无法保持的精确 ID/gid）→ `make_rollout_indices.py`（全量 r1 + 分层 pilot）→ `upload_all.py`
3. `lane_manager.py` / `run_rounds.py`（显式透传 model config/effort）→ `fire_high.py` → `submit_batch.py`
4. `mail_safety_audit.py` → `harvest_select.py`
5. `render_sft.py` / `render_dpo.py` → 可选 `split_sft_by_user.py`

新批次在题库已装配并通过 lint 回执后，标准下游入口是
`run_batch_to_sft.sh`。它要求显式传入批次、工作目录、数据集、模型、操作员邮箱和 PII 策略，
不再复制上一批的硬编码 launcher。`finalize_rq2d.sh` 是 rq2d 历史交付器，不是新批次入口。

依赖固定在 `requirements-pipeline.txt`；发车前先跑 `verify_manifest.py`。
`verify_manifest.py --update` 只能在脚本修改已 review、测试已通过后显式执行。

源数据收益审计使用 `audit_seed_sources.py`；它只做确定性统计，不调用模型。

发车闸：`audit_drive_identity.py`、`preflight_anchor.py`、`verify_model.py`、`verify_effort.py`、`midflight_check.py`、`safety_watch.sh`。

后处理：`cap_sft.py` 控制同题同答案堆积。当前主链只采信平台自动 GPT-5.5 判定；`local_judge.py` 在 rq2 那批只是诊断工具。
⚠️ 0818 补:**它在评测线上已经转正** —— 平台判 v11 的卷(0815 及更早)要靠
`local_judge.py --mode verify` 补出 v13 才能报分,见 `CLAUDE.md` 开头。这条别按 rq2 的旧口径理解。

`archive/rq1/` 仅保留事故复盘证据，里面脚本的固定路径、凭证、数据集名和数量都可能过期，不应直接运行。

离线改动后运行 `python3 -m pytest -q tests`；
`MANIFEST.json` 必须覆盖全部活动 `.py/.sh`、测试和依赖文件，且 SHA256 全部匹配后才允许发车。
