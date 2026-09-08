#!/usr/bin/env python3
"""全量重写 gold:只改 items 里的 `expectedOutput`,**seed 一个字不动**。

## 为什么值得做(实测,不是推测)

低档探路卷 30 条,同一批模型答案、同一个 v13 判官,唯一变量是 gold:

| gold 版本 | 要点数中位 | pass rate |
|---|---:|---:|
| 原版(`1-4 条`模板顶格出 4 元素) | 5 | 23.3% |
| v1 分号字符串 + 按极性限条数 | 4 | 40.0% |
| v2 硬压到 3 条 | 3 | 33.3% ↓ |
| **v3 原子化 + 不焊 seed 数值** | 4 | **46.7%** |

**23.4 分的差距只来自 gold**,占「rq3w 比评测集难 28 分」的 84% —— 所以
2,025 个世界**不用重生**,重写 gold 就够。7 条翻正、0 条翻负(没有放宽任务本身)。

## v2 那次为什么反而掉分(两个坑,规格里已堵)

1. **硬压条数导致合并**,合并出的「复合要求」判官当一条判,任何一半没满足整条挂。
   宁可 4 条原子的,不要 3 条复合的。
2. **压缩时把 seed 的具体数值焊进判分点**(`必须报出 09:30/10:00/11:30`),
   本来只要求「提到明早日程」,焊上精确时刻后少报一个就被判**幻觉**。

## 与 rollout 的关系

seed 没变 → `seed_sha` 不变 → **S3 的 seed 不用重传**,只要重传 items。

## 用法

    export COMPASS_API_KEY=$(cat relay/.compass_key)
    python3 regold_batch.py --items <items.jsonl> --seeds <seeds 目录> \\
        --out <新 items.jsonl> --cache <cache.jsonl> --workers 8

按 item id 缓存,中断了重跑不重复花钱。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_coverage_judge import call                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools/autotask_pipeline'))
from extract_source import infer_lang                             # noqa: E402

# 语种代码 → 给模型看的名字。写代码(`pt`)模型认得不牢,写名字最稳。
LANG_NAMES = {'pt': '葡萄牙语(Português)', 'en': '英语(English)',
              'id': '印尼语(Bahasa Indonesia)', 'vi': '越南语(Tiếng Việt)',
              'ja': '日语(日本語)', 'zh': '中文', 'es': '西班牙语(Español)',
              'thai': '泰语(ภาษาไทย)', 'fr': '法语(Français)'}
from regold_and_rejudge import (                                  # noqa: E402
    DRIVE_SYSTEM, REALTIME_SYSTEM, REGOLD_SYSTEM,
    merge_not_judged, parse_json, points_of,
)
from diagnose_failures import seed_summary                        # noqa: E402

ORIGINAL = re.compile(
    r'Original query at creation:\s*(.*?)(?:\n\n##|\n- Match evidence:|\nBased on)', re.S)


def rule_of(question: str) -> str:
    match = ORIGINAL.search(str(question or ''))
    return match.group(1).strip() if match else str(question or '')[:1500]


def load_seed(seeds_dir: Path, item_id: str) -> dict:
    """seed 在本地,不用走 langfuse。item id 形如 `rq3w7_rec_alrt_en_xxx__r1`,
    seed 文件名去掉 `__r1`。"""
    name = re.sub(r'__r\d+$', '', item_id or '')
    path = seeds_dir / f'{name}.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def regold_item(item: dict, seeds_dir: Path, system: str = REGOLD_SYSTEM) -> dict | None:
    inp = item.get('input') or {}
    old = item.get('expectedOutput') or {}
    seed = load_seed(seeds_dir, inp.get('id') or '')
    rule = rule_of(inp.get('question'))
    # ⛔ 语种必须**现算并显式钉在提示词开头**。0819 实案:SYSTEM 通篇中文、
    #    又没写语言要求,于是 84% 的 gold 被写成中文(规则语种最多的是葡语)。
    #    ⚠️ 别读 item metadata 里的 `lang` —— 池子那个标签 33.7% 是错的。
    lang = infer_lang(rule, None)
    prompt = (f"【gold 必须用这个语言】{LANG_NAMES.get(lang, lang)}\n\n"
              f"【规则原文】\n{rule}\n\n"
              f"【账号 seed(摘要,只为判断证据够不够,不要复述进 describe)】\n"
              f"{seed_summary(seed, 1200)}\n\n"
              f"【原 gold】\n{json.dumps(old, ensure_ascii=False)[:2500]}\n\n"
              f"【再说一遍】三个字段都用 {LANG_NAMES.get(lang, lang)} 写。")
    gold = parse_json(call(prompt, system=system), 'key_constraints')
    return merge_not_judged(gold) if gold else None


def gold_language_check(out_path: Path) -> tuple[int, int]:
    """数一数产物里有多少条 gold 的语种跟规则原文对不上。返回 (不符数, 够长可判数)。

    ⚠️ 太短的 gold 判不了语种,不计入分母(中日韩一个字顶拉丁两三个,所以按加权字数卡)。
    ⚠️ 别读 metadata 里的 `lang` —— 池子那个标签 33.7% 是错的,一律现算。
    """
    bad = n = 0
    for line in out_path.open(encoding='utf-8'):
        if not line.strip():
            continue
        item = json.loads(line)
        eo = item.get('expectedOutput') or {}
        kc = eo.get('key_constraints')
        kc = kc if isinstance(kc, str) else ' '.join(str(x) for x in (kc or []))
        text = ' '.join([str(eo.get('goal') or ''), str(eo.get('describe') or ''), kc])
        if sum(2.5 if ord(ch) > 0x2E80 else 1 for ch in text) < 80:
            continue
        want = infer_lang(rule_of((item.get('input') or {}).get('question')), None)
        n += 1
        if infer_lang(text, want) != want:
            bad += 1
    return bad, n


def load_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', required=True, type=Path)
    parser.add_argument('--seeds', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--realtime-ids', type=Path,
                        help='外部实时题清单 → 套「只判行为+格式」范式')
    parser.add_argument('--drive-ids', type=Path,
                        help='Drive/表格题清单 → 套「判语义定位」范式')
    parser.add_argument('--only-ids', type=Path,
                        help='只重写这些 id(定向补跑用),其余原样透传')
    args = parser.parse_args()

    realtime, drive = load_ids(args.realtime_ids), load_ids(args.drive_ids)
    only = load_ids(args.only_ids)
    both = realtime & drive
    if both:
        raise SystemExit(f'⛔ 实时题与 Drive 题清单重叠 {len(both)} 条,套哪个范式说不清:'
                         f'{sorted(both)[:3]}')

    def spec_of(item_id: str) -> tuple[str, str]:
        if item_id in realtime:
            return REALTIME_SYSTEM, 'realtime'
        if item_id in drive:
            return DRIVE_SYSTEM, 'drive'
        return REGOLD_SYSTEM, 'general'

    items = [json.loads(line) for line in args.items.open(encoding='utf-8') if line.strip()]
    if args.limit:
        items = items[:args.limit]
    cache: dict = {}
    if args.cache and args.cache.exists():
        for line in args.cache.open(encoding='utf-8'):
            if line.strip():
                row = json.loads(line)
                if row.get('id'):
                    cache[row['id']] = row.get('gold')
    print(f'{len(items)} 条 items,缓存命中 {sum(1 for i in items if (i.get("input") or {}).get("id") in cache)}',
          file=sys.stderr)

    done = [0]
    # ⚠️ 缓存必须**边跑边写**。0814 踩过:原实现攒到最后统一 append,跑到 88 分钟
    # 撞上 `timeout 5400` 被 SIGTERM,2,000 多次判官调用**一条没落盘**,全白花。
    # 每条一落盘 + flush,中断了重跑只补差集。
    cache_fp = args.cache.open('a', encoding='utf-8') if args.cache else None
    if cache_fp:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache_lock = threading.Lock()

    def remember(item_id: str, gold: dict) -> None:
        if not cache_fp:
            return
        with cache_lock:
            cache_fp.write(json.dumps({'id': item_id, 'gold': gold},
                                      ensure_ascii=False) + '\n')
            cache_fp.flush()

    def one(item):
        item_id = (item.get('input') or {}).get('id')
        # ⚠️ 顺序要紧:**先查缓存,再看 --only-ids**。反过来写的话,定向补跑时
        # 那些已经重写好、只是不在补跑清单里的题会被当成「清单外」原样透传,
        # 输出文件里就变回旧 gold —— 等于把前面几小时的成果冲掉(0815 差点踩)。
        # `--only-ids` 的语义是「只对这些花钱调判官」,不是「只保留这些的结果」。
        if item_id in cache and cache[item_id]:
            return item_id, cache[item_id], 'cached'
        if only and item_id not in only:
            return item_id, None, 'skipped'          # 没缓存又不在清单里:保留原 gold
        system, kind = spec_of(item_id)
        try:
            gold = regold_item(item, args.seeds, system)
        except Exception as exc:
            return item_id, None, f'ERROR: {str(exc)[:80]}'
        if gold:
            remember(item_id, gold)
        done[0] += 1
        if done[0] % 100 == 0:
            print(f'  重写 {done[0]} 条', file=sys.stderr)
        return item_id, gold, f'new:{kind}'

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, items))

    golds = {i: g for i, g, _ in results if g}
    status = Counter(s for _, _, s in results)
    skipped = {i for i, _, s in results if s == 'skipped'}
    before, after, failed = [], [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as handle:
        for item in items:
            item_id = (item.get('input') or {}).get('id')
            old = item.get('expectedOutput') or {}
            before.append(points_of(old))
            gold = golds.get(item_id)
            if gold:
                item = {**item, 'expectedOutput': {
                    'goal': gold.get('goal', ''), 'describe': gold.get('describe', ''),
                    'key_constraints': gold.get('key_constraints', '')}}
                after.append(points_of(item['expectedOutput']))
            elif item_id not in skipped:
                failed.append(item_id)          # 重写失败的**保留原 gold**,不丢题
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')

    if cache_fp:
        cache_fp.close()

    lang_bad, lang_n = gold_language_check(args.out)
    before.sort(); after.sort()
    print(json.dumps({
        'items': len(items), 'rewritten': len(golds), 'failed': len(failed),
        'status': dict(status),
        'points_median_before': before[len(before) // 2] if before else None,
        'points_median_after': after[len(after) // 2] if after else None,
        'gold_lang_mismatch': f'{lang_bad}/{lang_n}'
                              + (f' = {100 * lang_bad / lang_n:.1f}%' if lang_n else ''),
    }, ensure_ascii=False, indent=2))
    # ⛔ 出口闸:重写完的 gold 语种必须还跟规则原文一致。
    #    lint_cases 查的是**重写前**的世界,这一步在它之后,所以那边的 W13 拦不到 ——
    #    0815 那批就是这么让 79% 的 gold 变成中文并且一路发出去的。
    if lang_n >= 20 and lang_bad / lang_n > 0.15:
        raise SystemExit(
            f'⛔ {lang_bad}/{lang_n} = {100 * lang_bad / lang_n:.1f}% 的 gold 语种'
            f'跟规则原文不一致(上限 15%)。\n'
            f'   0815 全量那批就是 79% —— 提示词通篇中文、又没写语言要求,'
            f'模型顺手全写成了中文。\n'
            f'   产物已经写出来了({args.out}),但**别拿它发车** —— '
            f'先看 REGOLD_SYSTEM 的语言条款和 regold_item 里钉语种那两行还在不在。')
    if failed:
        print(f'⚠️ {len(failed)} 条重写失败,已**保留原 gold**(不丢题):{failed[:5]}')
    print(f'✅ {args.out}')


if __name__ == '__main__':
    main()
