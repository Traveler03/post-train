#!/usr/bin/env python3
"""把题库和种子里的日期模板**按锚点提前算成绝对日期**(烘焙),烘完一个 `{{ }}` 都不剩。

## 一句话

原来是「发车时让平台去算日期」,现在改成「发车前我们自己算好」。
平台拿到的是纯文本,它那套模板机制就不参与了 —— 三个坑一次全平。

## 为什么必须烘(0808 复盘,三条独立证据)

### ① 判官看到的是模板原文,模型看到的是真日期 → 判官判模型「编造日期」

灌种子时 `data/seed_inject.py` 会把 `emails.subject/body/body_html` 等字段过一遍
jinja 渲染,所以**模型**收件箱里是 `Tuesday, August 11, 2026`。
但喂判官的 `summarize_seed_for_judge()` 走的是 `_enrich_seed_for_span()` ——
**只算绝对时间、不跑 jinja**,所以**判官**看到的还是
`{{ fmt_date(add_days(run_date, 1), 'long') }}`。

实测抽样 116 条:**94.8% 的用例**判官的 seed_data 里带未渲染占位符。
抓到的现行(rq2_alrt_en_0000_fire__r4,判官原话):

    「正文把"Tue Aug 11"为截止日表述为确定值,但模板邮件中截止日期实际为
      占位符 {{ fmt_date(...) }},存在时间数据编造风险」  → 判 FAIL,归因『幻觉』

**日期题几乎全军覆没,而提醒类任务绝大多数都涉及日期。**

### ② drive_files 压根不在框架的渲染白名单里 → 连模型都看到模板

白名单只有 emails / drafts / calendar_events 那几个字段。
`drive_files[].content` 和 `.filename` 里的 **3,660 个占位符从来没被渲染过**,
波及 398 份种子(17.5%)。模型打开 Drive 文档,看到的就是 `{{ fmt_date(...) }}`。

### ③ 一个坏占位符会连坐整段

`_render()` 渲染抛错时**整段字符串原样返回**。所以非法 style(见 fix_templates.py)
不只是它自己不渲染,同一封信里其它好占位符也跟着废。

## 烘焙之后

    模型看到的  = 绝对日期   ✅
    判官看到的  = 同一个绝对日期  ✅(它读的就是种子原文)
    Drive 文档  = 绝对日期   ✅
    平台侧行为  = 零变化(没有 {{ 了,`_render` 直接原样透传)

**对平台零 diff** —— 我们没改平台任何东西,只是把它要算的东西提前算好。

## 前提(两条,都会硬闸)

1. **必须先跑 `fix_templates.py`** —— 有渲染不出来的占位符就烘不动,会退 2。
2. **锚点必须和 `override_timestamp_ms` 是同一个时刻**(`preflight_anchor.py` 的防漏烤闸)。
   ⚠️ 0819 更正:原来这条写的是「锚点必须等于**发车当天**」——那条规矩的前提
   (后端不认 override_timestamp_ms)0819 已被实测推翻,锚点可以是未来某天;
   现在要守的是**题面和锚点别脱节**。烘焙把日期钉死,
   隔天再发车 = 模型看到的世界跟它的钟差几天。

## 用法

    # 装配完、上传前跑
    python3 bake_dates.py --items out_v2/rq2_items.jsonl --seeds out_v2/seeds

    # 先看会烘出什么,不落盘
    python3 bake_dates.py --items ... --seeds ... --dry --sample 3

退出码:0 = 烘完零残留;2 = 有烘不动的(名单打出来)。

⚠️ 烘焙是**不可逆**的(模板被替换成具体日期)。原地改之前脚本会自动留一份
   `<seeds>.pre_bake/` 和 `<items>.pre_bake`。
"""
import argparse
import calendar
import importlib.util
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_TEMPLATE_RENDER = Path(
    '/home/work/migoo_ai_public/posttrain/code/beeai_eval/runner/template_render.py'
)
_SPEC = importlib.util.spec_from_file_location('_beeai_template_render', _TEMPLATE_RENDER)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
_ENV = _MODULE._ENV
INJECT_TIMEZONE = _MODULE.INJECT_TIMEZONE

TZ = ZoneInfo(INJECT_TIMEZONE)          # Asia/Shanghai,跟框架灌种子同一时区
TZ8 = timezone(timedelta(hours=8))
PH = re.compile(r'\{\{.{0,200}?\}\}', re.S)

# 框架没有、但我们题里用到的 helper。烘完就没占位符了,所以框架永远不需要认识它们。
def _add_months(value, months):
    dt = _ENV.globals['_to_dt_'](value)
    total = dt.year * 12 + dt.month - 1 + int(months)
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


EXTRA_HELPERS = {
    'iso_week': lambda v: f'W{_ENV.globals["_to_dt_"](v).isocalendar().week:02d}',
    'add_months': _add_months,
}


def _install_helpers():
    _ENV.globals['_to_dt_'] = _MODULE._to_dt
    _ENV.globals.update(EXTRA_HELPERS)


def bake_text(s, rd, stat):
    """把一段文本里的所有 `{{ }}` 用锚点 rd 渲染成绝对值。

    ⚠️ 逐个占位符渲染,**不整段渲染** —— 整段渲染时任何一个坏占位符都会让
    整段原样返回(框架就是这么丢日期的)。逐个渲染能把坏的隔离出来单独报。
    """
    if not isinstance(s, str) or '{{' not in s:
        return s

    def one(m):
        try:
            out = _ENV.from_string(m.group(0)).render(run_date=rd, seed={})
            stat['ok'] += 1
            return out
        except Exception as e:
            stat['bad'] += 1
            stat[f'bad::{type(e).__name__}: {str(e)[:50]} ← {m.group(0)[:60]}'] += 1
            return m.group(0)
    return PH.sub(one, s)


def walk_bake(o, rd, stat):
    if isinstance(o, str):
        return bake_text(o, rd, stat)
    if isinstance(o, dict):
        return {k: walk_bake(v, rd, stat) for k, v in o.items()}
    if isinstance(o, list):
        return [walk_bake(v, rd, stat) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True, help='装配好的题库 jsonl')
    ap.add_argument('--seeds', required=True, help='装配好的种子目录')
    ap.add_argument('--anchor', default='',
                    help='题里没写 override_timestamp_ms 时的兜底锚点 YYYY-MM-DD,'
                         '默认今天(+08:00)。⛔ 必须是发车当天')
    ap.add_argument('--dry', action='store_true', help='不落盘')
    ap.add_argument('--sample', type=int, default=0, help='--dry 时打印几条烘焙前后对照')
    a = ap.parse_args()
    _install_helpers()

    fb_date = (datetime.strptime(a.anchor, '%Y-%m-%d').date() if a.anchor
               else datetime.now(TZ8).date())
    fb = datetime(fb_date.year, fb_date.month, fb_date.day, 9, 0, tzinfo=TZ)
    drift = (fb_date - datetime.now(TZ8).date()).days
    print(f'兜底锚点 = {fb_date}({fb_date.strftime("%A")}),与今天差 {drift:+d} 天')
    if drift:
        print(f'⚠️ 兜底锚点不是今天 —— 烘出来的日期会跟发车当天差 {drift} 天')

    items_p, seeds_p = Path(a.items), Path(a.seeds)
    stat = Counter()

    # ---- ① 收每题的锚点,顺带建 题→种子 的映射 ----
    lines = [l for l in items_p.read_text(encoding='utf-8').splitlines() if l.strip()]
    anchors = {}          # 种子名(去 __rK) -> run_date
    n_frozen = n_fallback = 0
    for l in lines:
        o = json.loads(l)
        iid = o['input']['id']
        base = re.sub(r'__r\d+$', '', iid)
        ms = o['input'].get('override_timestamp_ms')
        if ms:
            rd = datetime.fromtimestamp(int(ms) / 1000, TZ)
            n_frozen += 1
        else:
            rd = fb
            n_fallback += 1
        prev = anchors.get(base)
        if prev is not None and prev != rd:
            # 同一道题的 5 次 rollout 必须同锚点,否则 5 份世界不一样、冻结就白做了
            sys.exit(f'⛔ 题 {base} 的多次 rollout 锚点不一致:{prev} vs {rd}')
        anchors[base] = rd
    print(f'题 {len(lines)} 条 / {len(anchors)} 道;带冻结锚点 {n_frozen} 条,'
          f'用兜底锚点 {n_fallback} 条')

    # ---- ② 烘题库 ----
    out_items, shown = [], 0
    for l in lines:
        o = json.loads(l)
        base = re.sub(r'__r\d+$', '', o['input']['id'])
        rd = anchors[base]
        before = l
        new = walk_bake(o, rd, stat)
        out_items.append(new)
        if a.dry and shown < a.sample and '{{' in before:
            q0 = o['input'].get('question', '')
            q1 = new['input'].get('question', '')
            i = q0.find('{{')
            if i >= 0:
                print(f"\n--- {o['input']['id']} 锚点 {rd:%F} ---")
                print('  前:…' + q0[max(0, i - 60):i + 90].replace('\n', ' ⏎ '))
                print('  后:…' + q1[max(0, i - 60):i + 90].replace('\n', ' ⏎ '))
                shown += 1

    # ---- ③ 烘种子 ----
    n_seed, n_seed_changed, missing = 0, 0, []
    baked_seeds = {}
    for f in sorted(seeds_p.glob('*.json')):
        base = f.stem
        rd = anchors.get(base)
        if rd is None:
            missing.append(base)
            rd = fb
        o = json.loads(f.read_text(encoding='utf-8'))
        new = walk_bake(o, rd, stat)
        n_seed += 1
        if new != o:
            n_seed_changed += 1
        baked_seeds[f] = new
    if missing:
        print(f'⚠️ {len(missing)} 份种子在题库里找不到对应题(用了兜底锚点):{missing[:5]}')
    # ⛔ 0821 加的 fail-closed 闸:对不上号的种子会被**兜底锚点**烤,而题面按各自的
    #    真锚点烤 —— 结果就是「一条消息里两个今天」。0815 那批 93.7% 中招,
    #    通过率 56.5% vs 64.3%(差 −7.8,五个族方向全一致)。
    #    原来这里只打一句 ⚠️ 就往下走:0821 重锚点时 **500/500 全没对上**,
    #    照样"烘焙完成、零占位符",一切看起来正常。
    #    配对判据是「种子文件名 == 题的 base id」,改了 item 前缀就必须一起改种子名。
    if n_seed and len(missing) / n_seed > 0.02:
        sys.exit(
            f'⛔ {len(missing)}/{n_seed} 份种子在题库里找不到对应题,拒绝烘焙。\n'
            '   它们会被兜底锚点烤,而题面按真锚点烤 → 一条消息两个今天。\n'
            f'   例:{missing[:3]}\n'
            '   配对是按**种子文件名 == 题的 base id**(去掉 __rN 后缀)。\n'
            '   多半是改过 item id 前缀却没改种子文件名 —— 用 reanchor_items.py '
            '(它会一起改名并有出口闸),别手工改一半。')

    # seed 内容变化后，item.metadata.seed_sha 也必须同步。否则远端 item 看似
    # upsert 成功，账本仍指向烘焙前的 seed，后续无法证明训练轨迹读的是哪一版世界。
    seed_hashes = {
        f.stem: hashlib.sha256(
            json.dumps(new, ensure_ascii=False, indent=1).encode()
        ).hexdigest()[:16]
        for f, new in baked_seeds.items()
    }
    for item in out_items:
        base = re.sub(r'__r\d+$', '', item['input']['id'])
        if base in seed_hashes:
            item.setdefault('metadata', {})['seed_sha'] = seed_hashes[base]
    out_lines = [json.dumps(item, ensure_ascii=False) for item in out_items]

    print(f'\n烘焙:成功 {stat["ok"]} 个占位符,失败 {stat["bad"]} 个')
    print(f'  题库 {len(out_lines)} 条 · 种子 {n_seed} 份(其中 {n_seed_changed} 份有改动)')

    bad_kinds = {k[5:]: v for k, v in stat.items() if k.startswith('bad::')}
    if bad_kinds:
        print(f'\n⛔ 烘不动的({len(bad_kinds)} 种):')
        for k, v in sorted(bad_kinds.items(), key=lambda x: -x[1])[:25]:
            print(f'   {v:4d}  {k}')
        print('\n   → 先跑 scripts/fix_templates.py 把它们修成合法模板')
        sys.exit(2)

    if a.dry:
        print('\n(--dry,没落盘)')
        return

    # ---- ④ 落盘,先留底 ----
    bak_i = items_p.with_suffix(items_p.suffix + '.pre_bake')
    if not bak_i.exists():
        shutil.copy2(items_p, bak_i)
    bak_s = seeds_p.parent / (seeds_p.name + '.pre_bake')
    if not bak_s.exists():
        shutil.copytree(seeds_p, bak_s)
    print(f'原件已留底:{bak_i} · {bak_s}')

    items_p.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    for f, new in baked_seeds.items():
        f.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- ⑤ 硬闸:落盘后再扫一遍,一个 {{ 都不许剩 ----
    left = 0
    for blob in [items_p.read_text(encoding='utf-8')] + \
                [f.read_text(encoding='utf-8') for f in seeds_p.glob('*.json')]:
        left += len(PH.findall(blob))
    if left:
        print(f'\n⛔ 落盘后还剩 {left} 个占位符 —— 不对,别上传')
        sys.exit(2)
    print('\n✅ 烘焙完成,题库和种子里零占位符')
    print('   下一步:upload_all.py 传 S3 + Langfuse → preflight_anchor.py → 发车')


if __name__ == '__main__':
    main()
