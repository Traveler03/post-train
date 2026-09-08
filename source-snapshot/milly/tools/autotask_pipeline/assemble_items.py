#!/usr/bin/env python3
"""阶段4a:装配 Langfuse items。

cases_raw/*.json(lint 无 ERROR)→
  out/seeds/<case_id>.json          规范化种子文件(S3 上传物)
  out/rq1_items_x1.jsonl            427 条(未扩充)
  out/rq1_items.jsonl               默认一题一槽(__r1);补轮在独立 run 重提同一槽
"""
import argparse, hashlib, json, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter

from pipeline_common import json_tree_digest, sha256_file
from audit_temporal_contract import source_contract

D = Path(__file__).resolve().parent.parent
RAW = D / 'out' / 'cases_raw'
SEEDS = D / 'out' / 'seeds'
# ⛔ 同上:模块级别不 mkdir,main() 里按 --out 建(0818)。

S3_PREFIX = 's3://live/seeds/qa-autotask-rq1'
# 账号池 = 平台吞吐的唯一杠杆(实测 ≈ 账号数 × 0.4 条/分钟)。rq1 中途从 31 个扩到 55 个,
# rq2 直接从 55 个起步。
# 能不能拿到 Google 直连工具,历史上分两种号。
# ⚠️ **0825 复测推翻旧白名单**:从当天 7 份 run 档案(1100 条 case)按账号统计实际工具调用,
#    **覆盖 46~100 共 64 个账号,全部调到 google_*,零个退化到 proxy_tool**。
#    0808 那份"46~58 / 70 / 71 / 76~80 没有 Google"的结论已过期(官方卷本来就在用那批号)。
#    ⇒ 白名单从 34 扩到 55。复测方法:扫 eval/runs/*_complete.json,按 input 里的账号统计
#    actual_outcome 里的 google_* vs proxy_tool —— 零成本,不用跑探针。
GOOGLE_OK = list(range(46, 101))
ACCOUNTS = [f'migoo_testbeeai_{i}@shopee.com' for i in GOOGLE_OK]
# 自适应补轮在独立 dataset run 中重提同一个稳定槽，不需要预建 __r2..__r6。
K_ROLLOUT = 1

ARM_SCENE = {
    'template_evening': '综合简报-晚间总结(官方模板)',
    'template_dawn': '综合简报-晨报(官方模板)',
    'alert': '邮件处理-重要邮件警报(官方半模板)',
    'template_variant': '其他官方模板/模板变体',
    'custom': '自定义任务',
}


def rebase_case_id(case_id: str, prefix: str) -> str:
    """Give a rebuilt dataset globally unique IDs while preserving lineage."""
    prefix = str(prefix or '').strip().strip('_')
    if not prefix or not prefix.replace('-', '').isalnum():
        raise ValueError(f'invalid id prefix: {prefix!r}')
    _, separator, suffix = str(case_id).partition('_')
    if not separator or not suffix:
        raise ValueError(f'case_id has no replaceable batch prefix: {case_id!r}')
    return f'{prefix}_{suffix}'

# ---------- 时间冻结:runner 日历只认 next_weekday / minutes_offset ----------
# 生成时用的 days_offset 是我们的中间表示,这里按冻结锚点机械转换。
#
# ⛔ 锚点必须 = **开跑那天**,别图省事定个"好看的周一"。
#    后端不认 X-Override-Time(0808 实测:框架发过去了,agent 的 <msg_time> 还是真实当天),
#    所以锚点只冻住"世界"(注入的邮件/日程),冻不住 agent 的钟。两者差几天,
#    模型就会看到"几天后的邮件"。rq2 因为锚在 08-10、实际 08-08 开跑,整批差了 2 天。
#    发车前跑 scripts/preflight_anchor.py 会拦住这种情况。
ANCHOR_DATE = None                    # 由 --anchor 决定,默认今天(+08:00)
ARM_HOUR = {'template_evening': 21, 'template_dawn': 8}


def day_positioned(seed):
    for ev in seed.get('calendar_events') or []:
        for k in ('start', 'end'):
            v = ev.get(k)
            if isinstance(v, dict) and ('days_offset' in v or 'next_weekday' in v):
                return True
    return False


def anchor_dt(case, anchor_date=None, hour=None, minute=0):
    """Build the freeze instant on the runner's UTC+8 calendar day.

    beeai_eval decodes ``override_timestamp_ms`` in Asia/Shanghai before seed
    construction. Encoding 14:00 in a western source offset would therefore
    move the runner date to the following day. The source offset remains in
    metadata, while the infrastructure timestamp uses the runner timezone.
    """
    anchor_date = anchor_date or datetime(*ANCHOR_DATE).date()
    hour = ARM_HOUR.get(case['arm'], 14) if hour is None else hour
    return datetime(anchor_date.year, anchor_date.month, anchor_date.day,
                    hour, minute, 0,
                    tzinfo=timezone(timedelta(hours=8)))


def convert_calendar(seed, warn, anchor_date=None):
    """days_offset -> next_weekday；对任意锚点星期都保持目标日期不变。"""
    anchor = anchor_date or datetime(*ANCHOR_DATE).date()
    this_monday = anchor - timedelta(days=anchor.weekday())
    kept = []
    for ev in seed.get('calendar_events') or []:
        bad = False
        for k in ('start', 'end'):
            v = ev.get(k)
            if not (isinstance(v, dict) and 'days_offset' in v):
                continue
            dd = int(v.pop('days_offset'))
            if dd < 0:
                warn.append(f"calendar {ev.get('id')} days_offset<0 dropped")
                bad = True
                break
            target = anchor + timedelta(days=dd)
            week_offset = (target - this_monday).days // 7
            ev[k] = {'next_weekday': target.weekday(), 'offset_weeks': week_offset,
                     'hour': int(v.get('hour', 9)), 'minute': int(v.get('minute', 0)),
                     **({'extra_minutes': v['extra_minutes']} if v.get('extra_minutes') else {})}
        if not bad:
            kept.append(ev)
    if 'calendar_events' in seed:
        seed['calendar_events'] = kept
    return seed


def fixed_offset(value: str) -> timezone:
    """Parse the case's reviewed UTC offset without introducing DST drift."""
    match = re.fullmatch(r'([+-])(\d{2}):(\d{2})', str(value or ''))
    if not match:
        raise ValueError(f'invalid fixed UTC offset: {value!r}')
    minutes = int(match.group(2)) * 60 + int(match.group(3))
    if match.group(1) == '-':
        minutes = -minutes
    return timezone(timedelta(minutes=minutes))


def _local_calendar_time(anchor: datetime, spec: dict) -> datetime:
    if spec.get('abs'):
        return datetime.fromisoformat(str(spec['abs']).replace('Z', '+00:00'))
    if 'minutes_offset' in spec:
        return anchor + timedelta(minutes=int(spec['minutes_offset']))
    if 'days_offset' in spec:
        target = anchor + timedelta(days=int(spec['days_offset']))
    elif 'next_weekday' in spec:
        weekday = int(spec['next_weekday'])
        weeks = int(spec.get('offset_weeks', 0))
        if weeks == 0:
            target = anchor + timedelta(days=(weekday - anchor.weekday()) % 7)
        else:
            monday = anchor - timedelta(days=anchor.weekday())
            target = monday + timedelta(days=weekday + 7 * weeks)
    else:
        raise ValueError(f'unsupported calendar time spec: {spec!r}')
    target = target.replace(hour=int(spec.get('hour', 0)),
                            minute=int(spec.get('minute', 0)), second=0,
                            microsecond=0)
    return target + timedelta(minutes=int(spec.get('extra_minutes', 0) or 0))


def freeze_cron_seed_times(seed: dict, anchor_date, hour: int, minute: int,
                           tz_offset: str) -> dict:
    """Resolve cron-relative seed times in the user's semantic timezone.

    The evaluator stores/injects datetimes through an Asia/Shanghai runner.
    Keeping ``days_offset/hour`` relative until that boundary silently turns an
    intended ``08:30 UTC-03`` event into ``08:30 UTC+08`` (eleven hours early).
    Absolute ISO instants retain the authored local wall time through the
    runner round trip. Event-triggered cases remain relative to actual launch.
    """
    local_tz = fixed_offset(tz_offset)
    anchor = datetime(anchor_date.year, anchor_date.month, anchor_date.day,
                      hour, minute, tzinfo=local_tz)
    for email in seed.get('emails') or []:
        offset = email.get('date_offset')
        if isinstance(offset, dict) and not offset.get('abs'):
            instant = anchor + timedelta(days=int(offset.get('days', 0) or 0),
                                         hours=int(offset.get('hours', 0) or 0))
            email['date_offset'] = {'abs': instant.isoformat()}
    for event in seed.get('calendar_events') or []:
        if event.get('all_day'):
            continue
        for key in ('start', 'end'):
            spec = event.get(key)
            if isinstance(spec, dict):
                event[key] = {'abs': _local_calendar_time(anchor, spec).isoformat()}
    return seed


def source_record_for_case(case, pool_group):
    traces = set(case.get('source_traces') or [])
    matches = [record for record in (pool_group or {}).get('records', [])
               if record.get('trace_id') in traces]
    if len(matches) != 1:
        raise SystemExit(
            f'⛔ {case.get("case_id")} 无法唯一定位 source record: '
            f'traces={sorted(traces)} matches={len(matches)}'
        )
    return matches[0]


# ── 规则绑没绑星期几 ────────────────────────────────────────────
# source_tick 会把每道 cron 题的锚点推到「源 tick 那天的星期几」下一次出现的日子,
# 于是一批题被摊到未来 7 天。0818 实测:**cron 题里只有 21.2% 的规则真提到星期几或
# 「每周/上周」**(全池 345/1624;试水 28 道里 19 个 cron 只有 2 个)。
# 另外八成被白等 —— 做合成数据时这个代价太大(全量 2025 道要拖一整周)。
# ⇒ source_tick_today:保留源的时/分(那才是「今天 8 点后」这类窗口要的),
#   锚点一律今天;**只有真绑了星期几的才顺延**。
WEEKDAY_BOUND = re.compile(
    r'周一|周二|周三|周四|周五|周六|周日|周末|礼拜[一二三四五六日天]|星期[一二三四五六日天]'
    r'|\bmon(day)?\b|\btue(s|sday)?\b|\bwed(nesday)?\b|\bthu(r|rs|rsday)?\b'
    r'|\bfri(day)?\b|\bsat(urday)?\b|\bsun(day)?\b|\bweekday\b|\bweekend\b'
    r'|senin|selasa|rabu|kamis|jumat|sabtu|minggu|akhir pekan|mingguan'
    r'|thứ hai|thứ ba|thứ tư|thứ năm|thứ sáu|thứ bảy|chủ nhật|hàng tuần|tuần trước'
    r'|segunda|terça|quarta|quinta|sexta|sábado|domingo|semanal|semana passada'
    r'|lunes|martes|miércoles|jueves|viernes'
    r'|月曜|火曜|水曜|木曜|金曜|土曜|日曜|平日|週末|毎週|先週'
    r'|วันจันทร์|วันอังคาร|วันพุธ|วันพฤหัส|วันศุกร์|วันเสาร์|วันอาทิตย์|รายสัปดาห์',
    re.I)


def rule_is_weekday_bound(rule):
    """这条规则的正确行为依赖「今天**星期几**」吗 —— 只认**具体星期名**。

    漏判的后果很重:「每周一汇总上周」在周三跑,模型会说"今天不是周一,不该跑",
    整题白费。所以判据要覆盖八种语言的星期名。

    ⛔ **但「每周 / weekly / 每星期 / 上周 / last week / weekday」不算绑**(0819 收窄)。
    老版本把它们也算进来,理由是"宁可误判成绑了也不要漏" —— 可是那条理由举的例子
    「每周一汇总上周」里**有「周一」,窄正则一样抓得到**,宽出来的部分买不到任何保险。
    实测宽出来的是这些,它们哪天跑都对:

        Every day at 6pm SGT, read the Google Sheet …
        Every weekday at 9:30AM SGT, read the Google Sheet …

    ⭐ **代价是实打实的**:源池 2025 条记录里,真绑具体星期的 340 条(16.8%),
    被宽正则多圈进来的 36 条(1.8%)——那 36 条会被推到未来某天,白等好几天。
    另注意「上周 / last week」说的是**取数范围**不是触发日,本来就不该算绑。

    ⚠️ `weekday` / `weekend` / 平日 / 週末 **仍然算绑**(实测只有 19 条 = 0.9%)——
    「每个工作日 9:30」确实排除周末,是真约束,只是没指定哪一天。
    为这 0.9% 去做「可接受星期集合」那套设计不划算,保守多等几天就行。
    """
    return bool(WEEKDAY_BOUND.search(str(rule or '')))


def case_temporal_contract(case, pool_group, base_date, schedule_mode):
    """Return the exact runner-side wall time used to render this case."""
    default_hour = ARM_HOUR.get(case['arm'], 14)
    if schedule_mode == 'batch_anchor':
        return {
            'mode': schedule_mode,
            'source_schedule': None,
            'anchor_date': base_date,
            'hour': default_hour,
            'minute': 0,
        }
    record = source_record_for_case(case, pool_group)
    source, raw = source_contract(record)
    if source is None:
        raise SystemExit(
            f'⛔ {case.get("case_id")} 缺 source scheduled_at/observed_at/cutoff_at'
        )
    if schedule_mode == 'source_tick_today' and not rule_is_weekday_bound(case.get('rule')):
        # 规则不看星期几 —— 今天就能跑,别白等
        target_date = base_date
    else:
        target_date = base_date + timedelta(
            days=(source.weekday() - base_date.weekday()) % 7
        )
    return {
        'mode': schedule_mode,
        'source_schedule': raw,
        'anchor_date': target_date,
        'hour': source.hour,
        'minute': source.minute,
    }


LIVE_EXTERNAL_TOOL = re.compile(
    r'(?:web_search|perplexity_search|(?:offline_)?news_search|weather_get|'
    r'finance_|twitter-v2__)', re.I,
)


def uses_live_external(pool_g):
    """Whether a case reads data that cannot follow our frozen seed clock.

    ``proxy_tool`` by itself is not evidence of live data: older source traces
    used it for seeded Gmail, Calendar, Drive and Sheets.  Treating every proxy
    call as web freshness incorrectly classified hundreds of closed-world cases.
    Only providers whose answer changes with real wall time belong here.
    """
    for r in pool_g.get('records', []):
        for t in r.get('tools_used') or []:
            if LIVE_EXTERNAL_TOOL.search(str(t)):
                return True
    return False


def add_event_time_context(question: str, anchor: datetime) -> str:
    """Make an event's execution instant explicit without changing its world.

    ⚠️ 0819 更正:这里原来写的是「后端仍然给真实墙钟的 ``<msg_time>``」——
    **已经不成立了**。0819 探针实测:``override_timestamp_ms`` 会把 agent 的钟
    一起改掉(连星期几都对),见 ``verify_clock.py`` 的说明。
    这行锚点仍然要写:它让事件题的"下一小时"这类相对窗口指向冻结的种子,
    而且现在它还是 ``preflight_anchor`` 那道**防漏烤**闸的比对基准 ——
    题面这行必须和 ``override_timestamp_ms`` 是同一个时刻。
    """
    if '## This Event' in question:
        return question
    stamp = anchor.isoformat(timespec='seconds')
    block = (
        '\n\n## This Event\n'
        f'- triggered_at: {stamp}\n'
        f'- now: {stamp}\n'
        '- timezone: UTC+08:00\n\n'
        'Use this event execution time as "now" when interpreting relative '
        'dates or times and when querying connected data.'
    )
    marker = '</system-reminder>'
    if marker not in question:
        raise ValueError('event question lacks </system-reminder>')
    return question.replace(marker, block + '\n' + marker, 1)


def require_lint_receipt(receipt_file, raw, pool_file):
    """只接受与当前输入逐字对应的成功 lint 回执。"""
    if not receipt_file.exists():
        raise SystemExit(f'⛔ 缺 lint 回执:{receipt_file}\n先对本批 raw/pool 运行 lint_cases.py')
    try:
        r = json.loads(receipt_file.read_text())
    except Exception as e:
        raise SystemExit(f'⛔ lint 回执不可读:{receipt_file}: {e}')
    actual = {
        'raw_sha256': json_tree_digest(raw),
        'pool_sha256': sha256_file(pool_file),
        'files': len(list(raw.glob('*.json'))),
    }
    problems = []
    if r.get('schema_version') != 1:
        problems.append(f'schema_version={r.get("schema_version")!r}')
    if not r.get('passed') or r.get('errors') != 0:
        problems.append(f'lint 未通过(errors={r.get("errors")})')
    if not r.get('strict_gate'):
        problems.append('回执来自 --no-strict 修复循环')
    for k, v in actual.items():
        if r.get(k) != v:
            problems.append(f'{k} 已变化(receipt={r.get(k)!r}, actual={v!r})')
    if problems:
        raise SystemExit('⛔ lint 回执与当前装配输入不一致:\n  - ' + '\n  - '.join(problems)
                         + '\n请重新运行 lint_cases.py')
    print(f'✅ lint 回执匹配:{receipt_file}', flush=True)


def main():
    # ⚠️ global 必须在**任何使用之前**声明 —— 下面 --rollouts 的 default=K_ROLLOUT
    # 就是一次使用,声明写在它后面会 SyntaxError(踩过)
    global RAW, SEEDS, S3_PREFIX, ACCOUNTS, ANCHOR_DATE, K_ROLLOUT, PIN_ACCOUNTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='起草稿目录 cases_raw')
    ap.add_argument('--out', required=True, help='本批产出目录')
    ap.add_argument('--tag', required=True, help='批次名:决定 S3 前缀和 items 文件名')
    ap.add_argument('--pin-accounts', action='store_true',
                    help='把 user_email 写死进 metadata(老行为,只为复现旧批次);默认交给平台账号池')
    ap.add_argument('--id-prefix', default='',
                    help='可选的新批次全局 ID 前缀。Langfuse dataset item ID 跨数据集全局唯一；'
                         '从旧 cases 重建新版本时必须传，例如 rq3w2。它只替换 case_id 第一个'
                         '下划线前的旧批次前缀，并保留 source_case_id 供追溯。')
    ap.add_argument('--accounts', default='',
                    help='测试账号。两种写法:区间 "46-100",或逗号列表 "59,60,61,..."。\n'
                         '⛔ **不是所有测试号都一样** —— 0808 实测 46~58 和 76~80 这 18 个号'
                         '拿不到 Google 直给工具(google_gmail_* / google_calendar_* 等 12 个),'
                         'agent 只能先 tool_search 找工具(一次吐 66k 字符目录)再 proxy_tool 转发。'
                         '而**正式评测集**(caseset_prof_多语言)跑在有 Google 工具的账号上'
                         '(三次 run 实测 proxy_tool 全 0、直接 google_* 200~253 次)。'
                         '训练数据的工具路径必须跟评测集一致,否则训了也考不出来。'
                         '缺省用本文件 GOOGLE_OK 中已验证的 34 个账号')
    ap.add_argument('--allow-non-google-accounts', action='store_true',
                    help='显式放行未验证 Google 直连工具的账号；正式 SFT 禁用')
    ap.add_argument('--pool', default='', help='阶段1的 source_pool.jsonl;决定"用不用实时联网→要不要冻结时间"。'
                                               '不传会退回 out/source_pool.jsonl(上一批的!),冻结判定会瞎给')
    ap.add_argument('--lint-receipt', default='',
                    help='lint_cases.py 回执;默认 <raw上级>/lint_receipt.json')
    ap.add_argument('--rollouts', type=int, default=K_ROLLOUT,
                    help='每个世界预建多少存储槽；当前补轮重提同一 __r1，正式批用 1')
    ap.add_argument('--anchor', default='', help='时间冻结锚点 YYYY-MM-DD。⛔ 必须是**开跑那天** '
                                                 '(后端不认 X-Override-Time,锚点只冻世界不冻 agent 的钟,'
                                                 '差几天模型就看到几天后的邮件)。默认=今天(+08:00)')
    ap.add_argument('--schedule-mode',
                    choices=['batch_anchor', 'source_tick', 'source_tick_today'],
                    default='batch_anchor',
                    help='source_tick: cron 保留所选源 tick 的 weekday/wall-clock，并按'
                         '未来 7 天分片；这是保留源 world/expected 时的正式模式。\n'
                         'source_tick_today(0818 加): 同样保留源的**时/分**,但锚点一律今天,'
                         '**只有规则真绑了星期几的才顺延**到那个星期几。'
                         '实测 cron 题里只有 21.2%% 真绑星期几 —— 另外八成被白等一周,'
                         '做合成数据时这个代价太大。\n'
                         'batch_anchor 仅兼容旧批，temporal hard gate 通常会拒绝。')
    a = ap.parse_args()

    K_ROLLOUT = a.rollouts
    _tz8 = timezone(timedelta(hours=8))
    _anchor = (datetime.strptime(a.anchor, '%Y-%m-%d').date() if a.anchor
               else datetime.now(_tz8).date())
    ANCHOR_DATE = (_anchor.year, _anchor.month, _anchor.day)
    _drift = (_anchor - datetime.now(_tz8).date()).days
    print(f'时间冻结锚点 = {_anchor}({_anchor.strftime("%A")}),与今天差 {_drift:+d} 天')
    if _drift:
        print(f'⚠️ 锚点不是今天。只有当天就发车才对得齐 —— 隔天再发,'
              f'模型会看到 {_drift} 天后的邮件。发车前务必跑 preflight_anchor.py')
    RAW = Path(a.raw)
    OUTD = Path(a.out); OUTD.mkdir(parents=True, exist_ok=True)
    SEEDS = OUTD / 'seeds'; SEEDS.mkdir(parents=True, exist_ok=True)
    S3_PREFIX = f's3://live/seeds/qa-autotask-{a.tag}'
    if not a.accounts:
        nums = list(GOOGLE_OK)                      # 缺省 = 验过有 Google 工具的 34 个
    elif ',' in a.accounts:
        nums = [int(x) for x in a.accounts.split(',') if x.strip()]
    else:
        lo, hi = (int(x) for x in a.accounts.split('-'))
        nums = list(range(lo, hi + 1))
    bad = sorted(set(nums) - set(GOOGLE_OK))
    if bad:
        if not a.allow_non_google_accounts:
            raise SystemExit(f'⛔ 账号不在 Google 直连工具白名单:{bad}；正式 SFT 禁止')
        print(f'⚠️⚠️ 账号池里有 {len(bad)} 个**没有 Google 直给工具**的号:{bad}\n'
              f'   这些号上跑出来的轨迹,工具路径跟正式评测集对不上(评测集 proxy_tool 全 0)。'
              f'   确认要用再继续,否则去掉它们。', flush=True)
    ACCOUNTS = [f'migoo_testbeeai_{i}@shopee.com' for i in nums]
    PIN_ACCOUNTS = bool(getattr(a, 'pin_accounts', False))
    print(f'稿={RAW} 出={OUTD} 批次={a.tag} S3={S3_PREFIX} 账号={len(ACCOUNTS)} 个', flush=True)
    pool = {}
    pool_f = Path(a.pool) if a.pool else (OUTD / 'source_pool.jsonl' if (OUTD / 'source_pool.jsonl').exists()
                                          else D / 'out' / 'source_pool.jsonl')
    print(f'源池={pool_f}', flush=True)
    receipt_f = Path(a.lint_receipt) if a.lint_receipt else RAW.parent / 'lint_receipt.json'
    require_lint_receipt(receipt_f, RAW, pool_f)
    with open(pool_f) as f:
        for l in f:
            g = json.loads(l)
            pool[g['rule_key']] = {'records': [
                {'trace_id': r.get('trace_id'),
                 'target_tick_index': r.get('target_tick_index'),
                 'judge_trigger_last': r.get('judge_trigger_last'),
                 'cutoff_at': r.get('cutoff_at'),
                 'ticks': r.get('ticks') or [],
                 'tools_used': r.get('tools_used')}
                for r in g['records']]}
    files = sorted(RAW.glob('*.json'))
    cases, skipped = [], []
    for fp in files:
        try:
            c = json.loads(fp.read_text())
        except Exception:
            skipped.append(fp.name + ' (unreadable/in-flight)')
            continue
        if 'llm' not in c or 'seed' not in c.get('llm', {}):
            skipped.append(fp.name + ' (incomplete)')
            continue
        if c.get('lint', {}).get('errors'):
            skipped.append(fp.name)
            continue
        cases.append(c)
    print(f'cases: {len(cases)} (lint-error skipped: {len(skipped)})')

    items_x1 = []
    n_frozen = 0
    shard_ids = {}
    for idx, c in enumerate(sorted(cases, key=lambda x: x['case_id'])):
        source_cid = c['case_id']
        cid = rebase_case_id(source_cid, a.id_prefix) if a.id_prefix else source_cid
        seed = c['llm']['seed'] or {}
        # 规范化种子:只留 runner 认的顶层键
        # ⛔ 这是**白名单**:清单外的顶层 key 会被静默丢掉。0818 照着
        #    beeai_eval/data/seed_inject.py 真正消费的那份清单核过一遍:
        #    · 加 user_notes —— 线上 `explicit_memories` 那段上下文就是从它来的,
        #      seed_test_data.seed_user_notes 存在,我们之前漏在白名单外,造了也白造;
        #    · 删 state_of_mind_schedule —— seed_inject.py **根本不读这个 key**(全仓只在
        #      cleanup 的 kkv 源清单里出现过 state_of_mind),留着是死键,会让人以为种得进去。
        #    还有 filters / gmail_settings / memory_records / photo_memories / *_face_id
        #    也被注入器支持,但目前没有任何一条规则用得上,先不放进来(用得上再加)。
        seed = {k: v for k, v in seed.items()
                if k in ('emails', 'drafts', 'labels', 'calendar_events', 'drive_files',
                         'contacts', 'auto_tasks', 'user_notes', 'timezone') and v}
        contract = case_temporal_contract(
            c, pool.get(c['rule_key']), _anchor, a.schedule_mode,
        )
        case_anchor = contract['anchor_date']
        live_external = uses_live_external(pool.get(c['rule_key'], {}))
        daypos = day_positioned(seed)
        _tickish = a.schedule_mode in ('source_tick', 'source_tick_today')
        freeze = ((c['trigger'] == 'cron' and _tickish)
                  or (not live_external) or daypos)
        run_anytime = bool(freeze and not live_external)
        if c['trigger'] == 'cron' and _tickish:
            seed = freeze_cron_seed_times(
                seed, case_anchor, contract['hour'], contract['minute'],
                c.get('tz_offset') or '+08:00',
            )
        elif c['trigger'] == 'event' and run_anytime:
            # The evaluator applies override to relative email offsets, but its
            # private Calendar import currently resolves minutes_offset against
            # the real execution clock.  Bake both to the runner contract so a
            # meeting "in 45 minutes" is actually queryable at virtual now+45.
            seed = freeze_cron_seed_times(
                seed, case_anchor, contract['hour'], contract['minute'], '+08:00',
            )
        # 冻结判定:source_tick 模式下 cron 必须冻结到源 wall-clock；运行仍必须
        # 等到 contract anchor_date 当天，避免 agent 真实时钟与世界跨日。
        # 旧 batch_anchor 模式保留原规则。
        conv_warn = []
        if live_external and daypos:
            conv_warn.append('live-external case forced frozen (day-positioned calendar)')
        if freeze:
            seed = convert_calendar(seed, conv_warn, case_anchor)
            adt = anchor_dt(c, case_anchor, contract['hour'], contract['minute'])
            override_ms = int(adt.timestamp() * 1000)
            n_frozen += 1
        else:
            override_ms = None
        seed_bytes = json.dumps(seed, ensure_ascii=False, indent=1).encode()
        (SEEDS / f'{cid}.json').write_bytes(seed_bytes)
        sha = hashlib.sha256(seed_bytes).hexdigest()[:16]

        exp = c['llm']['expected_output']
        question = c['question']
        run_anytime = bool(override_ms and not live_external)
        if c['trigger'] == 'event' and run_anytime:
            question = add_event_time_context(question, adt)
        # 0822 题族明示采样(仅 v3.1.9 的 draft/checklist 族):rollout 卷的题面尾拼条款,
        # 抬正样本密度(39%→67%,探针2);⛔ 渲染前必须按原串摘除(swap_rule_text.py 两道闸)
        if c.get('explicit_clause'):
            assert c['explicit_clause'] not in question, f'{cid} 条款已在题面里,别拼两遍'
            question = question + c['explicit_clause']
        item = {
            'input': {
                'id': cid,
                'question': question,
                'attachments': [],
                'tools_white_list': [],
                **({'override_timestamp_ms': override_ms} if override_ms else {}),
            },
            'expectedOutput': {
                'goal': exp.get('goal', ''),
                'describe': exp.get('describe', ''),
                'key_constraints': exp.get('key_constraints', []),
            },
            'metadata': {
                'tags': ['auto_task', 'execute', 'remind_agent_professional', a.tag,
                         f"arm_{c['arm']}"],
                'scene': ARM_SCENE[c['arm']],
                'arm': c['arm'],
                'mode': c.get('mode', 'engineered'),
                'variant': c.get('variant', ''),
                **({'family': c['family'], 'has_explicit_clause': bool(c.get('explicit_clause'))}
                   if c.get('family') else {}),
                'lang': c['lang'],
                'eval_sim': (c.get('eval_sim') or {}).get('J') if c.get('eval_sim') else None,
                'subset': 'A-执行',
                'turn_mode': 'single',
                'agent_name': 'remind_agent_professional',
                'risk_level': 'low',
                'source_file': f'{a.tag}_v1',
                'source_case_id': source_cid,
                'trigger_type': c['trigger'],
                'expected_polarity': 'silent' if c['polarity'] == 'skip' else 'push',
                'polarity_design': c['polarity'],
                'world_story': str(c['llm'].get('world_story', ''))[:800],
                'time_freeze': (f'override_timestamp_ms={override_ms}({case_anchor.isoformat()} '
                                f'{case_anchor.strftime("%A")} '
                                f'{contract["hour"]:02d}:{contract["minute"]:02d} +08:00 runner; '
                                f'source_tz={c.get("tz_offset")})——'
                                '种子注入与 agent 系统钟都锚到这个时刻'
                                '(0819 实测:平台会把 <msg_time> 一起改过来，连星期几都对；'
                                '0808 那条"agent 钟仍是真实时间"已过期)；'
                                '题面的显式时间必须与本锚点一致——不一致由 '
                                'preflight_anchor 的防漏烤闸拦下' if override_ms
                                else '不冻结——种子相对偏移+题面 jinja2,全部锚到 run 时刻'),
                'temporal_contract': {
                    'mode': contract['mode'],
                    'source_schedule': contract['source_schedule'],
                    'anchor_date': case_anchor.isoformat(),
                    'weekday': case_anchor.strftime('%A'),
                    'wall_clock': f'{contract["hour"]:02d}:{contract["minute"]:02d}',
                    'run_only_on_anchor_date': not run_anytime,
                    'run_anytime': run_anytime,
                    # 0819 更正:以前这里写 explicit_prompt / real_wall_clock,前提是
                    # 「后端不认 override_timestamp_ms」。实测已推翻 —— 冻结的题
                    # agent 钟也跟着锚点走(verify_clock.py 每批复核)。
                    'agent_clock_mode': ('override_timestamp_ms' if override_ms
                                         else 'real_wall_clock'),
                    'seed_clock_mode': ('override_timestamp_ms' if override_ms
                                        else 'real_wall_clock'),
                    'live_external_dependency': live_external,
                },
                'assemble_warnings': conv_warn or None,
                'data_dependency': 'seed_required' if seed else 'none',
                'seed_data_file': f'{S3_PREFIX}/{cid}.json' if seed else '',
                'seed_sha': sha if seed else '',
                'timeout_seconds': 3600,
                'clean_account_data': True,
                'source_rule_key': c['rule_key'],
                'source_user_key': c.get('user_key'),
                'source_account_keys': c.get('account_keys') or [],
                'source_tick_index': c.get('source_tick_index'),
                'source_cutoff_at': c.get('cutoff_at'),
                'source_traces': c.get('source_traces', [])[:3],
                'n_source_records': c.get('n_source_records'),
                'tz_offset': c.get('tz_offset'),
                'purpose': 'rq_factory_rejection_sampling',
            },
        }
        items_x1.append((idx, item))
        shard_ids.setdefault(case_anchor.isoformat(), []).append(cid)

    with open(OUTD / f'{a.tag}_items_x1.jsonl', 'w') as f:
        for _, it in items_x1:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    # ×K 扩充:同题 K 份,账号错开(idx+r 偏移),id 加 __rK
    with open(OUTD / f'{a.tag}_items.jsonl', 'w') as f:
        n = 0
        for idx, base in items_x1:
            for r in range(1, K_ROLLOUT + 1):
                it = json.loads(json.dumps(base, ensure_ascii=False))
                iid = f"{base['input']['id']}__r{r}"
                it['input']['id'] = iid
                it['metadata']['rollout_idx'] = r
                # 账号分配:默认**不写死**,交给平台账号池(SG 93 / US 83 个,`runner/account_pool.py`)
                # 运行时挑一个能加写锁的号。种子是**运行时注入当次分配到的账号**的
                # (`metadata.seed_data_file` + `clean_account_data`),账号和种子本来就解绑,
                # 写死账号只会把并发卡死在我们自己那几个号上(0825 用户指正)。
                # 发卷时钉 REGION=SG 即可只用 SG 号。
                # --pin-accounts 仅为复现老批次保留。
                if PIN_ACCOUNTS:
                    it['metadata']['user_email'] = ACCOUNTS[(idx * K_ROLLOUT + r - 1) % len(ACCOUNTS)]
                else:
                    it['metadata'].pop('user_email', None)
                f.write(json.dumps(it, ensure_ascii=False) + '\n')
                n += 1
    shard_dir = OUTD / 'schedule_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    for day, case_ids in sorted(shard_ids.items()):
        (shard_dir / f'{day}.txt').write_text(
            ''.join(f'{case_id}__r1\n' for case_id in sorted(case_ids))
        )
    print('schedule shards:', {day: len(ids) for day, ids in sorted(shard_ids.items())})
    print(f'items: x1={len(items_x1)} -> expanded={n} | frozen={n_frozen}')
    st = Counter((it['metadata']['arm'], it['metadata']['polarity_design']) for _, it in items_x1)
    print('x1 by arm/polarity:', dict(st))
    if skipped:
        print('skipped (lint errors):', skipped[:10])


    # 分臂索引:收割的覆盖表按这个分臂。不生成的话收割表全是 "?"
    with open(OUTD / 'remote_items_lite.jsonl', 'w') as f:
        for _, base in items_x1:
            m = base['metadata']
            for r in range(1, K_ROLLOUT + 1):
                f.write(json.dumps({'id': f"{base['input']['id']}__r{r}", 'arm': m.get('arm'),
                                    'merged_from': m.get('arm'), 'polarity': m.get('polarity_design'),
                                    'lang': m.get('lang'), 'mode': m.get('mode'),
                                    'user_key': m.get('source_user_key'),
                                    'rule_key': m.get('source_rule_key')},
                                   ensure_ascii=False) + '\n')
    print(f"分臂索引 -> {OUTD / 'remote_items_lite.jsonl'}", flush=True)

    _autofix_seeds(SEEDS, OUTD / f'{a.tag}_items.jsonl')
    refresh_seed_hashes(
        SEEDS,
        OUTD / f'{a.tag}_items_x1.jsonl',
        OUTD / f'{a.tag}_items.jsonl',
    )


def refresh_seed_hashes(seeds_dir, *item_files):
    """Refresh item provenance after the post-assembly seed normalizer."""
    changed = 0
    for item_file in item_files:
        rows = []
        for line in item_file.open():
            item = json.loads(line)
            metadata = item.get('metadata') or {}
            seed_name = str(metadata.get('seed_data_file') or '').rsplit('/', 1)[-1]
            if seed_name:
                seed_path = Path(seeds_dir) / seed_name
                if not seed_path.exists():
                    raise SystemExit(f'⛔ seed hash 刷新时缺文件:{seed_path}')
                digest = hashlib.sha256(seed_path.read_bytes()).hexdigest()[:16]
                if metadata.get('seed_sha') != digest:
                    metadata['seed_sha'] = digest
                    changed += 1
            rows.append(item)
        item_file.write_text(''.join(
            json.dumps(row, ensure_ascii=False) + '\n' for row in rows
        ))
    print(f'seed SHA 已按修复后文件刷新:{changed} item rows', flush=True)

def _autofix_seeds(seeds_dir, items_file):
    """⚠️ 装配会重新生成种子,把之前 fix_seeds_v2 修好的形状冲掉 —— 踩过一次:
    重装配后没重修就上传,中试 503 条全死在 seed inject failed。
    所以装配完自动修一遍,不依赖人记得。"""
    import subprocess, sys as _s, os as _o
    here = Path(__file__).resolve().parent
    env = dict(_o.environ)
    env['PYTHONPATH'] = _o.path.expanduser('~/.local/lib/python3.12/site-packages')
    print('\n--- 装配后自动修种子形状 ---', flush=True)
    # 0811 评审整改(点7):修种子失败必须炸出来 —— 静默失败=中试全死在 seed inject(踩过)
    subprocess.run([_s.executable, str(here / 'fix_seeds_v2.py'),
                    '--seeds', str(seeds_dir), '--items', str(items_file),
                    '--cases', str(RAW)], env=env, check=True)


if __name__ == '__main__':
    main()
