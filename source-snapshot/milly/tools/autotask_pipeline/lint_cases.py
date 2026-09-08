#!/usr/bin/env python3
"""阶段3:质检 cases_raw/*.json。ERROR=必须回炉,WARN=人工过目。

检查项:
  E1 seed schema(emails/calendar 必填字段、date_offset/start-end 形状)
  E2 安全域名:seed+trigger_event 里所有邮箱 ∈ mock.test / *.example.com / pool / self
  E3 绝对日期(YYYY-MM-DD 等)出现在 jinja 表达式之外(seed 文本或题面)
  E4 jinja2 真渲染:question + emails.subject/body + calendar summary/desc 全部渲一遍
  E5 event 题:trigger_event 必在;触发邮件必须能在 seed.emails 里对上(title 匹配 subject)
  E6 极性结构:skip/fire 必须是 event 或有明确判定物;quiet 的 expected 不应要求列内容
  E7 邮箱快照:email id 不重复、收件箱不出现相对运行时刻的未来邮件
  E8 lineage:用户/tick/cutoff/plan/source 指纹齐全,timezone 与 offset 同源一致
  E10 抄袭闸:seed 文本与「源轨迹真实工具返回」8-gram 重叠 > 0.30 → 阻断真实材料复用
  E11 评测种子闸:seed 文本与 prof 卷 77 份种子 8-gram 重叠 > 0.25 → 阻断评测泄漏
  E12 日历参会状态:带 attendees 的会议要有合法 self_response / responseStatus
      (四个值 needsAction/accepted/declined/tentative,从真实 trace 挖的)。
      「缺」只在 --require-rsvp 时才算 error,否则降级 W12;**非法取值任何时候都是 error**
  W3 规则点名实体缺席:规则里引号内的串 / 连续大写词组不在世界文本里
  W12 attendee 带了注入器不认的字段(displayName/self/organizer 等),会被丢弃
用法: python3 lint_cases.py [--fix-list out/relint.txt]
"""
import argparse, collections, json, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pipeline_common import json_tree_digest, sha256_file, normalize_realism, is_aligned_recipe, REALISM_CANON

D = Path(__file__).resolve().parent.parent
RAW = D / 'out' / 'cases_raw'
POOL = D / 'out' / 'source_pool.jsonl'
PROF_SEEDS = Path('/home/work/migoo_ai_public/haiyang/training/experiments/hy3_295b_lora/data_v7/caseset_seeds')

POOL_ACCOUNTS = re.compile(r'(testbeeai|onboarding_testmigoo)', re.I)
# ⚠️ 0806 改判据:黑名单 → 白名单。旧法只把免费邮箱和 @shopee.com 判错,
# sea.com / monee.com / seabank.co.id / e.read.ai 这类真实公司域只算警告就放行了
# —— rq1 实测有 40 题(1.8%)带着可投递地址上线,而评测账号是真会发信的。
# 只放行 RFC2606/6761 保留域(.test/.invalid/.example/.localhost 顶级域、example.com|net|org),
# 这些域全球不可投递,是唯一安全的。
SAFE_DOM = re.compile(r'(\.(test|invalid|example|localhost)|(^|\.)example\.(com|net|org))$', re.I)
# 日历参会状态:Google API 的四个合法值。0818 从线上真实 trace 里挖出来核对过
# (source_pool 里 1599 个 attendee 对象,取值只有这四个)。
RSVP_OK = {'needsAction', 'accepted', 'declined', 'tentative'}
REQUIRE_RSVP = False      # 兼容旧名;等价于 is_aligned_recipe(REALISM)
# 配方档位正名 v3.1.7 / v3.1.8,必须和 gen_worlds.py 的档位一致。
# 判「对齐线上那一档」走 is_aligned_recipe(),别写 == 'v3.1.8'(老 manifest 记的是 'v8'/'')
REALISM = 'v3.1.7'
# 平台注入器 seed_test_data.py 只透传这四个 attendee 字段,其余会被丢弃
ATT_KEEP = {'email', 'responseStatus', 'optional', 'comment'}


def calendar_rsvp_issues(event, require=True):
    """带 attendees 的会议必须有合法参会状态。返回 (errors, warnings)。

    ⚠️ `require=False`(默认关,由 `--require-rsvp` 打开)时**只降级成警告**:
    lint 的回执是 `assemble_items.py` 的发车硬闸,无条件启用会把所有旧批次
    (v3.1.7 实测 1292/2025 个世界)直接堵死。**非法取值任何时候都是错误** ——
    那是注入必失败的,和批次新旧无关。

    为什么要这道闸(0818,v3.1.8 起):线上真实 attendee **100% 带 responseStatus**
    (source_pool 里 1599/1599),而 v3.1.7 的世界只有 8/2025 写了 —— 模型看不到
    「这个会我还没回复」「谁接受了谁拒绝了」,而这正是提醒助手最该判断的东西之一。
    提示词里加了约束,但提示词是软的,这里做硬闸兜底。

    ⚠️ 取值必须和 Google API 对齐,写错了注入会失败。四个值是从真实 trace 挖的,
       不是照文档抄的。
    """
    errors, warnings = [], []
    attendees = event.get('attendees')
    if not attendees:
        return errors, warnings          # 没有参会者的会议不管
    eid = event.get('id')
    self_response = event.get('self_response')
    if self_response is None:
        # 「缺」只在本批要求时才算错;旧批次降级为提示
        (errors if require else warnings).append(
            f'{"E12" if require else "W12"} calendar {eid} 带 attendees 但缺 self_response')
    elif self_response not in RSVP_OK:
        errors.append(f'E12 calendar {eid} self_response={self_response!r} '
                      f'不是 {sorted(RSVP_OK)}')
    if not isinstance(attendees, list):
        errors.append(f'E12 calendar {eid} attendees 不是 list')
        return errors, warnings
    for attendee in attendees:
        if not isinstance(attendee, dict):
            continue                     # 纯字符串形式仍然允许(注入器认)
        status = attendee.get('responseStatus')
        if status is not None and status not in RSVP_OK:
            errors.append(f'E12 calendar {eid} attendee {attendee.get("email")} '
                          f'responseStatus={status!r} 不是 {sorted(RSVP_OK)}')
        extra = set(attendee) - ATT_KEEP
        if extra:
            warnings.append(f'W12 calendar {eid} attendee 多余字段 {sorted(extra)} '
                            f'会被注入器丢弃')
    return errors, warnings
REAL_DOM = re.compile(r'@(gmail|outlook|hotmail|yahoo|icloud|qq|163|foxmail|proton)\.', re.I)
ABS_DATE = re.compile(r'\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b')

# ---------- jinja 渲染环境(复刻 runner 的助手函数,最小实现) ----------
import jinja2

def _helpers(run_date):
    def add_days(v, n): return v + timedelta(days=n)
    def add_hours(v, n): return v + timedelta(hours=n)
    def add_minutes(v, n): return v + timedelta(minutes=n)
    def add_business_days(v, n):
        d, step = v, 1 if n >= 0 else -1
        for _ in range(abs(int(n))):
            d += timedelta(days=step)
            while d.weekday() >= 5: d += timedelta(days=step)
        return d
    def at_time(v, hour, minute=0, second=0): return v.replace(hour=hour, minute=minute, second=second)
    WD = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    def next_weekday(name, from_=None, strict=False):
        f = from_ or run_date
        t = WD.index(str(name).lower())
        ahead = (t - f.weekday()) % 7
        if ahead == 0 and strict: ahead = 7
        return f + timedelta(days=ahead)
    def fmt_date(v, style='long'):
        return v.strftime('%Y-%m-%d') if style == 'iso' else v.strftime('%B %d, %Y' if style=='long-no-day' else '%A, %B %d, %Y' if style=='weekday' else '%B %d, %Y')
    def fmt_time(v, style='hm'):
        return v.strftime({'hm':'%H:%M','hms':'%H:%M:%S','hm-12h':'%I:%M %p'}.get(style,'%H:%M'))
    def fmt_datetime(v, style='iso'):
        return v.strftime({'iso':'%Y-%m-%dT%H:%M:%S','iso-minute':'%Y-%m-%dT%H:%M','long':'%B %d, %Y %H:%M','long-12h':'%B %d, %Y %I:%M %p'}.get(style,'%Y-%m-%dT%H:%M:%S'))
    return dict(run_date=run_date, add_days=add_days, add_hours=add_hours, add_minutes=add_minutes,
                add_business_days=add_business_days, at_time=at_time, next_weekday=next_weekday,
                fmt_date=fmt_date, fmt_time=fmt_time, fmt_datetime=fmt_datetime)

ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)
HELP = _helpers(datetime(2026, 8, 3, 21, 0, 0))

def render_ok(s):
    try:
        ENV.from_string(s).render(**HELP)
        return None
    except Exception as e:
        return f'{type(e).__name__}: {e}'

def strip_jinja(s):
    return re.sub(r'\{\{.*?\}\}|\{%.*?%\}', ' ', s, flags=re.S)

def ngrams(s, n=8):
    s = re.sub(r'\s+', ' ', s.strip().lower())
    return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))

def overlap(a, b):
    if not a or not b: return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))

def seed_text(seed):
    parts = []
    for e in seed.get('emails') or []:
        parts.append(str(e.get('subject','')) + ' ' + str(e.get('body','')))
    for c in seed.get('calendar_events') or []:
        parts.append(str(c.get('summary','')) + ' ' + str(c.get('description','')))
    return '\n'.join(parts)


def question_without_original_rule(case):
    """Return generated question text without immutable user-rule examples.

    Absolute dates inside the original rule can be legitimate format examples
    (for example ``yyyy/mm/dd (a): 2026/07/17 (金)``).  E3 is meant to reject
    dates baked by world synthesis, not rewrite user intent.
    """
    question = str(case.get('question') or '')
    rule = str(case.get('rule') or '')
    return question.replace(rule, ' ') if rule else question


def trigger_title_matches_seed(title, subjects):
    """Exact normalized matches work even when a title is shorter than 6 chars."""
    norm = lambda value: re.sub(r'\s+', ' ', str(value)).strip().casefold()
    target = norm(title)
    if not target:
        return False
    normalized = [norm(subject) for subject in subjects]
    if target in normalized:
        return True
    target_ngrams = ngrams(target, 6)
    return bool(target_ngrams) and any(
        overlap(target_ngrams, ngrams(subject, 6)) > 0.5
        for subject in normalized)

def all_addresses(case):
    txt = json.dumps(case['llm'].get('seed'), ensure_ascii=False) + json.dumps(case['llm'].get('trigger_event') or {}, ensure_ascii=False)
    # ⚠️ 域名必须带点号,且左边不能紧跟反斜杠 —— 否则 json.dumps 后的 "\n@nicholas"
    # 会被误判成邮箱 n@nicholas(0806 实测 33 个假阳性全是这个)
    return set(a.lower().rstrip('.') for a in
               re.findall(r'(?<![\\\w])([\w.+-]+@[\w-]+(?:\.[\w-]+)+)', txt))


def seed_header_addresses(seed):
    """Yield SMTP addresses from fields consumed as addresses by injectors."""
    found = []
    address_keys = {'from', 'to', 'cc', 'bcc', 'reply_to', 'email', 'attendees'}

    def visit(value, address_context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, address_context or key in address_keys)
        elif isinstance(value, list):
            for item in value:
                visit(item, address_context)
        elif address_context and isinstance(value, str):
            found.extend(re.findall(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+', value))

    visit(seed)
    return found


def load_quarantine(raw_dir):
    """读 `<raw上级>/quarantine.json`,拿到「已隔离、不再期待出稿」的 case。

    ## 为什么要有这个

    手册写着「回炉两轮还不过的就隔离,移到 quarantine/ 留档」,但**以前没有配套通道**:
    隔离完文件数就比 `selected_cases` 少,E9 必炸;而 assemble 又要求
    `passed=True`。于是唯一能往下走的办法是手改 manifest —— 无痕、不可审计。
    (0815 实测踩到:一道题连造三次都不返回 `expected_output`/`seed`,
     按手册该隔离,却被自家闸门卡死。)

    ## 格式

        {"cases": [{"case_id": "...", "reason": "连续三次起草缺字段", "at": "2026-08-15"}]}

    `reason` **必填**且不能为空 —— 隔离是要留证据的,不是删数据的后门。

    隔离的 case 必须**确实不在** raw 目录里(移到 `quarantine/` 或别处),
    否则说明只是登记了没真挪走,当错误报。
    """
    path = Path(raw_dir).parent / 'quarantine.json'
    if not path.exists():
        return {}, []
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f'E9 quarantine.json 读不了: {exc}']
    out, errors = {}, []
    for row in (doc.get('cases') or []):
        cid = (row or {}).get('case_id')
        reason = str((row or {}).get('reason') or '').strip()
        if not cid:
            errors.append('E9 quarantine 条目缺 case_id')
        elif not reason:
            errors.append(f'E9 quarantine 条目 {cid} 没写 reason(隔离必须留原因)')
        else:
            out[cid] = reason
    return out, errors


def manifest_completeness_errors(manifest, files, quarantined=None):
    """Check that this directory contains the complete selected generation set.

    A smoke manifest deliberately selects only ten cases while retaining all plan
    fingerprints, so completeness is defined by ``selected_cases`` rather than
    the total fingerprint count.  Every present file must still belong to the
    fingerprinted plan.

    ``quarantined``(见 :func:`load_quarantine`)里的 case 从期望数里扣掉 ——
    它们是**明知故犯、留了原因**地不出稿,不是「漏生成」。
    """
    if not manifest:
        return ['E9 plan manifest missing']
    quarantined = quarantined or {}
    errors = []
    known = set((manifest.get('case_fingerprints') or {}).keys())
    stray = sorted(c for c in quarantined if c not in known)
    if stray:
        errors.append(f'E9 quarantine 里有不属于本计划的 case {stray[:5]}')
    present = {fp.stem for fp in files}
    still_here = sorted(c for c in quarantined if c in present)
    if still_here:
        errors.append(f'E9 已登记隔离但文件还在 raw 里(要移走){still_here[:5]}')
    expected = manifest.get('selected_cases')
    if not isinstance(expected, int) or expected < 0:
        errors.append(f'E9 invalid selected_cases {expected!r}')
    else:
        expected -= len([c for c in quarantined if c in known])
        if len(files) != expected:
            errors.append(f'E9 incomplete generation files={len(files)} expected={expected}'
                          + (f'(已扣除隔离 {len(quarantined)} 道)' if quarantined else ''))
    unexpected = sorted(fp.stem for fp in files if fp.stem not in known)
    if unexpected:
        errors.append(f'E9 files outside plan {unexpected[:5]}')
    return errors




NOREPLY_RE = re.compile(r'no[-_.]?reply|donotreply|notifications?@|mailer|newsletter|alerts?@',
                        re.I)
BRACKET_RE = re.compile(r'\[[^\]]{1,40}\]')


def _from_addr(email):
    """发件人的可匹配文本(名字 + 地址)。"""
    fr = email.get('from')
    if isinstance(fr, dict):
        return f"{fr.get('name', '')} <{fr.get('email', '')}>"
    return str(fr or '')


def _age_hours(email):
    """这封信比运行时刻早多少小时(正数=过去)。冻结成绝对时间的就算不出来。"""
    off = email.get('date_offset')
    if not isinstance(off, dict) or 'abs' in off:
        return None
    try:
        return -(float(off.get('days', 0)) * 24 + float(off.get('hours', 0)))
    except (TypeError, ValueError):
        return None

# ── v3.1.8「对齐线上」的形态闸 ────────────────────────────────────────
# ⚠️ 只在 --realism v3.1.8 时生效。v3.1.7 及更早的批次一条都不查 —— 它们的配方里
#    根本没要求这些字段,查了就是把 1292 个老世界一起判死。
#
# 每条阈值后面的百分比都是 0818 从线上真实 trace 量出来的(不是拍脑袋),
# 量法见 docs/rq3w/,要改阈值先重跑那次测量。

# 注入器只会把这五种扩展名转成真文件:.pdf/.docx 走生成器,.csv/.txt/.ics 直接落字节。
# .xlsx 和图片会拿到正确的 MIME 但内容是**纯文本**,产出的是损坏文件 —— 模型读不了。
# 依据:beeai_eval/data/seed_test_data.py `_create_email_message` 的分支。
RE_PREFIX = re.compile(r'^\s*(?:re|fwd|fw|回复|转发|答复)\s*[:：]\s*', re.I)
ATT_EXT_OK = ('.pdf', '.docx', '.csv', '.txt', '.ics')
CAL_STATUS_OK = {'confirmed', 'cancelled', 'tentative'}
CAL_TRIGGER_REQ = ('title', 'start_jinja', 'end_jinja')


def email_shape_issues(email, label_names):
    """v3.1.8:单封邮件的新字段形态。"""
    errors, warnings = [], []
    eid = email.get('id')
    for att in (email.get('attachments') or []):
        if not isinstance(att, dict):
            errors.append(f'E13 email {eid} attachment 不是 dict'); continue
        name = str(att.get('name') or '')
        if not name.lower().endswith(ATT_EXT_OK):
            errors.append(f'E13 email {eid} 附件 {name!r} 扩展名不在 {list(ATT_EXT_OK)} '
                          f'—— 注入器会产出损坏文件')
        if not str(att.get('content') or '').strip() and not att.get('url'):
            errors.append(f'E13 email {eid} 附件 {name!r} 没有 content')
    html = email.get('body_html')
    if html is not None:
        if not isinstance(html, str):
            errors.append(f'E13 email {eid} body_html 不是字符串')
        elif len(html) < 400:
            # ⚠️ 这个阈值 0818 下午从 800 降到 400。原来的 800 是"每封都要 ≥1500"
            #    那版配方定的;改成两档之后 **B 档背景噪声本来就只写 500~1000**,
            #    800 会把设计内的 B 档全报成警告(实测一批 27 个世界里误报 8 条)。
            #    400 以下才是真的"<p> 包一句话"交差。
            #    A 档写没写够**不在这里查** —— 那是世界级的事,见 world_shape_issues
            #    里的「≥2000 字符的至少 3 封」。
            warnings.append(f'W13 email {eid} body_html 只有 {len(html)} 字符 '
                            f'(连 B 档背景噪声的 500~1000 都不到)')
    labels = email.get('labels')
    if labels is not None:
        if not isinstance(labels, list):
            errors.append(f'E13 email {eid} labels 不是 list')
        else:
            unknown = [x for x in labels if x not in label_names]
            if unknown:
                # 注入顺序是 labels → emails,label 没在 seed.labels 里声明就映射不到 id,
                # 邮件上那个标签会**静默不生效**(seed_inject.py `_do_gmail` 的注释写了)
                warnings.append(f'W13 email {eid} 用了未声明的 label {unknown} '
                                f'—— 不会真的打上去')
    return errors, warnings


def calendar_shape_issues(event):
    """v3.1.8:单个日程的新字段形态。"""
    errors, warnings = [], []
    eid = event.get('id')
    # ⛔ recurrence / RRULE **一律禁用**(0819 定)。
    #    47 个世界最小二乘拟合逐条就绪率:普通日程 79%、全天事件≈100%、
    #    **带 RRULE 的重复事件 ≈0% —— 一条都进不了记忆索引**。
    #    平台把 RRULE 展开成实例后事件 id 变了,就绪检查永远对不上,
    #    于是 `memory_search` 查不到这半个日历,而卷照跑、分照出,事后看不出来。
    #    周期性会议改成**摊开写**:同一个 summary、日期按周期排、各给各的 id。
    #    那样既能进索引,也更像线上(线上给的本来就是"重复事件的实例")。
    if event.get('recurrence') is not None:
        errors.append(f'E13 calendar {eid} 带 recurrence —— v3.1.8 禁用 RRULE,'
                      f'周期性会议要摊成一条条独立日程(它进不了记忆索引,实测就绪率 ≈0%)')
    # ⛔ 过去的日程同样进不了记忆索引(0819 实测:51 个世界四参数拟合 R²=0.86,
    #    未来普通 84% / 未来全天 93% / **过去 ≈0%** / RRULE ≈0%)。
    #    这条是 0819 下午补的:我先修了 RRULE,却在提示词里写"周会排 -7/0/+7 天",
    #    于是每个世界都留一条对 memory_search 隐身的上期会议,就绪率卡在 68%。
    #    ⚠️ 只管日历 —— 过去的**邮件**正常入索引(实测 96%)。
    start = event.get('start') or {}
    if isinstance(start, dict) and isinstance(start.get('days_offset'), int) \
            and start['days_offset'] < 0:
        errors.append(f'E13 calendar {eid} 排在过去({start["days_offset"]} 天)—— '
                      f'过去的日程进不了记忆索引(实测 ≈0%),模型查不到。'
                      f'要表达"上次开过了"就写进本期日程的 description')
    for key in ('all_day',):
        v = event.get(key)
        if v is not None and not isinstance(v, bool):
            # 起草模型实测会写成字符串 "True"(0818 首条 v3.1.8 世界就是),
            # Python 里它是真值所以一路不报错,但存进 seed 的类型是错的
            errors.append(f'E13 calendar {eid} {key} 必须是 JSON 的 true/false,'
                          f'给的是 {type(v).__name__} {v!r}')
    st = event.get('status')
    if st is not None and st not in CAL_STATUS_OK:
        errors.append(f'E13 calendar {eid} status={st!r} 不是 {sorted(CAL_STATUS_OK)}')
    if event.get('all_day') and event.get('attendees'):
        warnings.append(f'W13 calendar {eid} 全天事件带 attendees —— 线上很少见')
    return errors, warnings


def calendar_trigger_issues(case, seed):
    """v3.1.8:日历触发题的 trigger_event 形态 + 触发源必须真在世界里。"""
    errors = []
    ev = (case.get('llm') or {}).get('trigger_event') or {}
    src = ev.get('source_type')
    if case.get('trigger') != 'event':
        return errors
    if src not in ('mail', 'calendar'):
        errors.append(f'E13 event 题的 trigger_event 缺 source_type(给的是 {src!r})')
        return errors
    if src != 'calendar':
        return errors
    miss = [k for k in CAL_TRIGGER_REQ if not ev.get(k)]
    if miss:
        errors.append(f'E13 日历触发缺字段 {miss}')
    # 触发的那条日程必须真的在 seed.calendar_events 里,否则模型查日历查不到触发源
    title = str(ev.get('title') or '').strip().lower()
    if title:
        summaries = {str(c.get('summary') or '').strip().lower()
                     for c in (seed.get('calendar_events') or [])}
        if title not in summaries:
            errors.append(f'E13 日历触发的日程 {ev.get("title")!r} 不在 seed.calendar_events 里')
    return errors


def needs_drive_issues(case, seed):
    """v3.1.8:计划说这题当年真用过网盘/表格,世界里就必须有文件(E13)。

    `needs_drive` 是 build_plan 从**真实轨迹**里读出来的(那天模型确实调了
    drive/sheets 工具),不是猜的。标了却没给 `drive_files`,模型一查网盘就是空的
    —— 那条规则等于没被考到,而且判官会按"漏做"扣分。
    0818 实测:7 道标了 needs_drive 的题里有 2 道没给文件。
    """
    if not case.get('needs_drive'):
        return []
    if seed.get('drive_files'):
        return []
    return ['E13 计划标了 needs_drive(当年真调过网盘/表格),但 seed.drive_files 是空的']


def user_notes_issues(seed):
    errors = []
    notes = seed.get('user_notes')
    if notes is None:
        return errors
    if not isinstance(notes, list):
        return ['E13 user_notes 不是 list']
    for i, nt in enumerate(notes):
        if not isinstance(nt, dict) or not str(nt.get('content') or '').strip():
            errors.append(f'E13 user_notes[{i}] 缺 content')
    if len(notes) > 3:
        errors.append(f'E13 user_notes 给了 {len(notes)} 条,上限 3 条')
    return errors




def world_shape_issues(seed):
    """v3.1.8:整个世界层面的形态(W13,只警告)。

    这几条**逐封看都合法,合起来才不对** —— 比如每封 800 字符的 HTML 单看没问题,
    但一个世界里一封长的都没有,「信号埋在版式里」这个训练价值就没了。
    """
    warnings = []
    emails = seed.get('emails') or []
    if not emails:
        return warnings

    n_long = sum(1 for e in emails if len(str(e.get('body_html') or '')) >= 2000)
    if n_long < 3:
        warnings.append(f'W13 世界里 ≥2000 字符的 body_html 只有 {n_long} 封'
                        f'(A 档要 4~6 封;判定相关的那几封必须写长)')
    if not (8 <= len(emails) <= 16):
        warnings.append(f'W13 世界里 {len(emails)} 封邮件(目标 10~14;线上一屏中位 14)')

    n_lab = sum(1 for e in emails if e.get('labels'))
    if n_lab > 2:
        warnings.append(f'W13 {n_lab} 封邮件带自定义标签(上限 2;线上只有 3% 的邮件有)')

    n_old = sum(1 for e in emails
                if (_age_hours(e) or 0) > 24 * 7)
    if n_old == 0 and len(emails) >= 8:
        warnings.append('W13 世界里没有 7 天以前的老邮件(线上占 22.9%,'
                        '没有的话「最近一个月」这类查询翻不到东西)')

    big = [c for c in (seed.get('calendar_events') or []) if len(c.get('attendees') or []) >= 6]
    withatt = [c for c in (seed.get('calendar_events') or []) if c.get('attendees')]
    if withatt and not big:
        warnings.append(f'W13 {len(withatt)} 个带参会者的会全都少于 6 人'
                        f'(线上中位 7 人、40% 在 10 人以上;两人会没什么可推理的)')
    return warnings


def kc_text(eo):
    """把 key_constraints 取成一段文本,**两种形状都吃**。

    v3.1.7 是列表(`["...", "..."]`),v3.1.8 改成了一条分号分隔的字符串。
    ⛔ 直接 `' '.join(str(x) for x in kc)` 对字符串会**逐字符拆开**,
    拼出 "应 f i r e ; ..." 这种东西 —— 语种判别当场废掉,而且一声不吭。
    """
    kc = eo.get('key_constraints')
    if isinstance(kc, str):
        return kc
    return ' '.join(str(x) for x in (kc or []))


# gold 粒度带:取自 QA 评测集实测(201 题,要点数均值 3.27、describe 中位 122 字)。
# 极性不同,正确终态的复杂度就不同 —— skip 只有"不输出"一件事。
GOLD_POINTS = {'skip': (1, 3), 'quiet': (1, 4), 'fire': (2, 7), 'deliver': (2, 7)}
GOLD_DESCRIBE_MAX = 400      # 字符;线上中位 122,留三倍余量
GOLD_TOTAL_MAX = 900         # 整个 expected_output 序列化后的字符上限


def gold_shape_issues(case):
    """v3.1.8:判分标准(gold)不许写成长篇大论。**这是四条闸里最贵的一条。**

    0819 实案:v3.1.8 试水四卷全部用主线 gen_worlds.py 跑,gold 中位 2072~2500 字符,
    通过率 17%;而 0815 实发的那批 gold 中位 196 字符,同样 26 条规则通过率 53.8%。
    我一开始把这 36 分的差当成"世界配方变难了"去查工具形态、查模型、查判官版本,
    全是弯路 —— **唯一变量是 gold**。

    为什么 gold 长了分就低:判官拿 gold 逐条对答案。判分点写到 5~7 个、describe
    复述整个世界,一份"基本做对"的答案总能被挑出没覆盖到的那一条。
    0815 用 n=2007 同一批答案换 gold 重判测过:粗粒度 gold 56.9% vs 细粒度 29.1%,
    **值 +27.8 分(±2.4)**。

    ⚠️ **这里只报警告(W9),硬闸在发车口** —— 0819 想清楚的分层:
    gold 粒度是 `regold_batch.py` 这个**独立步骤**在纠的,而 regold 跑在 **items** 上、
    items 又要先装配 —— 所以在 raw 阶段判死会形成死锁(lint 不过 → 装配被回执闸堵住
    → 永远跑不到 regold)。
    实测起草模型靠提示词只能压到 ~1050 字符,`regold_batch` 才能到 ~500。
    ⇒ raw 阶段**只提示**,真正拦人的是 `preflight_gold.py`(挂在 fire_high 上):
      不管上游走哪条路、跳过哪一步,发出去的东西必须满足粒度要求。
      那才是唯一拦得住"跳过 regold"的位置。
    """
    E = []
    eo = (case.get('llm') or {}).get('expected_output') or {}
    if not eo:
        return E                      # 缺 expected_output 是别处(E0)的事;E 恒为空,占位
    pol = str(case.get('polarity') or (case.get('llm') or {}).get('polarity') or '').lower()

    W = []
    kc = eo.get('key_constraints')
    if isinstance(kc, list):
        W.append(f'W9 key_constraints 还是数组({len(kc)} 条)—— v3.1.8 要求一条分号分隔的字符串')
        pts = len(kc)
    else:
        pts = len([x for x in str(kc or '').split(';') if x.strip()])
    lo, hi = GOLD_POINTS.get(pol, (1, 7))
    if pts and not (lo <= pts <= hi):
        W.append(f'W9 判分点 {pts} 个,{pol or "未知极性"} 该在 {lo}~{hi}(regold 会纠)')

    dsc = str(eo.get('describe') or '')
    if len(dsc) > GOLD_DESCRIBE_MAX:
        W.append(f'W9 describe {len(dsc)} 字符 > {GOLD_DESCRIBE_MAX}(别把世界复述进 gold)')

    total = len(json.dumps(eo, ensure_ascii=False))
    if total > GOLD_TOTAL_MAX:
        W.append(f'W9 gold 整体 {total} 字符 > {GOLD_TOTAL_MAX} —— **发车前必须跑 regold**,'
                 f'不跑会被 preflight_gold 拦下(实测这一项值 15.7 分)')
    return W


def language_issues(case):
    """v3.1.8:世界 / gold 的语种要跟着**规则原文**走(W13,只警告不阻断)。

    SYSTEM 第 6 条本来就写了「世界语言 = 用户语言;expected_output 三字段的语言 =
    用户语言」,但 0818 在 v3.1.7 的 2025 个世界上实测:

        世界与规则同语种  95.4%
        gold 与规则同语种 87.1%   ← 262 道不符,最多的是「规则英文、gold 写成中文」
                                    (起草提示词本身是中文,模型顺手就用了)

    ⚠️ **这条只报警告**:gold 是喂给判官的参考答案、不是喂给模型的,
    「gold 用中文描述一份葡语报告该有什么」到底扣不扣分**我们没测过**。
    在测出来之前不该用它堵发车 —— 但它违反的是我们自己写的约束,该看见。

    ⛔ 别用 case['lang'] 当基准:池子里那个标签 33.7% 是错的(0811 造池时缺
    langdetect,葡语被判成越南语)。一律现算,见 relabel_languages。
    """
    from extract_source import infer_lang

    def enough(text, need):
        """够不够长到值得判语种。

        ⚠️ 不能直接数字符:**中日韩一个字顶拉丁两三个**。按字符数卡 80,
        68 个汉字的 gold 会被判成"太短、跳过" —— 而「gold 写成中文」恰恰是
        我们最想抓的那一类(v3.1.7 实测 66 道),等于闸对着自己的盲区。
        """
        w = sum(2.5 if ord(ch) > 0x2E80 else 1 for ch in text)
        return w >= need

    warnings = []
    llm = case.get('llm') or {}
    rule_lang = infer_lang(case.get('rule') or '', case.get('lang'))

    eo = llm.get('expected_output') or {}
    gold = ' '.join(str(eo.get(k) or '') for k in ('goal', 'describe')) + ' ' + kc_text(eo)
    if enough(gold, 80):
        got = infer_lang(gold, rule_lang)
        if got != rule_lang:
            warnings.append(f'W13 gold 语种 {got} ≠ 规则语种 {rule_lang}')

    seed = llm.get('seed') or {}
    world = ' '.join(f"{e.get('subject', '')} {e.get('body', '')}"
                     for e in (seed.get('emails') or []))[:6000]
    if enough(world, 200):
        got = infer_lang(world, rule_lang)
        if got != rule_lang:
            warnings.append(f'W13 世界语种 {got} ≠ 规则语种 {rule_lang}')
    return warnings


# ── 批次级:按族统计形态比例 ──────────────────────────────────────
# 单个 case 合法 ≠ 整批不退化。上一次的教训就是 lint 全绿但 89% 的世界没有日历
# (漏传 --mailboxes),因为「比例」这种事逐个文件看不出来。
# 下界/上界取自线上实测,留了余量;n 不够时只报数不判死。
REALISM_TARGETS = {
    # 名字                    下界   上界   线上实测
    # —— 第一轮(0818 上午)——
    'body_html 占比':        (0.60, 0.90, 0.867),
    '长HTML(≥2k)邮件占比':    (0.20, 0.60, None),    # A 档:每世界 4~6 封 / 共 10~14 封
    '附件邮件占比':            (0.05, 0.25, 0.156),
    '周期性会议实例占比':      (0.25, 0.75, 0.531),
    '全天事件占比':            (0.20, 0.60, 0.446),
    'description 占比':      (0.10, 0.40, 0.243),
    'location 占比':         (0.08, 0.35, 0.198),
    '带参会状态占比':          (0.85, 1.00, 0.97),
    # —— 第二轮(0818 下午,逐维和线上对齐后补的)——
    '6人以上的会占比':          (0.20, 0.75, 0.62),   # 线上中位 7 人、40% 在 10 人以上
    '未读邮件占比':            (0.60, 0.95, 0.826),
    '带自定义标签的邮件':        (0.00, 0.12, 0.03),   # ⚠️ 上一批 56%,过度归档
    '超过7天的老邮件':          (0.10, 0.45, 0.229),
    'noreply类发件人':        (0.15, 0.50, 0.337),
    '主题带方括号':         (0.10, 0.45, 0.286),
}
MIN_N_FOR_GATE = 60      # 分母不足就只报数不判死(试水批 25 个世界会落在这条线附近)


def realism_stats(cases):
    """按 arm 统计形态比例。返回 {arm: {指标: (命中, 分母)}}。"""
    from collections import defaultdict as _dd
    out = _dd(lambda: _dd(lambda: [0, 0]))

    def bump(arm, key, hit, tot=1):
        out[arm][key][0] += hit; out[arm][key][1] += tot

    for case in cases:
        arm = case.get('arm') or '?'
        seed = (case.get('llm') or {}).get('seed') or {}
        emails = seed.get('emails') or []
        cals = seed.get('calendar_events') or []
        for e in emails:
            html = str(e.get('body_html') or '')
            bump(arm, 'body_html 占比', 1 if html else 0)
            bump(arm, '长HTML(≥2k)邮件占比', 1 if len(html) >= 2000 else 0)
            bump(arm, '附件邮件占比', 1 if e.get('attachments') else 0)
            bump(arm, '未读邮件占比', 1 if e.get('unread') else 0)
            bump(arm, '带自定义标签的邮件', 1 if e.get('labels') else 0)
            bump(arm, 'noreply类发件人', 1 if NOREPLY_RE.search(_from_addr(e)) else 0)
            bump(arm, '主题带方括号', 1 if BRACKET_RE.search(str(e.get('subject') or '')) else 0)
            bump(arm, '超过7天的老邮件', 1 if _age_hours(e) is not None
                 and _age_hours(e) > 24 * 7 else 0)
        _repeat_titles = {t for t, k in collections.Counter(
            (c.get('summary') or '').strip() for c in cals).items() if t and k >= 2}
        for c in cals:
            # 摊开写之后,"周期性"体现为**同一 summary 在世界里出现多次**。
            # ⚠️ 这个 bump 在逐个日程的循环里,拿不到整个世界 —— 所以用外面
            # 预先算好的 _repeat_titles(本世界出现 ≥2 次的 summary 集合)。
            bump(arm, '周期性会议实例占比',
                 1 if (c.get('summary') or '').strip() in _repeat_titles else 0)
            bump(arm, '全天事件占比', 1 if c.get('all_day') else 0)
            bump(arm, 'description 占比', 1 if c.get('description') else 0)
            bump(arm, 'location 占比', 1 if c.get('location') else 0)
            att = c.get('attendees') or []
            if att:
                bump(arm, '带参会状态占比', 1 if c.get('self_response') else 0)
                bump(arm, '6人以上的会占比', 1 if len(att) >= 6 else 0)
        # 线程:同一世界里去掉 Re:/Fwd: 前缀后有 3 封以上同主题
        base = _dd(int)
        for e in emails:
            base[RE_PREFIX.sub('', str(e.get('subject') or '')).strip().lower()] += 1
        bump(arm, '有3封+线程的世界占比', 1 if any(v >= 3 for v in base.values()) else 0)
    return out


def _pad(text, width, right=False):
    """按**显示宽度**补空格 —— 中文占两列,f'{s:<22}' 按字符数算会排歪。"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(text))
    fill = ' ' * max(0, width - w)
    return (fill + str(text)) if right else (str(text) + fill)


def print_realism_report(cases):
    """打形态报表;返回 global_errors(整批越界才判死)。"""
    per_arm = realism_stats(cases)
    arms = sorted(per_arm)
    total = {}
    for arm in arms:
        for k, (hit, tot) in per_arm[arm].items():
            a, b = total.get(k, (0, 0))
            total[k] = (a + hit, b + tot)

    print('\n── v3.1.8 形态报表(按族;括号里是线上实测值)──')
    keys = [k for k in REALISM_TARGETS] + ['有3封+线程的世界占比']
    head = _pad('指标', 24) + ''.join(_pad(a[:10], 12, right=True) for a in arms) \
        + _pad('合计', 12, right=True)
    print(head)
    errs = []
    for k in keys:
        row = _pad(k, 24)
        for arm in arms + ['<合计>']:
            hit, tot = (total.get(k, (0, 0)) if arm == '<合计>'
                        else per_arm[arm].get(k, [0, 0]))
            row += _pad(f'{100*hit/tot:.0f}% ({tot})' if tot else '—', 12, right=True)
        tgt = REALISM_TARGETS.get(k)
        print(row + (f'   线上 {100*tgt[2]:.0f}%' if tgt and tgt[2] is not None else ''))
        if not tgt:
            continue
        hit, tot = total.get(k, (0, 0))
        if tot < MIN_N_FOR_GATE:
            continue
        rate = hit / tot
        # ⛔ 按**置信区间**判越界,不按点估计。0819 实案:13 个世界的试水卷里
        #    noreply 占比算出 51.2%,带上界是 50% —— 判死了。但 n=125、p≈0.5 时
        #    单侧标准误就有 4.5 个点,51.2% 和 50% 根本分不开。
        #    点估计判死 = 小批次必然被噪声堵住,人就会去关闸(或者加 --no-strict),
        #    结果连真越界也一起放过了。**闸要挡真信号,不能挡噪声。**
        half = 1.96 * (rate * (1 - rate) / tot) ** 0.5
        if rate + half < tgt[0] or rate - half > tgt[1]:
            online = f',线上 {100*tgt[2]:.0f}%' if tgt[2] is not None else ''
            errs.append(f'v3.1.8 形态越界:{k} = {100*rate:.1f}% ±{100*half:.1f} '
                        f'(要求 {100*tgt[0]:.0f}~{100*tgt[1]:.0f}%{online},n={tot})')
        elif not (tgt[0] <= rate <= tgt[1]):
            print(f'   ⚠️ {k} 点估计 {100*rate:.1f}% 出带(要求 '
                  f'{100*tgt[0]:.0f}~{100*tgt[1]:.0f}%),但 ±{100*half:.1f} 的区间还压着边界,'
                  f'n={tot} 不够判死 —— 样本大了要复查')
    # HTML 正文长度(中位数,直接从 cases 算)
    allen = [len(str(e['body_html']))
             for c in cases for e in ((c.get('llm') or {}).get('seed') or {}).get('emails') or []
             if e.get('body_html')]
    if allen:
        allen.sort(); med = allen[len(allen) // 2]
        print(_pad('body_html 长度中位', 24) + _pad(med, 12, right=True) + '   线上 2882(纯文本正文线上中位也是这个量级)')
        # ⚠️ 这个中位数**只报不判**。0818 下午改成两档之后(A 档 4~6 封写 2000~4000、
        #    B 档 5~8 封只写 500~1000),中位会被 B 档压到 1000 上下 —— 那是设计如此,
        #    不是没做到。真正该判的是「长 HTML(≥2k)邮件占比」那一行,已经在上表里。
        #    留着这个数是因为它一眼能看出 A 档有没有偷懒(A 档塌了中位会掉到 700 以下)。
    # 某一族单独塌掉:整批有、这一族一个都没有 —— 这正是「89% 没日历」那次的样子
    for k in keys:
        hit, tot = total.get(k, (0, 0))
        if not tot or hit / tot < 0.30:
            continue
        for arm in arms:
            h, t = per_arm[arm].get(k, [0, 0])
            if t >= 10 and h == 0:
                errs.append(f'v3.1.8 形态:族 {arm} 的「{k}」是 0/{t},而整批是 {100*hit/tot:.0f}% '
                            f'—— 这一族单独退化了')
    return errs


def main():
    global RAW, POOL
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='起草稿目录 cases_raw')
    ap.add_argument('--pool', required=True, help='阶段1产物 source_pool.jsonl')
    ap.add_argument('--fix-list', default='')
    ap.add_argument('--receipt', default='',
                    help='机器可验的 lint 回执;默认 <raw上级>/lint_receipt.json')
    ap.add_argument('--manifest', default='',
                    help='gen_worlds plan manifest;默认 <raw上级>/<raw名>_plan_manifest.json')
    # 0811 评审整改(点7):有 error 默认非零退出,让下游关卡真正被闸住。
    # 三轮修循环里要「报告但继续」时显式传 --no-strict。
    ap.add_argument('--realism', default='', choices=sorted(REALISM_CANON),
                    help="配方档位,必须和 gen_worlds.py 的 --realism 一致。"
                         "''=v3.1.7 老配方(默认,只查老规则);"
                         'v3.1.8=额外查「对齐线上」的形态(参会状态/附件扩展名/recurrence 形状/'
                         '日历触发/user_notes),并打一张**按族的形态报表**。'
                         '⚠️ 默认关闭 —— 旧批次全都没写这些字段(v3.1.7 实测 1292/2025 个世界没参会状态),'
                         '无条件启用会把它们的发车堵死')
    ap.add_argument('--require-rsvp', action='store_true',
                    help='旧名,等价于 --realism v3.1.8')
    ap.add_argument('--no-strict', action='store_true',
                    help='有 error 也 exit 0(仅供修复循环;流水线正式关卡禁用)')
    args = ap.parse_args()
    global REQUIRE_RSVP, REALISM
    _raw_realism = args.realism or ('v8' if args.require_rsvp else '')
    REALISM = normalize_realism(_raw_realism)
    REQUIRE_RSVP = is_aligned_recipe(REALISM)          # 兼容旧名
    if _raw_realism != REALISM:
        print(f'ℹ️ --realism {_raw_realism!r} 是弃用旧名,已归到正名 {REALISM}', flush=True)
    if args.require_rsvp and not args.realism:
        print('ℹ️ --require-rsvp 是旧名,已当作 --realism v3.1.8 处理', flush=True)
    if is_aligned_recipe(REALISM):
        print(f'⚠️ --realism {REALISM} 已开启:除老规则外还查「对齐线上」的形态'
              f'(旧批次不满足这些,别拿它扫旧数据)', flush=True)
    RAW, POOL = Path(args.raw), Path(args.pool)
    print(f'稿={RAW} 源池={POOL}', flush=True)
    manifest_path = (Path(args.manifest) if args.manifest else
                     RAW.parent / f'{RAW.name}_plan_manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest_fingerprints = (manifest or {}).get('case_fingerprints') or {}
    pool_sha256 = sha256_file(POOL)
    pool = {json.loads(l)['rule_key']: json.loads(l) for l in open(POOL)}
    prof_ngrams = []
    for f in sorted(PROF_SEEDS.glob('*.json')):
        try:
            s = json.load(open(f))
            prof_ngrams.append((f.name, ngrams(seed_text(s))))
        except Exception:
            pass

    n_err = n_warn = n_ok = 0
    warns_by_kind = defaultdict(int)
    errs_by_kind = defaultdict(int)
    bad_files = []
    files = sorted(RAW.glob('*.json'))
    all_cases = []          # v3.1.8 形态报表要按族统计,得把 case 攒起来
    quarantined, quarantine_errors = load_quarantine(RAW)
    if quarantined:
        print(f'已隔离 {len(quarantined)} 道(不计入出稿数):')
        for cid, why in sorted(quarantined.items()):
            print(f'  · {cid} — {why}')
    global_errors = quarantine_errors + manifest_completeness_errors(manifest, files, quarantined)
    for error in global_errors:
        errs_by_kind[error.split()[0]] += 1
    for fp in files:
        case = json.loads(fp.read_text())
        all_cases.append(case)
        E, W = [], []
        llm = case['llm']; seed = llm.get('seed') or {}

        # E1 schema
        email_ids = []
        for e in seed.get('emails') or []:
            miss = [k for k in ('id','from','to','subject','body','date_offset') if k not in e]
            if miss: E.append(f'E1 email missing {miss}')
            do = e.get('date_offset')
            if isinstance(do, dict):
                numeric = all(isinstance(do.get(k, 0), (int, float)) for k in ('days','hours'))
                if not numeric:
                    E.append('E1 date_offset non-numeric')
                else:
                    if abs(do.get('days', 0)) > 60: W.append('E1 date_offset days>60')
                    total_hours = do.get('days', 0) * 24 + do.get('hours', 0)
                    if total_hours > 0:
                        E.append(f"E7 future inbox email {e.get('id')!r} offset={do}")
            elif do is not None: E.append('E1 date_offset not dict')
            if e.get('id') is not None:
                email_ids.append(str(e['id']))
        dup_ids = sorted({x for x in email_ids if email_ids.count(x) > 1})
        if dup_ids:
            E.append(f'E7 duplicate email ids {dup_ids[:5]}')
        for c in seed.get('calendar_events') or []:
            miss = [k for k in ('id','summary','start','end') if k not in c]
            if miss: E.append(f'E1 calendar missing {miss}')
            for kk in ('start','end'):
                v = c.get(kk)
                if isinstance(v, dict):
                    keys = set(v)
                    if not (keys & {'minutes_offset','days_offset','next_weekday'}):
                        E.append(f'E1 calendar {kk} shape {sorted(keys)}')
                elif v is not None: E.append(f'E1 calendar {kk} not dict')
            e8e, e8w = calendar_rsvp_issues(c, require=REQUIRE_RSVP)
            E.extend(e8e); W.extend(e8w)
            if is_aligned_recipe(REALISM):
                ce, cw = calendar_shape_issues(c)
                E.extend(ce); W.extend(cw)
        if is_aligned_recipe(REALISM):
            label_names = {str((l.get('name') if isinstance(l, dict) else l))
                           for l in (seed.get('labels') or [])}
            for e in seed.get('emails') or []:
                ee, ew = email_shape_issues(e, label_names)
                E.extend(ee); W.extend(ew)
            E.extend(calendar_trigger_issues(case, seed))
            E.extend(user_notes_issues(seed))
            E.extend(needs_drive_issues(case, seed))
            W.extend(gold_shape_issues(case))
            W.extend(world_shape_issues(seed))
            W.extend(language_issues(case))

        for address in seed_header_addresses(seed):
            if not address.isascii():
                E.append(f'E1 non-ASCII email address {address}')

        # E2 domains —— 白名单硬闸:凡不是保留域一律判错,不再放行"陌生公司域"
        for a in all_addresses(case):
            if POOL_ACCOUNTS.search(a): continue
            dom = a.rsplit('@', 1)[-1].strip('>\u00bb"\' ').lower()
            if SAFE_DOM.search(dom): continue
            kind = ('real mailbox' if REAL_DOM.search(a)
                    else 'shopee addr' if a.endswith('@shopee.com') else 'deliverable domain')
            E.append(f'E2 {kind} {a}')

        # E3 absolute dates (outside jinja)
        for label, txt in (('seed', seed_text(seed)),
                           ('question', question_without_original_rule(case))):
            if ABS_DATE.search(strip_jinja(txt)):
                E.append(f'E3 absolute date in {label}: {ABS_DATE.search(strip_jinja(txt)).group()}')

        # E4 jinja render
        for label, txt in [('question', case['question'])] + \
                [(f"email:{e.get('id')}", str(e.get('subject','')) + '\n' + str(e.get('body',''))) for e in seed.get('emails') or []] + \
                [(f"cal:{c.get('id')}", str(c.get('summary','')) + '\n' + str(c.get('description',''))) for c in seed.get('calendar_events') or []]:
            err = render_ok(txt)
            if err: E.append(f'E4 jinja {label}: {err[:80]}')

        # E5 event trigger in seed
        if case['trigger'] == 'event':
            ev = llm.get('trigger_event')
            if not ev: E.append('E5 no trigger_event')
            elif (ev or {}).get('source_type') == 'calendar':
                # 日历触发没有"触发邮件" —— 它的触发源是一条日程。
                # 对应的检查在 calendar_trigger_issues(要求那条日程真在
                # seed.calendar_events 里)。这里不查,否则日历触发题必然误报 E5。
                pass
            else:
                title = str(ev.get('title',''))
                subjects = [str(e.get('subject','')) for e in seed.get('emails') or []]
                if not trigger_title_matches_seed(title, subjects):
                    E.append(f'E5 trigger email not in seed (title={title[:50]!r})')

        # E6 polarity structure
        if case['polarity'] in ('fire','skip') and case['trigger'] != 'event':
            W.append('E6 fire/skip on cron case')
        if llm.get('polarity') and llm['polarity'] != case['polarity']:
            E.append(f"E6 llm polarity={llm['polarity']} != plan {case['polarity']}")

        # E8 v2 溯源与同源时间闸。旧存量 schema=1 不追溯报错，新生成数据必须齐全。
        if case.get('schema_version', 1) >= 2:
            missing_lineage = [k for k in ('user_key', 'cutoff_at', 'source_tick_index',
                                            'plan_fingerprint', 'generation') if case.get(k) is None]
            if missing_lineage:
                E.append(f'E8 missing lineage {missing_lineage}')
            if case.get('judge_target_aligned') is False:
                E.append('E8 source tick != judge target')
            generation = case.get('generation') or {}
            if generation.get('pool_sha256') != pool_sha256:
                E.append('E8 source pool fingerprint mismatch')
            expected_fp = manifest_fingerprints.get(case.get('case_id'))
            if not manifest:
                E.append('E8 plan manifest missing')
            elif expected_fp != case.get('plan_fingerprint'):
                E.append('E8 plan fingerprint mismatch')
            try:
                tz = ZoneInfo(case['tz_iana'])
                probe = datetime(2026, 8, 3, 12, 0, tzinfo=tz)
                seconds = int(probe.utcoffset().total_seconds())
                sign = '+' if seconds >= 0 else '-'
                seconds = abs(seconds)
                actual_offset = f'{sign}{seconds // 3600:02d}:{seconds % 3600 // 60:02d}'
                if actual_offset != case.get('tz_offset'):
                    E.append(f"E8 timezone {case.get('tz_iana')}={actual_offset} != {case.get('tz_offset')}")
            except (KeyError, ZoneInfoNotFoundError):
                E.append(f"E8 invalid timezone {case.get('tz_iana')!r}")

        # E10 copy from real material.  This used to be warning-only, which
        # allowed business details and identity-bearing prose into SFT even
        # when task semantics were otherwise valid.
        g = pool.get(case['rule_key'])
        if g:
            mat = '\n'.join(str(tr) for r in g['records'][:3] for t in r['ticks'][:2] for tr in t['tool_responses'][:4])
            ov = overlap(ngrams(seed_text(seed)), ngrams(mat))
            if ov > 0.30: E.append(f'E10 material-copy overlap={ov:.2f}')

        # E11 eval seed similarity is experiment leakage, not a soft style issue.
        sng = ngrams(seed_text(seed))
        for name, png in prof_ngrams:
            ov = overlap(sng, png)
            if ov > 0.25:
                E.append(f'E11 prof-seed overlap {name} {ov:.2f}')
                break

        # W3 rule-named entities present
        rule = case['rule']
        named = re.findall(r"[\'\"“”‘’「」]([^\'\"“”‘’「」]{3,40})[\'\"“”‘’「」]", rule)[:6]
        world_txt = (seed_text(seed) + json.dumps(llm.get('trigger_event') or {}, ensure_ascii=False)).lower()
        for nm in named:
            key = nm.strip().lower()
            if len(key) >= 4 and key not in world_txt and case['polarity'] in ('fire','deliver'):
                W.append(f'W3 named entity absent: {nm[:30]!r}')

        if E:
            n_err += 1; bad_files.append((fp.name, E, W))
            for e in E: errs_by_kind[e.split()[0]] += 1
        elif W:
            n_warn += 1
        else:
            n_ok += 1
        # 0818:警告以前只落进 case 文件、不上汇总行,等于「记了但没人看见」。
        # W12(参会状态缺失)默认就是走警告这条路的,不打出来这个设计就是空的。
        # ⛔ 这个循环**必须放在 if/elif/else 之后** —— 0818 第一版夹在 elif 和 else
        #    中间,于是 `else` 绑到了 for 上(Python 的 for-else),n_ok 变成每个文件都 +1,
        #    汇总行永远显示 ok=文件总数。py_compile 和单测都查不出来。
        for w in W:
            warns_by_kind[w.split()[0]] += 1
        # 每个文件都写本轮结果。旧实现只在本轮有 E/W 时覆盖,因此一个已经修好的
        # case 仍可能残留上轮 errors,也可能没有任何“确实 lint 过”的证据。
        lint = {'schema_version': 1, 'errors': E, 'warnings': W}
        if case.get('lint') != lint:
            case['lint'] = lint
            fp.write_text(json.dumps(case, ensure_ascii=False, indent=1))

    if is_aligned_recipe(REALISM):
        # 逐个 case 合法 ≠ 整批不退化 —— 上一次「89% 的世界没有日历」就是 lint 全绿出来的
        global_errors.extend(print_realism_report(all_cases))

    total_errors = n_err + len(global_errors)
    print(f'files={len(files)} ok={n_ok} warn-only={n_warn} '
          f'error-cases={n_err} global-errors={len(global_errors)}')
    print('errors by kind:', dict(errs_by_kind))
    if warns_by_kind:
        print('warnings by kind:', dict(warns_by_kind), '(不阻断,但要看)')
    for error in global_errors:
        print(f'  GLOBAL: {error}')
    for name, E, W in bad_files[:25]:
        print(f'  {name}: {E[:3]}')
    relint = Path(args.fix_list) if args.fix_list else RAW.parent / 'relint.txt'
    relint.write_text('\n'.join(n for n, _, _ in bad_files))
    print(f"error case list -> {relint} ({len(bad_files)})")
    # 回执把“哪一版 raw + 哪一版 source_pool 通过了 lint”钉死。assemble 会重算
    # 两个 SHA；lint 后哪怕只改一个字，旧回执也立即失效，不能再靠文件里的旧字段蒙混。
    receipt = Path(args.receipt) if args.receipt else RAW.parent / 'lint_receipt.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'passed': total_errors == 0,
        'strict_gate': not args.no_strict,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_dir': str(RAW.resolve()),
        'raw_sha256': json_tree_digest(RAW),
        'pool_file': str(POOL.resolve()),
        'pool_sha256': sha256_file(POOL),
        'files': len(files),
        'ok': n_ok,
        'warn_only': n_warn,
        'errors': total_errors,
        'error_cases': n_err,
        'global_errors': global_errors,
        # 隔离名单进回执:下游(assemble/审计)能一眼看到这批**少了哪几道、为什么少**,
        # 不用去翻 quarantine.json。以前手改 manifest 是查不出来的。
        'quarantined': quarantined,
        'lint_script_sha256': sha256_file(Path(__file__)),
    }
    tmp = receipt.with_suffix(receipt.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(receipt)
    print(f'lint receipt -> {receipt} (passed={payload["passed"]})')
    if total_errors and not args.no_strict:
        print(f'❌ lint 有 {total_errors} 个 error,非零退出(修复循环请用 --no-strict)')
        sys.exit(1)


if __name__ == '__main__':
    main()
