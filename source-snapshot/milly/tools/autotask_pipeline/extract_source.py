#!/usr/bin/env python3
"""阶段1:从 lsy 训练集提取工厂源数据并分四臂分组。

输入: **--src 必填,没有默认值**。rq3w 这条线用的是
      /home/work/migoo_ai_public/lsy/autotask/data/v3/v3_1/v3_1_balanced.jsonl.gz
      (「v3.1」这个批次命名就是从它来的);rq1/rq2 两条老线用的是 v1/train/v1_raw_best.jsonl.gz。
⛔ 这里**不写默认值**:0826 实案,有人(我)读到写死的 SRC 常量就当成"现行数据源",
   据此得出「我们的池子只有一天数据」的错误结论,连带推出一串错的分析。
   数据源身份必须显式传,查历史批次实际用了哪份 → 看产物的 plan_manifest。
输出: out/source_pool.jsonl  — 每行一个唯一规则,含:
        arm(template_evening/template_dawn/alert/custom)、rule、trigger、lang、
        records[]:每条真实记录的 (trace_id, timezone, ticks[], 每tick的工具调用/返回/信封)
      out/excluded_rules.jsonl — 被剔除的规则及原因
      out/arms_summary.json   — 分组统计

剔除规则:
  - agent != professional 或非标准 cron/event tick(lite/compact/skill-tick/user-context)
  - 记录文本含测试账号标识(评测机器人)
  - 非模板家族的自定义规则,与任何线上评测卷题面 8-gram Jaccard ≥ 0.70(撞评测题)
    (模板家族=晚报/晨报/警报骨架,题面同源是产品事实,靠世界隔离,不按 J 剔)
"""
import argparse, json, gzip, re, sys, hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SRC = ''   # ⛔ 不设默认值 —— 见模块 docstring。由 --src 必填注入。
OUT_DIR = Path(__file__).resolve().parent.parent / 'out'
LEAK_HITS = '/home/work/migoo_ai_public/haiyang/coding/claude_bak/jobs/3340f05e/tmp/leak_hits.json'

# ---------- 工具 ----------

def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())

def ngrams(s, n=8):
    s = norm(s)
    return set(s[i:i + n] for i in range(max(0, len(s) - n + 1)))

def jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def rule_key(rule):
    return hashlib.sha1(norm(rule).encode()).hexdigest()[:16]

# ---------- 模板家族识别(骨架样例) ----------
# 晚报 / 晨报:官方模板原文的多语言版(训练数据里的 8 个巨型重复规则)
EVENING_HEADS = [
    'wrap up my day. go through email conversations',
    'tutup hari saya. tinjau percakapan email',
    '本日の業務を締めくくります。午前8時00分以降に届いたメール',
    'hoàn thành công việc trong ngày. xem lại các cuộc trò chuyện email',
    'encerre meu dia. confira as conversas por e-mail',
]
DAWN_HEADS = [
    "use dawn-brief skill: prepare today's morning brief now",
    'gunakan keterampilan dawn-brief: siapkan pengarahan pagi ini',
    'dawn-brief skill を使用して、今日の朝のブリーフィングを今すぐ作成',
    'sử dụng skill dawn-brief để chuẩn bị bản tóm tắt buổi sáng',
    'use a habilidade dawn-brief: prepare o briefing matinal',
]
# 警报骨架:官方 Critical Email Alert 的标志短语(各语言)
ALERT_MARKERS = [
    'fire only when the email is critical',
    'aktifkan hanya', 'メールが重要または非常に重要で、即時対応が必要な場合にのみ発火',
    'chỉ kích hoạt khi email', 'dispare apenas quando o e-mail',
    'emails sent directly to me', 'email yang dikirim langsung ke saya',
    '私宛に直接送信されたメール', 'email được gửi trực tiếp đến tôi',
    'e-mails enviados diretamente para mim',
]

# 其余官方模板。它们在 source 中仍走通用 record 世界，不额外设计 engineered
# 变体；但必须识别为模板家族，否则与 official eval 的共同产品骨架会被误判为
# custom 泄题并整桶删除。锚点只取语义唯一的产品/skill 名或模板短语。
OTHER_TEMPLATE_MARKERS = [
    r'unread-email-processor',
    r'meeting-materials-summary',
    r'work progress report', r'laporan\s+(?:\w+\s+)?(?:kemajuan|progres)\s+kerja',
    r'作業進捗レポート|業務進捗レポート', r'báo cáo tiến độ công việc',
    r'relatório.{0,12}progresso',
    r'search_person', r'attendees i may not know',
    r'meeting_prep', r'meetings with my manager',
    r'linkedin\.com/messaging', r'profile-optimizer', r'job-apply',
    r'subscription renewal', r'monitor my subscription',
    r'renovações das minhas assinaturas', r'perpanjangan langganan',
    r'サブスクリプションの更新',
]
OTHER_TEMPLATE_RE = [re.compile(p, re.I) for p in OTHER_TEMPLATE_MARKERS]

def classify_arm(rule):
    r = norm(rule)
    for h in EVENING_HEADS:
        if r.startswith(h[:40]):
            return 'template_evening'
    for h in DAWN_HEADS:
        if r.startswith(h[:40]):
            return 'template_dawn'
    hits = sum(1 for m in ALERT_MARKERS if m in r)
    if hits >= 1 and re.match(r'^(trigger|pemicu|トリガー|kích hoạt|disparador)', r):
        return 'alert'
    # dawn/evening 的用户改版(骨架短语在但开头被改):也算模板家族
    for h in EVENING_HEADS + DAWN_HEADS:
        if h[:30] in r:
            return 'template_variant'
    if any(p.search(r) for p in OTHER_TEMPLATE_RE):
        return 'template_variant'
    return 'custom'

# ---------- 语言推断(rule_lang 缺失时兜底) ----------

LANGDETECT_AVAILABLE = None          # None = 还没试过;True/False = 试过的结果


def _langdetect_or_warn():
    """Import langdetect once, and make its absence loud instead of silent.

    这个导入以前包在裸 ``except Exception: pass`` 里。装没装都不报错,而缺了它就会
    掉进下面的正则兜底 —— 那条路把葡语判成越南语(见 ``_fallback_lang``)。
    rq3w v7 的 ``language_audit.json`` 就是这么来的:640/2025 标签与题面不符。
    所以这里只吞一次 ImportError,并且必须喊出来。
    """
    global LANGDETECT_AVAILABLE
    try:
        from langdetect import DetectorFactory, detect_langs
    except Exception as exc:                       # pragma: no cover - 环境问题
        if LANGDETECT_AVAILABLE is None:
            print(f'⚠️ langdetect 不可用({exc}),语言判定降级到正则兜底。'
                  f'兜底对拉丁字母语种(pt/vi/es/fr/id)不可靠,'
                  f'请先 pip install -r requirements-pipeline.txt 再跑。',
                  file=sys.stderr)
        LANGDETECT_AVAILABLE = False
        return None
    LANGDETECT_AVAILABLE = True
    DetectorFactory.seed = 0
    return detect_langs


# 拉丁字母语种的兜底判据。⛔ 只放**该语种独有**的记号,别放跨语种通用词
# (踩过:'disparador'/'apenas' 葡语西语都有,按顺序判会把西语判成葡语)。
# 尤其注意 â/ê/ô —— 葡语也有,不能算越南语记号;越南语只认 ă/đ/ơ/ư。
_FALLBACK_MARKS = {
    'pt': (r'[ãõ]|\bçã|\b(você|não|então|meu|minha|também não|'
           r'e-mails?\s+enviados|quando|assunto|remetente)\b'),
    'es': (r'[¿¡ñ]|\b(sólo|solo si|correo electrónico|cuando|reunión|'
           r'enviados directamente a mí|también)\b'),
    'fr': r'\b(lorsque|courriel|réunion|uniquement|également|déclencheur|objet)\b',
    'id': r'\b(yang|dengan|untuk|saya|email masuk|setiap|gunakan|kepada)\b',
    'vi': r'[ăđơư]|\b(của|và|các|cho|khi|gửi|quan trọng)\b',
}


def _fallback_lang(rule):
    """没有 langdetect 时的正则兜底:按「独有记号命中数」打分,取最高的那个。

    用打分而不是 if 顺序,是因为顺序判断的第一条分支会吃掉所有与它共享词的语种
    (0814 复现:西语规则被判成葡语)。平手或全零时返回空,交给上层用 meta 兜底。
    """
    low = norm(rule)
    lowered = rule.lower()
    scores = {lang: len(re.findall(pattern, low)) + len(re.findall(pattern, lowered))
              for lang, pattern in _FALLBACK_MARKS.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return ''
    if sorted(scores.values())[-2] == scores[best]:      # 并列第一 = 判不了
        return ''
    return best


def infer_lang(rule, meta_lang):
    """Infer the language of the rule text, treating metadata as a fallback.

    Production traces contain a material number of stale ``rule_lang`` values
    (for example Portuguese rules labelled ``vi``).  That field used to win
    unconditionally and then drove world generation and rollout stratification.
    Script-specific checks plus deterministic ``langdetect`` now take priority;
    metadata is used only when text detection is unavailable or uncertain.
    """
    r = rule
    if re.search(r'[぀-ヿ]', r):
        return 'ja'
    if re.search(r'[一-鿿]', r):
        return 'zh'
    if re.search(r'[฀-๿]', r):
        return 'thai'
    detect_langs = _langdetect_or_warn()
    if detect_langs is not None:
        try:
            candidates = detect_langs(str(r))
        except Exception:
            candidates = []                        # 文本太短/全是符号,交给兜底
        if candidates and candidates[0].prob >= 0.80:
            detected = {
                'zh-cn': 'zh', 'zh-tw': 'zh', 'th': 'thai',
            }.get(candidates[0].lang, candidates[0].lang)
            if detected in {'en', 'id', 'ms', 'vi', 'pt', 'es', 'fr',
                            'ja', 'zh', 'thai'}:
                return detected
    guessed = _fallback_lang(r)
    if guessed:
        return guessed
    normalized_meta = {'th': 'thai', 'zh-cn': 'zh', 'zh-tw': 'zh'}.get(
        str(meta_lang or '').lower(), str(meta_lang or '').lower())
    return normalized_meta or 'en'

# ---------- tick 切分:把多天会话拆成 (tick文本, 工具调用, 工具返回, 助手信封) ----------

MSG_TIME_RE = re.compile(
    r'<msg_time>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2})')
SCHEDULED_RE = re.compile(r'- scheduled_at:\s*([^\n]+)$', re.M)
EMAIL_RE = re.compile(r'[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}', re.I)


def tick_metadata(text):
    """从同一个 tick 提取时间、offset 和产品展示的时区标签。"""
    out = {'observed_at': None, 'tz_offset': None,
           'scheduled_at_raw': None, 'tz_label': None}
    m = MSG_TIME_RE.search(text)
    if m:
        out['observed_at'] = m.group(1)
        try:
            compact = datetime.fromisoformat(m.group(1)).strftime('%z')
            out['tz_offset'] = compact[:3] + ':' + compact[3:]
        except ValueError:
            pass
    m = SCHEDULED_RE.search(text)
    if m:
        raw = m.group(1).strip()
        lm = re.search(r'\s([A-Z]{2,5})$', raw)
        out['scheduled_at_raw'] = raw
        out['tz_label'] = lm.group(1) if lm else None
    return out

def split_ticks(msgs):
    ticks = []
    cur = None
    for m in msgs:
        role, c = m['role'], str(m.get('content', ''))
        if role == 'user':
            if c.startswith('<skills'):
                continue
            if cur:
                ticks.append(cur)
            cur = {'tick': c, 'tool_calls': [], 'tool_responses': [], 'assistant': [],
                   **tick_metadata(c)}
        elif cur is not None:
            if role == 'tool_call':
                cur['tool_calls'].append(c)
            elif role == 'tool_response':
                cur['tool_responses'].append(c)
            elif role == 'assistant':
                cur['assistant'].append(c)
    if cur:
        ticks.append(cur)
    return ticks


def extract_rule(msgs):
    for m in msgs:
        if m['role'] != 'user':
            continue
        c = str(m.get('content', ''))
        if c.startswith('<skills'):
            continue
        # Event ticks put per-run evidence immediately after the persistent
        # rule.  It is not part of the user's original query; including it in
        # the rule duplicates answer-bearing evidence in every reconstructed
        # question and can pin source-run dates into future rollouts.
        mm = re.search(
            r'Original query at creation:\s*(.*?)'
            r'(?=\n## |\n- (?:scheduled_at|Match evidence):|\Z)',
            c, re.S | re.I)
        if mm:
            return mm.group(1).strip(), ('cron_fmt' if '[Cron Auto-Task Tick]' in c else 'event_fmt')
        return c[:2000], 'other'
    return '', 'none'


BOT_PAT = re.compile(r'testbeeai|onboarding_testmigoo', re.I)


def extract_accounts(msgs):
    """只从认证 user_emails 区块取账号；不从正文或收发件人推断用户。"""
    for m in msgs:
        if m.get('role') != 'user':
            continue
        c = str(m.get('content', ''))
        mm = re.search(r'<user_emails[^>]*>(.*?)</user_emails>', c, re.S | re.I)
        if not mm:
            continue
        block = mm.group(1)
        accounts = list(dict.fromkeys(x.lower() for x in EMAIL_RE.findall(block)))
        pm = re.search(r'Primary account:\s*\n\s*-\s*([^\s<]+)', block, re.I)
        primary = pm.group(1).lower() if pm and EMAIL_RE.fullmatch(pm.group(1)) else None
        return primary, accounts
    return None, []


def _scoped_key(namespace, value):
    if value is None or value == '':
        return None
    return hashlib.sha256(f'autotask-rq2-{namespace}:v1:{value}'.encode()).hexdigest()[:20]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_judge_meta(path):
    if not path or not Path(path).exists():
        return {}, None
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            d = json.loads(line)
            if d.get('trace_id'):
                out[d['trace_id']] = {
                    'verdict': d.get('_judge'),
                    'trigger_last': d.get('trigger_last'),
                }
    return out, sha256_file(path)


def extract_eval_rule(question):
    """从本地评测 item 的 cron/event 题面提取规则，兼容多语言字段名。

    字段标签可以翻译，但题面结构稳定：规则是 Task/Matched Rule 段第一条
    `- 标签: 内容`，cron 到 `## This Tick`，event 到下一条 `- ...:`。
    """
    text = str(question or '')
    section = re.search(
        r'##\s+(?:Task|Matched Rule|Tugas|Peraturan Dipadankan|Regra Correspondente|'
        r'Tarefa|タスク|一致したルール|Nhiệm vụ|Quy tắc đã khớp)[^\n]*\n'
        r'-\s*[^:\n]{1,120}:\s*(.*)', text, re.S | re.I)
    if not section:
        return ''
    body = section.group(1)
    ends = [m.start() for pat in (r'\n##\s+', r'\n-\s*[^:\n]{1,120}:\s*')
            for m in [re.search(pat, body, re.S)] if m]
    return body[:min(ends)].strip() if ends else body.strip()


def load_eval_rules(paths):
    """读取本地 JSON/JSONL 评测快照，返回 (dataset,item,rule,ngrams)。"""
    rows = []
    for raw_path in paths or []:
        path = Path(raw_path)
        if path.suffix == '.jsonl':
            values = (json.loads(line) for line in path.open() if line.strip())
        else:
            obj = json.load(path.open())
            # offline bank 是 {item_id: {meta: {question: ...}}}；普通快照是 list。
            values = (obj if isinstance(obj, list) else obj.values()
                      if isinstance(obj, dict) and obj and
                      all(isinstance(v, (dict, str)) for v in obj.values()) else [obj])
        for index, value in enumerate(values):
            if isinstance(value, str):
                question, item, dataset = value, str(index), path.stem
            elif isinstance(value, dict):
                inp = value.get('input') or {}
                meta = value.get('meta') or {}
                question = (inp.get('question') if isinstance(inp, dict) else None) or value.get('question')
                question = question or (meta.get('question') if isinstance(meta, dict) else None)
                item = str((inp.get('id') if isinstance(inp, dict) else None)
                           or value.get('id') or (meta.get('id') if isinstance(meta, dict) else None)
                           or index)
                dataset = str(value.get('datasetName') or path.stem)
            else:
                continue
            rule = extract_eval_rule(question)
            if rule:
                rows.append((dataset, item, rule, ngrams(rule)))
    return rows


def same_instant(a, b, tolerance_seconds=5):
    if not a or not b:
        return None
    try:
        return abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()) <= tolerance_seconds
    except ValueError:
        return None


def resolve_judge_lineage(record_meta, external_meta):
    """兼容 v2 外置 meta 与 v3 record.meta 内嵌的 target lineage。"""
    trigger_last = external_meta.get('trigger_last') or record_meta.get('trigger_last')
    verdict = (external_meta.get('verdict') if external_meta.get('verdict') is not None
               else record_meta.get('_judge'))
    source = ('external_meta' if external_meta.get('trigger_last')
              else 'record_meta' if record_meta.get('trigger_last') else None)
    return trigger_last, verdict, source

# ---------- 主流程 ----------

def main():
    global SRC, OUT_DIR, LEAK_HITS
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='lsy 训练集 jsonl.gz')
    ap.add_argument('--out', required=True, help='本批产出目录,如 out_v3')
    ap.add_argument('--leak', default=LEAK_HITS, help='撞评测题留痕表;缺文件则跳过(只影响 eval_sim 留痕)')
    ap.add_argument('--meta', default='',
                    help='与 clean 配套的 v2_meta.jsonl.gz;默认自动找同目录文件')
    ap.add_argument('--keep-eval-collisions', action='store_true',
                    help='仅调试用:保留 custom 且 8-gram Jaccard>=0.70 的评测撞题')
    ap.add_argument('--eval-items', action='append', default=[],
                    help='本地评测 item JSON/JSONL；可重复传，直接按规则计算撞题，不依赖旧 trace 清单')
    a = ap.parse_args()
    SRC, OUT_DIR, LEAK_HITS = a.src, Path(a.out), a.leak
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'源={SRC}\n出={OUT_DIR}', flush=True)

    meta_path = a.meta
    if not meta_path:
        candidate = Path(SRC).with_name('v2_meta.jsonl.gz')
        meta_path = str(candidate) if candidate.exists() else ''
    judge_by_trace, meta_sha256 = load_judge_meta(meta_path)
    src_sha256 = sha256_file(SRC)
    eval_rules = load_eval_rules(a.eval_items)
    if eval_rules:
        print(f'本地评测规则 {len(eval_rules)} 条，来自 {len(a.eval_items)} 个快照', flush=True)

    # 撞题表:rule 前缀 -> (J, dataset, item)。leak_hits 每行 [trace_id, cluster, dsn, iid, J, rule_prefix70]
    leak_by_trace = {}
    if Path(LEAK_HITS).exists():
        for h in json.load(open(LEAK_HITS)):
            leak_by_trace[h[0]] = {'J': h[4], 'dataset': h[2], 'item': h[3]}
    else:
        print('⚠️ 没有撞题留痕表,eval_sim 全空(0801 口径:撞题本就保留,只影响留痕)', flush=True)

    groups = defaultdict(lambda: {'records': [], 'arm': None, 'lang': None,
                                  'trigger': None, 'rule': None, 'eval_sim': None})
    n_total = n_bot = n_nonstd = 0
    n_target_aligned = n_target_mismatch = n_target_unknown = 0
    excluded = []

    with gzip.open(SRC, 'rt') as f:
        for source_line_no, line in enumerate(f, 1):
            n_total += 1
            d = json.loads(line)
            primary_account, accounts = extract_accounts(d['messages'])
            # 认证账号区块是权威来源。兼容少量旧格式时只退回 user-context，
            # 不会因为工具返回或邮件正文提到测试地址而误删整条生产记录。
            user_context = next((str(m.get('content', '')) for m in d['messages']
                                 if m.get('role') == 'user' and
                                 '<user-context' in str(m.get('content', ''))), '')
            if any(BOT_PAT.search(x) for x in accounts) or BOT_PAT.search(user_context):
                n_bot += 1
                continue
            meta = d['meta']
            rule, fmt = extract_rule(d['messages'])
            if meta.get('agent') != 'professional' or fmt not in ('cron_fmt', 'event_fmt') or not rule:
                n_nonstd += 1
                continue
            k = rule_key(rule)
            g = groups[k]
            if g['rule'] is None:
                g['rule'] = rule
                g['arm'] = classify_arm(rule)
                g['lang'] = infer_lang(rule, meta.get('rule_lang'))
                g['trigger'] = 'cron' if fmt == 'cron_fmt' else 'event'
                if eval_rules:
                    source_grams = ngrams(rule)
                    best = max(((jacc(source_grams, grams), dataset, item)
                                for dataset, item, _, grams in eval_rules), default=(0, '', ''))
                    if best[0] > 0:
                        g['eval_sim'] = {'J': best[0], 'dataset': best[1], 'item': best[2],
                                         'method': 'direct_rule_8gram'}
            leak = leak_by_trace.get(meta.get('trace_id'))
            if leak and (g['eval_sim'] is None or leak['J'] > g['eval_sim']['J']):
                g['eval_sim'] = leak
            ticks = split_ticks(d['messages'])
            if not ticks:
                # 新源可能混入只有 system/profile、没有实际触发的记录。它们不能提供
                # cutoff/时区/目标信封，也不能作为可运行世界的 source record。
                n_nonstd += 1
                if not g['records']:
                    groups.pop(k, None)
                continue
            target_tick_index = len(ticks) - 1
            target = ticks[target_tick_index]
            judge = judge_by_trace.get(meta.get('trace_id'), {})
            # v2 把 trigger_last 放在逐行对齐的外置 v2_meta；v3_1w 直接放在
            # record.meta。外置 meta 优先，缺失时使用内嵌 lineage，不能因为换源
            # 就把整批 judge target 降级成 unknown。
            judge_trigger_last, judge_verdict, judge_lineage_source = resolve_judge_lineage(
                meta, judge)
            aligned = same_instant(target.get('observed_at'), judge_trigger_last)
            alignment_mode = 'same_tick_msg_time' if aligned is True else None
            # 少量 safety/skill 轨迹把 <msg_time> 放在紧邻目标 trigger 的上一条
            # user context 中。判官 meta.trigger_last 与该 context 对齐，最终 envelope
            # 仍属于最后这条 cron/event trigger。
            if aligned is None and ('[Cron Auto-Task Tick]' in target.get('tick', '') or
                                    '[Event Triggered]' in target.get('tick', '')):
                prior_times = [t.get('observed_at') for t in ticks[:target_tick_index]
                               if t.get('observed_at')]
                if prior_times and same_instant(prior_times[-1], judge_trigger_last) is True:
                    aligned = True
                    alignment_mode = 'adjacent_context_msg_time'
            cutoff_at = target.get('observed_at') or judge_trigger_last
            tz_offset = target.get('tz_offset')
            if not tz_offset and cutoff_at:
                try:
                    compact = datetime.fromisoformat(cutoff_at).strftime('%z')
                    tz_offset = compact[:3] + ':' + compact[3:]
                except ValueError:
                    pass
            if aligned is True:
                n_target_aligned += 1
            elif aligned is False:
                n_target_mismatch += 1
            else:
                n_target_unknown += 1
            uid = meta.get('user_id')
            user_basis = str(uid) if uid is not None else primary_account
            g['records'].append({
                'trace_id': meta.get('trace_id'),
                'source_line_no': source_line_no,
                'user_id': uid,
                'user_key': _scoped_key('user', user_basis),
                'primary_account_key': _scoped_key('account', primary_account),
                'account_keys': [_scoped_key('account', x) for x in accounts],
                'n_authenticated_accounts': len(accounts),
                'tz_offset': tz_offset,
                'tz_label': target.get('tz_label'),
                'cutoff_at': cutoff_at,
                'target_tick_index': target_tick_index,
                'judge_verdict': judge_verdict,
                'judge_trigger_last': judge_trigger_last,
                'judge_lineage_source': judge_lineage_source,
                'judge_target_aligned': aligned,
                'judge_target_alignment_mode': alignment_mode,
                'rule_lang': meta.get('rule_lang'),
                'answer_lang': meta.get('answer_lang'),
                'n_tool_calls': meta.get('n_tool_calls'),
                'tools_used': meta.get('tools_used'),
                'source_quality': {k: meta.get(k) for k in
                                   ('v3_1w_src', 'v13_score', 'judge_path')
                                   if meta.get(k) is not None},
                'ticks': ticks,
            })

    # 默认硬隔离 custom 撞题；仅显式调试开关允许保留。
    kept, arm_stat = [], Counter()
    for k, g in groups.items():
        sim = g['eval_sim']['J'] if g['eval_sim'] else 0.0
        if g['arm'] == 'custom' and sim >= 0.70:
            action = 'KEPT by override' if a.keep_eval_collisions else 'EXCLUDED'
            excluded.append({'rule_key': k, 'reason': f'eval_collision J={sim} ({action})',
                            'vs': g['eval_sim'], 'rule': g['rule'][:200],
                            'n_records': len(g['records'])})
            if not a.keep_eval_collisions:
                continue
        arm_stat[(g['arm'], g['trigger'], g['lang'])] += 1
        kept.append((k, g))

    with open(OUT_DIR / 'source_pool.jsonl', 'w') as f:
        for k, g in kept:
            row = {'rule_key': k, **{x: g[x] for x in ('arm', 'trigger', 'lang', 'rule', 'eval_sim')},
                   'n_records': len(g['records']), 'records': g['records']}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(OUT_DIR / 'excluded_rules.jsonl', 'w') as f:
        for e in excluded:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    kept_records = [r for _, g in kept for r in g['records']]
    kept_alignment = Counter(
        'aligned' if r.get('judge_target_aligned') is True else
        'mismatch' if r.get('judge_target_aligned') is False else 'unknown'
        for r in kept_records)
    summary = {
        'schema_version': 2,
        'source_file': str(Path(SRC).resolve()), 'source_sha256': src_sha256,
        'judge_meta_file': str(Path(meta_path).resolve()) if meta_path else None,
        'judge_meta_sha256': meta_sha256,
        'total_records': n_total, 'bot_records': n_bot, 'nonstandard_records': n_nonstd,
        'standard_records_before_eval_isolation': n_total - n_bot - n_nonstd,
        'usable_records_kept': len(kept_records),
        'unique_users_kept': len({r.get('user_key') for _, g in kept for r in g['records']
                                  if r.get('user_key')}),
        'unique_rules_kept': len(kept), 'excluded_eval_collision': len(excluded),
        'eval_collisions_kept': bool(a.keep_eval_collisions),
        'judge_target_alignment_all_standard': {'aligned': n_target_aligned,
                                                'mismatch': n_target_mismatch,
                                                'unknown': n_target_unknown},
        'judge_target_alignment_kept': {k: kept_alignment.get(k, 0)
                                        for k in ('aligned', 'mismatch', 'unknown')},
        'arms': {f'{a}/{t}/{l}': c for (a, t, l), c in sorted(arm_stat.items())},
    }
    with open(OUT_DIR / 'arms_summary.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print('\nexcluded:')
    for e in excluded:
        print(f"  J={e['vs']['J']} vs {e['vs']['dataset']}/{e['vs']['item']}: {e['rule'][:80]!r}")


if __name__ == '__main__':
    main()
