#!/usr/bin/env python3
"""阶段2:为每个工厂题起草种子世界 + 预期输出(claude-opus-4-6 via compass)。

用法:
  python3 gen_worlds.py --smoke        # 每臂×语言抽 1 题,共 ~10 题,人工过目
  python3 gen_worlds.py                # 全量(断点续跑,已有 case 文件跳过)
  python3 gen_worlds.py --only rq1_alrt_id_0007   # 单题重跑(先删文件)

输出: out/cases_raw/<case_id>.json
"""
import argparse, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, '/home/work/migoo_ai_public/haiyang/training/experiments/hy3_295b_lora/data_v6')
import _api
from pipeline_common import canonical_json, sha256_file, normalize_realism, is_aligned_recipe, is_family_recipe, REALISM_CANON

D = Path(__file__).resolve().parent.parent
POOL = D / 'out' / 'source_pool.jsonl'
OUT = D / 'out' / 'cases_raw'   # 仅占位:真实路径一律由 --out 给
# ⛔ 别在这里 mkdir —— 模块级建目录会在**任何人 import 本模块时**
#    在工具链目录下凭空造出 tools/out/cases_raw(0818 搬家后实际发生过)。
#    main() 里拿到 --out 之后才建。
MODEL = 'claude-opus-5'   # 起草老师。⚠️ 0825 更正:默认值一直停在 claude-opus-4-6,
                         # 而 v3.1.9 各批实际都靠 --model 传的 claude-opus-5
                         # (证据:cases_raw_*_plan_manifest.json)。默认值和实跑不一致会让
                         # 「照默认重跑一遍」得到另一个起草模型 —— 已对齐。
random.seed(20260801)

# ---------------- 世界变体轴(模板臂 12 个/语言) ----------------
VARIANTS = [
    ('quiet', '安静日:窗口内没有任何够格的邮件(只有时间窗外/促销/FYI噪声),正确行为是如实说安静;日历有 1-2 个普通事项'),
    ('busy_mixed', '普通忙碌日:2 封够格重要邮件 + 5-6 封各类噪声(促销/newsletter/CC线程/会议邀请);明日日历 2 项'),
    ('overload', '爆炸日:4-5 封都算重要 + 大量噪声,考验单屏取舍与压缩;明日日历 3 项'),
    ('deadline_tomorrow', '明早硬截止:一封邮件或日历事项指向明早必须交付的东西,今晚就该准备;应当被点名'),
    ('vip_urgent', '关键人物急件:上级/大客户的紧急请求埋在噪声里,必须置顶'),
    ('conflict', '日历冲突:明早两个会重叠,应当指出;邮件面平常'),
    ('travel', '差旅变动:航班/酒店变更邮件 + 明日行程;需要提醒准备'),
    ('payment_security', '账务/安全类:一封支付失败或安全告警(第一方)夹在噪声中,该报'),
    ('already_handled', '已处理干扰:窗口内的"重要"邮件其实用户已回复过(线程里有用户回信),不该再报;另有 1 封真该报的'),
    ('borderline', '边界日:邮件都在灰区(措辞急但无真实风险/常规审批),多数应被过滤,最多 1 封勉强够格'),
    ('empty_calendar', '日历空白:明日无任何日程,邮件面普通(1 封够格);日历部分应如实说空'),
    ('foreign_noise', '跨语言噪声:收件箱混着英语和用户语言的邮件,重要的那封是英语的;测语言混排下的筛选'),
]

# ---------------- 0822 四题族(仅 --realism v3.1.9;病根与判据见 docs/rq3w/13、15) ----------------
# ⛔ 条款只随族下发进 user_prompt,不进 system —— 条款会抢产出预算(0819 实测,20号文)。
FAMILY_VARIANTS = {
    'draft': [
        ('draft_clean2', '收件箱含 2 封「线程内事实足以答复」的邮件(确认时间/确认收到/答应提供线程里点名的东西),线程里必须真的有那个事实;另配 3-4 封普通噪声。正确行为=给这 2 封各建一份 Gmail 草稿(不发送),重要邮件保持未读+分析'),
        ('draft_mixed3', '收件箱含 3 封可答复邮件 + 1 封「需要用户拍板」的邮件(报价二选一/日程冲突取舍),后者不建草稿只保持未读;考验两类的区分'),
        ('draft_borderline', '收件箱含 1 封勉强可答复(答复所需事实藏在线程更早的引用里,要挖)+ 2 封需要用户拍板的;正确行为=只给那 1 封建草稿'),
        ('draft_none', '压舱反例:收件箱全是需要用户拍板/纯 FYI 的邮件,没有任何一封该建草稿;正确行为=零草稿,该未读的未读、该标读的标读'),
    ],
    'progress_dedupe': [
        ('overlap_partial', '记忆/邮件里的工作进展与「上次已报告」部分重叠(世界里要埋上次报告的痕迹:一封已发出的进展邮件或对应记忆条目);正确行为=去重后照常报告一次,绝不因重叠而静默'),
        ('overlap_heavy', '进展几乎全部与上次报过的重叠,只有一小条是新的;正确行为=报那一小条,其余可一句带过「无变化」;不许整报告静默'),
        ('fresh_normal', '正常日:进展全是新的,常规报告(族内对照锚,防分布漂移)'),
    ],
    'checklist': [
        ('empty_category', '规则要求的清单里有一类当日为空(如无会议/无紧急待办);正确行为=明确说出「该类今天没有」,不许整类跳过不提'),
        ('exclusion_trap', '世界里埋 1-2 个规则明确排除的干扰项(FYI 公告/他人负责事项/窗口外条目),它们要造得看起来很值得报;正确行为=完全不提它们'),
        ('empty_plus_trap', '组合题:一类为空要明说 + 一个排除项要忍住;清单其余项照常覆盖'),
    ],
}
FAM_SHORT = {'draft': 'drf', 'progress_dedupe': 'pgd', 'checklist': 'chk'}
# 明示条款 = 采样密度杠杆(39%→67%,探针2实测),⛔ 渲染前必须换回含糊版(swap_rule_text.py 两道闸)
FAMILY_EXPLICIT_CLAUSE = {
    'en': "\n\nHard rule for this run: treat the rule's checklist as mandatory. Cover EVERY required item; if a category has nothing to report, say so explicitly; include ONLY what the rule asks for and exclude everything else; if any required item exists, never suppress the output — deduplicate repeated progress instead of skipping the report.",
    'zh': "\n\n本次执行的硬性要求:把规则里的清单当作强制项。要求覆盖的每一项都必须覆盖;某一类没有内容时要明确说明「无」;只输出规则要求的内容,规则以外的一律不写;只要存在任何一项要求的内容,就不允许不输出——重复的进展做去重后照常报告,不要因重复而跳过。",
    'ja': "\n\n今回の実行における必須ルール:ルールのチェックリストを必須項目として扱ってください。要求されているすべての項目を必ず網羅すること。あるカテゴリに報告すべき内容がない場合は、その旨を明示的に述べること。ルールが求める内容のみを含め、それ以外は一切含めないこと。要求された項目が一つでも存在する限り、出力を省略してはいけません——重複する進捗はスキップせず、重複を除いた上で報告してください。",
    'pt': "\n\nRegra obrigatória desta execução: trate a lista de itens da regra como obrigatória. Cubra TODOS os itens exigidos; se uma categoria não tiver nada a reportar, diga isso explicitamente; inclua APENAS o que a regra pede e exclua todo o resto; se existir qualquer item exigido, nunca suprima o relatório — deduplique o progresso repetido em vez de omitir o relatório.",
    'id': "\n\nAturan wajib untuk eksekusi ini: perlakukan daftar dalam aturan sebagai wajib. Cakup SEMUA item yang diminta; jika suatu kategori tidak ada isinya, nyatakan secara eksplisit; sertakan HANYA yang diminta aturan dan kecualikan semua yang lain; selama ada item yang diminta, jangan pernah menahan laporan — deduplikasi kemajuan yang berulang alih-alih melewatkan laporan.",
    'vi': "\n\nQuy tắc bắt buộc cho lần chạy này: coi danh sách trong quy tắc là bắt buộc. Bao phủ TẤT CẢ các mục được yêu cầu; nếu một hạng mục không có gì để báo cáo, hãy nói rõ điều đó; chỉ đưa vào những gì quy tắc yêu cầu và loại trừ mọi thứ khác; miễn là còn bất kỳ mục được yêu cầu nào, tuyệt đối không được im lặng — hãy khử trùng lặp tiến độ thay vì bỏ qua báo cáo.",
}
# 族题语种钉死:0819 补题批实证「世界语种≠题面语种」是最大单项失败(24%),
# 而 draft 族的源记录多为印尼语背景、规则却是英文 —— 材料会把世界带跑(试造 3/13 实测)。
FAMILY_LANG_PIN = ('\n⚠️ 世界语言必须与规则语言(上面标的“用户语言”)完全一致:邮件主题/正文、'
                   '日历、人名与写作风格全部用该语言;真实材料语言不同时**只借结构与难度,不借语言**。')

FAMILY_EXPLICIT_DRAFT = {
    'en': "\n\nHard rule for this run: draft a reply for EVERY email that can be answered from the retrieved thread facts (confirming a time, acknowledging a request, agreeing to provide something the thread names) by creating a Gmail draft; do NOT downgrade a replyable email to a suggestion-only note. Emails that still need the user's own decision keep following the skill (no draft).",
}

# rq2e-s58 三轮稳定失败归因（2026-08-11）显示，剩余错误不是“不会交付”，
# 而是时间窗、排除项、动作落地、身份/责任边界与静默语义。下面只用于给世界起草
# 追加通用难度要求，不复制任何评测题文本或实体。
RULE_GUIDANCE_PATTERNS = {
    'time_window': re.compile(
        r'last week|this week|previous day|yesterday|past 24|within|before|after|overdue|'
        r'renew|deadline|上周|本周|昨天|过去|窗口|截止|续费|minggu|semalam|sebelum|'
        r'selepas|期限|昨日', re.I),
    'email_action': re.compile(
        r'draft|reply|respond|mark.{0,8}read|unread|trash|email|mail|草稿|回复|已读|'
        r'未读|邮件|删除|balas|draf|e-mail|e-mel|trả lời|thư', re.I),
    'ownership': re.compile(
        r'\bfyi\b|my work|i own|responsib|owner|我负责|本人负责|我的工作|他人|'
        r'tanggungjawab|milik saya|phụ trách', re.I),
    'people': re.compile(
        r'attendee|contact|familiar|unfamiliar|participant|参会|联系人|熟人|陌生|'
        r'kenal|peserta|liên hệ|người tham dự', re.I),
    'report_completeness': re.compile(
        r'brief|recap|report|summary|digest|weekly|daily|morning|evening|ticker|'
        r'简报|簡報|报告|報告|总结|總結|汇总|彙總|日报|日報|周报|週報|早报|晚报|'
        r'ringkas|rangkuman|laporan|resumo|relat[oó]rio|t[oó]m tắt', re.I),
}

# 空结果必须先读规则，不能按 rule 序号机械造 quiet。尤其要避免把“无命中则静默”
# 训成“无命中也发一条没事通知”。模式写得保守：只有明确空条件 + 明确终态才分类。
# Keep the alternatives grouped.  This fragment is concatenated with the
# terminal-action fragment below; without the outer group, regex alternation
# precedence makes a bare ``if there are no`` a complete match and labels any
# such rule as both silent and report-empty.
_EMPTY = r'(?:(?:if|when|unless)\s+(?:there (?:is|are) )?(?:no|none|nothing)|if no|when no)'
SILENT_EMPTY_PATTERNS = [
    re.compile(_EMPTY + r'.{0,180}(?:stay|remain|keep).{0,30}(?:silent|quiet)', re.I | re.S),
    re.compile(_EMPTY + r'.{0,180}(?:do not|don\'t|never).{0,30}(?:send|notify|output|message)', re.I | re.S),
    re.compile(r'(?:only notify|only send (?:a )?(?:message|notification)) if', re.I),
    re.compile(r'(?:stay|remain) completely silent|no notification|without notifying', re.I),
    re.compile(r'(?:如果|若|如)(?:没有|无|未发现).{0,100}(?:不发送|不要发送|无需通知|不需要通知|保持静默|不推送|不输出|跳过)', re.S),
    re.compile(r'(?:没有|无).{0,80}(?:则|时).{0,60}(?:不发送|无需通知|保持静默|不推送|不输出)', re.S),
    re.compile(r'(?:jika|kalau) (?:tidak ada|tiada).{0,160}(?:tidak perlu|jangan).{0,40}(?:kirim|hantar|notifikasi|pesan)', re.I | re.S),
    re.compile(r'(?:jika|kalau) (?:tidak ada|tiada).{0,160}(?:diam|senyap)', re.I | re.S),
    re.compile(r'(?:se não houver|caso não haja|si no hay).{0,160}(?:não envie|não enviar|não notifique|silêncio|no envíe|no enviar|silencio)', re.I | re.S),
    re.compile(r'(?:không có|nếu không).{0,160}(?:không gửi|không thông báo|im lặng)', re.I | re.S),
    re.compile(r'(?:ない場合|なければ).{0,100}(?:通知しない|何も送らない|出力しない|スキップ|静かに(?:する|しておく|して))', re.S),
]
REPORT_EMPTY_PATTERNS = [
    re.compile(_EMPTY + r'.{0,180}(?:say|state|report|inform|tell|indicate).{0,80}(?:no|none|nothing|free|quiet)', re.I | re.S),
    re.compile(r'(?:if|when) there (?:are|is) no .{0,120}(?:say|state|report|inform|tell)', re.I | re.S),
    re.compile(r'(?:如果|若|如)(?:没有|无|未发现).{0,100}(?:说明|告知|回复|报告|发送|输出|说)', re.S),
    re.compile(r'(?:jika|kalau) (?:tidak ada|tiada).{0,160}(?:sampaikan|informasikan|laporkan|katakan)', re.I | re.S),
    re.compile(r'(?:se não houver|caso não haja|si no hay).{0,160}(?:informe|informar|diga|indique|avise)', re.I | re.S),
    re.compile(r'(?:nếu không có|nếu không).{0,160}(?:hãy nói|chỉ cần nói|thông báo|cho biết|báo rằng)', re.I | re.S),
    re.compile(r'(?:ない場合|なければ).{0,100}(?:知らせ|伝え|述べ|報告|空いて|ありません)', re.S),
    re.compile(r'(?:say|state|report|reply|output).{0,40}(?:no changes?|nothing new|agenda (?:is )?free)', re.I),
    re.compile(r'只(?:回|回复|輸出|输出|說|说)(?:一句)?[「“\" ]{0,3}(?:今日無|无变化|沒有新)', re.I),
]

NEGATED_REPORT = re.compile(
    r"do not|don't|never|no need to|"
    r"不(?:发送|發送|通知|输出|輸出|报告|報告|告知|回复|回覆)|无需|無需|不要|"
    r"tidak perlu|jangan|não (?:envie|enviar|informe|informar)|no (?:envíe|enviar)",
    re.I,
)


def empty_behavior(rule):
    """Return skip (true silence), quiet (explicit no-result report), or None."""
    text = re.sub(r'\s+', ' ', rule or '')
    # An explicit positive no-result report wins over a nearby suppression
    # qualifier.  A reporting verb inside a negative clause ("do NOT send a
    # message saying no changes") is not positive evidence.
    # Example: "say there is no Hypercare; no need to send a long briefing" is
    # a short quiet report, not complete silence.
    report_matches = [match for pattern in REPORT_EMPTY_PATTERNS
                      if (match := pattern.search(text))]
    if any(not NEGATED_REPORT.search(match.group(0)) for match in report_matches):
        return 'quiet'
    if any(pattern.search(text) for pattern in SILENT_EMPTY_PATTERNS):
        return 'skip'
    return None


def eval_guidance(rule):
    guidance = []
    if RULE_GUIDANCE_PATTERNS['time_window'].search(rule or ''):
        guidance.append('时间窗两侧都放近边界干扰项；expected 必须逐项写清窗口内纳入、窗口外排除。')
    if RULE_GUIDANCE_PATTERNS['email_action'].search(rule or ''):
        guidance.append('需要 draft/已读/删除/发送等动作时，正确终态必须是真实工具动作，不得用“建议文案/我会处理”替代。')
    if RULE_GUIDANCE_PATTERNS['ownership'].search(rule or ''):
        guidance.append('同时放本人负责与他人/FYI 项，expected 明列纳入与排除清单。')
    if RULE_GUIDANCE_PATTERNS['people'].search(rule or ''):
        guidance.append('联系人/参会人要同时覆盖熟悉、陌生或不确定边界，并让 contacts/历史互动提供可核验证据。')
    if RULE_GUIDANCE_PATTERNS['report_completeness'].search(rule or ''):
        guidance.append('把规则要求的每个报告区块/字段都放入可验证材料，并加入一个容易漏掉但仍未解决的项目；expected 按字段列完整清单。')
    guidance.append('不得补写 seed/工具证据之外的业务承诺、数字、日期、关系或完成状态。')
    return '\n'.join(f'- {item}' for item in guidance)

TZ_MAP = {'+07:00': 'Asia/Jakarta', '+08:00': 'Asia/Singapore', '+09:00': 'Asia/Tokyo',
          # 只有 offset、没有可靠地区时必须用固定 offset zone。把 -07/-08 猜成
          # Los Angeles 会在 DST 切换后改变 offset；v3 还新增了 -04/-05/-06。
          '-03:00': 'Etc/GMT+3', '-04:00': 'Etc/GMT+4', '-05:00': 'Etc/GMT+5',
          '-06:00': 'Etc/GMT+6', '-07:00': 'Etc/GMT+7', '-08:00': 'Etc/GMT+8',
          '+03:00': 'Etc/GMT-3', '+00:00': 'UTC',
          # 0815 补:整点档补齐。原表只有上面那几个,别的 offset 全走兜底
          # `or 'UTC'` —— 于是造出「offset=+03:30 而 tz_iana=UTC」这种
          # **自己跟自己打架的世界**(实测 145 道里中了 6 道,lint E8 才抓到)。
          '-11:00': 'Etc/GMT+11', '-10:00': 'Etc/GMT+10', '-09:00': 'Etc/GMT+9',
          '-02:00': 'Etc/GMT+2', '-01:00': 'Etc/GMT+1',
          '+01:00': 'Etc/GMT-1', '+02:00': 'Etc/GMT-2', '+04:00': 'Etc/GMT-4',
          '+05:00': 'Etc/GMT-5', '+06:00': 'Etc/GMT-6', '+10:00': 'Etc/GMT-10',
          '+11:00': 'Etc/GMT-11', '+12:00': 'Etc/GMT-12',
          # 半小时/三刻钟档 Etc/GMT±N 表达不了,只能挑**全年不跳 DST** 的真实地区:
          # 伊朗 2022 年起废除夏令时、印度/尼泊尔/阿富汗/缅甸从来没有、
          # 澳洲北领地(Darwin)不跟南澳走夏令时。都已用 zoneinfo 双日期验过。
          '+03:30': 'Asia/Tehran', '+04:30': 'Asia/Kabul', '+05:30': 'Asia/Kolkata',
          '+05:45': 'Asia/Kathmandu', '+06:30': 'Asia/Yangon',
          '+09:30': 'Australia/Darwin'}
TZ_LABEL_MAP = {'WIB': 'Asia/Jakarta', 'WITA': 'Asia/Makassar', 'WIT': 'Asia/Jayapura',
                'JST': 'Asia/Tokyo', 'PST': 'Asia/Manila', 'CST': 'Asia/Singapore'}
TZ_LABEL_OFFSET = {'WIB': '+07:00', 'WITA': '+08:00', 'WIT': '+09:00',
                   'JST': '+09:00', 'PST': '+08:00', 'CST': '+08:00'}

ACCOUNTS = [f'migoo_testbeeai_{i}@shopee.com' for i in range(48, 79)]

# ---------------- 源材料摘要 ----------------

def target_tick(rec):
    """取 extractor 已与 v2 judge trigger_last 对齐的最终 cron/event trigger。"""
    ticks = rec.get('ticks') or []
    if not ticks:
        return None, None
    index = rec.get('target_tick_index', len(ticks) - 1)
    if not isinstance(index, int) or index < 0 or index >= len(ticks):
        index = len(ticks) - 1
    return index, ticks[index]


def tz_for_record(rec):
    _, tick = target_tick(rec)
    off = (tick or {}).get('tz_offset') or rec.get('tz_offset') or '+08:00'
    label = (tick or {}).get('tz_label') or rec.get('tz_label')
    # CST 本身有歧义：+08 是中国/新加坡，-06 是北美中部。只有 label 与已知
    # offset 一致时才用 label map，否则以同 tick 的数值 offset 为权威。
    by_label = TZ_LABEL_MAP.get(label) if TZ_LABEL_OFFSET.get(label) == off else None
    iana = by_label or TZ_MAP.get(off)
    if not iana:
        # ⛔ 这里**不许兜底**。原来写的是 `or 'UTC'`,查不到就悄悄安一个 UTC,
        # 造出来的世界 offset 和时区名对不上 —— 邮件头一个时间、模型的钟另一个时间,
        # 而且只有 lint E8 事后能看出来。宁可当场炸,让人去补 TZ_MAP。
        raise ValueError(
            f'tz_offset {off!r} 不在 TZ_MAP 里(label={label!r})。'
            f'请往 TZ_MAP 补一个**全年不跳 DST** 的 zone,别用 Europe/* 或 America/* 这种有夏令时的。')
    return off, iana, label or f'UTC{off}'


def stable_record_index(case_id, g, ordinal=0):
    """同一规则内按用户/时区/工具画像稳定轮转，禁止 first-N 和跨用户拼接。"""
    records = g.get('records') or []
    if not records:
        raise ValueError(f"rule {g.get('rule_key')} has no records")
    order = sorted(range(len(records)), key=lambda i: hashlib.sha256(canonical_json({
        'user': records[i].get('user_key') or records[i].get('user_id'),
        'tz': records[i].get('tz_offset'),
        'tools': sorted(records[i].get('tools_used') or []),
        'trace': records[i].get('trace_id'),
    }).encode()).hexdigest())
    start = int.from_bytes(hashlib.sha256(
        str(g.get('rule_key') or case_id).encode()).digest()[:8], 'big') % len(order)
    return order[(start + ordinal) % len(order)]


def material_digest(g, rec_idx, cap=5000):
    """只从一个确定用户的目标轨迹及其历史取参照，避免把不同用户拼成一个世界。"""
    parts = []
    r = g['records'][rec_idx]
    target_index, _ = target_tick(r)
    start = max(0, target_index - 1)
    for t in r['ticks'][start:target_index + 1]:
        for tr in t['tool_responses'][:4]:
            parts.append('[真实工具返回节选] ' + re.sub(r'\s+', ' ', str(tr))[:600])
        if t['assistant']:
            parts.append('[当时的线上回答节选] ' + re.sub(r'\s+', ' ', t['assistant'][-1])[:500])
    s = '\n'.join(parts)
    return s[:cap] if s else '(无轨迹材料)'


def load_mailboxes(path):
    if not path:
        return {}
    return {row['user_key']: row for row in
            (json.loads(line) for line in open(path))}


def mailbox_snapshot_material(mailbox, cutoff_at, source_trace, cap_items=5, cap=4000):
    """物化 cutoff 前的同用户观测；未来版本永远不会进入 prompt。"""
    if not mailbox or not cutoff_at:
        return ''
    try:
        cutoff_dt = datetime.fromisoformat(cutoff_at)
    except (TypeError, ValueError):
        return ''
    chosen = []
    for item in mailbox.get('observations') or []:
        versions = []
        for version in item.get('versions') or []:
            try:
                observed_dt = datetime.fromisoformat(version.get('observed_at'))
            except (TypeError, ValueError):
                continue
            if observed_dt <= cutoff_dt:
                versions.append((observed_dt, version))
        if not versions:
            continue
        observed_dt, version = max(versions, key=lambda x: x[0])
        if version.get('source_trace') == source_trace:
            continue
        fields = version.get('fields') or {}
        if not fields:
            continue
        chosen.append((observed_dt, version['observed_at'], item, fields))
    chosen.sort(key=lambda x: (x[0], x[2]['stable_id']), reverse=True)
    parts = []
    for _, observed_at, item, fields in chosen[:cap_items]:
        safe = {k: fields.get(k) for k in
                ('subject', 'from', 'to', 'cc', 'sent_at', 'labels', 'unread', 'important', 'body')
                if fields.get(k) not in (None, '', [])}
        if 'body' in safe:
            safe['body'] = str(safe['body'])[:500]
        parts.append(f"[同用户历史邮箱观测 {observed_at} {item['stable_id']}] " +
                     json.dumps(safe, ensure_ascii=False))
    return '\n'.join(parts)[:cap]


_RESPONSE_CN = {'accepted': '已接受', 'declined': '已拒绝',
                'tentative': '待定', 'needsAction': '未回复'}


def calendar_snapshot_material(mailbox, cutoff_at, cap_items=8, cap=1200):
    """把同用户被真实观测到的日程压成**只有形态、没有日期**的几行参照。

    ⛔ 一律不给绝对日期,也不给"距今几天" —— 硬约束 2 要求世界里的时间全部相对,
    而 3.7 又要求日程一律往后排;把真实日期或负偏移放进提示词,模型会照抄。
    这里只留"几点开始 / 多长 / 几个人 / 我回没回 / 是不是周期实例 / 有没有地点说明"。

    ⚠️ 和邮箱那份不同,**本次执行自己那条 trace 的日程也收**:它就是我们要重建的
    那一天的真实日历,而且原始工具返回在源数据里被截到 2000 字符、模型本来就只能
    看到半截;这里给的是同一份数据的干净版本,不引入新信息。
    """
    if not mailbox or not cutoff_at:
        return ''
    try:
        cutoff_dt = datetime.fromisoformat(cutoff_at)
    except (TypeError, ValueError):
        return ''
    chosen = []
    for item in mailbox.get('calendar_observations') or []:
        versions = []
        for version in item.get('versions') or []:
            try:
                observed_dt = datetime.fromisoformat(version.get('observed_at'))
            except (TypeError, ValueError):
                continue
            if observed_dt <= cutoff_dt:
                versions.append((observed_dt, version))
        if not versions:
            continue
        observed_dt, version = max(versions, key=lambda x: x[0])
        fields = version.get('fields') or {}
        if fields:
            chosen.append((observed_dt, item['stable_id'], fields))
    if not chosen:
        return ''
    chosen.sort(key=lambda x: (x[0], x[1]), reverse=True)
    lines = []
    for _, _, f in chosen[:cap_items]:
        bits = []
        if f.get('all_day'):
            bits.append('全天')
        else:
            start_at = f.get('start_at')
            clock = ''
            if start_at:
                try:
                    clock = datetime.fromisoformat(start_at).strftime('%H:%M') + ' 起'
                except ValueError:
                    clock = ''
            duration = f.get('duration_min')
            bits.append(' '.join(x for x in (clock, f'{duration} 分钟' if duration else '') if x)
                        or '定时')
        n = f.get('n_attendees') or 0
        if n:
            who = _RESPONSE_CN.get(f.get('self_response'), f.get('self_response') or '未知')
            bits.append(f'{n} 人会(我:{who})')
        else:
            bits.append('无参会者')
        if f.get('recurring'):
            bits.append('周期会议的一次')
        if f.get('has_location'):
            bits.append('有地点')
        if f.get('has_description'):
            # ⛔ 只给派生特征,不给说明正文(正文里有真文档直链和真人名字)。
            detail = [f'{f["desc_len"]} 字'] if f.get('desc_len') else []
            if f.get('desc_has_html'):
                detail.append('带 HTML')
            if f.get('desc_has_link'):
                detail.append('含链接')
            bits.append('有说明(' + '、'.join(detail) + ')' if detail else '有说明')
        title = str(f.get('summary') or '(无标题)')[:60]
        lines.append(f'- 「{title}」 ' + ' · '.join(bits))
    return '\n'.join(lines)[:cap]


def drive_snapshot_material(mailbox, cutoff_at, cap_items=4, cap=900):
    """把同用户被真实观测到的网盘/表格压成几行**只有形态、没有内容**的参照。

    ⛔ 不给表格里的数据行,只给"多大、几列、列名叫什么、什么类型" ——
    真实表动辄上千行、几万个单元格,数据本身既进不了提示词预算,也不该照抄。
    """
    if not mailbox or not cutoff_at:
        return ''
    try:
        cutoff_dt = datetime.fromisoformat(cutoff_at)
    except (TypeError, ValueError):
        return ''
    chosen = []
    for item in mailbox.get('drive_observations') or []:
        versions = []
        for version in item.get('versions') or []:
            try:
                observed_dt = datetime.fromisoformat(version.get('observed_at'))
            except (TypeError, ValueError):
                continue
            if observed_dt <= cutoff_dt:
                versions.append((observed_dt, version))
        if not versions:
            continue
        observed_dt, version = max(versions, key=lambda x: x[0])
        fields = version.get('fields') or {}
        if fields:
            chosen.append((observed_dt, item['stable_id'], fields))
    if not chosen:
        return ''
    chosen.sort(key=lambda x: (x[0], x[1]), reverse=True)
    lines = []
    for _, _, f in chosen[:cap_items]:
        name = str(f.get('name') or '(无名)')[:80]
        if f.get('n_rows'):
            bits = [f'约 {f["n_rows"]} 行 × {f.get("n_cols") or "?"} 列']
            if f.get('n_tabs', 1) > 1:
                bits.append(f'{f["n_tabs"]} 个工作表')
            lines.append(f'- 表格「{name}」 ' + ' · '.join(bits))
        else:
            mime = str(f.get('mime') or '').rsplit('.', 1)[-1]
            lines.append(f'- 文件「{name}」' + (f' [{mime}]' if mime else ''))
    return '\n'.join(lines)[:cap]


# ---------------- 题面渲染 ----------------

CRON_TICK = """<system-reminder>[Cron Auto-Task Tick]

The user previously created a cron auto-task, and its schedule has just fired -- please produce this run's reminder/result according to the rule's intent.

## Task
- Original query at creation: {rule}

## This Tick
- scheduled_at: {{{{ fmt_date(run_date, 'iso') }}}} {{{{ fmt_time(run_date, 'hms') }}}} {tzlabel}
- now: {{{{ fmt_date(run_date, 'iso') }}}} {{{{ fmt_time(run_date, 'hms') }}}} {tzlabel}

Execute the task per the "Original query at creation."

The user must not perceive that this was generated ahead of time. Write the content as if it is being delivered at scheduled_at. Do not mention the gap between now and scheduled_at.</system-reminder>"""

EVENT_TICK = """<system-reminder>[Event Triggered]

The user previously created an important_auto_task. An external trigger source has now matched that rule -- please act on this event according to the rule's intent.

## Matched Rule
- Original query at creation: {rule}
- Match evidence: {match_evidence}

## Trigger Source Event (source_type = mail)
New Gmail message: Related People: {related_people}

Title: {title}

Sender: {sender}

Receivers: {receivers}

CC: {cc}

Send Time: {send_time}

Summary: {summary}

Topics: {topics}

Based on the "Original query at creation," decide whether to surface anything to the user and, if so, what to say.</system-reminder>"""


# 线上 event 类触发里 20.4% 是日历触发(0818 实测 82/401),原文形状和邮件完全不同。
# 下面这套字段是照线上原文一比一抄的,别改动标签文字。
CAL_EVENT_TICK = """<system-reminder>[Event Triggered]

The user previously created an important_auto_task. An external trigger source has now matched that rule -- please act on this event according to the rule's intent.

## Matched Rule
- Original query at creation: {rule}
- Match evidence: {match_evidence}

## Trigger Source Event (source_type = calendar)
[CALENDAR] {title}

Description: {description}
Start: {start}
End: {end}
Organizer: {organizer}
Status: {status}
Is All Day: {is_all_day}

Based on the "Original query at creation," decide whether to surface anything to the user and, if so, what to say.</system-reminder>"""


def render_question(case, llm):
    rule = case['rule']
    if case['trigger'] == 'cron':
        return CRON_TICK.format(rule=rule, tzlabel=case['tz_label'])
    ev = llm['trigger_event']
    if (ev or {}).get('source_type') == 'calendar':
        return CAL_EVENT_TICK.format(
            rule=rule, match_evidence=ev.get('match_evidence', ''),
            title=ev.get('title', ''), description=ev.get('description', ''),
            start=ev.get('start_jinja', "{{ fmt_datetime(add_hours(run_date, 2), 'iso') }}"),
            end=ev.get('end_jinja', "{{ fmt_datetime(add_hours(run_date, 3), 'iso') }}"),
            organizer=ev.get('organizer', ''),
            status=ev.get('status', 'confirmed'),
            is_all_day=str(bool(ev.get('is_all_day'))).lower())
    return EVENT_TICK.format(
        rule=rule, match_evidence=ev.get('match_evidence', ''),
        related_people=ev.get('related_people', ''), title=ev.get('title', ''),
        sender=ev.get('sender', ''), receivers=ev.get('receivers', 'self'),
        cc=ev.get('cc', ''), send_time=ev.get('send_time_jinja', "{{ fmt_datetime(add_hours(run_date, -1), 'iso') }}"),
        summary=ev.get('summary', ''), topics=ev.get('topics', ''))

# ---------------- 生成 prompt ----------------

# ── 可选提示词片段:日历参会状态(v3.1.8 起,--rsvp 开启)────────────────────
# ⚠️ **默认关闭**。gen_worlds.py 是 rq1/rq2/rq3w 三条线共用的,
#    无条件改提示词等于**悄悄改掉别人批次的配方** —— 而「造世界的配方必须和上一批
#    对齐」是踩过坑的铁律(漏传 --mailboxes 那次造出 89% 没日历的退化世界)。
#    所以做成显式开关,并把开关值写进 plan manifest,事后能查这批是哪个配方。
# 配方档位。正名 v3.1.7(老配方)/ v3.1.8(对齐线上)。判「是不是对齐线上那一档」
# 一律走 is_aligned_recipe(),别写 == 'v3.1.8' —— 老 manifest 里记的是 'v8'/''。
REALISM = 'v3.1.7'
RSVP_ENABLED = False    # 兼容旧名:--rsvp 等价于 --realism v3.1.8;别在别处改它


def system_prompt():
    """SYSTEM 提示词。--realism v3.1.8 时追加「对齐线上」的那几条要求。

    ⚠️ 每一条都对应一个 0818 实测出来的差距(线上真实 trace vs 我们 v3.1.7 的种子),
    数字写在条文里了 —— 以后要改配方,先重跑那次测量,别凭感觉调。
    """
    if not is_aligned_recipe(REALISM):
        return SYSTEM
    # ⛔ 四处改写全部先断言锚点存在。`str.replace` 找不到就**原样返回**,
    #    没有任何报错 —— 前两处一旦失效,整段 v3.1.8 条款 / 字段清单会**静默不插入**,
    #    造出来的还是 v3.1.7 的世界,而 manifest 里却记着 realism=v3.1.8。那种批次是查不出来的。
    for anchor in ('4. 极性可判:', 'cron 题不需要 trigger_event'):
        assert anchor in SYSTEM, f'SYSTEM 里找不到锚点 {anchor!r},v3.1.8 条款插不进去'
    out = SYSTEM.replace('4. 极性可判:', V318_CLAUSES + '4. 极性可判:', 1)
    out = out.replace('cron 题不需要 trigger_event', V318_SCHEMA + 'cron 题不需要 trigger_event', 1)
    # ⛔ 数量必须**在原地改**,不能靠后面的条款去"盖掉"。0818 实测:附加条款里明写了
    #    「本条盖掉硬约束 5 的 3-8 封」,模型照样锚在前面那个数上,造出来中位仍是 7 封。
    #    模型对**同一句话里的数字**远比对"后面某条说不算"敏感。
    # ⛔ **0819 账本实测:这些数字必须写成「产出真实是多少」,不是「理想是多少」。**
    #   原来主提示词要 A 档 4~6 + B 档 5~8 = 10~14 封,模型照做就必然超 24000 输出上限:
    #     main 调用 86 次,**平均输出 23,962 tok = 顶满 99.8%** → JSON 截断
    #     → 88% 的 case 落进 repair(那句"压缩数量"的重试)才写得完
    #   而 repair 产出的世界(实测 288 个)中位就是 **8 封邮件 / A 档 3 封 / 日历 4 条 /
    #   world_story 220 字** —— **我们一直在用的就是压缩版**,10~14 封那个数从没落地过。
    #   把真实数字原地写进来:形态一点不变,省掉那次 24k 的废调用。
    #   账:$1.185/世界 → 约 $0.42,1.9 次调用 → 1.1 次。全量 2025 省约 $1,560、少约 4 小时。
    assert '3-8 封' in out, 'SYSTEM 第 5 条的封数说法变了,v3.1.8 的改写锚点失效'
    out = out.replace(
        '3-8 封',
        '**A 档 3 封 + B 档背景噪声 4~5 封,整个世界共 7~8 封**,分档要求见下面 3.6',
        1)
    # 同一句里还有个"长度 40-200 词"(≈250~1300 字符),和 A 档的 2000~4000 字符打架。
    # 同样必须原地改 —— 留着它模型就会把 A 档写短。
    assert '长度 40-200 词' in out, 'SYSTEM 第 5 条的正文长度说法变了,v3.1.8 的改写锚点失效'
    out = out.replace('长度 40-200 词',
                      '长度按 3.6 的两档来:A 档 body_html 2000~4000 字符、B 档 500~1000', 1)
    # ── gold 粒度(0815 实测 +27.8 分的那一刀,0819 才合进主线)──────────
    # 先塞条款 12/13,再把 JSON 模板里的 expected_output 那一行**原地换掉** ——
    # 只加条款不换模板没用:模板里的 `["硬性判分点,1-4 条"]` 是个数组、还给了范围,
    # 模型会顶格取上界(0814 实测 99.3% 的世界给满 4 个)。
    assert '11. 严格区分静默:' in out, 'SYSTEM 硬约束 11 的措辞变了,v3.1.8 gold 条款插不进去'
    assert '\n12.' not in out, 'SYSTEM 已经有硬约束 12 了,v3.1.8 gold 条款会撞号'
    out = out.replace('\n输出严格 JSON', '\n' + V318_GOLD_CLAUSES + '输出严格 JSON', 1)
    assert V318_GOLD_SCHEMA_OLD in out, 'expected_output 模板变了,v3.1.8 gold 模板改写锚点失效'
    out = out.replace(V318_GOLD_SCHEMA_OLD, V318_GOLD_SCHEMA, 1)
    return out


# ── v3.1.8 输出字段补充 ─────────────────────────────────────────────
# ⛔ 这些字段注入器都真的会用(0818 逐个查过 beeai_eval/data/seed_test_data.py 的透传清单),
#    但**清单以外的字段会被静默丢掉** —— 别自己加新键。
# ── v3.1.8 gold(判分标准)粒度 ────────────────────────────────────
# ⛔ 这一段是 0815 实测**值 +27.8 分(±2.4,n=2007)** 的那个改动,原本只活在
#    `projects/autotask_testset_aligned/scripts/gold_spec.py` 产出的**副本**里,
#    主线 gen_worlds.py 从没合过。0819 因此栽了一次:v3.1.8 试水四卷全用主线跑,
#    gold 中位 2072~2500 字符(0815 实发的是 196),通过率从 57% 塌到 17%,
#    我差点把它当成「世界配方变难了」去查。**别再把它留在副本里。**
V318_GOLD_CLAUSES = """12. 【gold 粒度】key_constraints 写成**一条分号分隔的字符串**,每个要点是一句可核查的短句
    (范例:`应fire;基于邮件不编造;含建议回复或下一步;简洁action alert 80-180词`)。
    条数**按极性给,不许一律顶格**:
      · skip(真静默):**1~2 个**。正确终态只有一件事——不输出。
        写成「content 必须为空且 pass=false;不得解释为什么没提醒」就够了,
        不要把「不许出现 A/B/C/D/E」拆成五条,那会让判官逐条挑刺。
      · quiet(显式报无事项):**2~3 个**。
      · fire / deliver:**3~6 个**,只写「不写就判错」的点;
        锦上添花的格式偏好一律不进 key_constraints。
    自检:如果某一条去掉之后,一份合理答案仍然会被判错 —— 它才该留;否则删掉。
13. 【别把世界复述进 gold】describe 只说判定依据,不要罗列世界里有什么。
    世界内容已经在 seed 里了,gold 再抄一遍只会让判分点膨胀。

"""

V318_GOLD_SCHEMA = ('"expected_output": {"goal": "一句话:这题考什么、正确终态(≤80 字)", '
                  '"describe": "为什么这么判:把触发/不触发的依据点出来(≤150 字,别复述世界全貌)", '
                  '"key_constraints": "硬判分点,**一条分号分隔的字符串**,条数按上面【gold 粒度】给"}')

V318_GOLD_SCHEMA_OLD = ('"expected_output": {"goal": "一句话:这题考什么、正确终态", '
                      '"describe": "详细:世界里有什么、正确回答必须覆盖/不许出现什么", '
                      '"key_constraints": ["硬性判分点,1-4 条"]}')


V318_SCHEMA = """【v3.1.8 新增字段(都是可选的,按上面 3.5~3.11 的目标比例给)】
seed.emails 追加:`body_html`(HTML 正文)、`attachments`([{"name","content"}])、`labels`(字符串列表,给这封邮件打用户标签)、`starred`(bool)。
seed.calendar_events 追加:`all_day`(bool)、`status`("confirmed"/"cancelled")、`self_response`、`organizer`。⛔ 没有 `recurrence` 字段 —— 周期性会议按 3.7 摊成一条条独立日程写。
seed 顶层追加:`user_notes`([{"content":"...","date_offset":{"days":-30}}])。
trigger_event 追加**必填** `"source_type"`:`"mail"` 或 `"calendar"`。
  source_type="calendar" 时 trigger_event 换成这套字段(和线上日历触发的原文一一对应):
  {"source_type":"calendar","match_evidence":"...","title":"...","description":"...",
   "start_jinja":"{{ fmt_datetime(add_hours(run_date, 2), 'iso') }}","end_jinja":"{{ fmt_datetime(add_hours(run_date, 3), 'iso') }}",
   "organizer":"someone@mock.test","status":"confirmed","is_all_day":false}
  ⚠️ 这个日程**必须同时出现在 seed.calendar_events 里**(标题/时间/参会者一致),否则模型查日历会查不到触发源。
"""



V318_CLAUSES = """⚠️ 下面 3.5~3.12 是「**把世界造得像线上**」那一组,每条后面的百分比都是
   0818 从线上真实 trace 量出来的。**但你一次只造一个世界,所以每条都给了「这个世界应该有几个」
   的具体条数 —— 照条数做,百分比只是让你知道为什么。**

3.5 **参会状态必须写**(线上:有参会者的日程 97% 带回复状态;我们上一批是 0%):
   - **事件级** `self_response` = 用户本人的回复,**带 attendees 的会议一律要写**。
     四个合法值(和 Google API 一致):`needsAction` / `accepted` / `declined` / `tentative`。
     线上真实分布(n=1599):needsAction 63%、accepted 32%、declined 5%、tentative 1%
     —— **大多数会用户还没回复**,别一律写 accepted。这个世界里带参会者的会,
     **多数应该是 `needsAction`**。
   - **参会者级** 用 dict:`{"email": "...", "responseStatus": "..."}`,同样四个值;
     可选参会加 `"optional": true`,附言用 `"comment"`。
   - ⛔ attendee 只有这四个字段会被保留:`email` / `responseStatus` / `optional` / `comment`。
     `displayName`/`self`/`organizer` 写了会被丢掉,别浪费。
   - 状态要**服务于判定**:"催一下还没回复的人" → 世界里就得真有 needsAction 的人;
     "会议被拒了所以不用准备" → 就得真有 declined。别随机撒。
   - ⭐ **会的人数按比例配**(0818 实测线上:带参会者的会**中位 7 人、40% 在 10 人以上**,
     同时**近四成是 5 人以下的小会** —— 1:1、三人同步、小组站会本来就占相当比例):
     **带参会者的会里,6~15 人的大会占「六成」,2~5 人的小会占「四成」。**
     ⚠️ 这是**两头都卡的比例,不是下限**。上一版写的是「至少一半要有 6~15 个参会者」,
     模型直接顶到 88%(线上 62%)—— **只给下限,模型就会顶格取**。
     大会用 `needsAction` 铺开、个别 `declined`、个别 `optional: true`,
     这样"谁还没回""谁是可选的"才有得判;小会让"该催谁"能指到具体的人。
3.6 **邮件正文分两档写**(线上 86.7% 的正文是 HTML、长度中位 2882 字符;
   我们上一批是 0% HTML、中位 379 字符。但**整屏都写长会把输出撑爆**,所以分档):
   - ⭐ **A 档 · 和判定相关的 3 封**(触发邮件 + 主要干扰项 + 线程里的一次往返):
     `body_html` **2000~4000 字符**。真实的 HTML 邮件里有:`<table>` 排版、头部 logo 行、
     正文段落、`<a href>` 按钮、分隔线、页脚的公司地址 + 退订链接 + 法律免责声明、
     以及**下方引用的历史往来**。把这些写出来长度自然就够。
   - ⭐ **B 档 · 背景噪声 4~5 封**:`body_html` **只要 500~1000 字符**就够
     —— 促销、系统通知、newsletter、订阅摘要这类,给个精简版式(一个 `<table>` + 页脚)即可。
     **这批的作用是把收件箱撑到真实密度,不是考点,别在它们身上花篇幅。**
   - ⭐ 于是**整个世界 7~8 封邮件**。⚠️ 线上一屏中位是 14 封、p90 25 —— 我们**故意少写**:
     一次输出写不下 14 封 2000~4000 字符的正文(实测顶满 24000 token 上限还没写完,
     要重来一次才行,那一刀占 37% 的成本)。**宁可少几封,不许把正文写短** ——
     正文长度才是线上那条差距的大头。
   - ⭐ **日历 ≤4 条、`world_story` ≤200 字。** 同理:它们和邮件抢同一个输出预算。
   - ⭐ **留 1~2 封同事随手写的私信只给纯文本、不给 `body_html`** ——
     线上是 **87%** 带 HTML,不是 100%(0818 三次试水:49% → 70% → 100%,最后一次冲过头了)。
   - `body`(纯文本)两档都要给,内容和 HTML 对得上 —— 它是纯文本回退版。
     B 档的 `body` 写 1~3 句话就行。
   - ⭐ **关键那句话不要放第一行**,让它跟在问候、免责声明、历史引用后面。
3.7 **日历要有重复事件和全天事件**(线上 53.1% 的条目是重复事件的实例、44.6% 是全天;
   我们上一批两者都≈0):
   - ⭐ **这个世界的日程里,大约一半该是"周期性会议"** —— 但**要把每一次单独写出来**,
     同一个 summary、日期**一律往后排**(例:周会写成 0 天、+7 天、+14 天三条独立日程),
     每条给自己的 id。周会/日会/双周同步/月度评审都这么写。
     ⛔ **一条日程都不许排在过去**(`days_offset` 必须 ≥ 0)。0819 实测(51 个世界拟合,
     R²=0.86):**过去的日程进记忆索引的概率 ≈0%**,未来的普通日程 84%、全天 93%。
     我上一版写的是"-7 天、0 天、+7 天",于是每个世界都有一条上期会议**对
     `memory_search` 完全隐身** —— 逐条就绪率因此卡在 68% 而不是 85%。
     ⚠️ 邮件不受这条限制(过去的邮件正常入索引,实测就绪 96%);**这是日历专有的**。
     要表达"上次会开过了",写进本期日程的 description,别真造一条过去的日程。
     ⛔ **不许用 `recurrence` / RRULE 字段。** 0819 实测(47 个世界最小二乘):
     普通日程 79% 能进记忆索引、全天事件≈100%,而**带 RRULE 的重复事件 ≈0%,一条都进不去**
     —— 平台把它展开成实例后 id 变了,就绪检查永远对不上,于是 `memory_search`
     查不到那半个日历。摊开写既躲开这个洞,**也更像线上** —— 线上给的本来就是
     "重复事件的实例"(一条条具体日程),不是一条 RRULE 规则。
   - ⭐ **再放 2 个全天事件**:`"all_day": true`(⚠️ **JSON 的 true,不是字符串 "True"**),
     start/end 给**含两端**的日期(单日事件 start 和 end 同一天)。
     公共假期、请假、出差、在家办公、生日属于这类。
   - `"status"` 可写 `confirmed`(默认)或 `cancelled` —— 后者用来出"会议取消了别准备"这类题。
   - 时长别都写 60 分钟(线上 60 分钟占 38%、30 分钟占 19%,其余散在 15/45/90/120)。
3.8 **别过度装饰**(反过来的一条:我们上一批 83.6% 的日程写了 description、75.4% 写了 location,
   线上只有 24.3% 和 19.8%。每个都写满反而不真实,还白烧 token):
   - ⭐ **一个世界里通常只有 1 个日程配 description、1 个配 location**,其余都不写。
   - 只在**内容对判定有用**时写(议程、要准备的东西、会议链接)。
3.9 **邮件附件**(线上 15.6% 的邮件带附件,我们上一批是 0):
   - ⭐ **每个世界放 1 封带附件的邮件**(规则明确涉及文件/报表时放 2 封);
     线上 55% 的情况只带一个,别一封挂五个。附件只挂 A 档邮件。
   - 形式:`"attachments": [{"name": "对账单.pdf", "content": "附件里的纯文本内容"}]`。
   - ⛔ **只许这五种扩展名**:`.pdf` / `.docx` / `.csv` / `.txt` / `.ics`。
     注入器会把 content 真转成 PDF/DOCX;**`.xlsx` 和图片(.png/.jpg)会生成损坏文件,严禁使用**。
   - 规则里提到"看附件 / lampiran / anexo / attachment"的题,**必须**有附件,
     而且**判定要点要藏在附件内容里** —— 不然那条规则等于没被考到。
3.10 **要有多消息线程**(线上**逐封邮件** 18.2% 带 `Re:/Fwd:`;我们上一批只有 18.8% 的世界有线程):
   - ⭐ **每个世界必须有且只有一条 3~4 封的线程** —— 「必须有」和「只要一条」同样重要:
     0818 实测把「至少一条」松成「一条就够」之后,**有线程的世界从 93% 塌到 12%**,
     模型直接不做了。**这是硬要求,不是可选项。**
     同一话题,后续几封主题加 `Re: ` 前缀(注入器靠主题前缀归线程),
     发件人在对方和 `self` 之间交替,`date_offset` 递增。线程算 A 档,要写长。
   - 线程里放"最后一封才翻案"的情况(前面同意、最后一封改口),能考出模型有没有读完整条线。
3.11 **显式备忘 user_notes**(线上 `explicit_memories` 那段上下文会把用户存过的长期偏好
   直接注进提示词;我们的测试账号默认是空的):
   - 形式:seed 顶层 `"user_notes": [{"content": "我周五下午不排会", "date_offset": {"days": -30}}]`。
   - **只在规则确实依赖长期偏好时给 1~3 条**(如"按我的习惯处理"、"别让 XX 打扰我");
     ⭐ **大多数题应该一条都不给** —— 线上大多数用户也没存过。
     0818 首跑 9/11 个世界都塞了,那是过度使用:规则里没有"按我的偏好/习惯/口味"
     这类字眼就别给。
3.12 **收件箱要像个真收件箱**(0818 第二轮实测出来的几条,都是小改动但差得不小):
   - ⭐ **未读要多**:线上 **82.6%** 的邮件是未读的,我们上一批只有 58%。
     **除了明确"已经处理过/已回复"的干扰项,其余一律 `"unread": true`。**
   - ⭐ **标签要少**:线上只有 **3%** 的邮件带用户自己打的标签,
     我们上一批 56% —— 几乎每封都归了档,很假。
     **一个世界最多给 1~2 封邮件写 `labels`**,而且只在规则真的按标签筛选时才给。
   - ⭐ **机器发的要多**:线上 **33.7%** 的发件人是 `no-reply@` / `notifications@` /
     `newsletter@` 这类。⭐ **全世界给 4~5 封这种发件人就够**
     (0818 试水按"B 档至少一半"写,做到了 50%,超了)。
   - ⭐ **老邮件要有**:线上 **22.9%** 的邮件是 7 天以前的(年龄 p90 是 17 天),
     我们上一批只有 6.9%。**B 档噪声里放 3~4 封 `date_offset` 在 -7 到 -30 天的**,
     这样"最近一个月""上次那封"这类查询才有东西可翻。
   - **主题写得像真的**:线上 **28.6%** 的主题带方括号前缀(`[Need Approval] …`、
     `[JIRA] …`、`[Alerta] …`),**8.3%** 带 emoji。我们上一批是 13% 和 2%。
     ⭐ **全世界放 2 个方括号主题、1 个带 emoji 的就够**
     ⛔ **改总封数的时候,所有"给绝对个数"的条款都要跟着重算** —— 这条 0820 栽过:
     总封数从 10~14 压到 7~8 之后,"放 3 个"的比例被动从 25% 涨到 **51.7%**
     (带 10~45%、线上 29%),而提示词一个字没改。
     3 个 / 8 封 = 37.5%,再加上模型本来就爱多给,就顶出去了。改成 2 个 / 8 封 = 25%。
     (0818 试水按"3~4 个"写做到 52%;当时是 12 封的世界,同一个数在不同总量下比例完全不同。)
"""



SYSTEM = """你在为"自动任务提醒助手"的数据工厂造可运行的题:给定一条真实用户的自动任务规则(原文,不许改),你要设计一个全新的账号世界(种子数据,会被注入测试 Gmail 账号),让这条规则在这个世界里可以被执行、且正确行为无歧义。

硬约束(违反即废):
1. 实体一致性优先:**规则原文里点名的实体(人名/主题词/标签名/发件人名)必须原名出现在世界里**——它们是判定条件的一部分,改名会把条件规则弄断(比如规则说"Dhika 回复要保持未读",世界里就必须有叫 Dhika 的人)。邮箱地址一律安全域化:规则点名的地址保留 local-part 和显示名、域名替换为 mock.test(如 rizqi.amalia@spxexpress.com → "Rizqi Amalia" <rizqi.amalia@mock.test>)。**邮箱地址本身必须全 ASCII**；人名可以保留重音符号,但地址要转写(如 ``Cauã Brandão <caua.brandao@mock.test>``),否则评测注入器会整题失败。**未被规则点名的其他人物**(噪声发件人、路人)才必须虚构,严禁照抄材料里的真实姓名。域名规则——凡是助手可能"回复/转发/发送"到的地址一律用 @mock.test;纯装饰性的只读发件人可用 *.example.com;严禁 gmail.com/outlook.com/真实公司域名等可投递域名;严禁编造 @shopee.com 地址。
2. 时间全部相对:邮件用 date_offset {"days":int,"hours":int}(相对运行时刻,负数=过去);正文/主题里出现日期时用 jinja2 助手如 {{ fmt_date(add_days(run_date, 1), 'long') }};严禁写死 2026-xx-xx 之类绝对日期。
3. 日历事件 start/end:"今天/几小时后"用 {"minutes_offset": int};"明天/后天"用 {"days_offset": int, "hour": h, "minute": m};不要用 next_weekday 表达"今天/明天"。
4. 极性可判:我给你的目标极性(该报 fire / 该忍 skip / 正常交付 deliver / 如实说安静 quiet)必须由世界内容唯一决定,不能模棱两可。skip 型:被触发的邮件要"表面能被笨闸匹配、按规则细读却明确不够格"(如 CC-only、常规审批、无真实风险的急措辞、纯营销)。
5. 真实感:收件箱要有噪声(促销/newsletter/系统通知/CC 线程,3-8 封),重要信号埋在噪声里;邮件正文要像真的(有称呼、正文、落款,长度 40-200 词),语言跟随该用户的语言环境(可混少量英语噪声)。
6. 世界语言 = 用户语言;expected_output 三字段的语言 = 用户语言。
7. 纳入/排除成对:规则只要含时间窗、已处理状态、本人责任、To/CC、熟悉/陌生等筛选条件,世界里必须同时有够格项和至少一个近似但不够格的干扰项(空结果题则至少两个近似干扰项);expected_output 要把“必须出现”和“绝不能出现”分别列清。
8. 动作必须落地:规则要求建草稿、标已读、删除、发送、更新表格等动作时,expected_output 必须要求真实工具动作完成;只给建议文案、说“我会做”、或把需建草稿邮件仅保留未读都算失败。
9. 报告必须完整:规则要求多个区块/字段/时间段时,seed 必须逐项提供证据,并放一个容易漏掉但仍未解决的项目;expected_output 按字段列出完整清单,不能只写主题正确但缺段落的答案。
10. 证据闭环:任何业务承诺、数字、时间、人物关系、是否已处理都必须能在 seed 中找到证据;不知道就写未知/缺数据,严禁为了让答案完整而脑补。
11. 严格区分静默:skip = 面向用户 content 为空且 pass=false(不得解释“为什么没提醒”);quiet = 规则明确要求空窗时也报告“无事项”,content 非空且 pass=true。不得互换。

输出严格 JSON(不要 markdown 代码块):
{
 "world_story": "一段中文内部说明:这个世界怎么回事、为什么极性成立",
 "seed": {"emails": [...], "calendar_events": [...], "labels": [...], "contacts": [...], "drive_files": [...]},
 "trigger_event": {"match_evidence": "...", "related_people": "...", "title": "...", "sender": "...", "receivers": "self 的邮箱或姓名", "cc": "", "send_time_jinja": "{{ fmt_datetime(add_minutes(run_date, -35), 'iso') }}", "summary": "...", "topics": "逗号分隔主题词"},
 "expected_output": {"goal": "一句话:这题考什么、正确终态", "describe": "详细:世界里有什么、正确回答必须覆盖/不许出现什么", "key_constraints": ["硬性判分点,1-4 条"]},
 "polarity": "fire|skip|deliver|quiet"
}
seed.emails 字段:id/from{name,email}/to("self")/subject/body/date_offset/unread(bool)/important(bool)/cc(列表,可省);calendar_events 字段:id/summary/start/end(+可选 attendees/location/description)。
drive_files 字段(仅任务需要文件/表格时给):id/filename/mime/content——Google 表格 mime="application/vnd.google-apps.spreadsheet"、content 放 TSV(制表符分隔);文档 mime="application/vnd.google-apps.document"、content 纯文本;文件名与内容要能支撑任务里的查询。
cron 题不需要 trigger_event(填 null);event 题的触发邮件必须同时出现在 seed.emails 里(字段一致,date_offset 跟 send_time_jinja 对应)。"""


POL_MAP = {'fire': '该报(fire):触发邮件按规则确实够格,应当推送提醒',
           'skip': '真静默(skip):按规则不应面向用户输出,信封 pass=false 且 content 为空；若规则要求静默执行工具动作可执行,但不能发“已忽略/没有更新”提示',
           'deliver': '正常交付(deliver):世界里有本任务要的内容,正常产出',
           'quiet': '显式空结果(quiet):仅当规则明确要求空窗时也要告知“没有/无变化”才使用,信封 pass=true；不得用于要求静默的规则'}


# ⛔⛔ **这一条不进 SYSTEM,只在"这道题当年真调过网盘/表格"时拼进该题的提示词。**
#
# 0820 同题配对实测:把它作为第 9 条形态条款放进 SYSTEM(所有题都读)之后,
#   A 档长邮件(≥2000 字符)  3.0 封 → **0.5 封**
#   最长正文中位             3648  → 1994 字符
#   world_story              222   → 396 字符(反而超了 3.6 的 200 上限)
# 也就是说:**多加一条带自己诉求的条款,会把模型的产出预算从 3.6 的长正文上挪走。**
# 第一版里那句"别真写上千行,会和 3.6 抢预算"更是雪上加霜(那是我们的成本考量,
# 根本不该进提示词),但删掉它之后 A 档仍然只回到 0.5 —— 说明**位置本身就是代价**。
#
# 而这条**天然是有条件的**:只有 33.6% 的世界需要 drive_files。
# ⇒ 放进 needs_drive 分支,其余 66% 的题拿到的提示词与已验证过的批次逐字相同。
# ⚠️ 它仍然计入 prompt_sha256(见 main),否则两批不同的网盘要求会共用一个指纹。
DRIVE_CLAUSE = """3.13 **网盘/表格**(0820 从源池子实测 172 个真实文件、84 张真表):
   ⚠️ 只有这道题**当年真的调过网盘/表格工具**时才给 `drive_files`;没调过就不给。
   - ⭐ **一个世界给 1 份文件**(线上 222 条用过网盘的记录里,174 条只碰 1 个文件 = 78%)。
   - ⭐ **表格给 20~40 行 × 8~12 列的代表性切片**。真表中位 **998 行 × 11 列** ——
     列数要够、规模感要像真的,行数取有代表性的一段即可。
   - ⭐ **列名要像真业务表**:`Date raised (Ctrl + ;)` / `Escalated by who (SM PIC)` /
     `Topic of Complaint` / `Case Status` / `Meet SLA (First contact)` / `L1 Reason Code` ——
     特征是带括号说明、带责任人缩写、带状态/编码列。
   - ⭐ **类型别全给原生 Google 表格**:线上是原生表格 51 · 演示 9 · PDF 6 · 原生文档 6 ·
     纯文本/CSV 5。约一半的题给 `text/csv` 或 `application/pdf`
     (CSV 放逗号分隔内容;PDF/演示只给文件名和一段纯文本摘要)。
   - ⭐ **文件名带日期戳和版本号**(`[FM Sorter] Reconciliation Form 2.0 (Responses)`),
     ⛔ 日期必须用 jinja2 助手写,不许写死绝对日期。
"""


def user_prompt(case, material):
    v = dict(VARIANTS).get(case.get('variant', ''), '')
    if case.get('family'):
        v = dict(FAMILY_VARIANTS[case['family']]).get(case.get('variant', ''), v) + FAMILY_LANG_PIN
    if case.get('variant') == 'empty_boundary':
        v = ('空结果边界题:按规则制造没有任何够格项的世界,但至少放两个很像、因时间窗/'
             '已处理状态/责任人/收件人身份等明确不够格的干扰项；严格遵守目标是 skip 真静默'
             '还是 quiet 明示无结果。')
    pol_map = POL_MAP
    lines = [
        f"【规则原文(用户语言={case['lang']},触发类型={case['trigger']})】\n{case['rule']}",
        f"\n【目标极性】{pol_map[case['polarity']]}",
    ]
    if v:
        lines.append(f"\n【世界变体要求】{v}")
    lines.append(f"\n【用户时区】UTC{case['tz_offset']}({case['tz_iana']})")
    lines.append(f"\n【真实线上材料(只作噪声构成/难度/文风参照,身份和内容必须重造,不许照抄)】\n{material}")
    if case['trigger'] == 'event':
        if case.get('trigger_source') == 'calendar':
            lines.append('\n提醒:这是 event 题,而且**当年真实的触发源是一条日历日程,不是邮件**。'
                         'trigger_event 必须写 "source_type":"calendar" 那一套字段,'
                         '并且同一条日程要出现在 seed.calendar_events 里(标题/时间/参会者一致)。')
        else:
            lines.append('\n提醒:这是 event 题,必须给 trigger_event("source_type":"mail"),'
                         '且触发邮件要进 seed.emails。')
    else:
        lines.append('\n提醒:这是 cron 题,trigger_event 填 null;世界要覆盖规则里查询窗口(如"今天 8 点后"→邮件 date_offset 落在今天窗口内,用 hours 负偏移)。')
    if case.get('needs_drive'):
        # ⚠️ 规模数字**只写一处**(DRIVE_CLAUSE),别在这句里再写一遍 —— 两处各写一个数,
        #    模型会锚在先出现的那个上(0818 实测,见 3.6 那条的注释)。
        lines.append('\n⚠️ 该任务当年真实用到了网盘/表格工具:seed.drive_files 必须提供'
                     '任务所需的文件,规模、类型和命名按下面这条来:\n' + DRIVE_CLAUSE
                     if is_aligned_recipe(REALISM) else
                     '\n⚠️ 该任务当年真实用到了网盘/表格工具:seed.drive_files 必须提供任务所需的'
                     '文件(表格给 TSV 内容,行数 5-15,列名和数据要能支撑任务的查询/汇总)。')
    lines.append('\n【rq2e 稳定失败反馈转成的通用造题要求】\n' + eval_guidance(case['rule']))
    return '\n'.join(lines)


def record_material(g, rec_idx, cap=13000):
    """按记录模式的材料:双判官实际评审的最后一个 tick。"""
    r = g['records'][rec_idx]
    _, best = target_tick(r)
    if best is None:
        return '(目标 tick 缺失)'
    parts = [f"[当天的触发文本]\n{best['tick'][:3000]}"]
    for i, tc in enumerate(best['tool_calls'][:10]):
        parts.append(f'[工具调用{i}] {str(tc)[:400]}')
    for i, tr in enumerate(best['tool_responses'][:10]):
        parts.append(f'[工具返回{i}] {str(tr)[:2500]}')
    if best['assistant']:
        parts.append(f"[当时的最终回答(双判官判OK,极性以它为准)]\n{best['assistant'][-1][:2000]}")
    return '\n\n'.join(parts)[:cap]


def user_prompt_record(case, material):
    lines = [
        '这是【重建】任务:下面是一个真实用户这条自动任务当天真实执行的完整轨迹。'
        '请把当时的收件箱/日历重建成种子数据,让同一条规则在测试账号里重演同一个决策情境。',
        f"\n【规则原文(用户语言={case['lang']},触发类型={case['trigger']})】\n{case['rule']}",
        f"\n【目标极性(=当时真实结果)】{POL_MAP[case['polarity']]}",
        f"\n【用户时区】UTC{case['tz_offset']}({case['tz_iana']})",
        f'\n【当天真实轨迹】\n{material}',
        '\n重建要求:'
        '\n1. 重建而非再创作:工具返回里出现的邮件/日程照实重建(主题、发件人显示名、时间先后、未读状态、正文要点);'
        '[REDACTED]/[PHONE_x]/[EMAIL_x] 等脱敏洞用合理虚构内容补齐;正文可按要点扩写成自然邮件'
        # ⚠️ 这里原来写死「(40-200词)」。record 模式用的就是这份提示词,
        #    那个数(≈250~1300 字符)会把 v3.1.8 的 A 档 2000~4000 直接压回去 ——
        #    SYSTEM 里改了、这里不改等于白改。
        # ⛔ v3.1.8 这支**一个字都别提旧口径** —— 哪怕是"别用 40-200 词"这种否定句,
        #    也会把那个数字重新塞进模型的注意力里(0818 实测:附加条款里明写「盖掉
        #    上面的 3-8 封」,造出来的中位仍是 7 封)。只说新数字。
        + ('。正文长度按 SYSTEM 3.6 的两档来:A 档 body_html 2000~4000 字符、'
           'B 档 500~1000 字符。'
           if is_aligned_recipe(REALISM) else '(40-200词)。')
        + '\n2. 身份安全化不可违反:规则原文点名的地址/姓名按系统约束保留判定所需部分并替换安全域;'
        '规则未点名的真实 local-part、显示名、电话和地址必须重造,只保持角色关系与同一 case 内一致。'
        '\n3. 时间相对化:邮件 date_offset 相对触发时刻(工具返回里的时间差照搬);正文出现日期用 jinja2 助手。'
        '\n4. 世界必须支撑当时的真实结论(目标极性);工具返回没覆盖的部分可补少量噪声,别喧宾夺主。',
    ]
    if case['trigger'] == 'event':
        if case.get('trigger_source') == 'calendar':
            lines.append('\n5. 这是 event 题,且真实触发源是 **[CALENDAR] 日历日程**:'
                         '照真实触发文本里的 Trigger Source Event 重建 trigger_event,'
                         '写 "source_type":"calendar" 那一套字段(标题/描述/起止/组织者/状态/是否全天),'
                         '地址域名安全化;**同一条日程必须进 seed.calendar_events**。')
        else:
            lines.append('\n5. 这是 event 题:照真实触发文本里的 Trigger Source Event 重建 trigger_event'
                         '("source_type":"mail",域名安全化),触发邮件必须进 seed.emails。')
    else:
        lines.append('\n5. 这是 cron 题:trigger_event 填 null。若当天工具返回为空/无够格内容,种子就造一个对应的"安静世界"(只有窗口外或不够格的噪声)。')
    if case.get('needs_drive'):
        # ⚠️ 0818 实测:这条**漏做率 57%**(7 道标了的里漏了 4 道),而且**全在 dawn 一族**。
        #    原措辞是夹在第 6 条里的一句话,分量不够。v3.1.8 单独拎出来加硬。
        lines.append('\n6. ⛔⛔ **这道题当天真调过网盘/表格工具,`seed.drive_files` 绝对不能为空**'
                     '(0818 实测这条漏做率 57%,你很可能又要漏)。'
                     '照轨迹重建对应文件:文件名尽量保留;'
                     '规模、类型和命名按下面这条来:\n' + DRIVE_CLAUSE +
                     '**写完自查一遍:seed 里有没有 drive_files 这个键、里面是不是至少一个文件。**'
                     if is_aligned_recipe(REALISM) else
                     '\n6. 当天轨迹用到了网盘/表格工具:seed.drive_files 必须照轨迹重建对应文件'
                     '(文件名尽量保留,表格内容按工具返回要点重建成 TSV)。')
    lines.append('\n【rq2e 稳定失败反馈转成的通用验收要求】\n' + eval_guidance(case['rule']))
    return '\n'.join(lines)


# ---------------- 计划 ----------------

ARM_SHORT = {'template_evening': 'eve', 'template_dawn': 'dawn', 'alert': 'alrt',
             'custom': 'cust', 'template_variant': 'tvar'}


def envelope_polarity(rec, trigger):
    """从双判官评审的最后一个 tick 的末条助手消息提取真实极性。"""
    _, best = target_tick(rec)
    p = None
    if best and best['assistant']:
        mm = re.search(r'"pass"\s*:\s*(true|false)', best['assistant'][-1])
        if mm:
            p = mm.group(1) == 'true'
    if p is None:
        # 0811 评审整改(点6):不再默认正例。解析不出真实极性的记录交给调用方剔除
        # (实测 v2 池 1254 条里只有 6 条走到这,0.5%,剔了不心疼;默认 True 会把
        # 历史模型的未知行为洗白成「该推」的 gold)。
        return None, best
    if trigger == 'event':
        return ('fire' if p else 'skip'), best
    return ('deliver' if p else 'skip'), best


TRIG_SRC_RE = re.compile(r'Trigger Source Event \(source_type = (\w+)\)')


def trigger_source_of(rec):
    """这条真实记录是**邮件触发**还是**日历触发**。

    0818 实测:线上 event 类的 401 条记录里 82 条(20.4%)是日历触发,原文形状
    (``[CALENDAR] 标题 / Description / Start / End / Organizer / Status / Is All Day``)
    和邮件那套完全不同。v3.1.7 把它们全按邮件造了 —— 那 82 道题的触发形状是错的。
    """
    srcs = {m.group(1) for t in (rec.get('ticks') or [])
            for m in [TRIG_SRC_RE.search(t.get('tick') or '')] if m}
    return 'calendar' if srcs == {'calendar'} else 'mail'


def relabel_languages(pool):
    """v3.1.8:用**规则原文**重新判语种,覆盖池子里烤着的 lang。

    为什么必须做:`source_pool.jsonl` 是 0811 造的,**当时机器上没装 langdetect**,
    `extract_source.infer_lang` 降级到兜底规则,把葡语大面积判成越南语/英语。
    0818 用装好 langdetect 的检测器复算:**683/2025 = 33.7% 的标签是错的**。

        池子烤着的  en 907 / id 439 / vi 339 / pt 135 / ja 115 / thai 61 / zh 29
        重新判的    pt 651 / en 499 / id 397 / vi 160 / ja 115 / zh 70 / es 64 / thai 64 / fr 5

    **葡语才是最大语种(32%),我们一直以为是 6.7%;西班牙语 64 条、法语 5 条
    以前根本没有这两个类别。**

    危害不是"标签不好看":`user_prompt` 会把这个标签当成「用户语言」写进起草提示词,
    而 SYSTEM 里写着「世界语言 = 用户语言」—— 于是有 179 道葡语题,我们告诉起草模型
    "用户语言=vi"。而「语种一致」正是 v13 判官的判据之一,也是我们低档失分的大项。

    ⚠️ 只在 --realism v3.1.8 生效:改了 lang 就会改 case_id 和 plan 指纹,
    老批次不能动。审计器见 `audit_item_language.py`(它 0814 就报过 640/2025 不符)。
    """
    from extract_source import infer_lang
    changed = defaultdict(int)
    for g in pool:
        old_lang = g.get('lang')
        new_lang = infer_lang(g['rule'], old_lang)
        if new_lang != old_lang:
            changed[(old_lang, new_lang)] += g.get('n_records', 1)
            g['lang'] = new_lang
    if changed:
        total = sum(changed.values())
        print(f'⚠️ 语种重判:{total} 条记录的 lang 变了(池子是 0811 缺 langdetect 时造的)',
              flush=True)
        for (o, n), c in sorted(changed.items(), key=lambda kv: -kv[1])[:10]:
            print(f'     {o} → {n}: {c} 条', flush=True)
    return pool


def build_plan(pool, allow_eval_collisions=False, case_prefix='rq1'):
    if is_aligned_recipe(REALISM):
        pool = relabel_languages(pool)
    by_arm_lang = defaultdict(list)
    for g in pool:
        sim = ((g.get('eval_sim') or {}).get('J') or 0)
        if g.get('arm') == 'custom' and sim >= 0.70 and not allow_eval_collisions:
            raise ValueError(f"评测撞题未隔离: rule={g['rule_key']} J={sim}")
        by_arm_lang[(g['arm'], g['lang'])].append(g)

    cases = []
    engineered_ordinals = defaultdict(int)

    def add(case_id, g, variant, polarity, mode='engineered', rec_idx=None):
        if rec_idx is None:
            ordinal = engineered_ordinals[g['rule_key']]
            engineered_ordinals[g['rule_key']] += 1
            rec_idx = stable_record_index(case_id, g, ordinal)
        rec = g['records'][rec_idx]
        tick_index, _ = target_tick(rec)
        off, iana, label = tz_for_record(rec)
        cases.append({
            'schema_version': 2,
            'case_id': case_id, 'arm': g['arm'], 'trigger': g['trigger'], 'lang': g['lang'],
            'rule_key': g['rule_key'], 'rule': g['rule'], 'variant': variant, 'polarity': polarity,
            'mode': mode, 'rec_idx': rec_idx,
            'source_tick_index': tick_index,
            'cutoff_at': rec.get('cutoff_at') or (target_tick(rec)[1] or {}).get('observed_at'),
            'user_key': rec.get('user_key'),
            'primary_account_key': rec.get('primary_account_key'),
            'account_keys': rec.get('account_keys') or [],
            'judge_target_aligned': rec.get('judge_target_aligned'),
            'tz_offset': off, 'tz_iana': iana, 'tz_label': label,
            'n_source_records': g['n_records'],
            # 仅 v3.1.8:老配方的 case 结构保持原样,免得 plan 指纹/下游 schema 跟着变
            **({'trigger_source': trigger_source_of(rec)} if is_aligned_recipe(REALISM)
                and g['trigger'] == 'event' else {}),
            'source_traces': [rec['trace_id']],
            'eval_sim': g.get('eval_sim'),
        })

    # ===== 设计变体臂(engineered):模板 12 变体/语言 + 警报每规则1题 + 自定义按规则 =====
    for arm, short in (('template_evening', 'eve'), ('template_dawn', 'dawn')):
        for lang in ('en', 'id', 'ja', 'vi'):
            gs = sorted(by_arm_lang.get((arm, lang), []), key=lambda g: -g['n_records'])
            if not gs:
                continue
            for i, (vkey, _) in enumerate(VARIANTS):
                g = gs[i % len(gs)] if len(gs) > 1 and i >= 8 else gs[0]
                # ``quiet`` here is a world-content variant, not an output
                # label.  Mandatory briefs remain deliver; only the rule's
                # explicit empty-result contract can turn it into skip/quiet.
                pol = ((empty_behavior(g['rule']) or 'deliver')
                       if vkey == 'quiet' else 'deliver')
                add(f'{case_prefix}_{short}_{lang}_{i:02d}_{vkey}', g, vkey, pol)

    for (arm, lang), gs in sorted(by_arm_lang.items()):
        if arm != 'alert':
            continue
        for i, g in enumerate(sorted(gs, key=lambda g: g['rule_key'])):
            pol = 'fire' if i % 2 == 0 else 'skip'
            add(f'{case_prefix}_alrt_{lang}_{i:04d}_{pol}', g, '', pol)

    # 自定义臂:id 用 rule_key 前 8 位(池子增删不迁移已有 case)。
    # cron 先固定造一个正例；只有规则**明确写出**空结果终态时才追加负例/空结果。
    # 旧实现按 i%3 机械把 1/3 规则变 quiet，实测会把“每天必须播报价格/报表”也造
    # 成空窗标签，且把“无命中应完全静默”误训成“发一条今天没事”。
    for (arm, lang), gs in sorted(by_arm_lang.items()):
        if arm != 'custom':
            continue
        for i, g in enumerate(sorted(gs, key=lambda g: g['rule_key'])):
            rk = g['rule_key'][:8]
            if g['trigger'] == 'cron':
                add(f'{case_prefix}_cust_{lang}_{rk}_deliver', g, '', 'deliver')
                empty_pol = empty_behavior(g['rule'])
                if empty_pol:
                    add(f'{case_prefix}_cust_{lang}_{rk}_{empty_pol}', g,
                        'empty_boundary', empty_pol)
            else:
                add(f'{case_prefix}_cust_{lang}_{rk}_fire', g, '', 'fire')
                add(f'{case_prefix}_cust_{lang}_{rk}_skip', g, '', 'skip')

    # ===== 按记录臂(record):每条源记录 1 题,世界=该记录自己的轨迹重建,极性=真实信封 =====
    seen_ids = {c['case_id'] for c in cases}
    for g in pool:
        short = ARM_SHORT.get(g['arm'], 'x')
        for ri, r in enumerate(g['records']):
            pol, _ = envelope_polarity(r, g['trigger'])
            if pol is None:
                print(f"  剔除:极性解析不出(不默认正例,点6)rule={g['rule_key']} rec#{ri}")
                continue
            cid = f"{case_prefix}_rec_{short}_{g['lang']}_{(r['trace_id'] or f'{g['rule_key']}{ri}')[:12]}"
            if cid in seen_ids:
                cid = f'{cid}_{ri}'
            seen_ids.add(cid)
            add(cid, g, 'record', pol, mode='record', rec_idx=ri)

    # ===== 0822 四题族(仅 v3.1.9;A=建草稿 B=进展去重 C=清单纪律;D 压舱=现成 quiet/borderline/already_handled) =====
    if is_family_recipe(REALISM):
        prog_re = re.compile(r'progress report|work progress|relat[óo]rio.{0,20}progresso|'
                             r'laporan kemajuan|进展|進捗|báo cáo tiến độ', re.I)

        def addfam(family, g, vkey, pol, sfx=''):
            cid = (f"{case_prefix}_fam{FAM_SHORT[family]}_{ARM_SHORT.get(g['arm'], 'x')}_"
                   f"{g['lang']}_{g['rule_key'][:8]}_{vkey}{sfx}")
            assert cid not in seen_ids, f'题族 case_id 撞了: {cid}'
            seen_ids.add(cid)
            add(cid, g, vkey, pol)
            cases[-1]['family'] = family
            exp = None
            if family == 'draft' and sfx != '_w4':
                # _w4 遍 = 免条款采样臂(0824 第三圈实验):不拼明示条款,
                # 对照「条款拉高收割密度但复读闸吃掉 ~63%」的净产出。
                exp = FAMILY_EXPLICIT_DRAFT.get('en')
            elif family == 'checklist':
                exp = FAMILY_EXPLICIT_CLAUSE.get(g['lang'])
            if exp:
                # 只存纯条款:装配时拼到 question 尾,渲染前按原串精确摘除(swap_rule_text.py)
                cases[-1]['explicit_clause'] = exp
            if family == 'draft' and sfx == '_w4':
                cases[-1]['clause_free'] = True

        # 四遍:''/_w2(0822 用户定翻倍)+ _w3/_w4(0824 题库再翻倍,176→352)。
        # ⛔ 前两遍的调用顺序必须与老版完全一致 —— engineered_ordinals 是顺序敏感的,
        #    乱序会改已造世界的 rec_idx/指纹;新遍只许**追加在后**。
        for dup, sfx in ((0, ''), (1, '_w2'), (2, '_w3'), (3, '_w4')):
            for g in pool:
                r = g['rule']
                if 'when appropriate' in r and 'draft' in r.lower():
                    for vkey, _ in FAMILY_VARIANTS['draft']:
                        addfam('draft', g, vkey, 'deliver', sfx)
                elif prog_re.search(r):
                    for vkey, _ in FAMILY_VARIANTS['progress_dedupe']:
                        addfam('progress_dedupe', g, vkey, 'deliver', sfx)
            for arm in ('template_evening', 'template_dawn'):
                for lang in ('en', 'id', 'ja', 'vi', 'pt', 'zh'):   # 重判后 pt 是最大语种,别漏
                    gs = sorted(by_arm_lang.get((arm, lang), []), key=lambda g: -g['n_records'])
                    if not gs:
                        continue
                    for vkey, _ in FAMILY_VARIANTS['checklist']:
                        addfam('checklist', gs[dup % len(gs)] if len(gs) > 1 else gs[0],
                               vkey, 'deliver', sfx)
        nfam = Counter(c.get('family') for c in cases if c.get('family'))
        print(f"  题族计划: {dict(nfam)}", flush=True)
    return cases


def select_plan_mode(plan, mode):
    """Select the requested world source without silently changing its size."""
    if mode == 'all':
        return list(plan)
    if mode not in {'record', 'engineered'}:
        raise ValueError(f'unknown plan mode: {mode}')
    return [case for case in plan if case.get('mode') == mode]


def cap_per_rule(plan, cap):
    """同一条规则最多留 ``cap`` 道题。

    为什么要这个:题目高度集中在少数几条**官方模板**上 —— 0820 实测,最大的一条
    (葡语晨报)独占 2025 道里的 208 道 = 10.3%,前 20 条规则吃掉 47.5%。
    这些题的世界内容各不相同(不是重复数据),但**题面文本一模一样**,
    钱和梯度都压在少数模板上,而 506 条只出现一次的自定义规则(最多样、最难的那批)
    只占 25%。设上限就是把预算从"第 31 遍晨报"挪到别处。

    ⛔ **只对 record 模式的题生效**。另外两种出题模式本来就是一条规则一道题。
    ⛔ 选哪 30 条按 ``rec_idx`` 取前 N —— 必须是**纯函数**,不能看目录里已经造了什么,
       否则同样的输入会因为磁盘状态不同排出不同的计划,批次就不可复现了。
    """
    if not cap or cap <= 0:
        return list(plan), {}
    kept, dropped = [], Counter()
    for case in plan:
        if case.get('mode') != 'record' or case.get('rec_idx', 0) < cap:
            kept.append(case)
        else:
            dropped[case['rule_key']] += 1
    return kept, dict(dropped)


# ---------------- 执行 ----------------

# ══════════════════════════════════════════════════════════════════════════════
# 计费仪表(0819 加)。**没有它就只能靠猜**:用户问"这批要花多少钱"时,我只能
# 拿分词器估,而 DSv4 的分词器对中文只有 Claude 的一半,估出来能差一倍。
# 网关在 usage 里直接回 `price_cost_usd`(真实计费),白拿的东西之前一直没接。
#
# 三个调用点分开记,因为它们的**优化手段完全不同**:
#   main    主调用           —— 降不了,除非精简提示词
#   repair  JSON 不合法重来   —— 撞 24000 上限居多,抬 cap 或压输出量
#   trigger event 题补字段    —— 提示词里把 trigger_event 说得更硬就能省掉
# 0819 实测 cal2/cal3 都是 59 次调用 / 30 个世界 ≈ 1.97 次,**近一半的钱花在重来**。
CALL_STATS = {}


def _billed(site, usage):
    """把一次调用的用量记进账本。usage 是网关原样返回的 dict。"""
    row = CALL_STATS.setdefault(site, {'n': 0, 'in': 0, 'out': 0, 'usd': 0.0})
    row['n'] += 1
    row['in'] += int(usage.get('input_tokens') or usage.get('prompt_tokens') or 0)
    row['out'] += int(usage.get('output_tokens') or usage.get('completion_tokens') or 0)
    try:
        row['usd'] += float(usage.get('price_cost_usd') or usage.get('price_cost') or 0)
    except (TypeError, ValueError):
        pass


def _snapshot_billing(out_dir):
    """账本随跑随落。⛔ 只在结束时写的话,中途 kill 就把这批的计费全丢了 ——
    而"跑一小批看账本再优化"正是要中途停的场景。"""
    try:
        (out_dir.parent / f'{out_dir.name}_billing.json').write_text(
            json.dumps(billing_report(), ensure_ascii=False, indent=2))
    except OSError:
        pass


def billing_report():
    tot = {'n': 0, 'in': 0, 'out': 0, 'usd': 0.0}
    for row in CALL_STATS.values():
        for k in tot:
            tot[k] += row[k]
    return {'by_site': CALL_STATS, 'total': tot}


# ══════════════════════════════════════════════════════════════════════════════
# 出口闸:A 档长正文(0820 加)
#
# 3.6 要求「和判定相关的 3 封」body_html 写到 2000~4000 字符 —— 判据往往就藏在
# 这三封里(引用的历史往来、表格里的数字、页脚的免责声明)。写短了题目会变简单。
#
# ⛔ **为什么要闸,而不是继续改提示词**:起草模型是外部依赖,它的行为会漂。
# 0820 实测:用 0819 那份代码原样重跑同 10 道题(提示词指纹同为 3aec7fab),
# A 档 3.0 → **1.0**、判定相关那封的正文 3648 → **2016 字符**,而中位长度没变
# —— 长尾被削掉了。提示词一个字没改。⇒ 靠措辞维持形态本来就不牢靠,
# 把要求变成**可检可修的出口条件**才稳。
A_TIER_MIN_CHARS = 2000
A_TIER_TARGET = 3
GATE_STATS = Counter()


def count_a_tier(obj, min_chars=A_TIER_MIN_CHARS):
    emails = ((obj.get('seed') or {}).get('emails') or []) if isinstance(obj, dict) else []
    return sum(1 for x in emails
               if isinstance(x, dict) and len(str(x.get('body_html') or '')) >= min_chars)


LENGTHEN_INSTRUCTION = (
    '你的 seed 里 body_html ≥{min_chars} 字符的邮件只有 {have} 封,3.6 要求 {want} 封。\n'
    '⛔ **只改正文长度,别的一个字都不许动**:邮件封数、主题、发件人、日期偏移、未读状态、'
    '日历、通讯录、标签、网盘文件、trigger_event、expected_output、polarity、world_story '
    '全部原样保留。\n'
    '把**和判定相关的那 {want} 封**(触发邮件 + 主要干扰项 + 线程里的一次往返)的 `body_html` '
    '扩写到 **{min_chars}~4000 字符**:补 `<table>` 排版、头部 logo 行、正文段落、'
    '`<a href>` 按钮、分隔线、页脚的公司地址 + 退订链接 + 法律免责声明,'
    '以及**下方引用的历史往来**。\n'
    '⛔ 扩写只能加**已经能从这个世界里推出来的**内容,不许引入新的事实、数字、日期或人物。\n'
    '重新输出【完整】严格 JSON,不要任何解释。')



def merge_lengthened(obj, obj2):
    """只把**变长了的 body_html** 回填进原世界,其余一概不动,返回回填了几封。

    ⛔ **不整份采纳补写结果。** 实测:即使指令明写"邮件封数/主题/发件人全部原样保留",
    模型在这一轮里仍然会增删邮件(10 道题里 5 次因此被拒,那 5 次调用全白花)。
    按 id / 主题匹配、只覆盖 body_html,就与它是否动了别的字段完全无关 ——
    动了也不要紧,我们只取想要的那一个字段。
    """
    source = {}
    for mail in ((obj2.get('seed') or {}).get('emails') or []):
        if not isinstance(mail, dict):
            continue
        for key in (mail.get('id'), str(mail.get('subject') or '').strip()):
            if key:
                source.setdefault(key, mail)
    filled = 0
    for mail in ((obj.get('seed') or {}).get('emails') or []):
        if not isinstance(mail, dict):
            continue
        cand = source.get(mail.get('id')) or source.get(str(mail.get('subject') or '').strip())
        if not cand:
            continue
        longer = str(cand.get('body_html') or '')
        if len(longer) > len(str(mail.get('body_html') or '')):
            mail['body_html'] = longer
            filled += 1
    return filled



def run_case(case, material):
    up = user_prompt_record(case, material) if case.get('mode') == 'record' else user_prompt(case, material)
    msgs = [{'role': 'system', 'content': system_prompt()},
            {'role': 'user', 'content': up}]
    # v3.1.8 要求 HTML 正文,输出体积是老配方的 2~3 倍 —— 上限不抬会被截断,
    # 表现是 `missing fields after retry: ['expected_output', 'seed']`
    # (JSON 截在半截,解析不出这两个键)。0818 首跑 28 道就撞了。
    cap = 24000 if is_aligned_recipe(REALISM) else 12000
    _u = {}
    txt = _api.chat(MODEL, msgs, max_tokens=cap, temperature=0.7, usage_out=_u)
    _billed('main', _u)
    obj = _api.parse_json(txt, want='obj')
    if not obj or 'seed' not in obj or 'expected_output' not in obj:
        # 一次修复重试:加大输出额度 + 要求压缩世界(截断是主要死因)
        # Claude /messages rejects an empty text block.  A transient empty
        # first response must still be repairable, so only carry forward
        # non-empty model text.
        if txt and txt.strip():
            msgs.append({'role': 'assistant', 'content': txt[:6000]})
        # ⚠️ 这句重试指令必须**跟着配方走**。v3.1.7 那版写的是「每封正文 ≤80 词」——
        #    在 v3.1.8 下它和「body_html 2000~4000 字符」直接打架:要么继续失败,
        #    要么**重试成功但悄悄退回 v3.1.7 形状**(比失败还坏,因为没人会发现)。
        #    v3.1.8 的压缩方向是**砍数量,不砍长度**。
        if is_aligned_recipe(REALISM):
            fix = ('你的输出不是合法 JSON、被截断、或缺 seed/expected_output 字段。'
                   # ⚠️ 0819:主提示词已经写成 A3+B4~5 了,所以这里必须**再压一档**,
                   #    否则和主提示词一个量,救不回来。
                   '重新输出【完整】严格 JSON;为防截断请**再压一档数量**:'
                   'A 档(判定相关)压到 2 封、B 档(背景噪声)压到 3 封、日历 ≤3 个、'
                   '思路说明(world_story)≤120 字。'
                   '⛔ **两档的长度要求都不变** —— A 档仍要 2000~4000 字符,B 档仍要 500~1000。'
                   '宁可少几封,不许把正文写短。不要任何解释。')
        else:
            fix = ('你的输出不是合法 JSON、被截断、或缺 seed/expected_output 字段。'
                   '重新输出【完整】严格 JSON;为防截断请压缩:噪声邮件最多 4 封、每封正文 ≤80 词、'
                   '思路说明(world_story)≤150 字。不要任何解释。')
        msgs.append({'role': 'user', 'content': fix})
        _u = {}
        txt = _api.chat(MODEL, msgs, max_tokens=cap + 8000, temperature=0.3, usage_out=_u)
        _billed('repair', _u)
        obj = _api.parse_json(txt, want='obj')
    if not obj:
        raise ValueError('JSON parse failed twice')
    if 'seed' not in obj or 'expected_output' not in obj:
        raise ValueError(f"missing fields after retry: {sorted(set(('seed','expected_output')) - set(obj))}")
    # ⭐ 出口闸:A 档不够就定向返工一次(只加长,不动别的)
    if is_aligned_recipe(REALISM) and A_TIER_TARGET:
        have = count_a_tier(obj)
        if count_a_tier(obj) < A_TIER_TARGET:
            GATE_STATS['a_tier_short'] += 1
        # ⛔ 给两次机会:一次就放弃时,10 道里有 2 道留下 A 档为 0 的世界。
        for _ in range(2):
            have = count_a_tier(obj)
            if have >= A_TIER_TARGET:
                break
            if txt and txt.strip():
                msgs.append({'role': 'assistant', 'content': txt[:6000]})
            msgs.append({'role': 'user', 'content': LENGTHEN_INSTRUCTION.format(
                min_chars=A_TIER_MIN_CHARS, have=have, want=A_TIER_TARGET)})
            _u = {}
            txt = _api.chat(MODEL, msgs, max_tokens=cap + 8000, temperature=0.3, usage_out=_u)
            _billed('lengthen', _u)
            obj2 = _api.parse_json(txt, want='obj')
            if not (obj2 and obj2.get('seed')):
                GATE_STATS['lengthen_reject_broken'] += 1
                continue
            if not merge_lengthened(obj, obj2) or count_a_tier(obj) <= have:
                GATE_STATS['lengthen_no_gain'] += 1
        if count_a_tier(obj) >= A_TIER_TARGET:
            GATE_STATS['lengthened'] += 1
        elif GATE_STATS['a_tier_short']:
            GATE_STATS['still_short'] += 1

    if case['trigger'] == 'event' and not obj.get('trigger_event'):
        # 针对性补一轮:只要 trigger_event
        if txt and txt.strip():
            msgs.append({'role': 'assistant', 'content': txt[:6000]})
        msgs.append({'role': 'user', 'content':
                     '你漏了 trigger_event 字段(这是 event 题,必须有)。基于你刚才的 seed,'
                     '重新输出【完整】JSON(world_story/seed/trigger_event/expected_output/polarity 全带),'
                     'trigger_event 的触发邮件必须与 seed.emails 里的第一封触发邮件一致。只输出 JSON。'})
        _u = {}
        txt = _api.chat(MODEL, msgs, max_tokens=12000, temperature=0.3, usage_out=_u)
        _billed('trigger', _u)
        obj2 = _api.parse_json(txt, want='obj')
        if obj2 and obj2.get('trigger_event') and obj2.get('seed'):
            obj = obj2
        else:
            raise ValueError('event case missing trigger_event')
    question = render_question(case, obj)
    return {**case, 'llm': obj, 'question': question}


def main():
    global OUT, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--pool', required=True, help='阶段1产物 source_pool.jsonl')
    ap.add_argument('--out', required=True, help='起草稿输出目录 cases_raw')
    ap.add_argument('--only')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--model', default=MODEL, help='起草老师;0806 起 compass 老 key 掐了 claude,新 key 走 /messages')
    ap.add_argument('--realism', default='', choices=sorted(REALISM_CANON),
                    help='世界配方档位。**正名 v3.1.7 / v3.1.8** —— 这几批数据都是从 '
                         'v3.1 那份语料来的,所以按 v3.1.x 编号。\n'
                         '  v3.1.7 = 老配方(默认;旧写法是空串或 v7,别的线在用,不许动)\n'
                         '  v3.1.8 = 对齐线上 trace 的新配方(参会状态/HTML 正文/'
                         '重复与全天日程/附件/多消息线程/日历触发形状/user_notes;旧写法 v8)\n'
                         '⛔ 两档造出来的世界难度不同,教材不能混着训、分数也不可比。\n'
                         'v7/v8/空串仍然认(弃用旧名),但会打提示并在 manifest 里记正名。')
    ap.add_argument('--rsvp', action='store_true',
                    help='(v3.1.8 起)要求世界给带 attendees 的会议写参会状态 '
                         'self_response / responseStatus。⚠️ 默认关闭 —— 打开会改变造世界配方,'
                         '新批次和旧批次不可比;lint 侧要配 lint_cases.py --require-rsvp')
    ap.add_argument('--max-per-rule', type=int, default=0,
                    help='同一条规则最多造几道题(0=不限)。⭐ 0820 起新批次一律 30。\n'
                         '理由:最大的一条规则(葡语晨报模板)独占 2025 道里的 208 道\n'
                         '= 10.3%%,前 20 条吃掉 47.5%% —— 题面完全一样,钱和梯度\n'
                         '白压在少数模板上。设 30 之后题数 2025 → 1637。\n'
                         '⛔ 只对 record 模式生效;已经造好的稿子不删(装配扫目录)。')
    ap.add_argument('--mailboxes', default='',
                    help='build_user_mailboxes.py 产物;提供后只注入 cutoff 前的同用户观测')
    ap.add_argument('--allow-eval-collisions', action='store_true',
                    help='仅调试用:允许 custom 规则与评测题 Jaccard>=0.70')
    ap.add_argument('--max-new', type=int, default=0,
                    help='本次最多新造几个就**干净收尾**(打完整账本再退出)。\n'
                         '⭐ 长跑先跑一小批看账本再决定优化,就用它 —— 直接 kill 会把\n'
                         '   内存里的计费账本一起丢掉。0 = 不限。')
    ap.add_argument('--reforge', action='store_true',
                    help='允许 --only 删掉目录里过半的已有稿重造。⛔ 只有确实要整批回炉时才加 —— '
                         '「补跑失败的几条」不需要它(只传失败的 id 即可)')
    ap.add_argument('--accept-stale', action='store_true',
                    help='**知情地接受混批**:旧稿的 plan_fingerprint 就地更新到本次口径,\n'
                         '不删不重造。⭐ 提示词微调后想接着往同一个目录里补,用这个。\n'
                         '⚠️ 只有当**语义指纹里除 prompt_sha256 之外的部分都没变**时才允许 ——\n'
                         '  源池/邮箱/模型/配方档位任何一个变了都会拒绝(那才是真的两批不同东西)。\n'
                         '混批本身可审计:每个世界稿自带 generation.prompt_sha256,事后按它分组即可。\n'
                         '⛔ 别和 --overwrite-stale 混:那个是**删掉重造**(贵),这个是**认下来**。')
    ap.add_argument('--overwrite-stale', action='store_true',
                    help='移除并重生成 fingerprint 与本次计划不一致的旧 case')
    ap.add_argument('--plan-only', action='store_true',
                    help='只生成并校验 plan manifest，不调用起草模型')
    ap.add_argument('--case-prefix', default='rq1',
                    help='case id 前缀；新远端批次必须换，避免与历史全局 item id 冲突')
    ap.add_argument('--plan-mode', default='all', choices=['all', 'record', 'engineered'],
                    help='all=源记录重建+额外边界扩增；record=严格一条有效源记录一个世界；'
                         'engineered=只要额外设计题。约 2k 源数据批应显式用 record。')
    args = ap.parse_args()

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.case_prefix):
        raise SystemExit('⛔ --case-prefix 只允许字母、数字、下划线和连字符')

    MODEL = args.model
    global RSVP_ENABLED, REALISM
    _raw_realism = args.realism or ('v8' if args.rsvp else '')
    REALISM = normalize_realism(_raw_realism)
    RSVP_ENABLED = is_aligned_recipe(REALISM)          # 兼容旧名
    if args.rsvp and not args.realism:
        print('ℹ️ --rsvp 是旧名,已当作 --realism v3.1.8 处理', flush=True)
    if _raw_realism != REALISM:
        print(f'ℹ️ --realism {_raw_realism!r} 是弃用旧名,已归到正名 {REALISM}'
              f'(manifest 里记正名)', flush=True)
    if is_aligned_recipe(REALISM):
        print(f'⚠️ 配方档位 --realism {REALISM}:本批要求「对齐线上」的种子形态,'
              f'与 v3.1.7 老批次配方不同 —— **两档的教材不能混着训,分数也不可比**', flush=True)
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    pool = [json.loads(l) for l in open(args.pool)]
    pool_sha256 = sha256_file(args.pool)
    generator_sha256 = sha256_file(Path(__file__))
    prompt_sha256 = hashlib.sha256(canonical_json({
        'system': system_prompt(), 'drive_clause': DRIVE_CLAUSE,
        'variants': VARIANTS, 'cron_tick': CRON_TICK,
        'event_tick': EVENT_TICK, 'polarity': POL_MAP,
        'eval_guidance_patterns': {k: v.pattern for k, v in RULE_GUIDANCE_PATTERNS.items()},
        'silent_empty_patterns': [v.pattern for v in SILENT_EMPTY_PATTERNS],
        'report_empty_patterns': [v.pattern for v in REPORT_EMPTY_PATTERNS],
        **({'family_variants': FAMILY_VARIANTS,
            'family_lang_pin': FAMILY_LANG_PIN,
            'family_explicit_clause': FAMILY_EXPLICIT_CLAUSE,
            'family_explicit_draft': FAMILY_EXPLICIT_DRAFT} if is_family_recipe(REALISM) else {}),
    }).encode()).hexdigest()
    mailbox_sha256 = sha256_file(args.mailboxes) if args.mailboxes else None
    # ⭐ 批次的**身份**是这两个指纹,不是批次名、也不是版本号。
    #   同一个配方档位(v3.1.8)在补漏洞的过程中提示词改过好几次 —— 0818 的 28 个世界、
    #   0819 的 13 个、0819 下午的 30 个,三批 realism 都是 v3.1.8 而 prompt_sha256 各不相同。
    #   **补漏洞不升配方版本**(见 data/autotask_rq3w/README.md),所以要指认「这批到底是
    #   什么」只能看指纹。打进日志,免得事后只能靠 manifest 翻。
    print(f'配方={REALISM}  提示词指纹={prompt_sha256[:16]}…  '
          f'生成器指纹={generator_sha256[:16]}…', flush=True)
    mailboxes = load_mailboxes(args.mailboxes)
    print(f'源池={args.pool}({len(pool)} 规则) 出={OUT}', flush=True)
    plan = build_plan(pool, allow_eval_collisions=args.allow_eval_collisions,
                      case_prefix=args.case_prefix)
    plan = select_plan_mode(plan, args.plan_mode)
    plan, capped = cap_per_rule(plan, args.max_per_rule)
    if capped:
        total = sum(capped.values())
        print(f'ℹ️ --max-per-rule {args.max_per_rule}:{len(capped)} 条规则超限,'
              f'共砍掉 {total} 道题(剩 {len(plan)} 道)。被砍最多的:', flush=True)
        for rule_key, n in sorted(capped.items(), key=lambda x: -x[1])[:5]:
            print(f'     {rule_key} −{n}', flush=True)
        print('   ⚠️ 目录里已经造好、但这次落在上限之外的稿子**不会被删** ——'
              '装配那步是扫目录的,它们照样会进最终数据集。', flush=True)
    if not plan:
        raise SystemExit(f'⛔ --plan-mode {args.plan_mode} 选完是空计划')
    generation = {
        'model': MODEL,
        # 只在开了思考时才记字段 —— 平时不加键,免得老批次的逐题指纹整体变掉(会把已有稿判 stale)
        **({'thinking': os.environ['V6_THINKING'].strip()}
           if os.environ.get('V6_THINKING', '').strip() else {}),
        'case_prefix': args.case_prefix,
        'plan_mode': args.plan_mode,
        'pool_sha256': pool_sha256,
        'mailbox_sha256': mailbox_sha256,
        'prompt_sha256': prompt_sha256,
        'generator_sha256': generator_sha256,
        'realism': REALISM,            # 正名(v3.1.7 / v3.1.8);老批次记的是 ''/'v3.1.8'
        'rsvp': RSVP_ENABLED,          # 兼容旧字段;等价于 realism == 'v3.1.8'
    }
    # ⛔ **`generator_sha256` 不进逐条指纹**(0819 改)。
    #   它进来的话,**改一行注释就没法续跑** —— 0819 实案:全量跑到 165/2025 时我给
    #   gen_worlds 加了计费仪表(没碰提示词),重启就被判「165 个旧 case 指纹不一致」,
    #   要么 --overwrite-stale 把 165 个重造,要么换目录从头来。十小时量级的批次上
    #   这条闸的代价远大于它挡住的风险。
    #   真正会改变世界内容的东西各有自己的 sha,**全都还在指纹里**:
    #   pool / mailbox / **prompt** / model / realism / case 本身的字段。
    #   代码指纹仍然记进 manifest(下面 generator_shas),只是**不判死**,改成打警告。
    fp_generation = {k: v for k, v in generation.items() if k != 'generator_sha256'}
    for case in plan:
        case['generation'] = generation
        case['plan_fingerprint'] = hashlib.sha256(canonical_json({
            'case': {k: case.get(k) for k in (
                'case_id', 'rule_key', 'variant', 'polarity', 'mode', 'rec_idx',
                'source_tick_index', 'cutoff_at', 'user_key', 'tz_offset', 'tz_iana')},
            'generation': fp_generation,
        }).encode()).hexdigest()
    full_plan = list(plan)
    plan_quality = defaultdict(int)
    for case in full_plan:
        plan_quality['cases'] += 1
        plan_quality['missing_user_key'] += not bool(case.get('user_key'))
        plan_quality['missing_cutoff'] += not bool(case.get('cutoff_at'))
        plan_quality['judge_target_mismatch'] += case.get('judge_target_aligned') is False
        try:
            probe = datetime(2026, 8, 3, 12, 0, tzinfo=ZoneInfo(case['tz_iana']))
            seconds = int(probe.utcoffset().total_seconds())
            sign = '+' if seconds >= 0 else '-'
            seconds = abs(seconds)
            actual = f'{sign}{seconds // 3600:02d}:{seconds % 3600 // 60:02d}'
            plan_quality['timezone_iana_offset_mismatch'] += actual != case.get('tz_offset')
        except Exception:
            plan_quality['timezone_iana_offset_mismatch'] += 1
        history = mailbox_snapshot_material(
            mailboxes.get(case.get('user_key')), case.get('cutoff_at'), case['source_traces'][0])
        if history:
            plan_quality['cases_with_prior_mailbox_context'] += 1
            plan_quality['prior_mailbox_items'] += history.count('[同用户历史邮箱观测')
        calendar = calendar_snapshot_material(
            mailboxes.get(case.get('user_key')), case.get('cutoff_at'))
        if calendar:
            plan_quality['cases_with_calendar_context'] += 1
            plan_quality['calendar_context_items'] += calendar.count('\n- ') + 1
        drive = drive_snapshot_material(
            mailboxes.get(case.get('user_key')), case.get('cutoff_at'))
        if drive:
            plan_quality['cases_with_drive_context'] += 1
            plan_quality['drive_context_items'] += drive.count('\n- ') + 1
    print(f'total plan: {len(plan)} cases')
    print(Counter((c['arm'], c['polarity']) for c in plan).most_common())
    print('plan quality:', dict(plan_quality))

    if args.only:
        if args.only.startswith('@'):
            wanted = set(Path(args.only[1:]).read_text().split())
            wanted = {w.replace('.json', '') for w in wanted}
        else:
            wanted = set(args.only.split(','))
        plan = [c for c in plan if c['case_id'] in wanted]
        # ⛔ **`--only` 会把命中的稿删掉重造**(回炉语义)。0819 我因此把一批已经造好的
        #    28 个世界整批删了:想「补跑那 2 条失败的」,却把**完整的 30 条 id 列表**
        #    传了进来。全量 2025 条上犯这个错 = 删掉约 1850 个世界重造 = **再烧 13 小时**。
        #    ⇒ **补跑只传失败的那几条 id**(它们本来就不存在,删除是空操作);
        #      传全表 = 回炉全批。下面这道闸就是拦「一不小心回炉了大半个目录」。
        existing = [c for c in plan if (OUT / f"{c['case_id']}.json").exists()]
        n_dir = len(list(OUT.glob('*.json'))) if OUT.exists() else 0
        if existing:
            print(f'⚠️ --only 命中 {len(existing)} 个**已有稿**的 case,它们会被删掉重造'
                  f'(目录里现共 {n_dir} 个)', flush=True)
        if n_dir and len(existing) * 2 > n_dir and not args.reforge:
            sys.exit(
                f'⛔ --only 会删掉目录里 {len(existing)}/{n_dir} 个已有稿(过半)。\n'
                f'   要「补跑失败的那几条」→ --only 只传**失败的 id**,别传全表;\n'
                f'   确实要整批回炉重造 → 显式加 --reforge。\n'
                f'   (0819 实案:传了全表,28 个造好的世界被整批删掉重来。)')
        for c in plan:
            (OUT / f"{c['case_id']}.json").unlink(missing_ok=True)
    elif args.smoke:
        seen, sm = set(), []
        for c in plan:
            k = (c['arm'], c['trigger'])
            k2 = (c['arm'], c['polarity'])
            if k not in seen or k2 not in seen:
                seen.add(k); seen.add(k2); sm.append(c)
        plan = sm[:10]
        print('smoke cases:', [c['case_id'] for c in plan])

    manifest_path = OUT.parent / f'{OUT.name}_plan_manifest.json'
    # `selected_cases` 是 lint 的 E9 拿来对「出稿数够不够」的期望值。
    #
    # ⚠️ 回炉(`--only @relint.txt`)时它会被写成子集大小,于是下一轮质检报
    #    `E9 incomplete generation files=146 expected=3` —— 一个纯假的错,
    #    还得再跑一次 `--plan-only` 才能修回来。手册里那个「回炉 → 再质检」
    #    的循环每转一圈都会踩(0815 实测踩到)。
    #
    #    修法:回炉是**往已有目录里补几个**,目录该有的总数不变,所以沿用
    #    原 manifest 的期望值。只有当原 manifest 认的计划跟这次不是同一个
    #    (指纹对不上)才重新计数。
    selected = len(plan)
    # ⚠️ `--only` 有两种完全不同的用法,期望出稿数正好相反:
    #    ① **回炉**:往一个已经有稿的目录里补几个 —— 目录该有的总数不变,沿用旧值;
    #    ② **故意的子集**(试水卷):打到一个**空目录**上 —— 期望数就是这次这几条。
    #    0819 踩到②:先跑了 --plan-only(写下 selected=2025)、再 --only 造 13 条,
    #    于是 lint 报 `E9 incomplete generation files=13 expected=2025`,
    #    一个纯假的错,还把下游的装配回执闸一起堵死了。
    #    判据用「跑之前目录里有没有稿」——自证、不用人记。
    had_drafts = OUT.exists() and any(OUT.glob('*.json'))
    if args.only and manifest_path.exists() and had_drafts:
        try:
            old = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            old = {}
        same_plan = (old.get('case_fingerprints') or {}) == {
            c['case_id']: c['plan_fingerprint'] for c in full_plan}
        if same_plan and isinstance(old.get('selected_cases'), int):
            selected = old['selected_cases']
            print(f'回炉(目录里已有稿):沿用原 manifest 的 selected_cases={selected}'
                  f'(不按本次子集 {len(plan)} 覆盖)')
    elif args.only:
        print(f'子集卷(目录原本是空的):selected_cases={selected} —— 期望出稿数就是这几条')

    manifest = {
        'schema_version': 2,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'out_dir': str(OUT.resolve()),
        'generation': generation,
        # ⚠️ 上限不进逐题指纹 —— 它改的是"有哪些题",不是"每道题怎么造"。
        #    进了指纹会把已有稿子全判成作废。
        'max_per_rule': args.max_per_rule,
        'capped_rules': capped,
        'quality': dict(plan_quality),
        'planned_cases': len(full_plan),
        'selected_cases': selected,
        'case_fingerprints': {c['case_id']: c['plan_fingerprint'] for c in full_plan},
    }
    mtmp = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    mtmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    mtmp.replace(manifest_path)
    if args.plan_only:
        print(f'plan-only: {len(plan)} cases; manifest={manifest_path}')
        return

    # 代码指纹变了:不判死,但要留痕 —— 事后能查"这批世界是几版代码混着造的"
    prev_shas = []
    if manifest_path.exists():
        try:
            prev_shas = list(json.loads(manifest_path.read_text()).get('generator_shas') or [])
        except (OSError, json.JSONDecodeError):
            prev_shas = []
    if generator_sha256 not in prev_shas:
        prev_shas.append(generator_sha256)
    if len(prev_shas) > 1 and any(OUT.glob('*.json')):
        print(f'⚠️ 本目录的世界是 {len(prev_shas)} 版 gen_worlds 代码造出来的'
              f'(提示词/源池/模型都没变,所以不判死;代码指纹全记在 manifest 的 generator_shas)',
              flush=True)
    try:
        _m = json.loads(manifest_path.read_text())
        _m['generator_shas'] = prev_shas
        manifest_path.write_text(json.dumps(_m, ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError):
        pass

    stale = []
    for c in plan:
        path = OUT / f"{c['case_id']}.json"
        if not path.exists():
            continue
        try:
            old = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            stale.append(path)
            continue
        if old.get('plan_fingerprint') != c['plan_fingerprint']:
            stale.append(path)
    if stale and args.accept_stale:
        # 只允许「差别仅在 prompt_sha256」这一种。别的都是真的两批不同东西。
        by_case = {c['case_id']: c for c in plan}
        diffs, patched = set(), 0
        for path in stale:
            try:
                old_world = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            og = old_world.get('generation') or {}
            for key in ('pool_sha256', 'mailbox_sha256', 'model', 'realism',
                        'case_prefix', 'plan_mode'):
                if og.get(key) != generation.get(key):
                    diffs.add(key)
        if diffs:
            sys.exit(f'⛔ --accept-stale 只认「只有提示词变了」这一种,但这次 {sorted(diffs)} '
                     f'也变了 —— 那是两批不同的东西,别混。换个输出目录。')
        for path in stale:
            try:
                old_world = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            want = by_case.get(path.stem)
            if not want:
                continue
            old_world['plan_fingerprint'] = want['plan_fingerprint']
            path.write_text(json.dumps(old_world, ensure_ascii=False, indent=1))
            patched += 1
        old_shas = {(json.loads(p.read_text()).get('generation') or {}).get('prompt_sha256')
                    for p in list(OUT.glob('*.json'))[:400]}
        print(f'ℹ️ --accept-stale:认下 {patched} 个旧稿(只有提示词变了)。'
              f'本目录现有提示词版本 {len([x for x in old_shas if x])} 个 —— '
              f'每个世界自带 generation.prompt_sha256,事后按它分组', flush=True)
        stale = []
    if stale and not args.overwrite_stale:
        print(f'❌ {len(stale)} 个旧 case 与本次 source/prompt/model fingerprint 不一致。')
        print('  · 只是提示词微调、想接着往同一目录补 → --accept-stale(认下来,不重造)')
        print('  · 确实要按新配方重造这些 → --overwrite-stale(删掉重跑,贵)')
        print('  · 两批本来就不该混 → 换个输出目录')
        sys.exit(2)
    if stale:
        for path in stale:
            path.unlink()
        print(f'已移除 {len(stale)} 个 stale case，准备重生成。')

    todo = [c for c in plan if not (OUT / f"{c['case_id']}.json").exists()]
    print(f'todo: {len(todo)} (skip {len(plan)-len(todo)} done)')
    if not todo:
        print(f'plan complete: {len(plan)}/{len(plan)}')
        return

    pool_by_key = {g['rule_key']: g for g in pool}

    # 标注需要网盘/表格素材的题(该规则的真实轨迹用过 drive/sheets/docs 工具)
    DRIVE_PAT = ('drive', 'sheets', 'docs')
    drive_rules = set()
    for g in pool:
        for r in g['records']:
            if any(any(p in str(t) for p in DRIVE_PAT) for t in (r.get('tools_used') or [])):
                drive_rules.add(g['rule_key'])
                break
    def rec_uses_drive(g, ri):
        r = g['records'][ri]
        return any(any(p in str(t) for p in DRIVE_PAT) for t in (r.get('tools_used') or []))

    for c in todo:
        if c.get('mode') == 'record':
            c['needs_drive'] = rec_uses_drive(pool_by_key[c['rule_key']], c['rec_idx'])
        else:
            c['needs_drive'] = c['rule_key'] in drive_rules

    def mat(c):
        g = pool_by_key[c['rule_key']]
        material = (record_material(g, c['rec_idx']) if c.get('mode') == 'record'
                    else material_digest(g, c['rec_idx']))
        history = mailbox_snapshot_material(
            mailboxes.get(c.get('user_key')), c.get('cutoff_at'), c['source_traces'][0])
        if history:
            material += ('\n\n[同一用户在本次执行前的去重邮箱观测；只能补背景/噪声，'
                         '不得覆盖目标 tick 的真实结论]\n' + history)
        calendar = calendar_snapshot_material(
            mailboxes.get(c.get('user_key')), c.get('cutoff_at'))
        if calendar:
            material += (
                '\n\n[这个用户日历的真实形态参照 —— **只照抄形态,不照抄内容**]\n'
                '用途:让 seed.calendar_events 的会议规模/全天比例/周期会议密度/'
                '有没有地点说明,贴近这个用户真实的样子。\n'
                '⛔ 标题只作措辞风格参照,**人名一律换成虚构的**;\n'
                '⛔ 这里不给日期是故意的 —— 时间按 3.7 自己排,且一条都不许排在过去。\n'
                + calendar)
        drive = drive_snapshot_material(
            mailboxes.get(c.get('user_key')), c.get('cutoff_at'))
        if drive:
            material += (
                '\n\n[这个用户网盘/表格的真实形态参照 —— **只照抄形态,不照抄内容**]\n'
                '用途:让 seed.drive_files 的**规模感、列名风格、文件命名风格**贴近真实。\n'
                '⛔ 真表动辄上千行,你**不要**照着写 —— 按 3.13 给 20~40 行的代表性切片,'
                '但列数要够、列名要像。\n'
                '⛔ 文件名只作**命名风格**参照,里面的人名/项目名一律换成虚构的;'
                '列名怎么写看 3.13 给的例子。\n'
                + drive)
        return material

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_case, c, mat(c)): c for c in todo}
        for f in as_completed(futs):
            c = futs[f]
            try:
                res = f.result()
                path = OUT / f"{c['case_id']}.json"
                tmp = path.with_suffix(path.suffix + '.tmp')
                tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
                tmp.replace(path)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {c['case_id']}: {e}", flush=True)
            if (ok + fail) % 20 == 0:
                print(f'progress {ok+fail}/{len(todo)} ok={ok} fail={fail}', flush=True)
                _snapshot_billing(OUT)          # 账本随跑随落,kill 也不丢
            if args.max_new and ok >= args.max_new:
                print(f'⏹ 到达 --max-new {args.max_new},停止收新结果并收尾'
                      f'(已提交但没跑完的会被丢弃,下次续跑会补上)', flush=True)
                for pending in futs:
                    pending.cancel()
                break
    missing = [c['case_id'] for c in plan if not (OUT / f"{c['case_id']}.json").exists()]
    rep = billing_report()
    t = rep['total']
    print(f'done ok={ok} fail={fail} missing={len(missing)}; api stats={_api.stats()}')
    if GATE_STATS:
        rej = {k[9:]: v for k, v in GATE_STATS.items() if k.startswith('lengthen_')}
        print(f"A 档出口闸:{GATE_STATS['a_tier_short']} 个世界不足 {A_TIER_TARGET} 封 → "
              f"补齐 {GATE_STATS['lengthened']} · 仍不足 {GATE_STATS['still_short']}"
              + (f" · 被拒的补写 {rej}" if rej else ''), flush=True)
    print(f'计费:{t["n"]} 次调用 · 入 {t["in"]:,} 出 {t["out"]:,} · '
          f'**${t["usd"]:.2f}**' + (f' · 每个成稿 ${t["usd"]/ok:.3f}' if ok else ''))
    for site in ('main', 'repair', 'trigger'):
        r = CALL_STATS.get(site)
        if r:
            print(f'   {site:8s} {r["n"]:5d} 次 · ${r["usd"]:7.2f} '
                  f'({100*r["usd"]/max(t["usd"],1e-9):.0f}%) · 出 {r["out"]:,} tok')
    try:
        (OUT.parent / f'{OUT.name}_billing.json').write_text(
            json.dumps(rep, ensure_ascii=False, indent=2))
    except OSError:
        pass
    if fail or missing:
        print(f'❌ plan incomplete: generated={len(plan)-len(missing)}/{len(plan)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
