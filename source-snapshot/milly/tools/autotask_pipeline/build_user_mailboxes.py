#!/usr/bin/env python3
"""把 rule-centric source_pool 转成 user-centric 邮箱 + 日历观测仓库。

这里构建的是“被工具看见过的邮件/日程观测”，不是完整 Gmail / Calendar 镜像：
  * user_key 来自 extract_source.py 对 meta.user_id 的稳定匿名映射；
  * message_id 去重，thread_id 只用于会话归组；
  * 每个稳定对象保留按 observed_at 排序的版本，物化 case 时只能取 cutoff 之前
    的最后一版；
  * 搜索未返回某邮件不代表删除，因此本脚本只做正向观测合并。

日历观测(0820 新增)另有两条硬约束：
  * **只出形态、不出身份**：存 summary(已抹掉邮箱)、是否全天、时长、参会人数、
    自己的回复状态、是否周期实例、有没有地点/说明；**不存任何参会者地址**；
  * 源数据把每条 tool_response 截到 2000 字符,中间层常常**不是**合法 JSON,
    所以解析必须能在截断的字符串上继续剥(见 ``peel``),不能只靠 ``json.loads``。

输出含真实轨迹里的邮件片段，只能留在 raw_internal 数据域。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

from pipeline_common import canonical_json, sha256_file


READ_TOOLS = {
    'google_gmail_search', 'google_gmail_messages_search',
    'google_gmail_thread_get', 'google_gmail_get',
}
# 只收**读**日历的工具。google_calendar_create 是写动作,收进来会把"用户当时新建的
# 日程"和"用户本来就有的日程"混成一谈。
CALENDAR_TOOLS = {'google_calendar_events', 'google_calendar_get'}
THREAD_TEXT_RE = re.compile(r'(?im)^Thread ID:\s*([a-f0-9]{12,32})\s*$')
# 源数据的截断尾巴,形如 "\n[...truncated, 2482 chars total]"
TRUNC_RE = re.compile(r'\n\[\.\.\.truncated,\s*\d+\s*chars total\]\s*$')
# Google 会在时间后面挂状态/说明尾巴:"... +0700 [upcoming]"、
# "2026-07-20 (all-day date; not timezone-shifted)"
WHEN_SUFFIX_RE = re.compile(r'\s*(?:\[[^\]]*\]|\([^)]*\))\s*$')
EVENTS_KEY_RE = re.compile(r'"events"\s*:\s*\[')
EMPTY_EVENTS_RE = re.compile(r'"events"\s*:\s*\[\s*\]')
EMAIL_RE = re.compile(r'[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}', re.I)
DESC_HTML_RE = re.compile(r'<(?:br|b|i|u|a|div|p|ul|ol|li|span)\b', re.I)
# 只收**读**网盘/表格的工具。append/update 是写动作,收进来会把"用户当时写进去的"
# 和"表里本来就有的"混成一谈(和日历那边不收 calendar_create 同理)。
DRIVE_TOOLS = {'google_drive_search', 'google_drive_get', 'google_drive_download',
               'google_sheets_get', 'google_sheets_metadata', 'google_docs_cat'}
FILES_KEY_RE = re.compile(r'"files"\s*:\s*\[')
# 形如 "range": "'[SM/CCS] Sentiment Tracker'!A1:Z50492"
SHEET_RANGE_RE = re.compile(
    r'"range"\s*:\s*"(?:\'?([^\'!"]{0,80})\'?!)?([A-Z]{1,3})(\d+):([A-Z]{1,3})(\d+)"')
GRID_RE = re.compile(r'"columnCount"\s*:\s*(\d+)\s*,\s*"rowCount"\s*:\s*(\d+)')
SHEET_TITLE_RE = re.compile(r'"title"\s*:\s*"([^"]{1,60})"')
DOC_ID_RE = re.compile(r'"(?:spreadsheetId|fileId|documentId)"\s*:\s*"([\w-]{10,})"')
# 真实文件名常带日期戳(``..._2026-07-02_v1.csv``)。⛔ 硬约束 2 不许绝对日期进提示词
# —— 模型会连日期一起照抄,把世界钉死在过去某一天。形状(带日期戳)要留,日期本身打码。
NAME_DATE_RE = re.compile(r'\d{4}[-_/.]\d{1,2}[-_/.]\d{1,2}|\d{1,2}[-_/.]\d{1,2}[-_/.]\d{4}')


def decode_json(value):
    """解开 tool_response 外壳及 result 的一到两层 JSON 字符串。"""
    cur = value
    for _ in range(3):
        if not isinstance(cur, str):
            break
        try:
            cur = json.loads(cur)
        except json.JSONDecodeError:
            break
    if isinstance(cur, dict) and 'result' in cur:
        cur = cur['result']
        for _ in range(2):
            if not isinstance(cur, str):
                break
            try:
                cur = json.loads(cur)
            except json.JSONDecodeError:
                break
    return cur


def call_name(raw):
    """取工具真名。⛔ 必须拆 ``proxy_tool`` —— 真名在 ``arguments.tool_name``,
    线上过半调用走这层包装,按外层函数名统计会整片漏掉。
    这个池子里 gmail 读工具恰好都没走包装,但别指望下一批也是。"""
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get('name')
    if name != 'proxy_tool':
        return name
    arguments = obj.get('arguments') or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return name
    return arguments.get('tool_name') if isinstance(arguments, dict) else name


def peel(text, max_layers=8):
    """把工具返回逐层剥到能看见明文 JSON 为止,**允许中途是残缺 JSON**。

    日历返回长这样:``{"result": "{\\"result\\": \\"{\\\\\\"content\\\\\\": [...]}"}"``,
    而源数据把每条返回截到 2000 字符 —— 于是中间层 ``json.loads`` 必然失败。
    这时改用"手工退一层转义"继续剥,截断的尾巴交给 ``iter_calendar_events`` 丢掉。
    """
    cur = text
    for _ in range(max_layers):
        if not isinstance(cur, str):
            if isinstance(cur, dict):
                if 'result' in cur:
                    cur = cur['result']
                    continue
                if isinstance(cur.get('content'), list):
                    cur = '\n'.join(str(x.get('text', '')) if isinstance(x, dict) else str(x)
                                    for x in cur['content'])
                    continue
                return json.dumps(cur, ensure_ascii=False)
            if isinstance(cur, list):
                cur = '\n'.join(json.dumps(x, ensure_ascii=False) for x in cur)
                continue
            return str(cur)
        stripped = TRUNC_RE.sub('', cur)
        try:
            cur = json.loads(stripped)
            continue
        except (json.JSONDecodeError, TypeError):
            pass
        if '\\"' not in stripped:
            return stripped
        # 退一层转义。先把 \\ 藏成 NUL,免得下面两步把它和后面的字符连读。
        cur = (stripped.replace('\\\\', '\x00').replace('\\"', '"')
               .replace('\\n', '\n').replace('\\t', '\t').replace('\\/', '/')
               .replace('\x00', '\\'))
    return cur if isinstance(cur, str) else json.dumps(cur, ensure_ascii=False)


def iter_array_items(text, key_re, opener='{'):
    """从(可能被截断的)文本里,逐个抠出 ``"<key>": [ ... ]`` 数组的元素。

    不整体 ``json.loads`` 那个数组 —— 截断会让整个数组作废,而前面几条其实是完整的。
    改成逐个括号配对:配平一个就收一个,最后那个残缺的自然收不进来。
    ``opener`` 决定收的是对象(``{``,如 events/files)还是数组(``[``,如 sheets 的 values)。
    """
    closer = '}' if opener == '{' else ']'
    out = []
    for match in key_re.finditer(text):
        index, depth, start = match.end(), 0, None
        in_string = escaped = False
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == opener:
                if depth == 0:
                    start = index
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        out.append(json.loads(text[start:index + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
                elif depth < 0:
                    break
            elif char == ']' and depth == 0 and opener == '{':
                break
            index += 1
    return out


def iter_calendar_events(text):
    """从(可能被截断的)文本里逐个抠出 ``events`` 数组的元素。"""
    return [x for x in iter_array_items(text, EVENTS_KEY_RE, '{') if isinstance(x, dict)]


def parse_when(value):
    """解析 Google 给的三种时间写法,返回 (datetime, 'date'|'datetime')。

    实测三种都出现过:
      ``Tue, 14 Jul 2026 15:00:00 +0700 [upcoming]`` / ``2026-07-14`` / ISO-8601。
    """
    if not isinstance(value, str) or not value.strip():
        return None, None
    raw = WHEN_SUFFIX_RE.sub('', value.strip())
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', raw):
        return datetime.fromisoformat(raw + 'T00:00:00'), 'date'
    try:
        return parsedate_to_datetime(raw), 'datetime'
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return datetime.fromisoformat(raw.replace('Z', '+00:00')), 'datetime'
    except ValueError:
        return None, None


def _when_value(node):
    if isinstance(node, dict):
        return node.get('dateTime') or node.get('date')
    return node if isinstance(node, str) else None


def calendar_fields(event):
    """只留形态,不留身份。⛔ 一个参会者地址都不许进来 —— 这份材料会被拼进
    造世界的提示词,地址进去模型就会照抄,而硬约束 1 要求未点名的人一律重造。"""
    start_value, end_value = _when_value(event.get('start')), _when_value(event.get('end'))
    start_dt, start_kind = parse_when(start_value)
    end_dt, _ = parse_when(end_value)
    start_node = event.get('start')
    all_day = bool(start_kind == 'date' or (isinstance(start_node, dict)
                                            and 'date' in start_node
                                            and 'dateTime' not in start_node))
    duration = None
    # 全天事件的"时长"只是天数,拼进提示词是噪声(会写成"全天 · 1440 分钟"),不留。
    if not all_day and start_dt and end_dt and start_dt.tzinfo == end_dt.tzinfo:
        minutes = int((end_dt - start_dt).total_seconds() // 60)
        if 0 <= minutes <= 60 * 24 * 14:
            duration = minutes
    attendees = event.get('attendees')
    attendees = attendees if isinstance(attendees, list) else []
    self_response = next((x.get('responseStatus') for x in attendees
                          if isinstance(x, dict) and x.get('self')), None)
    organizer = event.get('organizer')
    fields = {
        'summary': EMAIL_RE.sub('<addr>', str(event.get('summary') or ''))[:200],
        'all_day': all_day,
        'start_at': start_dt.isoformat() if start_dt else None,
        'duration_min': duration,
        'recurring': bool(event.get('recurringEventId')),
        'status': event.get('status'),
        'event_type': event.get('eventType'),
        'n_attendees': len(attendees),
        'n_resource_attendees': sum(1 for x in attendees
                                    if isinstance(x, dict) and x.get('resource')),
        'self_response': self_response,
        'organizer_self': bool(isinstance(organizer, dict) and organizer.get('self')),
        'has_location': bool(event.get('location')),
        'has_description': bool(event.get('description')),
    }
    # ⛔ 说明的**正文**一个字都不带出去:实测 86 条真实说明里,19 条含 Google 文档
    # 直链、10 条含 Zoom 个人会议室链接,还有大量真人名字。只带三个派生特征 ——
    # 足够让模型知道"说明该写多长、要不要带格式和链接",又不泄露任何内容。
    description = event.get('description')
    if isinstance(description, str) and description.strip():
        fields['desc_len'] = len(description)
        fields['desc_has_html'] = bool(DESC_HTML_RE.search(description))
        fields['desc_has_link'] = 'http://' in description or 'https://' in description
    return {k: v for k, v in fields.items() if v not in (None, '')}


def _col_index(letters):
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - 64
    return value


# ⛔⛔ **不抓表头,这是想清楚之后的决定,别再加回来。**
#
# 0820 试过两版:第一版"取第一条非空 ≥3 格的行",对一张 A3 起读的表抓到了**数据行**
# ——里面是真实用户投诉原文和某人的 Twitter handle。第二版加了"只看前两行 + 单格
# ≤60 字符 + 日期列打码",**同一张表照样漏过去**(那格投诉原文只有 48 字符)。
#
# 根子在于:**从返回里分不出表头行和数据行**。范围可能从 A3 起读、可能有两行合并
# 表头、可能整张表就没有表头。判错的代价是把真实用户数据拼进造世界的提示词,
# 而判对的收益很小 —— "列名该长什么样"这个信号已经用**人工挑过的例子**写死在
# 提示词 3.13 里了(`Date raised (Ctrl + ;)`、`Escalated by who (SM PIC)` 那几个),
# 覆盖 100% 的网盘题,不用每道题现抓。
#
# ⇒ 这里只收**形态**:文件名、多大(行×列)、什么类型、几个工作表。


def _safe_name(value):
    """文件名/表名照收(命名风格是有用信号),但抹掉邮箱和绝对日期。"""
    text = EMAIL_RE.sub('<addr>', str(value))
    return NAME_DATE_RE.sub('<日期>', text)[:120]


def drive_observations(call_text, response_text):
    """从一次网盘/表格调用里抠出**文件形态**(不抠内容)。

    收三种:
      * ``google_sheets_get``      → range 里带着真实行列数 + 表头行
      * ``google_sheets_metadata`` → 每个工作表的标题和网格大小
      * ``google_drive_search/get``→ files[] 的文件名和类型
    """
    out = []
    doc_ids = DOC_ID_RE.findall(call_text or '')
    doc_id = doc_ids[0] if doc_ids else None

    # 1) sheets_get:"range": "'Tab'!A1:Z50492"
    for tab, col1, row1, col2, row2 in SHEET_RANGE_RE.findall(response_text):
        n_rows = int(row2) - int(row1) + 1
        n_cols = _col_index(col2) - _col_index(col1) + 1
        if not (1 <= n_rows <= 500000 and 1 <= n_cols <= 200):
            continue
        tab = _safe_name(tab)[:80] if tab else None
        out.append({'kind': 'sheet', 'name': tab or None, 'tab': tab or None,
                    'n_rows': n_rows, 'n_cols': n_cols,
                    'doc_id': doc_id})

    # 2) sheets_metadata:每个 tab 的 gridProperties
    grids = GRID_RE.findall(response_text)
    if grids:
        titles = SHEET_TITLE_RE.findall(response_text)
        for index, (n_cols, n_rows) in enumerate(grids[:8]):
            out.append({'kind': 'sheet',
                        'name': titles[index] if index < len(titles) else None,
                        'tab': titles[index] if index < len(titles) else None,
                        'n_rows': int(n_rows), 'n_cols': int(n_cols),
                        'n_tabs': len(grids), 'doc_id': doc_id})

    # 3) drive_search / drive_get:files[]
    for item in iter_array_items(response_text, FILES_KEY_RE, '{'):
        if not isinstance(item, dict):
            continue
        name = item.get('name') or item.get('title')
        if not name:
            continue
        out.append({'kind': 'file', 'name': _safe_name(name),
                    'mime': str(item.get('mimeType') or '')[:80] or None,
                    'doc_id': item.get('id') or doc_id})

    cleaned = []
    for row in out:
        fields = {k: v for k, v in row.items() if v not in (None, '', [])}
        stable = fields.get('doc_id') or fields.get('name')
        if not stable:
            continue
        fields.pop('doc_id', None)
        cleaned.append({'stable_id': f"file:{stable}", 'kind': 'drive_file',
                        'message_id': None, 'thread_id': None, 'fields': fields})
    return cleaned


def calendar_observation(event):
    fields = calendar_fields(event)
    if not fields.get('start_at') and not fields.get('summary'):
        return None
    uid = event.get('iCalUID') or event.get('id')
    if uid:
        # 周期会议的每个实例 id 里就带了实例时间,不用再拼 start。
        stable_id = f'event:{uid}'
    else:
        stable_id = 'event-fingerprint:' + hashlib.sha256(
            canonical_json([fields.get('summary'), fields.get('start_at')]).encode()
        ).hexdigest()[:20]
    return {'stable_id': stable_id, 'kind': 'calendar_event',
            'message_id': None, 'thread_id': None, 'fields': fields}


def first_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, '', []):
            return value
    return None


def compact_fields(row, body_chars):
    """仅保留重建邮箱有用的字段，避免复制整个工具响应。"""
    fields = {}
    aliases = {
        'subject': ('subject',),
        'from': ('from', 'sender'),
        'to': ('to', 'recipients'),
        'cc': ('cc',),
        'sent_at': ('date', 'sentAt', 'sent_at', 'internalDate', 'timestamp'),
        'labels': ('labels', 'labelIds'),
        'unread': ('unread',),
        'important': ('important',),
        'body': ('body', 'text', 'snippet'),
    }
    for out_key, in_keys in aliases.items():
        value = first_value(row, *in_keys)
        if value is None:
            continue
        if out_key == 'body':
            value = str(value)[:body_chars]
        fields[out_key] = value
    return fields


def observation(row, kind, body_chars):
    if not isinstance(row, dict):
        return None
    message_id = first_value(row, 'messageId', 'message_id', 'id')
    thread_id = first_value(row, 'threadId', 'thread_id')
    # thread 搜索结果的 id 是 threadId；message 列表的 id 才是 message id。
    if kind == 'thread' and not thread_id:
        thread_id = message_id
        message_id = None
    if message_id:
        stable_id = f'message:{message_id}'
        kind = 'message'
    elif thread_id:
        stable_id = f'thread:{thread_id}'
        kind = 'thread'
    else:
        fields = compact_fields(row, body_chars)
        basis = [fields.get(k) for k in ('from', 'subject', 'sent_at', 'body')]
        if not any(basis):
            return None
        stable_id = 'fingerprint:' + hashlib.sha256(
            canonical_json(basis).encode()).hexdigest()[:20]
        kind = 'fingerprint'
    return {
        'stable_id': stable_id,
        'kind': kind,
        'message_id': str(message_id) if message_id else None,
        'thread_id': str(thread_id) if thread_id else None,
        'fields': compact_fields(row, body_chars),
    }


def extract_structured(payload, body_chars):
    out = []
    if isinstance(payload, dict):
        if isinstance(payload.get('threads'), list):
            out.extend(x for row in payload['threads']
                       if (x := observation(row, 'thread', body_chars)))
        if isinstance(payload.get('messages'), list):
            out.extend(x for row in payload['messages']
                       if (x := observation(row, 'message', body_chars)))
        if not {'threads', 'messages'} & set(payload):
            item = observation(payload, 'message', body_chars)
            if item:
                out.append(item)
    elif isinstance(payload, list):
        out.extend(x for row in payload if (x := observation(row, 'message', body_chars)))
    return out


def extract_text_fallback(payload):
    if not isinstance(payload, str):
        return []
    out = []
    hits = list(THREAD_TEXT_RE.finditer(payload))
    for i, hit in enumerate(hits):
        block = payload[hit.end():(hits[i + 1].start() if i + 1 < len(hits) else len(payload))]
        fields = {}
        for key, pat in (
                ('subject', r'(?im)^\s*Subject:\s*(.+)$'),
                ('from', r'(?im)^\s*From:\s*(.+)$'),
                ('sent_at', r'(?im)^\s*Date:\s*(.+)$')):
            m = re.search(pat, block)
            if m:
                fields[key] = m.group(1).strip()
        out.append({'stable_id': f'thread:{hit.group(1)}', 'kind': 'thread',
                    'message_id': None, 'thread_id': hit.group(1), 'fields': fields})
    return out


def calendar_shape(rows):
    """把所有用户的日历观测汇成一张形态表。

    ⭐ 这才是这份数据的**主要用途**:它是我们自己这份源池子里量出来的地面真值,
    可以直接拿去核 gen_worlds 提示词 3.7 / 3.8 那几条的靶子对不对。
    ⚠️ 和 0818 那份线上 trace 统计**不是同一批数据**,对不上先当口径差,别急着改靶子。
    """
    events = [item['versions'][-1]['fields']
              for row in rows for item in row['calendar_observations']
              if item.get('versions')]
    total = len(events)
    if not total:
        return {'events': 0}
    with_attendees = [e for e in events if e.get('n_attendees')]
    durations = sorted(e['duration_min'] for e in events if e.get('duration_min'))
    sizes = sorted(e['n_attendees'] for e in with_attendees)

    def pct(n):
        return round(n / total * 100, 1)

    responses = Counter(e.get('self_response') for e in with_attendees
                        if e.get('self_response'))
    return {
        'events': total,
        'users_with_events': sum(1 for row in rows if row['calendar_observations']),
        'all_day_pct': pct(sum(1 for e in events if e.get('all_day'))),
        'recurring_instance_pct': pct(sum(1 for e in events if e.get('recurring'))),
        'has_location_pct': pct(sum(1 for e in events if e.get('has_location'))),
        'has_description_pct': pct(sum(1 for e in events if e.get('has_description'))),
        'with_attendees_pct': pct(len(with_attendees)),
        'attendees_median': sizes[len(sizes) // 2] if sizes else None,
        'attendees_ge6_pct_of_meetings': (round(sum(1 for n in sizes if n >= 6)
                                                / len(sizes) * 100, 1) if sizes else None),
        'attendees_2to5_pct_of_meetings': (round(sum(1 for n in sizes if 2 <= n <= 5)
                                                 / len(sizes) * 100, 1) if sizes else None),
        'duration_median_min': durations[len(durations) // 2] if durations else None,
        'self_response_mix': dict(responses.most_common()),
    }


def drive_shape(rows):
    """网盘/表格的地面真值。⭐ 用来核提示词里"表格给几行几列、给几份文件"那条。"""
    files = [item['versions'][-1]['fields']
             for row in rows for item in row['drive_observations']
             if item.get('versions')]
    sheets = [f for f in files if f.get('n_rows')]
    per_user = [len(row['drive_observations']) for row in rows if row['drive_observations']]
    if not files:
        return {'files': 0}
    mimes = Counter(str(f.get('mime') or '').rsplit('.', 1)[-1] for f in files if f.get('mime'))

    def median(values):
        values = sorted(values)
        return values[len(values) // 2] if values else None

    return {
        'files': len(files),
        'users_with_drive': len(per_user),
        'files_per_user_median': median(per_user),
        'sheets': len(sheets),
        'sheet_rows_median': median([f['n_rows'] for f in sheets]),
        'sheet_rows_p90': (sorted(f['n_rows'] for f in sheets)[int(len(sheets) * 0.9)]
                           if sheets else None),
        'sheet_cols_median': median([f['n_cols'] for f in sheets if f.get('n_cols')]),
        'mime_mix': dict(mimes.most_common(8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary', default='')
    ap.add_argument('--body-chars', type=int, default=1200)
    args = ap.parse_args()

    pool_path, out_path = Path(args.pool), Path(args.out)
    summary_path = Path(args.summary) if args.summary else out_path.with_name('mailbox_summary.json')
    users = defaultdict(lambda: {
        'account_keys': set(), 'traces': set(), 'rules': set(),
        'records': 0, 'items': defaultdict(list), 'cal_items': defaultdict(list),
        'drive_items': defaultdict(list),
    })
    stats = Counter()

    with pool_path.open() as f:
        for line in f:
            group = json.loads(line)
            for rec in group['records']:
                stats['records'] += 1
                user_key = rec.get('user_key')
                if not user_key:
                    stats['records_without_user_key'] += 1
                    continue
                user = users[user_key]
                user['records'] += 1
                user['traces'].add(rec.get('trace_id'))
                user['rules'].add(group['rule_key'])
                user['account_keys'].update(x for x in rec.get('account_keys', []) if x)
                for tick_index, tick in enumerate(rec.get('ticks') or []):
                    observed_at = tick.get('observed_at')
                    if observed_at is None and tick_index == rec.get('target_tick_index'):
                        observed_at = rec.get('cutoff_at')
                    calls, responses = tick.get('tool_calls') or [], tick.get('tool_responses') or []
                    if len(calls) != len(responses):
                        stats['ticks_call_response_count_mismatch'] += 1
                    for idx, raw_response in enumerate(responses):
                        name = call_name(calls[idx]) if idx < len(calls) else None
                        if name in CALENDAR_TOOLS:
                            stats['calendar_tool_responses'] += 1
                            text = peel(str(raw_response))
                            events = iter_calendar_events(text)
                            if not events:
                                # 空日历和"截断到读不出"要分开记:前者是用户那天真没日程,
                                # 后者是我们丢了数据。混在一起会把覆盖率算高。
                                stats['calendar_empty' if EMPTY_EVENTS_RE.search(text)
                                      else 'calendar_unparsable'] += 1
                            for event in events:
                                obs = calendar_observation(event)
                                if not obs:
                                    stats['calendar_events_dropped'] += 1
                                    continue
                                stats['calendar_occurrences'] += 1
                                user['cal_items'][obs['stable_id']].append((
                                    {k: obs[k] for k in
                                     ('stable_id', 'kind', 'message_id', 'thread_id')},
                                    {'observed_at': observed_at,
                                     'source_trace': rec.get('trace_id'),
                                     'source_tick_index': tick_index,
                                     'fields': obs['fields']}))
                            continue
                        if name in DRIVE_TOOLS:
                            stats['drive_tool_responses'] += 1
                            text = peel(str(raw_response))
                            found = drive_observations(
                                str(calls[idx]) if idx < len(calls) else '', text)
                            if not found:
                                stats['drive_empty_or_unparsable'] += 1
                            for obs in found:
                                stats['drive_occurrences'] += 1
                                user['drive_items'][obs['stable_id']].append((
                                    {k: obs[k] for k in
                                     ('stable_id', 'kind', 'message_id', 'thread_id')},
                                    {'observed_at': observed_at,
                                     'source_trace': rec.get('trace_id'),
                                     'source_tick_index': tick_index,
                                     'fields': obs['fields']}))
                            continue
                        if name not in READ_TOOLS:
                            continue
                        stats['read_tool_responses'] += 1
                        payload = decode_json(raw_response)
                        observations = extract_structured(payload, args.body_chars)
                        if not observations:
                            observations = extract_text_fallback(payload)
                        for obs in observations:
                            stats['observation_occurrences'] += 1
                            version = {
                                'observed_at': observed_at,
                                'source_trace': rec.get('trace_id'),
                                'source_tick_index': tick_index,
                                'fields': obs['fields'],
                            }
                            item = {k: obs[k] for k in
                                    ('stable_id', 'kind', 'message_id', 'thread_id')}
                            user['items'][obs['stable_id']].append((item, version))

    def collapse(bucket, dup_stat, version_stat):
        """同一个对象的多次观测按时间排好、去掉状态没变的那些。"""
        items = []
        for _, occurrences in bucket.items():
            head = occurrences[0][0]
            versions, seen = [], set()
            for _, version in sorted(occurrences, key=lambda x: (
                    x[1].get('observed_at') or '', x[1].get('source_trace') or '',
                    x[1].get('source_tick_index', -1))):
                state = canonical_json(version.get('fields') or {})
                # 相同状态重复出现只留第一次；last_observed_at 仍记录在 item 上。
                if state in seen:
                    stats[dup_stat] += 1
                    continue
                seen.add(state)
                versions.append(version)
            stats[version_stat] += len(versions)
            observed = [v.get('observed_at') for _, v in occurrences if v.get('observed_at')]
            items.append({
                **head,
                'first_observed_at': min(observed) if observed else None,
                'last_observed_at': max(observed) if observed else None,
                'n_occurrences': len(occurrences),
                'versions': versions,
            })
        items.sort(key=lambda x: (x.get('first_observed_at') or '', x['stable_id']))
        return items

    rows = []
    for user_key, user in users.items():
        items = collapse(user['items'], 'duplicate_observation_occurrences',
                         'observation_versions')
        calendar_items = collapse(user['cal_items'], 'duplicate_calendar_occurrences',
                                  'calendar_versions')
        drive_items = collapse(user['drive_items'], 'duplicate_drive_occurrences',
                               'drive_versions')
        rows.append({
            'schema_version': 2,
            'user_key': user_key,
            'account_keys': sorted(user['account_keys']),
            'n_records': user['records'],
            'n_rules': len(user['rules']),
            'source_traces': sorted(x for x in user['traces'] if x),
            'observations': items,
            'calendar_observations': calendar_items,
            'drive_observations': drive_items,
        })
        stats['unique_observations'] += len(items)
        stats['unique_calendar_events'] += len(calendar_items)
        stats['unique_drive_files'] += len(drive_items)
        if calendar_items:
            stats['users_with_calendar'] += 1
        if drive_items:
            stats['users_with_drive'] += 1

    rows.sort(key=lambda x: x['user_key'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(out_path)

    summary = {
        'schema_version': 2,
        'pool_file': str(pool_path.resolve()),
        'pool_sha256': sha256_file(pool_path),
        'user_mailboxes': len(rows),
        **dict(stats),
        'calendar_shape': calendar_shape(rows),
        'drive_shape': drive_shape(rows),
    }
    stmp = summary_path.with_suffix(summary_path.suffix + '.tmp')
    stmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    stmp.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
