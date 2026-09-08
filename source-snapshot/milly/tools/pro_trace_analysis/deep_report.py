#!/usr/bin/env python3
"""raw_deep_*.jsonl(线上真相)+ per_trace_v2.jsonl(训练语料对照)→ 双口径汇总。

输出 data/pro_trace_analysis_0826/deep_report.json + 终端摘要。
口径:请求级 = 一次用户请求;会话级 = 按 session_id 聚合所有请求的当前轮信息。
"""
import collections, glob, json, pathlib, statistics as st

OUT = pathlib.Path(__file__).resolve().parents[2] / 'data' / 'pro_trace_analysis_0826'


def pct(d, ps=(50, 90, 99)):
    if not d:
        return {}
    s = sorted(d)
    return {f'p{p}': s[min(len(s) - 1, int(len(s) * p / 100))] for p in ps} | {'max': s[-1], 'n': len(s)}


def dist(counter, total=None):
    total = total or sum(counter.values())
    return {k: {'n': v, 'pct': round(v / total * 100, 2)}
            for k, v in counter.most_common()}


raw = []
for f in sorted(glob.glob(str(OUT / 'raw_deep_20*.jsonl'))):
    for l in open(f, encoding='utf-8'):
        raw.append(json.loads(l))

rep = {'window': '2026-08-19~21', 'n_requests': len(raw)}

# ---------- 请求级(全流量) ----------
rep['request_level'] = {
    'form_all': dist(collections.Counter(r['form'] for r in raw)),
    'pro_lite': dist(collections.Counter(
        'pro' if r['is_pro'] else ('lite' if r['is_lite'] else 'no_llm_seen')
        for r in raw)),
}
pro = [r for r in raw if r['is_pro']]
rep['request_level']['form_pro_only'] = dist(collections.Counter(r['form'] for r in pro))
fam = collections.Counter()
for r in pro:
    for x in r['families']:
        fam[x] += 1
rep['request_level']['families_pro(request含该族=1)'] = dist(fam, total=len(pro))

# ---------- 上下文构成(pro 主链) ----------
pp = [r['prof'] for r in pro if r.get('prof')]
rep['context_pro'] = {
    '真历史轮数max_msg_time': pct([p['max_msg_time'] for p in pp]),
    '历史函数调用max_fc': pct([p['max_fc'] for p in pp]),
    'total_chars': pct([p['max_total_chars'] for p in pp]),
    'sys_chars': pct([p['sys_chars'] for p in pp]),
    'tools_chars': pct([p['tools_chars'] for p in pp]),
    'out_chars': pct([p['out_chars'] for p in pp]),
    'total_tokens': pct([p['total_tokens_max'] for p in pp]),
    'llm_steps_per_request': pct([p['n_calls'] for p in pp]),
}

# ---------- QRC 校准抽样 ----------
qs = []
for f in sorted(glob.glob(str(OUT / 'raw_deep_qrc_sample_*.jsonl'))):
    for l in open(f, encoding='utf-8'):
        qs.append(json.loads(l))
if qs:
    rep['qrc_sample'] = {
        'n': len(qs),
        'QRC>0占比%': round(sum(q['qrc_like'] > 0 for q in qs) / len(qs) * 100, 1),
        'qrc对数': pct([q['qrc_like'] for q in qs]),
    }

# ---------- 会话级(全流量;聚合每请求当前轮,不拆最长条的肚子) ----------
sess = {}
for r in raw:
    s = sess.setdefault(r['session_id'], {
        'n_req': 0, 'forms': collections.Counter(), 'fams': set(),
        'pro': False, 'lite': False, 'max_depth_turns': 0, 'max_chars': 0})
    s['n_req'] += 1
    s['forms'][r['form']] += 1
    s['fams'].update(r['families'])
    s['pro'] |= r['is_pro']
    s['lite'] |= r['is_lite']
    if r.get('prof'):
        s['max_depth_turns'] = max(s['max_depth_turns'], r['prof']['max_msg_time'])
        s['max_chars'] = max(s['max_chars'], r['prof']['max_total_chars'])
rep['session_level'] = {
    'n_sessions': len(sess),
    'requests_per_session': pct([s['n_req'] for s in sess.values()]),
    '会话含该形态=1': dist(collections.Counter(
        f for s in sess.values() for f in s['forms']), total=len(sess)),
    '会话含该族=1(pro会话)': dist(collections.Counter(
        f for s in sess.values() if s['pro'] for f in s['fams']),
        total=sum(1 for s in sess.values() if s['pro'])),
    '会话深度(最长那条的真历史轮数,只量深度用)': pct(
        [s['max_depth_turns'] for s in sess.values() if s['pro']]),
}

# ---------- merged_clean 对照(训练语料 vs 线上真相的扭曲) ----------
try:
    mrows = [json.loads(l) for l in open(OUT / 'per_trace_v2.jsonl', encoding='utf-8')]
    rep['merged_clean_control'] = {
        'n': len(mrows),
        'form': dist(collections.Counter(r['form'] for r in mrows)),
        '真历史轮数': pct([r['n_real_hist_turns'] for r in mrows]),
        'QRC>0占比%': round(sum(r['n_qrc_like'] > 0 for r in mrows) / len(mrows) * 100, 1),
        '历史工具返回被硬切>0占比%': round(
            sum(r['hist_trunc_tools'] > 0 for r in mrows) / len(mrows) * 100, 1),
    }
except FileNotFoundError:
    pass

json.dump(rep, open(OUT / 'deep_report.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
