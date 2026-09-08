#!/usr/bin/env python3
"""线上原始 trace 深描 · 第 1 步:轨迹形态 + 上下文构成(全流量,双口径)。

为什么在原始 parquet 上做而不是 merged_clean:后者是 13 道过滤只留 28.8% 的训练候选,
统计线上真相会失真(0826 sheets 假象实案)。merged_clean 只做对照,不做主统计。

口径(0826 和用户对齐):
  - 请求级:一个 trace = 用户的一次请求(平台一个 trace 就是一轮)。
  - 会话级:按 session_id 聚合该会话**所有请求行的当前轮信息**;
    ⛔ 不能只解析"最长那条"的肚子——历史只留近 5 个请求轮、QRC 是压缩改写、
    工具返回超 2000 字符从中间硬切,最长那条看不见早期轮的真实动作。
    最长那条只用来量"上下文最深能到多深"。

轨迹形态标签(docs/pro/02 §3.5 的三形态):
  A_sync(imagen/music/tts 同步回主链)/ B_video_*(状态机,按 status 细分)/
  C_sandbox(transfer 单程票)/ tool_other / zero_tool。
  工具三层拆分:模型可见 / 平台机件(checkpoint 等,不算模型行为)/ 沙箱内部。

每次 call_llm 沿 parent 链爬到 agent_run 归属(单文件断链 96%,必须全天建表)。
上下文构成走正则计数(<msg_time>=真历史轮;functionCall/Response=历史工具轮),
另抽 1/50 全量 JSON 解析校准 QRC 对(压缩召回的早期对话,无 msg_time、非平台标签)。

用法:python3 raw_deep.py 2026-08-19 2026-08-20 2026-08-21
输出:data/pro_trace_analysis_0826/raw_deep_<day>.jsonl(请求级)
      + raw_deep_qrc_sample_<day>.jsonl(QRC 校准抽样)
"""
import collections, glob, json, pathlib, re, sys
import pyarrow.parquet as pq

SRC = pathlib.Path('/home/work/migoo_ai_public/posttrain/trace/main')
OUT = pathlib.Path(__file__).resolve().parents[2] / 'data' / 'pro_trace_analysis_0826'
OUT.mkdir(parents=True, exist_ok=True)

PROXY = re.compile(r'^proxy_tool \[([^\]]+)\]$')
SANDBOX_INTERNAL = re.compile(r'^\[')
# 平台机件:raw 里量大但模型从不直接调用(02 号文第二层)
PLATFORM_TOOLS = re.compile(r'^(video_generate|video_task_checkpoint|load_video_creative_resource|get_video_task)$')
SYNC_GEN = {'imagen_generate', 'music_generate', 'google_tts', 'edit_image'}
FAMILY_RULES = [(n, re.compile(p)) for n, p in [
    ('drive', r'^google_drive_'), ('docs', r'^google_docs_'),
    ('sheets', r'^google_sheets_'), ('slides', r'^google_slides_'),
    ('gmail', r'^(google_gmail_|gmail_)'), ('calendar', r'^(google_calendar_|calendar_)'),
    ('memory', r'^(memory_|hierarchy_node_search)'), ('contacts', r'^contact_'),
    ('auto_task', r'^auto_task_'), ('weather', r'^weather_'),
    ('web_search', r'^web_search$'), ('video', r'^video_agent$'),
    ('image_gen', r'^(imagen_|edit_image)'), ('music_gen', r'^music_'),
    ('tts', r'^google_tts$'),
    ('skill', r'^(load_skill|load_skill_resource)$'),
    ('tool_infra', r'^(tool_search|load_google_tools|load_lazy_tools|load_connector)$'),
    ('transfer', r'^transfer_to_agent$'), ('file', r'^(file_read|file_context_get)'),
    ('temporal', r'^temporal_resolve$'), ('people', r'^people_search'),
    ('places', r'^(search_places|get_place_details)'), ('flight', r'^flight_search'),
    ('channel', r'^channel_list'),
]]
VIDEO_STATUS = re.compile(r'"status"\s*:\s*\\?"(plan_ready|quick_completed|project_ready|failed|planning|committed)')
PLATFORM_TAG = re.compile(r'^\s*<(system-reminder|skills|skill_prefetch|knowledge-context|'
                          r'available_tools|overview|user_uploaded_files|seatalk-context|'
                          r'query-context|temporal_context|skill |quoted_message)')


def family(tool):
    for n, rx in FAMILY_RULES:
        if rx.search(tool):
            return n
    return 'other'


def one_day(day):
    files = sorted(glob.glob(str(SRC / f'dt={day}' / '*.parquet')))
    if not files:
        print(f'⛔ {day} 没有文件'); return

    # ---- 第一遍:全天建 parent 链表 + trace 骨架(轻列)
    parent = {}          # obs_id -> parent_obs_id
    agent_of = {}        # obs_id(agent_run span) -> agent名
    tr = {}              # trace_id -> 骨架
    for f in files:
        t = pq.read_table(f, columns=['observation_id', 'parent_observation_id', 'name',
                                      'type', 'trace_id', 'session_id', 'user_id']).to_pydict()
        for i in range(len(t['name'])):
            oid = t['observation_id'][i]
            parent[oid] = t['parent_observation_id'][i]
            nm = t['name'][i] or ''
            if nm.startswith('agent_run ['):
                agent_of[oid] = nm[11:-1]
            tid = t['trace_id'][i]
            r = tr.get(tid)
            if r is None:
                r = tr[tid] = {'session': t['session_id'][i], 'user': t['user_id'][i],
                               'agents': set(), 'mv_tools': [], 'plat_tools': 0,
                               'sbx_tools': 0, 'route_tools': 0, 'unattr_tools': 0,
                               'prof_transfer_to': set(),
                               'video_status': collections.Counter(),
                               'llm': {}}
            if nm.startswith('agent_run ['):
                r['agents'].add(nm[11:-1])
    print(f'{day} pass1: {len(tr)} traces, {len(parent)} spans, {len(agent_of)} agent spans', flush=True)

    def up_agent(oid):
        d = 0
        while oid and d < 30:
            a = agent_of.get(oid)
            if a:
                return a
            oid = parent.get(oid)
            d += 1
        return None

    # ---- 第二遍:call_llm 的输入构成(重列,逐行正则计数)+ video TOOL 的 status
    qrc_fh = open(OUT / f'raw_deep_qrc_sample_{day}.jsonl', 'w', encoding='utf-8')
    sample_i = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000,
                columns=['observation_id', 'parent_observation_id', 'name', 'type',
                         'trace_id', 'provided_model_name', 'input_propery',
                         'output_propery', 'input_encrypt', 'output_encrypt',
                         'usage_details']):
            b = batch.to_pydict()
            for i in range(len(b['name'])):
                nm = b['name'][i] or ''
                tid = b['trace_id'][i]
                r = tr.get(tid)
                if r is None:
                    continue
                if b['type'][i] == 'TOOL':
                    inner = (PROXY.match(nm).group(1) if PROXY.match(nm) else nm)
                    if inner == 'video_agent':
                        m = VIDEO_STATUS.search(b['output_encrypt'][i] or '')
                        if m:
                            r['video_status'][m.group(1)] += 1
                    owner = up_agent(b['parent_observation_id'][i])
                    if SANDBOX_INTERNAL.match(nm) or owner == 'sandbox_runner':
                        r['sbx_tools'] += 1
                    elif owner == 'assistant_router':
                        r['route_tools'] += 1          # router 的移交/路由管道,不算模型行为
                    elif PLATFORM_TOOLS.match(inner) or owner == 'video_agent':
                        r['plat_tools'] += 1           # 平台机件/video 内部执行
                    elif owner in ('assistant_professional', 'assistant_lite'):
                        r['mv_tools'].append(inner)
                        if inner == 'transfer_to_agent':
                            mt = re.search(r'"agent_name\\?"\s*:\s*\\?"([a-zA-Z0-9_]+)',
                                           b['input_encrypt'][i] or '')
                            if mt:
                                r['prof_transfer_to'].add(mt.group(1))
                    else:
                        r['unattr_tools'] += 1         # parent 链断(应<5%,报告里记着)
                    continue
                if nm != 'call_llm':
                    continue
                ag = up_agent(b['parent_observation_id'][i]) or '?'
                s = b['input_encrypt'][i] or ''
                model = b['provided_model_name'][i] or ''
                try:
                    ip = json.loads(b['input_propery'][i] or '{}')
                except Exception:
                    ip = {}
                try:
                    ud = json.loads(b['usage_details'][i] or '{}')
                except Exception:
                    ud = {}
                slot = r['llm'].setdefault(ag, {
                    'n_calls': 0, 'models': collections.Counter(),
                    'max_msg_time': 0, 'max_fc': 0, 'max_total_chars': 0,
                    'sys_chars': 0, 'tools_chars': 0, 'out_chars': 0,
                    'total_tokens_max': 0})
                slot['n_calls'] += 1
                slot['models'][model] += 1
                slot['max_msg_time'] = max(slot['max_msg_time'], s.count('<msg_time>'))
                slot['max_fc'] = max(slot['max_fc'], s.count('"function_call"'))
                tc = int(ip.get('total_characters') or 0)
                if tc >= slot['max_total_chars']:
                    slot['max_total_chars'] = tc
                    slot['sys_chars'] = int(ip.get('system_instruction_characters') or 0)
                    slot['tools_chars'] = int(ip.get('tools_characters') or 0)
                try:
                    op = json.loads(b['output_propery'][i] or '{}')
                    slot['out_chars'] += int(op.get('total_characters') or 0)
                except Exception:
                    pass
                slot['total_tokens_max'] = max(slot['total_tokens_max'],
                                               int(ud.get('total') or 0))
                # 抽 1/50 的主链调用做 QRC 校准(全量 JSON 解析)
                if ag == 'assistant_professional':
                    sample_i += 1
                    if sample_i % 50 == 0:
                        try:
                            contents = json.loads(s).get('contents') or []
                            qrc, seen_mt, users = 0, False, 0
                            for c in contents:
                                if (c.get('role') or '') != 'user':
                                    continue
                                users += 1
                                txt = ''.join(p.get('text') or '' for p in (c.get('parts') or []))
                                if '<msg_time>' in txt[:200]:
                                    seen_mt = True
                                elif users > 1 and not seen_mt and not PLATFORM_TAG.match(txt):
                                    qrc += 1
                            qrc_fh.write(json.dumps({
                                'trace_id': tid, 'n_contents': len(contents),
                                'qrc_like': qrc, 'n_user': users,
                                'n_msg_time': s.count('<msg_time>')},
                                ensure_ascii=False) + '\n')
                        except Exception:
                            pass
    qrc_fh.close()

    # ---- 落盘请求级记录
    n = 0
    with open(OUT / f'raw_deep_{day}.jsonl', 'w', encoding='utf-8') as fh:
        for tid, r in tr.items():
            mv = r['mv_tools']
            fams = sorted({family(x) for x in mv})
            vs = r['video_status']
            if 'sandbox_runner' in r['prof_transfer_to'] or \
                    ('transfer_to_agent' in mv and
                     ('sandbox_runner' in r['agents'] or r['sbx_tools'])):
                form = 'C_sandbox'
            elif 'video_agent' in mv:
                if vs.get('project_ready'):
                    form = 'B_video_project'
                elif vs.get('quick_completed'):
                    form = 'B_video_quick'
                elif vs.get('plan_ready') or vs.get('planning'):
                    form = 'B_video_plan'
                else:
                    form = 'B_video_other'
            elif set(mv) & SYNC_GEN:
                form = 'A_sync_gen'
            elif mv:
                form = 'tool_other'
            else:
                form = 'zero_tool'
            prof = r['llm'].get('assistant_professional') or {}
            lite = r['llm'].get('assistant_lite') or {}
            fh.write(json.dumps({
                'trace_id': tid, 'session_id': r['session'], 'user_id': r['user'],
                'day': day, 'agents': sorted(r['agents']), 'form': form,
                'families': fams, 'mv_tools': mv[:40],
                'n_mv': len(mv), 'n_plat': r['plat_tools'], 'n_sbx': r['sbx_tools'],
                'n_route': r['route_tools'], 'n_unattr': r['unattr_tools'],
                'prof_transfer_to': sorted(r['prof_transfer_to']) or None,
                'video_status': dict(vs) or None,
                'is_pro': bool(prof), 'is_lite': bool(lite) and not prof,
                'prof': {k: (dict(v) if isinstance(v, collections.Counter) else v)
                         for k, v in prof.items()} or None,
            }, ensure_ascii=False) + '\n')
            n += 1
    print(f'{day} done: {n} requests -> raw_deep_{day}.jsonl', flush=True)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        one_day(d)
