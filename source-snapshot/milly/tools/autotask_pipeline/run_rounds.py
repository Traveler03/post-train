#!/usr/bin/env python3
"""按平台判分做 effort 漏斗：全量首轮 → 仅错题补轮。

## 跟固定 N 轮的区别

固定 3 轮:简单题也跑满 3 遍(第三遍几乎必过,纯浪费),难题只有 3 次机会。
这个脚本:**每轮只发「到现在一次都没做对」的题**,省下的次数全给难题。

当前主策略不让已经通过的题重复消耗预算。R1 由外部首发；本脚本收取平台
GPT-5.5 判定后，每轮只发尚无可靠 PASS 的题。档位可以由真正生效的 override
逐轮改变，也可以由精确 endpoint 固定（983552-max 当前使用 max,max,max）。

## 停止判据(谁先到算谁)

1. 一轮**新增覆盖 < --min-gain**(默认 5%)→ 停
2. 一道题**连挂 --max-attempts 轮**(默认 4)→ 不再追。
   多半是**题本身有问题**(expected_outcome 写歪/歧义),硬逼出来的那次
   PASS 反而是脏数据。这些落 `hard_questions.txt` 交给人看
3. 达到 --max-rounds
4. **任何一道闸没过 → 立刻停,不再发车**

## ⛔ 「再三确定」怎么落地:PASS 要两条独立通道都认

只信平台的 `eval_result.状态` 是不够的 —— 它出过错(判官截断拿不到判定、
respond 被拼接污染)。这里要求:

    通道①  平台判定 == PASS
    通道②  **live trace 末轮正文是可解析的信封**({content,reason,pass} 齐全)

**两条都成立才算这题做对了。** 只有①的记为 `suspect`,不算覆盖、也不进教材,
单独落盘供复查。

## 用法

    nohup python3 run_rounds.py \\
        --dataset benchmark/AutoTask_homolog/rq2d_v1 --batch rq2d \\
        --all-ids out_v3/rq2d_all_ids.txt --workdir out_v3 \\
        --emails a@x.com,b@x.com,c@x.com \\
        --initial-effort high --retry-efforts high,max \
        --model-config-overrides '{...high...}' > out_v3/rounds.log 2>&1 &

产物:
    <workdir>/rounds_state.json     每轮的覆盖率/执行数/判据,人可读
    <workdir>/hard_questions.txt    连挂 N 轮的难题(题本身可能有问题)
    <workdir>/suspect_pass.txt      平台说 PASS 但 trace 里没有合法信封的
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline_common import model_embedded_effort, model_matches

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EV = 'https://langfuse.us.migoo.shopee.io'
EA = os.environ.get('LANGFUSE_EVAL_AUTH_B64', '')
LA = os.environ.get('LANGFUSE_LIVE_AUTH_B64', '')


def g(path, host=EV, auth=EA, retries=3):
    for i in range(retries):
        try:
            r = urllib.request.Request(host + path, headers={'Authorization': 'Basic ' + auth})
            return json.load(urllib.request.urlopen(r, timeout=90))
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(3)


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def parse_effort_funnel(initial, retries):
    """Return the exact effort assigned to R1, R2...; empty retries = legacy mode."""
    retry = [item.strip() for item in (retries or '').split(',') if item.strip()]
    schedule = [initial] + retry
    bad = [item for item in schedule if item not in {'low', 'high', 'max'}]
    if bad:
        raise ValueError(f'illegal effort schedule: {bad}')
    return schedule


def overrides_for_effort(raw, effort):
    """Clone request overrides and pin effort for every configured model entry."""
    if not raw:
        raise ValueError('effort funnel requires --model-config-overrides')
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not obj:
        raise ValueError('--model-config-overrides must be a non-empty JSON object')
    for config in obj.values():
        if not isinstance(config, dict):
            raise ValueError('each model override must be an object')
        kwargs = config.setdefault('extraBody', {}).setdefault('chat_template_kwargs', {})
        kwargs['reasoning_effort'] = effort
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def model_name_from_mapping(raw):
    """Read the first concrete model name from the agent->model JSON mapping."""
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw or '')
    if isinstance(obj, dict):
        return str(next(iter(obj.values()), ''))
    return str(raw or '')


def uncovered_ids(allids, passed):
    """Stable retry list: only questions with no accepted PASS so far."""
    return [item_id for item_id in allids if item_id not in passed]


# ---------------------------------------------------------------- 收判定
def all_run_items(dataset, batch=''):
    """0811 评审整改(点9):runs 必须按本批前缀过滤 —— 数据集里还躺着 pilot/旧模型/
    别的 effort 的 run,不滤会污染覆盖率与停轮判据。batch 为空 = 旧行为(仅调试用)。"""
    ds = urllib.parse.quote(dataset, safe='')
    did = (g(f'/api/public/v2/datasets/{ds}') or {}).get('id')
    runs, page = [], 1
    while True:
        response = g(f'/api/public/datasets/{ds}/runs?page={page}&limit=100') or {}
        data = response.get('data') or []
        runs.extend(r['name'] for r in data if r.get('name'))
        if not data or page >= ((response.get('meta') or {}).get('totalPages') or 1):
            break
        page += 1
    if batch:
        skipped = [rn for rn in runs if not rn.startswith(batch)]
        runs = [rn for rn in runs if rn.startswith(batch)]
        if skipped:
            log(f'  run 过滤:纳入 {len(runs)} 个(前缀 {batch}),排除无关 {len(skipped)} 个')
    items = []
    for rn in runs:
        page = 1
        while True:
            r = g(f'/api/public/dataset-run-items?datasetId={did}'
                  f'&runName={urllib.parse.quote(rn, safe="")}&page={page}&limit=100')
            data = (r or {}).get('data') or []
            if not data:
                break
            items += [(rn, x['datasetItemId'], x['traceId']) for x in data if x.get('traceId')]
            if page >= (((r or {}).get('meta') or {}).get('totalPages') or 1):
                break
            page += 1
    return items


def judge_one(t):
    """两条独立通道:平台判定 + live trace 里的合法信封。"""
    rn, iid, tid = t
    tr = g(f'/api/public/traces/{tid}') or {}
    out = tr.get('output') or {}
    if out.get('error'):
        return {'iid': iid, 'run': rn, 'tid': tid, 'state': 'ERR'}
    ev = out.get('eval_result') or {}
    d1 = next(iter(ev.values()), {}) if ev else {}
    plat = str(d1.get('状态') or '')
    envelope_ok = False
    model = ''
    m = re.search(r'https?://([^/]+)/.*?traces/([0-9a-f]+)', out.get('langfuse_trace_link', '') or '')
    if m:
        ob = g(f'/api/public/observations?traceId={m.group(2)}&limit=100', 'https://' + m.group(1), LA)
        gens = sorted([o for o in (ob or {}).get('data', [])
                       if o.get('type') == 'GENERATION' and str(o.get('name')).strip() == 'call_llm'],
                      key=lambda x: x['startTime'])
        if gens:
            model = str(gens[0].get('model') or '')
            parts = ((gens[-1].get('output') or {}).get('content') or {}).get('parts') or []
            fin = '\n'.join(p['text'] for p in parts
                            if isinstance(p, dict) and p.get('text') is not None and not p.get('thought'))
            # 与最终 SFT 渲染使用同一套生产口径（包括生产允许的 repair），避免
            # 编排器把实际有效轨迹误记为 SUSPECT、继而多发补轮。
            from render_sft import check_envelope
            envelope_ok, _ = check_envelope(fin)
    # ⛔ 两条都成立才算做对
    if plat == 'PASS' and envelope_ok:
        st = 'PASS'
    elif plat == 'PASS':
        st = 'SUSPECT'          # 平台说过了,但 trace 里没有合法信封 —— 不采信
    elif plat:
        st = 'FAIL'
    else:
        st = 'NOJUDGE'          # 判官没出结果(截断等),这次不算数
    return {'iid': iid, 'run': rn, 'tid': tid, 'state': st, 'model': model,
            'plat': plat, 'env': envelope_ok}


def collect(dataset, batch='', workers=14):
    items = all_run_items(dataset, batch)
    rows = list(ThreadPoolExecutor(workers).map(judge_one, items))
    by_q = defaultdict(list)
    for r in rows:
        by_q[r['iid']].append(r)
    return rows, by_q


# ---------------------------------------------------------------- 等一轮跑完
def running(batch):
    from qa_tasks import tasks
    return [t for t in tasks() if t.get('task_name', '').startswith(batch)]


def wait_done(batch, poll_s=300, stall_min=60):
    last, last_change = -1, time.time()
    while True:
        ts = running(batch)
        if not ts:
            log('本轮平台任务已清空'); return 'done'
        done = sum(t['completed_cases'] for t in ts)
        tot = sum(t['total_cases'] for t in ts)
        log(f'  在跑 {len(ts)} 个任务 · {done}/{tot}')
        if done > last:
            last, last_change = done, time.time()
        elif time.time() - last_change > stall_min * 60:
            log(f'⚠️ 连续 {stall_min} 分钟零进展,当本轮结束处理'); return 'stall'
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--batch', required=True)
    ap.add_argument('--all-ids', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--emails', required=True)
    ap.add_argument('--model', default='{"remind_agent_professional": "deepseek-v4-flash-0731-online-autotask"}')
    ap.add_argument(
        '--model-config-overrides', default='',
        help='JSON 串，逐轮透传到 lane_manager/fire_high/submit_batch。'
             '仅用于已经验证会生效的运行时配置；精确 983552-max endpoint 不传。'
             '例如某个真正支持动态 effort 的模型: '
             '{"<模型名>":{"extraBody":{"chat_template_kwargs":'
             '{"reasoning_effort":"high"}}}}',
    )
    ap.add_argument('--app-name', default='remind_agent_professional')
    ap.add_argument('--judge-model', default='gpt-5.5-2026-04-23')
    ap.add_argument('--judge-prompt', default='migoo-autotask-prompt')
    ap.add_argument('--judge-prompt-version', type=int, default=13)
    ap.add_argument('--wall-clock-tolerance-min', type=int, default=15)
    ap.add_argument('--expect-model', default='',
                    help='live trace 中必须出现的实际模型名子串；默认从 --model JSON 推导')
    ap.add_argument('--base-rounds', type=int, default=2,
                    help='旧自适应模式的全员打底轮数；设置 --retry-efforts 后忽略')
    ap.add_argument('--max-rounds', type=int, default=5)
    ap.add_argument('--max-attempts', type=int, default=4,
                    help='一题连挂几轮就不再追(多半是题本身有问题)')
    ap.add_argument('--min-gain', type=float, default=0.05, help='一轮新增覆盖低于这个比例就停')
    ap.add_argument('--min-envelope', type=float, default=0.55,
                    help='信封率低于这个就停(服务过载的特征)。0808 塌的时候是 6.5%%,'
                         '正常是 95%%+,门槛设 55%% 既能挡住塌陷又不会被正常抖动误触')
    ap.add_argument('--lanes', type=int, default=12)
    ap.add_argument('--conc', type=int, default=20)
    ap.add_argument('--per-lane', type=int, default=190)
    ap.add_argument('--initial-effort', default='high', choices=['low', 'high', 'max'],
                    help='已经在跑/刚结束的首轮档位，仅用于漏斗记账')
    ap.add_argument('--retry-efforts', default='',
                    help='启用定档漏斗；high,max 表示 R2 错题 high、R3 仍错题 max。'
                         '已 PASS 的题绝不进入下一轮；空值保留旧自适应模式。')
    a = ap.parse_args()

    funnel = bool(a.retry_efforts.strip())
    effort_schedule = parse_effort_funnel(a.initial_effort, a.retry_efforts)
    if funnel:
        a.max_rounds = len(effort_schedule)
        # Fail closed before waiting for R1.  Effort can be pinned either by a
        # real runtime override, or by a reviewed exact deployment alias such
        # as dsv4-flash-online-autotask-983552-max.  The latter must stay fixed
        # across all retries and must not be rewritten into a fake override.
        for effort in effort_schedule:
            if a.model_config_overrides:
                overrides_for_effort(a.model_config_overrides, effort)
            elif model_embedded_effort(a.model) != effort:
                raise SystemExit(
                    '⛔ effort 漏斗既没有可生效的 model_config_overrides，'
                    f'模型端点也不保证 {effort}: {a.model}'
                )

    W = Path(a.workdir)
    allids = [l.strip() for l in open(a.all_ids) if l.strip()]
    state = {
        'rounds': [],
        'started': time.strftime('%F %T'),
        'model': a.model,
        'model_config_overrides': a.model_config_overrides,
        'effort_schedule': effort_schedule if funnel else None,
    }
    statef = W / 'rounds_state.json'

    lm_prev_pid = None   # 本编排器自己起的 lane_manager(精确杀用,0811 点9)
    for rnd in range(1, a.max_rounds + 1):
        log(f'===== 第 {rnd} 轮:等跑完 =====')
        wait_done(a.batch)
        time.sleep(120)                       # 等判分尾巴落库

        if funnel:
            verify_cmd = [sys.executable, str(HERE / 'verify_effort.py'),
                          '--dataset', a.dataset, '--run-filter', a.batch,
                          '--sample', '2', '--manifest', str(W / 'effort_manifest.json'),
                          '--update-manifest']
            log(f'核验实际 effort={effort_schedule[rnd - 1]}...')
            if subprocess.run(verify_cmd).returncode != 0:
                raise SystemExit('⛔ 实际 effort 与发车账本不一致，停止补轮')

        log('收判定(平台 GPT-5.5 + live trace 合法信封)...')
        rows, by_q = collect(a.dataset, a.batch)
        st = Counter(r['state'] for r in rows)
        passed = {q for q, rs in by_q.items() if any(x['state'] == 'PASS' for x in rs)}
        attempts = {q: len([x for x in rs if x['state'] in ('PASS', 'FAIL', 'SUSPECT')]) for q, rs in by_q.items()}
        cov = len(passed) / len(allids)
        prev_cov = state['rounds'][-1]['coverage'] if state['rounds'] else 0.0
        gain = cov - prev_cov
        # 单次通过率(用于估下一轮)
        judged = st['PASS'] + st['FAIL'] + st['SUSPECT']
        p = st['PASS'] / max(judged, 1)
        # 闸:模型必须还是对的
        # 信封率:服务过载时会塌(0808 实测掉到 6.5%)。这道闸独立于巡检,
        # 免得巡检被杀了/没跑,编排器还在傻乎乎往下发车。
        env_ok = sum(1 for r in rows if r.get('env')) / max(len([r for r in rows if r['state'] != 'ERR']), 1)
        mods = Counter(r.get('model', '') for r in rows if r.get('model'))
        want = a.expect_model or model_name_from_mapping(a.model)
        mod_ok = sum(v for k, v in mods.items() if model_matches(want, k)) / max(sum(mods.values()), 1)

        rec = {'round': rnd, 'effort': (effort_schedule[rnd - 1] if funnel else None),
               'judged': judged, 'pass': st['PASS'], 'fail': st['FAIL'],
               'suspect': st['SUSPECT'], 'nojudge': st['NOJUDGE'], 'err': st['ERR'],
               'single_pass_rate': round(p, 4), 'coverage': round(cov, 4),
               'gain': round(gain, 4), 'model_ok': round(mod_ok, 4),
               'envelope_rate': round(env_ok, 4),
               'covered_questions': len(passed), 'total_questions': len(allids),
               't': time.strftime('%F %T')}
        state['rounds'].append(rec)
        statef.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        log(f"  判定 {judged} 条 · 单次通过率 {100*p:.1f}% · 覆盖 {len(passed)}/{len(allids)} = {100*cov:.1f}% "
            f"(本轮 +{100*gain:.1f}pp)")
        log(f"  存疑(平台说PASS但没合法信封) {st['SUSPECT']} · 判官没出结果 {st['NOJUDGE']} · 跑挂 {st['ERR']}")
        log(f"  模型正确率 {100*mod_ok:.1f}% · 信封率 {100*env_ok:.1f}%")

        # 存疑清单落盘
        sus = [r['iid'] for r in rows if r['state'] == 'SUSPECT']
        if sus:
            (W / 'suspect_pass.txt').write_text('\n'.join(sorted(set(sus))) + '\n')

        # ---- 停止判据 ----
        if mod_ok < 0.9:
            log('⛔ 停:模型不对(<90%),不再发车'); break
        if judged >= 30 and env_ok < a.min_envelope:
            log(f'⛔ 停:信封率 {100*env_ok:.1f}% < {100*a.min_envelope:.0f}% '
                f'—— 多半是模型服务过载,继续发车等于烧钱'); break
        if rnd >= a.max_rounds:
            log('⛔ 停:到达最大轮数'); break
        if not funnel and rnd > a.base_rounds and gain < a.min_gain:
            log(f'⛔ 停:本轮新增覆盖 {100*gain:.1f}pp < {100*a.min_gain:.0f}pp'); break

        # ---- 下一轮发谁 ----
        next_effort = None
        if funnel:
            todo = uncovered_ids(allids, passed)
            next_effort = effort_schedule[rnd]
            why = f'只补仍未通过的 {len(todo)} 道，effort={next_effort}'
        elif rnd < a.base_rounds:
            todo = list(allids)               # 打底轮:全员
            why = f'打底第 {rnd+1}/{a.base_rounds} 轮,全员再跑一遍(为拿到 ≥2 个判定)'
        else:
            todo = [q for q in allids
                    if q not in passed and attempts.get(q, 0) < a.max_attempts]
            hard = [q for q in allids if q not in passed and attempts.get(q, 0) >= a.max_attempts]
            if hard:
                (W / 'hard_questions.txt').write_text('\n'.join(sorted(hard)) + '\n')
                log(f'  连挂 {a.max_attempts} 轮的难题 {len(hard)} 道 → hard_questions.txt(不再追)')
            why = f'只补没做对的 {len(todo)} 道'
        if not todo:
            log('⛔ 停:没有要补的题了'); break

        suffix = f'_{next_effort}' if next_effort else ''
        idsf = W / f'{a.batch}_round{rnd+1}{suffix}_ids.txt'
        idsf.write_text('\n'.join(todo) + '\n')
        log(f'===== 发第 {rnd+1} 轮:{why} =====')
        # ⛔ 先把上一轮的调度器收干净再起新的。
        # 两个 lane_manager 并存会互相抢线号和记账文件 —— 0809 已经因为
        # 「线号从 L1 重来」覆盖过一次记账(踩坑 48)。守车(fire_high)不用杀,
        # 它们守的任务已经跑完了、自己会退。
        # 0811 评审整改(点9):只杀**自己上一轮起的那一个** lane_manager(记账 pid),
        # 严禁 ps 扫全机 —— 别的批次/别人的调度器不归本编排器管(0810 误杀连坐同款教训)。
        if lm_prev_pid:
            try:
                cmdline = open(f'/proc/{lm_prev_pid}/cmdline').read().replace('\0', ' ')
                if 'lane_manager.py' in cmdline and a.batch in cmdline:
                    subprocess.run(['kill', str(lm_prev_pid)])
                    log(f'  收掉上一轮的调度器 pid={lm_prev_pid}')
                    time.sleep(3)
                else:
                    log(f'  上一轮调度器 pid={lm_prev_pid} 已不是本批 lane_manager(pid 复用?),不杀')
            except FileNotFoundError:
                log(f'  上一轮调度器 pid={lm_prev_pid} 已自然退出')
            except Exception as e:
                log(f'  ⚠️ 收旧调度器失败(不致命):{e}')
        # ⛔ 每轮必须用**独立的批次前缀**。
        # lane_manager 的记账是「一批 id 只发一次」:它 glob `lanes_<batch>_*.json`
        # 把里面的 id 全当成已提交。打底第 2 轮要把**同样的 2,270 条**再发一遍,
        # 用同一个 batch 就会被 R1 的记账挡住 —— 0809 实测:
        #     「还没提交 0 题 → 没有待提交的题了 → 全部线跑完,退出」
        # 一条都没发出去,而且不报错。
        # 用 rq2d_r2 / rq2d_r3 … 记账文件互不干扰;run 名也带上轮次,收割能分辨。
        # ⚠️ 编排器自己的 running() 用 startswith(a.batch),'rq2d' 仍能匹配 'rq2d_r2_*'。
        round_batch = f'{a.batch}_r{rnd + 1}{suffix}'
        next_overrides = (
            overrides_for_effort(a.model_config_overrides, next_effort)
            if funnel and a.model_config_overrides else a.model_config_overrides
        )
        cmd = [sys.executable, str(HERE / 'lane_manager.py'),
               '--dataset', a.dataset, '--batch', round_batch, '--ids', str(idsf),
               '--workdir', str(W), '--model', a.model, '--emails', a.emails,
               '--app-name', a.app_name, '--judge-model', a.judge_model,
               '--judge-prompt', a.judge_prompt,
               '--judge-prompt-version', str(a.judge_prompt_version),
               '--wall-clock-tolerance-min', str(a.wall_clock_tolerance_min),
               '--lanes', str(a.lanes), '--conc', str(a.conc),
               '--per-lane', str(max(40, len(todo) // a.lanes + 1)),
               '--allow-drift', '0', '--loop', '300', '--interval', '300',
               *(['--model-config-overrides', next_overrides]
                 if next_overrides else [])]
        lg = open(W / f'lm_{a.batch}_r{rnd+1}.log', 'w')
        _p = subprocess.Popen(cmd, stdout=lg, stderr=lg, start_new_session=True)
        lm_prev_pid = _p.pid
        log(f'  lane_manager 已起,pid={lm_prev_pid}(记账供下轮精确收)')
        time.sleep(180)                        # 等它把线发出去再进下一轮等待

    log('===== 全部结束 =====')
    log(json.dumps(state['rounds'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
