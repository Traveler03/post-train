#!/usr/bin/env python3
"""把收割选中的轨迹渲染成**能直接喂进训练的样本**(流水线最后一步)。

## 它在哪一环

    跑题 → 判分 → 收割(挑出做对的)→ **渲染(这一步)** → 训练

收割器产出的 `sft_pass.jsonl` 只是一张**清单**(哪道题、推荐哪条轨迹、轨迹链接),
不是样本。这个脚本顺着链接把 live trace 拉回来,还原成对话消息。

## 为什么必须从 trace 渲染,不能自己拼

评测下发给模型的是**四条消息**:

    system            11,742 字   系统提示词
    user              30,359 字   skills 大块(工具/技能清单)
    assistant            144 字   握手("好的,我已阅读")
    user               7,193 字   真正的任务(触发事件 + 规则)

hm 系列四个臂全部作废,就是因为自己拼只拼出**两条**(system + 任务),
训练时的形状跟推理时对不上。从 trace 渲染天然是对的 —— 这就是模型当时真看到的。

原料位置:
  - system  = 最后一次 `call_llm` 的 `input.config.system_instruction`
  - 前面所有轮 = 那次调用的 `input.contents`(ADK/Gemini 形状)
  - 最后一轮 = 那次调用的 `output.content.parts`

## 四条红线(全部做成硬闸,过不了就不出样本)

1. **思考写 `reasoning_content`,绝不写 `content`**
   Hy3 的思考是独立通道。塞进 content = 旁白泄露给用户。
   trace 里思考带 `thought: true` 标记,分得干净。

2. **末条 content 只留信封,不许带旁白前缀**
   v8b 实测 62.2% 的末条先写一段旁白再写信封,而底座原始输出是 0.0% ——
   这个病是教出来的。这里硬判:末条必须以 `{` 开头且能解析成
   `{content, reason, pass}`,否则整条丢掉。

3. **信封必须完整可解析**(截断/少字段一律丢)

4. **PII 脱敏**:收割器标了 `pii: true` 的样本,把真人邮箱换成 mock 域。
   名单来自 `mail_safety_audit.py --dump-pii`(它扫的是整条 trace 正文)。
   ⚠️ 真人邮箱主要来自**测试账号 Drive 里前人留下的通讯录**,模型一读就抄进轨迹,
   洗种子挡不住,只能在这一步脱。

## 用法

    python3 render_sft.py \\
        --sft out_v3/harvest/sft_pass.jsonl \\
        --out out_v3/train/sft.jsonl \\
        --pii-traces out_v3/harvest/pii_traces.jsonl \\
        --workers 16

    # 先看一条长什么样,不落盘
    python3 render_sft.py --sft ... --dry --sample 1

退出码:0 = 出样本了;2 = 一条都没出(多半是链接/权限问题)。
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LIVE_AUTH = os.environ.get('LANGFUSE_AUTH_B64', '')
ENVELOPE_KEYS = {'content', 'reason', 'pass'}
_BEEAI = '/home/work/migoo_ai_public/posttrain/code/beeai_eval'


def get(host, path, retries=3, timeout=90):
    for i in range(retries):
        try:
            req = urllib.request.Request(host + path,
                                         headers={'Authorization': 'Basic ' + LIVE_AUTH})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                return None


def _txt(parts):
    """把非思考的 text part 拼起来。⚠️ 必须排掉 thought=True —— 那是另一条通道。"""
    return '\n'.join(p['text'] for p in parts
                     if isinstance(p, dict) and p.get('text') is not None and not p.get('thought'))


def _think(parts):
    return '\n'.join(p['text'] for p in parts
                     if isinstance(p, dict) and p.get('thought') and p.get('text'))


def _calls(parts):
    out = []
    for p in parts:
        fc = p.get('function_call') if isinstance(p, dict) else None
        if not fc:
            continue
        out.append({'id': fc.get('id') or '', 'type': 'function',
                    'function': {'name': fc.get('name') or '',
                                 'arguments': json.dumps(fc.get('args') or {}, ensure_ascii=False,
                                                         sort_keys=True)}})
    return out


def to_messages(system, contents, final_parts):
    """ADK/Gemini 的 contents → OpenAI 风格 messages。

    role 映射有个坑:ADK 把**工具返回**也放在 `role: "user"` 里(用
    function_response 区分),不是单独的 tool role。按 role 直接映射会把
    工具返回当成用户说的话 —— 那是训练数据里最难查的一类错。
    """
    msgs = [{'role': 'system', 'content': system}]
    for m in contents:
        role, parts = m.get('role'), (m.get('parts') or [])
        fr = [p for p in parts if isinstance(p, dict) and p.get('function_response')]
        if fr:                                   # 工具返回:一个 response 一条 tool 消息
            for p in fr:
                r = p['function_response']
                msgs.append({'role': 'tool', 'name': r.get('name') or '',
                             'tool_call_id': r.get('id') or '',
                             'content': json.dumps(r.get('response'), ensure_ascii=False,
                                                   sort_keys=True)})
            continue
        if role == 'model':
            msg = {'role': 'assistant', 'content': _txt(parts)}
            th = _think(parts)
            if th:
                msg['reasoning_content'] = th    # ⛔ 红线①:思考走这里,不进 content
            tc = _calls(parts)
            if tc:
                msg['tool_calls'] = tc
            msgs.append(msg)
        else:
            msgs.append({'role': 'user', 'content': _txt(parts)})
    # 最后一轮(成品)
    last = {'role': 'assistant', 'content': _txt(final_parts)}
    th = _think(final_parts)
    if th:
        last['reasoning_content'] = th
    msgs.append(last)
    return msgs


def _parse_like_prod(s):
    """用**生产同款**的信封解析(带 repair)。拿不到就退回裸 json.loads。

    生产是 `reminder_agent_response_postprocess` 那套:未转义引号 / 裸控制字节
    都能修。评测侧的 `autotask_envelope_check._parse_envelope` 就是包的它。
    """
    try:
        import sys as _s
        if _BEEAI not in _s.path:
            _s.path.insert(0, _BEEAI)
        from autotask_envelope_check import _parse_envelope
        env, _ = _parse_envelope(s, reasoning_model=False)
        return env
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None


def check_envelope(text):
    """红线②③:末条必须**只有信封**,不许带旁白前缀,且必须完整可解析。

    返回 (ok, 原因)。宁可丢样本也不放病数据进去 —— 旁白前缀是教出来的病:
    v8b 实测 62.2% 的末条先写一段旁白再写信封,而底座原始输出是 0.0%。
    """
    s = (text or '').strip()
    if not s:
        return False, 'empty'
    # ⛔ 这两件事必须分开判,别混:
    #   ① 末条纯净:信封前面不许有字。旁白前缀是**教出来的病**
    #      (v8b 实测 62.2% 末条先写旁白再写信封,而底座原始输出 0.0%)
    #      → 这一关用我们自己的严格判据,以 { 开头才行
    #   ② 格式合法:信封解不解得出来。**这一关必须跟生产口径一致** ——
    #      生产的 reminder Go 带 repair(修未转义引号、裸控制字节),
    #      它能解析的,用户就真能看到内容,轨迹就是有效的。
    #      裸 json.loads 比生产严格,会错杀(0809 实测 40 条 SUSPECT 里 9 条是这么冤死的)。
    if not s.startswith('{'):
        return False, 'narration_prefix'
    env = _parse_like_prod(s)
    if env is None:
        # ⚠️ 生产的解析器**要求信封字段齐全**才返回,所以「缺字段」也会走到这里。
        # 再用裸 json.loads 分辨一次:能解析 = 结构问题(缺字段/类型错),
        # 解不了 = 真坏(截断/坏 JSON)。两者都拒,但原因标准确,免得诊断被误导。
        try:
            env = json.loads(s)
        except Exception:
            return False, 'unparsable'
    if not isinstance(env, dict) or not ENVELOPE_KEYS <= set(env):
        return False, 'missing_keys'
    if not isinstance(env.get('pass'), bool):
        return False, 'pass_not_bool'
    if env['pass'] and not str(env.get('content') or '').strip():
        return False, 'pass_true_empty'          # 说要推却没内容
    return True, ''


def canonical_envelope(text):
    """把生产可解析的合法信封规范成严格、稳定的 JSON。"""
    ok, why = check_envelope(text)
    if not ok:
        raise ValueError(why)
    return json.dumps(_parse_like_prod((text or '').strip()), ensure_ascii=False,
                      separators=(',', ':'), sort_keys=True, allow_nan=False)


def base_qid(x):
    """`xxx__r3` → `xxx`。题库 id 带 rollout 后缀,sft_pass 的 question 不带。"""
    x = str(x or '')
    i = x.rfind('__r')
    return x[:i] if i > 0 and x[i + 3:].isdigit() else x


def load_constraint_map(paths):
    """题库 jsonl → {base_id: key_constraints 条数}。

    「漏做」(该交付的没交付)是官方卷第一大失败(0821 实测占失败 40.5%,
    我们合成卷只有 13.1%)。一道题能不能考出「漏」,取决于它要交付几件事 ——
    也就是 expectedOutput.key_constraints 有几条。
    """
    m = {}
    for path in paths:
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            iid = ((d.get('input') or {}).get('id')) or d.get('id')
            kc = ((d.get('expectedOutput') or {}).get('key_constraints')) or ''
            n = len([x for x in str(kc).split(';') if x.strip()])
            if iid:
                m[base_qid(iid)] = n
    return m


def filter_sft_rows(rows, keep_low_confidence=False, min_score=None,
                    relax_score=None, relax_when_nkc_ge=None, nkc_map=None):
    """返回训练候选、默认剔除的低置信候选、以及被分数闸剔掉的。

    `min_score`(判官 D1 的整数分,None=不按分数过滤)。判官标尺:
      5 = 完全达标
      4 = **单点轻微瑕疵**仍算 PASS —— 原文举的例子是
          「星期几标错但日期对、时区差 1h 但绝对时间对、多加格式修饰、单个来源引用偏弱」
      ≤3 = FAIL(本来就不在 PASS 里)

    ⛔ **一刀切卡 5 分会把难题剔光**(0821 实测,1781 条 rollout):
       约束条数越多越难拿满分 —— 1 条约束 100% 是 5 分,4 条约束只剩 77.2%。
       结果被 5 分闸剔掉的 152 条里 **84.9% 是 ≥4 条约束的多交付题**
       (留下来的教材里只有 52.0%)。而「漏做」正是官方卷第一大失败。
       ⇒ 卡满分等于系统性地把教材推向简单题,和我们想补的方向相反。

    所以支持**分层**(`relax_score` + `relax_when_nkc_ge` + `nkc_map`):
       约束数 ≥ K 的题放宽到 relax_score,其余仍卡 min_score。
       难题不丢、简单题不带瑕疵。

    ⛔ 不给默认值式的隐式过滤:不传 min_score 行为和以前完全一样,
      免得别人复现旧实验时静默拿到被砍过的语料。
    """
    low = [r for r in rows if r.get('low_confidence')]
    keep = list(rows) if keep_low_confidence else [r for r in rows if not r.get('low_confidence')]
    dropped_score = []
    if min_score is None:
        return keep, low, dropped_score

    relax_on = relax_score is not None and relax_when_nkc_ge is not None
    nkc_map = nkc_map or {}

    def floor_for(r):
        if not relax_on:
            return min_score
        n = nkc_map.get(base_qid(r.get('question') or r.get('recommended')))
        if isinstance(n, int) and n >= relax_when_nkc_ge:
            return relax_score
        return min_score

    ok, bad = [], []
    for r in keep:
        sc = r.get('score')
        (ok if isinstance(sc, int) and sc >= floor_for(r) else bad).append(r)
    return ok, low, bad


def assert_score_gate_can_fire(rows, min_score):
    """⛔ fail-closed:分数闸只有在行里真有 `score` 时才有意义。

    0821 实案:`harvest_select.py` 写的 sft_pass.jsonl **压根没有 score 字段**,
    而闸把「取不到分数」当作不合格 —— 于是 `--min-score 5` 会静默剔光 100% 的行,
    渲出 0 条教材,而日志看起来一切正常。契约测试用手搓的行(带 score)所以没响。
    ⇒ 闸的验收标准是「在真路径上会响」,不是「写了」。
    """
    if min_score is None:
        return
    n = len(rows)
    if not n:
        return
    have = sum(1 for r in rows if isinstance(r.get('score'), int))
    if have / n < 0.5:
        sys.exit(
            f'⛔ 传了 --min-score {min_score},但 {n} 条里只有 {have} 条带整数 score。\n'
            '   分数闸把「取不到分数」当作不合格,再走下去会静默剔光大半语料。\n'
            '   多半是 sft_pass.jsonl 是旧版 harvest_select.py 收的(那版不写 score)——\n'
            '   重跑一次收割即可(harvest_select.py 0821 起会写 score/scores_seen)。')


def mask_pii(obj, addrs):
    """把真人邮箱换成稳定的 mock 地址(同一地址在整条样本里换成同一个)。"""
    if not addrs:
        return obj, 0
    blob = json.dumps(obj, ensure_ascii=False)
    n = 0
    for a in sorted(set(addrs), key=len, reverse=True):   # 长的先换,免得被短的截断
        if a not in blob:
            continue
        h = hashlib.sha1(a.encode()).hexdigest()[:10]
        blob = blob.replace(a, f'person.{h}@mock.test')
        n += 1
    return json.loads(blob), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft', required=True, help='收割器的 sft_pass.jsonl')
    ap.add_argument('--out', default='', help='样本落盘路径')
    ap.add_argument('--pii-traces', default='', help='mail_safety_audit --dump-pii 的产物')
    ap.add_argument('--allow-raw-pii', action='store_true',
                    help='显式允许 pii=true 的样本保留原文(仅限内网自部署训练)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-score', type=int, default=None,
                    help='判官 D1 分数下限(5=只要满分示范)。不传=不按分数过滤,行为同以前。'
                         ' 4 分是「单点轻微瑕疵」——星期几标错/时区差 1h 这类,'
                         '正好撞在超窗时效那个失败桶上,当教材会教出同样的毛病')
    ap.add_argument('--relax-score', type=int, default=None,
                    help='分层放宽:多交付题的分数下限(配 --relax-when-constraints-ge 用)')
    ap.add_argument('--relax-when-constraints-ge', type=int, default=None,
                    help='key_constraints 条数 ≥ 该值的题走 --relax-score,其余仍卡 --min-score')
    ap.add_argument('--items', default='',
                    help='题库 jsonl(逗号分隔多个),用来读 key_constraints 条数;分层放宽必须给')
    ap.add_argument('--keep-low-confidence', action='store_true',
                    help='保留只有 1 个有效判定的 PASS；默认剔除，避免判官抖动进入 SFT')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--sample', type=int, default=0, help='--dry 时打印几条')
    a = ap.parse_args()
    if not a.dry and not a.out:
        sys.exit('⛔ 要么给 --out,要么加 --dry')

    pii = {}
    if a.pii_traces and Path(a.pii_traces).exists():
        for l in open(a.pii_traces):
            try:
                x = json.loads(l)
                pii[x['trace']] = x.get('addrs') or []
            except Exception:
                pass
        print(f'真人邮箱名单 {len(pii)} 条 trace')
    elif not a.allow_raw_pii and not a.dry:
        # ⛔ 0818 修:原来这里只打印一句警告就过去了。
        # 老逻辑的闸是 `if flagged_pii and not pii and not allow_raw_pii`,
        # 而 flagged_pii 来自上一步 harvest_select 有没有传 --pii-traces ——
        # **完全跳过审计的人 flagged_pii=0,这道闸就一声不吭**,逻辑是反的:
        # 做过审计的被拦,没做审计的畅通无阻。0816 那批教材就是这么渲染出来的。
        # 现在改成:没给名单就必须显式表态,二选一。
        sys.exit(
            '⛔ 没给 --pii-traces,也没显式传 --allow-raw-pii —— 拒绝渲染。\n'
            '   这不是说教材里一定有真人信息,是说**没人查过**。二选一:\n'
            '   ① 先审计再脱敏(推荐):\n'
            '        python3 mail_safety_audit.py --dataset <卷> --dump-pii pii.jsonl\n'
            '        python3 harvest_select.py ... --pii-traces pii.jsonl [--pii-mode mark|drop]\n'
            '        python3 render_sft.py ... --pii-traces pii.jsonl\n'
            '   ② 按本批政策原样保留(0809 用户定的口径就是不脱敏):\n'
            '        加 --allow-raw-pii,产物仅限内网自部署训练\n'
            '   ⚠️ 只想看看渲染成什么样、不落盘,用 --dry。')
    elif a.allow_raw_pii:
        print('⚠️ 没给 --pii-traces —— 教材里可能混进真人邮箱且无人知晓;'
              '本次由 --allow-raw-pii 显式放行,按「不脱敏」政策原样保留')
    else:
        print('⚠️ --dry:没给 --pii-traces,只是预览、不落盘,所以放行。'
              '真要出产物必须二选一(--pii-traces 或 --allow-raw-pii)')

    rows = [json.loads(l) for l in open(a.sft) if l.strip()]
    assert_score_gate_can_fire(rows, a.min_score)

    relax_on = a.relax_score is not None and a.relax_when_constraints_ge is not None
    if (a.relax_score is None) != (a.relax_when_constraints_ge is None):
        sys.exit('⛔ --relax-score 和 --relax-when-constraints-ge 必须同时给')
    nkc_map = {}
    if relax_on:
        paths = [x.strip() for x in a.items.split(',') if x.strip()]
        if not paths:
            sys.exit('⛔ 分层放宽要读 key_constraints,必须给 --items <题库 jsonl>')
        missing = [x for x in paths if not Path(x).exists()]
        if missing:
            sys.exit(f'⛔ --items 里这些文件不存在:{missing}')
        nkc_map = load_constraint_map(paths)
        hit = sum(1 for r in rows if base_qid(r.get('question')) in nkc_map)
        print(f'题库约束表 {len(nkc_map)} 题;{len(rows)} 条候选里对上号 {hit} 条 '
              f'({hit / max(len(rows), 1) * 100:.1f}%)')
        # ⛔ fail-closed:对不上号的题拿不到约束数 → 一律走严的那档,
        #    于是「分层」静默退化成一刀切,而日志上看不出来。
        if hit / max(len(rows), 1) < 0.9:
            sys.exit('⛔ 超过 10% 的候选在题库里查不到 key_constraints —— '
                     '分层会静默退化成一刀切。请确认 --items 给全了(主批 + 补题批)。')

    rows, low, dropped_score = filter_sft_rows(
        rows, a.keep_low_confidence, a.min_score,
        relax_score=a.relax_score, relax_when_nkc_ge=a.relax_when_constraints_ge,
        nkc_map=nkc_map)
    if a.min_score is not None:
        import collections as _c
        dist = _c.Counter(r.get('score') for r in dropped_score)
        gate = (f'min_score={a.min_score}(约束数≥{a.relax_when_constraints_ge} 的放宽到 {a.relax_score})'
                if relax_on else f'min_score={a.min_score}')
        print(f'分数闸 {gate}:剔除 {len(dropped_score)} 条 '
              f'(分数分布 {dict(sorted(dist.items(), key=lambda kv: (kv[0] is None, kv[0])))})')
        if relax_on:
            kept4 = sum(1 for r in rows if (nkc_map.get(base_qid(r.get('question'))) or 0)
                        >= a.relax_when_constraints_ge)
            print(f'  留下的 {len(rows)} 条里,≥{a.relax_when_constraints_ge} 条约束的多交付题 '
                  f'{kept4} 条({kept4 / max(len(rows), 1) * 100:.1f}%)')
    if low and not a.keep_low_confidence:
        print(f'低置信 PASS 默认剔除 {len(low)} 条(--keep-low-confidence 可显式保留)')
    elif low:
        print(f'⚠️ 显式保留低置信 PASS {len(low)} 条')
    flagged_pii = sum(bool(r.get('pii')) for r in rows)
    if flagged_pii and not pii and not a.allow_raw_pii:
        sys.exit(f'⛔ {flagged_pii} 条候选标了 pii=true,但既未提供 --pii-traces 脱敏,'
                 '也未显式传 --allow-raw-pii')
    if flagged_pii and a.allow_raw_pii:
        print(f'⚠️ 显式保留原始 PII {flagged_pii} 条；产物仅限内网自部署训练')
    if a.limit:
        rows = rows[:a.limit]
    print(f'待渲染 {len(rows)} 条', flush=True)

    stat = Counter()
    lock = threading.Lock()
    def one(r):
        link = r.get('rec_lf_link') or ''
        m = re.search(r'https?://([^/]+)/.*?traces/([0-9a-f]+)', link)
        if not m:
            with lock:
                stat['no_link'] += 1
            return None
        host, tid = 'https://' + m.group(1), m.group(2)
        ob = get(host, f'/api/public/observations?traceId={tid}&limit=100') or {}
        gens = sorted([o for o in (ob.get('data') or [])
                       if o.get('type') == 'GENERATION' and str(o.get('name')).strip() == 'call_llm'],
                      key=lambda x: x['startTime'])
        if not gens:
            with lock:
                stat['no_call_llm'] += 1
            return None
        # ⚠️ 取**最后一次**调用:它的 contents 里带着前面所有轮,一次拿全。
        # 逐轮拼会漏掉框架在中途改写过的消息(实测有 skills 注入)。
        g = gens[-1]
        inp = g.get('input') or {}
        cfg = inp.get('config') or {}
        sysin = cfg.get('system_instruction') or cfg.get('systemInstruction') or ''
        if not isinstance(sysin, str):
            sysin = json.dumps(sysin, ensure_ascii=False)
        contents = inp.get('contents') or []
        final = ((g.get('output') or {}).get('content') or {}).get('parts') or []

        # ⛔ 工具清单必须跟着样本走(0815 补,之前一直漏)。
        #
        # `prepare_dsv4_data.py` 会把每条样本的 `tools` 渲进 prompt 第一条
        # (`sub[0]["tools"] = wrap_tools(tools)`);团队那份基准语料
        # `v1_clean_best.jsonl.gz` **每条都带**(1871/1871,96% 是 27 个工具)。
        # 我们这边一直没带 → `d.get("tools")` 拿到 None → **训练 prompt 里没有工具清单,
        # 而推理时是有的**。除了训推不一致,它还让「混不同工具形态的教材」变成投毒:
        # 模型看不到清单,15 工具那套的 `tool_search` 和 27 工具那套的 `google_*`
        # 就成了两套无从区分的动作词表。
        #
        # trace 里的 `config.tools[*].function_declarations` 和基准语料的条目
        # **同形**(都是 {name, description, parameters}),直接搬即可。
        tools = []
        for blk in (cfg.get('tools') or []):
            if isinstance(blk, dict):
                tools += [d for d in (blk.get('function_declarations')
                                      or blk.get('functionDeclarations') or [])
                          if isinstance(d, dict) and d.get('name')]
        if not tools:
            with lock:
                stat['warn::no_tools'] += 1

        ok, why = check_envelope(_txt(final))
        if not ok:
            with lock:
                stat[f'drop::{why}'] += 1
            return None
        msgs = to_messages(sysin, contents, final)
        # 生产解析器可能修复未转义引号/控制字节；训练文件不能继续保留这份“宽松
        # 可解析但不是严格 JSON”的原文。用解析后的对象重新序列化，保证下游 json.loads
        # 与线上看到的 content/reason/pass 语义一致。
        msgs[-1]['content'] = canonical_envelope(_txt(final))

        # 红线①的硬闸:任何 assistant 的 content 里不许出现思考通道的痕迹
        for mm in msgs:
            if mm.get('role') == 'assistant' and re.search(r'<think>|</think>', mm.get('content') or ''):
                with lock:
                    stat['drop::think_in_content'] += 1
                return None

        sample = {'question': r.get('question'), 'arm': r.get('arm'),
                  'user_key': r.get('user_key'),
                  'source_rule_key': r.get('source_rule_key'),
                  'iid': r.get('recommended'), 'trace': tid,
                  'teacher': r.get('rec_teacher'),
                  # ⭐ 这条轨迹是哪个思考档跑出来的(low/high/max)。
                  # 混档位的教材训出来不知道在学什么,所以每条都必须带。
                  'effort': r.get('effort', 'low'),
                  'run': r.get('rec_run'),
                  # 收割可信度/安全标记必须跟样本走到训练输入，不能渲染后丢失。
                  'n_verdicts': r.get('n_verdicts'),
                  'low_confidence': bool(r.get('low_confidence')),
                  'all_pass': bool(r.get('all_pass')),
                  'pii': bool(r.get('pii')),
                  # 这条轨迹当时挂了哪些工具 —— 训练时要渲进 prompt,
                  # 也是「这条教材属于哪套工具形态」的唯一可靠依据(15 还是 27)
                  'tools': tools,
                  'tool_count': len(tools),
                  # 这题的世界比发车日早/晚几天(repick_teacher_by_shape.py 打的标)。
                  # ⚠️ 渲染是按固定清单重拼样本的,不在清单里的字段一律丢失;
                  # 而锚点偏移**没有第二个来源**(不像 tool_count 能从 trace 实测),
                  # 丢了就再也分不出「主批 −2~+4」和「补题 0~+6」。
                  'anchor_offset_days': r.get('anchor_offset_days'),
                  'messages': msgs}
        if r.get('pii'):
            addrs = pii.get(r.get('rec_trace')) or []
            if a.pii_traces and not addrs:
                with lock:
                    stat['drop::pii_mapping_missing'] += 1
                return None
            if addrs:
                sample, n = mask_pii(sample, addrs)
                if not n:
                    with lock:
                        stat['drop::pii_not_found_in_sample'] += 1
                    return None
                sample['pii_masked'] = n
                with lock:
                    stat['pii_masked'] += bool(n)
        with lock:
            stat['ok'] += 1
            stat['turns'] += len(msgs)
        return sample

    with ThreadPoolExecutor(a.workers) as ex:
        # executor.map 保持输入顺序；线程只负责拉取/转换，统一由主线程顺序写盘。
        # 因而同一份原料重复渲染可得到逐字节相同的 JSONL。
        samples = [x for x in ex.map(one, rows) if x is not None]
    sizes = sorted(len(json.dumps(x, ensure_ascii=False).encode('utf-8')) for x in samples)
    if sizes:
        print(f'样本字节:p50={sizes[len(sizes)//2]:,} '
              f'p95={sizes[min(len(sizes)-1, int(len(sizes)*0.95))]:,} max={sizes[-1]:,}')
    if not a.dry:
        with open(a.out, 'w') as fout:
            for sample in samples:
                fout.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + '\n')
    else:
        for sample in samples[:a.sample]:
            print(f"\n===== 样例 {sample['iid']} · {len(sample['messages'])} 条消息 =====")
            for mm in sample['messages']:
                body = (mm.get('content') or '')[:110].replace('\n', ' ⏎ ')
                extra = ''
                if mm.get('reasoning_content'):
                    extra += f" +思考{len(mm['reasoning_content'])}字"
                if mm.get('tool_calls'):
                    extra += ' +调用' + ','.join(t['function']['name'] for t in mm['tool_calls'])
                print(f"  {mm['role']:9s} {len(mm.get('content') or ''):6d}字{extra}\n"
                      f"            {body}")

    print(f"\n渲染成功 {stat['ok']}/{len(rows)}"
          f"(平均 {stat['turns'] / max(stat['ok'], 1):.1f} 条消息/样本)")
    for k, v in sorted(stat.items()):
        if k.startswith('drop::'):
            print(f"  丢弃 {k[6:]:22s} {v}")
    for k in ('no_link', 'no_call_llm'):
        if stat[k]:
            print(f"  拉不到 {k:20s} {stat[k]}")
    if stat['pii_masked']:
        print(f"  真人邮箱已脱敏 {stat['pii_masked']} 条")
    if not stat['ok']:
        print('\n⛔ 一条样本都没出来')
        sys.exit(2)
    if not a.dry:
        print(f'\n✅ 落盘 {a.out}')


if __name__ == '__main__':
    main()
