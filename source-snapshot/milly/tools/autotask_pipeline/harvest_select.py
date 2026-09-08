#!/usr/bin/env python3
"""拒绝采样收割选择器(增量,随时可跑):

1. 从 out/poll_*.jsonl 取每个 run 的精确名(绕开会被大 run 压垮的 runs 列表接口),
   拉全部 run items → 逐 trace 取判官结果(16 线程,增量缓存 out/harvest/rows.jsonl)。
2. 按题聚合(去 __rK)输出三份:
   - out/harvest/sft_pass.jsonl   每题的 PASS 轨迹(含全对题,带 all_pass 标记;推荐条=分数最高)
   - out/harvest/dpo_pairs.jsonl  混合题(既有PASS又有FAIL):推荐 最好PASS × 最差FAIL 对
   - out/harvest/allfail_questions.txt  全败题(老师补做队列)
   - out/harvest/coverage.md      按臂覆盖/收成统计
"""
import argparse, json, os, re, sys, threading, queue, time, urllib.parse, urllib.request
from collections import defaultdict, Counter
from pathlib import Path

from pipeline_common import sha256_file, stable_best, stable_worst

ROOT = Path(__file__).resolve().parent.parent
# ⚠️ 下面四个都由 main() 按参数覆盖。写死会去收上一批 —— 别改成常量。
DATASET = ''   # ⛔ 不设默认值:由 --dataset(必填)注入。
               #    原来写死的是 **rq1** 的卷 —— 直接 import 会去收错卷的教材。
WORKDIR = ROOT / 'out'
OUT = WORKDIR / 'harvest'
LOCAL_JUDGE = None    # 有本地判分结果时优先用它,平台判定只作对照
TEACHER_DEFAULT = 'unknown-teacher'   # run 名猜不出老师时用它;--teacher 覆盖
# ⛔ 这里曾经写死 `hy3-low`。0819 实测:别名 `hy3-online-autotask-979884-low` 底下
#    跑的其实是 DeepSeek-V4-Flash(live trace 的 call_llm 184/184),
#    这条线**早就不在混元上跑了**。一个会骗人的默认值比没有默认值更糟 ——
#    它让教材看起来像是有出处,于是没人去查。猜不出来就写 unknown,逼人传 --teacher。
PII_FILE = ''         # mail_safety_audit --dump-pii 的产物
PII_MODE = 'mark'     # mark=照进教材只打标(默认,不丢数据) / drop=整条剔除
EH = 'https://langfuse.us.migoo.shopee.io'
AU = os.environ.get('LANGFUSE_AUTH_B64', '')
PASS_SET = {'PASS', '成功', '通过', 'pass', 'success'}


# run 名 → 思考档位。**每条轨迹必须知道自己是哪个档跑出来的** ——
# 混了档位的教材,训出来不知道在学什么。
# 约定:批次前缀带档位(rq2dhigh_* / rq2dmax_*),不带的就是默认 low。
_MANIFEST = None
LOCAL_JUDGE_SHA = ''
EXCLUDE_RE = None     # --exclude-run-re 编译后的正则;None=不排除任何 run
ONLY_MODEL = ''       # --only-requested-model 白名单;''=不按 endpoint 过滤
ALLOW_PARTIAL_VERDICTS = False   # --allow-partial-verdicts:明知卷没判完也要收


def partial_verdict_gate(verdicts_per_q, allow_partial):
    """卷还没判完就收割 → 一批题的判定数会明显少于同批其它题。这道闸抓的就是那个。

    ⛔ **为什么值得一道闸**:`low_confidence = n_verdicts < 2` 是**收割那一刻的瞬时值**,
    写进 `sft_pass.jsonl` 之后就成了永久属性,渲染那步照着它剔样本,**没人回头重算**。

    0819 复盘 v7:`sft_pass.jsonl` 是 0816 05:19 收的,那时 r2p2 三块还没跑完
    → 97 个任务当时只有 1 个判定 → 全被标 low_confidence → 渲染剔掉。
    等 r2p2 跑完后回查 rows.jsonl:**73/97 现在已经有 ≥2 个判定**,
    其中 34 个是两轮全 PASS 的好题。**白扔了 5.7% 的教材,而且全程没有一句话提示。**

    判据故意做得很笨:和**同批中位数**比,不套绝对阈值 ——
    轮次策略会变(历史上从固定 5 轮改成自适应 2 轮),写死"至少 2 个"这种门槛
    在改策略那天就会失效(`all_pass` 那个判据已经因此失效过一次)。

    fail-closed,和本文件里 `--only-requested-model` / `--allow-missing` 一个路子:
    滚动收割是正当用法,但要**显式**说"我知道这批还没判完"。
    """
    if not verdicts_per_q:
        return
    counts = sorted(verdicts_per_q.values())
    n = len(counts)
    med = counts[n // 2]
    low = {q: c for q, c in verdicts_per_q.items() if c < med}
    dist = Counter(counts)
    print(f'判定数分布(每题拿到过几个判定):{dict(sorted(dist.items()))} · 中位 {med}')
    if not low:
        return
    frac = len(low) / n
    listing = OUT / 'below_median_verdicts.txt'
    listing.write_text(''.join(f'{q}\t{c}\n' for q, c in sorted(low.items())))
    msg = (f'⚠️ {len(low)}/{n} = {100 * frac:.1f}% 的题判定数少于同批中位数 {med}。\n'
           f'   这通常意味着**卷还没判完就在收割** —— 它们会被标 low_confidence,\n'
           f'   渲染那步默认剔掉,而且之后不会重算(0819 实案:v7 因此白扔 73 个任务,\n'
           f'   其中 34 个是两轮全 PASS 的好题)。\n'
           f'   名单:{listing}')
    # n 太小时中位数本身不稳,只提示不判死
    if n < 100 or frac <= 0.02 or allow_partial:
        print(msg + ('\n   (--allow-partial-verdicts:已显式放行)' if allow_partial else
                     '\n   (占比小或样本少,只提示不判死)'))
        return
    raise SystemExit(
        msg + '\n\n⛔ 不出教材。二选一:\n'
              '   ① 等卷都收官 + 判官判完(收官后再等 8 分钟)再重跑这条收割命令;\n'
              '   ② 滚动收割是有意的 → 加 --allow-partial-verdicts 明确放行\n'
              '      (那就要记得**卷跑完后再收一次**,否则这些题永久丢失)。')


def enforce_single_model(only_model, run_filter, allow_mixed):
    """开工前先钉死"这批教材的老师是谁",钉不死就停。

    ⛔ **这是 fail-closed 的**:查不出来就不出教材,而不是"查不出来就当没这回事"。
    理由是这个错误**事后完全看不出来** —— 混了两个 endpoint 的教材长得和干净的一模一样,
    只有分数会莫名其妙。0817 踩过一次(同题的高/低档 rollout 聚在一起挑最优,
    22 道必然挑中高档当老师);0819 又发现 `requested_model` 有 dict / 字符串两种形状,
    比较永远不相等 —— 那次是**反向失效**:闸看着开着,其实把所有 run 都剔了。

    三种情形:
      · 显式传了 --only-requested-model  → 就用它(人明确说了要哪个);
      · 账本里本批只有一个 endpoint      → **自动钉死**,并打印出来。
        这样不用每次手打一长串,又不会因为忘了传而放行;
      · 查不到、或多于一个                → 停,除非 --allow-mixed-models。
    """
    if only_model:
        print(f'✅ endpoint 白名单(显式指定):{only_model}')
        return only_model
    man = _load_manifest()
    # ⚠️ requested_model 记的是**泳道名**(ais-relay-a/b/h/l),不是 endpoint。
    #    一个 endpoint 挂多条泳道是常态(并行发卷),按泳道名比会把它误判成
    #    "混了 4 个模型" 并停住 —— 闸的本意是"这批教材只能有一个老师",
    #    泳道只是通往老师的路。所以账本里有 upstream_model 时**以它为准**。
    #    0821 实案:r1/r2/r3/later 四轮走 4 条泳道,上游全是 Compass 的
    #    deepseek-v4-flash,旧判据会拒收整批。
    hit = {k: endpoint_of_entry(v)
           for k, v in man.items()
           if not run_filter or k.startswith(run_filter) or run_filter.startswith(k)}
    found = sorted({v for v in hit.values() if v})
    if len(found) == 1 and len(found) == len({v for v in hit.values()}):
        print(f'✅ endpoint 白名单(账本里本批只有这一个,自动钉死):{found[0]}')
        return found[0]
    if allow_mixed:
        print(f'⚠️ --allow-mixed-models:不按 endpoint 过滤。账本里查到 {found or "(无)"}')
        return ''
    where = WORKDIR / 'effort_manifest.json'
    if not hit:
        raise SystemExit(
            f'⛔ 发车账本 {where} 里找不到 run 前缀 {run_filter!r} —— '
            f'查不出这批是哪个 endpoint 跑的,不出教材。\n'
            f'   要么补账本,要么显式传 --only-requested-model,'
            f'要么明知混了就传 --allow-mixed-models。')
    raise SystemExit(
        f'⛔ 本批混了多个 endpoint,不出教材:\n' +
        ''.join(f'     {k} -> {v}\n' for k, v in sorted(hit.items())) +
        f'   用 --only-requested-model 指定要收哪一个,或 --allow-mixed-models 明确放行。')


def run_rejected(run_name):
    """这条 run 该不该被踢出教材。返回理由字符串;None=留着。

    两道闸都在这里,**拉取和聚合两处共用同一份判断** —— 只在拉取那边拦是不够的,
    缓存里可能还留着上一次(没带这两个参数时)收进来的高档 rollout。
    """
    rn = run_name or ''
    if EXCLUDE_RE is not None and EXCLUDE_RE.search(rn):
        return 'exclude-run-re'
    if ONLY_MODEL:
        rm = requested_model_of(rn)
        if rm is None:
            return '发车账本里查无此 run'
        if rm != ONLY_MODEL:
            return f'endpoint={rm}'
    return None


def _load_manifest(workdir=None):
    global _MANIFEST
    if _MANIFEST is None:
        f = (Path(workdir) if workdir is not None else WORKDIR) / 'effort_manifest.json'
        try:
            _MANIFEST = json.loads(f.read_text()) if f.exists() else {}
        except Exception:
            _MANIFEST = {}
    return _MANIFEST


def endpoint_of_entry(v):
    """一条账本记录 → 这批教材的**老师 endpoint**。

    ⚠️ `requested_model` 记的是**泳道名**(ais-relay-a/b/h/l),不是 endpoint。
    一个 endpoint 挂多条泳道是常态(并行发卷),按泳道名比会把它误判成"混了多个模型"。
    账本里有 `upstream_model` 就以它为准,没有才回落到泳道名。
    ⛔ 闸(enforce_single_model)和过滤(run_rejected)**必须共用这一个函数** ——
    两边用不同判据会造成"闸看着开着、其实把所有 run 都剔了"的反向失效
    (0819 踩过一次:dict/字符串两种形状比较永远不相等)。
    """
    up = v.get('upstream_model')
    return str(up) if up else norm_requested_model(v.get('requested_model'))


def requested_model_of(run_name, workdir=None):
    """这条 run 发车时**实际请求的是哪个 endpoint**(发车账本里的原始字段)。

    比 `effort_of` 硬:effort 是推出来的,账本是发车当时记下来的原文。
    账本里没有这条 run → 返回 None,调用方必须显式处理(别默认放行)。
    """
    rn = run_name or ''
    man = _load_manifest(workdir)
    for prefix in sorted(man, key=len, reverse=True):
        if rn.startswith(prefix):
            return endpoint_of_entry(man[prefix])
    return None


def norm_requested_model(value):
    """把账本里的 requested_model 归一成一个字符串。

    ⛔ 这个字段**有两种形状**,发车脚本按走的通道不同各写各的:
        老的  'hy3-online-autotask-979884-low'
        新的  {'remind_agent_professional': 'deepseek-v4-flash-0731-online-autotask'}
    直接拿 dict 去和 --only-requested-model 的字符串比,**永远不相等** ——
    于是这道白名单闸会把整批 run 判成 endpoint 不符、静默剔光,收出 0 条教材。
    多 agent 的情况按 agent 名排序拼起来,这样"混了两个模型"也比得出来。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return ','.join(f'{k}={v}' for k, v in sorted(value.items())) if value else None
    return str(value)


def effort_of(run_name, workdir=None):
    """这条轨迹是哪个思考档跑出来的。

    ① 先查发车账本(submit_batch 发车时写的,权威)
    ② 查不到再按 run 名猜(兜底,**不可靠**:`rq2d_MAXSMOKE` 曾被认成 low)

    langfuse 的 run 名可能带后缀(` - 2026-08-09...`),所以按最长前缀匹配。
    """
    rn = run_name or ''
    man = _load_manifest(workdir)
    for prefix in sorted(man, key=len, reverse=True):
        if rn.startswith(prefix):
            return man[prefix].get('effort', 'low')
    r = rn.lower()
    for k in ('nothink', 'no_think', 'max', 'high'):
        if f'rq2d{k}' in r or f'_{k}_' in r or r.startswith(k):
            return 'no_think' if k in ('nothink', 'no_think') else k
    return 'low'          # 主批 rq2d_* / rq2d_r2_* / rq2d_r3_* 都是默认档


def get(p, retries=3, timeout=60):
    for a in range(retries):
        try:
            req = urllib.request.Request(EH + p, headers={'Authorization': 'Basic ' + AU})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(2 * (a + 1))


def exact_run_names(run_filter=''):
    """两个来源合并:① 守护日志里的 task_name ② langfuse 的 runs 列表。
    ②是必须的——手工提交(没走守护)的批次只有 runs 接口知道,曾漏收 1,254 条。"""
    names = set()
    for p in WORKDIR.glob('poll_*.jsonl'):
        for l in open(p):
            try:
                o = json.loads(l)
            except Exception:
                continue
            for f in o.get('frag') or []:
                tn = f.get('task_name')
                if tn:
                    names.add(tn)
    enc = urllib.parse.quote(DATASET, safe='')
    page = 1
    while True:
        r = get(f'/api/public/datasets/{enc}/runs?page={page}&limit=100')
        if r is None:
            raise RuntimeError(f'拉 run 列表第 {page} 页失败，拒绝拿半截列表继续收割')
        data = r.get('data') or []
        for x in data:
            if x.get('name'):
                names.add(x['name'])
        if not data or page >= ((r.get('meta') or {}).get('totalPages') or 1):
            break
        page += 1
    # run 名由本厂批次前缀生成；substring 会把 rq2d_extra_r1 之类旁批误收进来。
    return sorted(n for n in names if not run_filter or n.startswith(run_filter))


def write_cache(path, cache):
    """按稳定键原子重写，避免线程完成顺序改变 rows.jsonl。"""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        for k in sorted(cache):
            f.write(json.dumps(cache[k], ensure_ascii=False, sort_keys=True) + '\n')
    tmp.replace(path)


def dataset_id():
    d = get(f'/api/public/v2/datasets/{urllib.parse.quote(DATASET, safe="")}')
    return (d or {}).get('id')


def run_items(did, run_name):
    items, page = [], 1
    rn = urllib.parse.quote(run_name, safe='')
    while True:
        r = get(f'/api/public/dataset-run-items?datasetId={did}&runName={rn}&page={page}&limit=100')
        if r is None:
            raise RuntimeError(f'拉 run={run_name!r} 第 {page} 页失败，拒绝部分收割')
        data = (r or {}).get('data') or []
        if not data:
            break
        items += data
        if page >= (((r or {}).get('meta') or {}).get('totalPages') or 1):
            break
        page += 1
    return items


def harvestable(row, infra_tids, excluded_ids):
    """A static item exclusion must reject every rollout, including old PASSes."""
    return (not row.get('err') and row.get('iid') not in excluded_ids and
            row.get('tid') not in infra_tids)


def main():
    global DATASET, WORKDIR, OUT, LOCAL_JUDGE, LOCAL_JUDGE_SHA, PII_FILE, PII_MODE, TEACHER_DEFAULT, _MANIFEST, EXCLUDE_RE, ONLY_MODEL
    global ALLOW_PARTIAL_VERDICTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='本批数据集全名')
    ap.add_argument('--workdir', required=True, help='本批产出目录(poll_*.jsonl 在这)')
    ap.add_argument('--out', default='', help='收割产物目录,默认 <workdir>/harvest')
    ap.add_argument('--run-filter', default='', help='只收 run 名以这个前缀开头的')
    ap.add_argument('--exclude-run-re', default='',
                    help='再把 run 名匹配这个正则的踢掉。⚠️ 混档必须用它:同一道题的'
                         '低档和高档 rollout 会被聚到一起,stable_best 可能挑中**高档**那条'
                         '当老师,于是低档训练样本里被写进 max 前缀,而且全程不报错。')
    ap.add_argument('--only-requested-model', default='',
                    help='白名单:只收发车账本里 requested_model 正好等于这个值的 run。'
                         '比 --exclude-run-re 硬 —— 排除法要靠人记全所有高档 run 名,'
                         '明天新起一个名字就静默漏进来。账本里查不到的 run 一律**剔除并点名**。')
    ap.add_argument('--allow-partial-verdicts', action='store_true',
                    help='明知本批还没判完也要收(滚动收割)。⚠️ 那些判定数少于同批中位数的题'
                         '会被标 low_confidence 并在渲染时剔掉,**卷跑完后必须再收一次**,'
                         '否则永久丢失(0819 实案:v7 因此白扔 73 个任务)。')
    ap.add_argument('--allow-mixed-models', action='store_true',
                    help='⛔ 逃生口:明知本批混了多个 endpoint 还要收,才加这个。'
                         '不加时,发车账本里查不到本批 run、或查到多于一个 endpoint,一律**停住不出教材**。')
    ap.add_argument('--local-judge', default='', help='local_judge.py 的产物;给了就用它的判定')
    ap.add_argument('--keep-local-judged', action='store_true',
                    help='采信 local_judge 兜底拼证据(src=local)的判定;默认剔除')
    ap.add_argument('--lite', default='', help='remote_items_lite.jsonl,按批次给')
    ap.add_argument('--teacher', default='',
                    help='本批老师的模型名。一个批次只用一个老师时**必须传** —— '
                         '不传会按 run 名猜,猜不中就全标成 unknown-teacher,下游按老师过滤全错。'
                         '⚠️ 写**实测的模型名**,别抄别名:0819 查实 hy3-* 这个别名底下'
                         '跑的是 DeepSeek-V4-Flash,教材里那个 hy3-online-low 标签是假的')
    ap.add_argument('--pii-traces', default='',
                    help='mail_safety_audit.py --dump-pii 的产物(命中真人邮箱的 trace 名单)')
    ap.add_argument('--pii-mode', choices=['mark', 'drop'], default='mark',
                    help='mark(默认)=照常进教材、只打 pii 标记,一条不丢,'
                         '交给渲染那步脱敏;drop=整条剔出教材(会丢数据)')
    ap.add_argument('--infra-traces', required=True,
                    help='account_isolation_audit.py 产物；这些 trace 强制剔除')
    ap.add_argument('--excluded-ids', default='',
                    help='静态不可执行/不可作教材的完整 item ID，一行一个；'
                         '例如 audit_drive_identity.py 的 blocked IDs')
    ap.add_argument('--allow-missing', type=int, default=0,
                    help='容忍几条 trace 拉不到就继续出教材(默认 0=一条都不容忍)。'
                         '两千多条里总有个别永远落不了库,卡在这里等于整批教材出不来;'
                         '给了这个数就只警告并**点名**漏的是哪几条,超过还是停。')
    a = ap.parse_args()
    DATASET = a.dataset
    PII_FILE = a.pii_traces
    PII_MODE = a.pii_mode
    EXCLUDE_RE = re.compile(a.exclude_run_re) if a.exclude_run_re else None
    ONLY_MODEL = a.only_requested_model
    ALLOW_PARTIAL_VERDICTS = a.allow_partial_verdicts
    if a.teacher:
        TEACHER_DEFAULT = a.teacher
    else:
        print('⚠️ 没传 --teacher —— 老师名只能按 run 名猜,猜不中会全标成 '
              f'{TEACHER_DEFAULT},下游按老师过滤会错')
    WORKDIR = Path(a.workdir)
    _MANIFEST = None
    OUT = Path(a.out) if a.out else WORKDIR / 'harvest'
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'数据集={DATASET}\n工作目录={WORKDIR}\n产物={OUT}', flush=True)
    ONLY_MODEL = enforce_single_model(ONLY_MODEL, a.run_filter, a.allow_mixed_models)
    if a.local_judge and Path(a.local_judge).exists():
        LOCAL_JUDGE_SHA = sha256_file(a.local_judge)
        LOCAL_JUDGE = {}
        n_fb = 0
        for l in open(a.local_judge):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get('状态'):
                continue
            # 0811 评审整改(点10):local_judge 的兜底路径(src=local,平台没有
            # judge-model span、证据是本地拼的)与线上不可逐字比对 —— 默认不当 gold,
            # 显式 --keep-local-judged 才采信。
            if r.get('src') == 'local' and not a.keep_local_judged:
                n_fb += 1
                continue
            LOCAL_JUDGE[f"{r.get('run','')}|{r['iid']}"] = r
        print(f'本地判分 {len(LOCAL_JUDGE)} 条,sha256={LOCAL_JUDGE_SHA[:12]},优先采用'
              + (f';剔除兜底拼证据的 {n_fb} 条(--keep-local-judged 可保留)' if n_fb else ''),
              flush=True)

    # ⚠️ 键必须是 run|iid:同一个执行槽位会被不同老师(hy3-low / hy3-high / opus)重跑,
    # 按 iid 去重会把所有补做结果丢掉(踩过:high 296 + opus 1140 全部消失)
    cache = {}
    rows_f = OUT / 'rows.jsonl'
    if rows_f.exists():
        for l in open(rows_f):
            try:
                r = json.loads(l)
                cache[f"{r.get('run','')}|{r['iid']}"] = r
            except Exception:
                continue
    print(f'缓存 {len(cache)} 条')

    # ⚠️ 滚动收割的静默坑(0815 发现):卷还没判完就收割,拉到的 trace 判定是空的,
    # 却照样进缓存;下次收割因为「已在缓存里」永远不再回源 → 这道题的判定永久丢失,
    # n_verdicts 少算一个,连带把题打成 low_confidence 被渲染剔掉,而且全程不报错。
    # 所以缓存装载时一律淘汰空判定行,让它们下轮回源重拉。
    stale = [k for k, r in cache.items() if not str(r.get('judge') or '').strip()]
    for k in stale:
        del cache[k]
    if stale:
        print(f'缓存策略:淘汰空判定 {len(stale)} 条(判官当时还没判完),回源重拉', flush=True)

    if LOCAL_JUDGE is None:
        # 本轮明确没提供 local judge，就不能继续沿用历史缓存里的外部判定。
        # 删掉后会从评测 trace 重新取平台判定，落实“平台默认 gold”的策略。
        old_local = [k for k, r in cache.items() if r.get('src') == 'local']
        for k in old_local:
            del cache[k]
        if old_local:
            print(f'缓存策略:淘汰历史 local 判定 {len(old_local)} 条，回源平台', flush=True)

    # 0811 评审整改(点10):缓存对账 —— 缓存行的判定必须跟当前本地判分一致,
    # 否则「先收割后重判」的批次会把旧判定固化(判官/prompt 升级同理:换新的
    # local_judge 产物文件,这里就会整体刷新)。
    if LOCAL_JUDGE is not None:
        n_fix = n_evict = 0
        for k in list(cache):
            row = cache[k]
            lj = LOCAL_JUDGE.get(k)
            if row.get('src') == 'local' and not lj:
                del cache[k]  # 新判分文件不再包含它：回源平台，不能沿用旧 local 结论
                n_evict += 1
                continue
            if lj:
                fresh = {'judge': str(lj.get('状态') or ''), 'score': lj.get('分数'),
                         'attr': str(lj.get('失败归因') or '')[:40], 'src': 'local',
                         'judge_source_sha256': LOCAL_JUDGE_SHA}
                if any(row.get(x) != v for x, v in fresh.items()):
                    row.update(fresh)
                    n_fix += 1
        if n_fix or n_evict:
            print(f'缓存对账:刷新 {n_fix} 条,淘汰旧 local 判定 {n_evict} 条', flush=True)

    did = dataset_id()
    if not did:
        print('❌ datasetId 拿不到(langfuse 又病了?)'); sys.exit(1)
    names = exact_run_names(a.run_filter)
    dropped = [(n, run_rejected(n)) for n in names]
    dropped = [(n, why) for n, why in dropped if why]
    names = [n for n in names if not run_rejected(n)]
    for n, why in sorted(dropped):
        print(f'  ⊘ 不收 {n.split(" - ")[0]}:{why}', flush=True)
    print(f'run 名 {len(names)} 个(踢掉 {len(dropped)} 个)')
    todo = []
    for rn in names:
        for it in run_items(did, rn):
            iid = it.get('datasetItemId') or ''
            if iid and f'{rn}|{iid}' not in cache:
                todo.append((iid, it.get('traceId'), rn))
    seen = set()
    todo = [t for t in todo if not (f'{t[2]}|{t[0]}' in seen or seen.add(f'{t[2]}|{t[0]}'))]
    print(f'需新拉 {len(todo)} 条 trace')

    q = queue.Queue()
    for t in todo:
        q.put(t)
    lock = threading.Lock()
    n_done = [0]
    fetch_failed = [0]
    missing = []          # (run, iid, traceId) —— 拉不到的要点名,不能只报个数

    def worker():
        while True:
            try:
                iid, tid, rn = q.get_nowait()
            except queue.Empty:
                return
            if not tid:
                with lock:
                    fetch_failed[0] += 1
                    missing.append((rn, iid, '(没有 traceId)'))
                continue
            tr = get(f'/api/public/traces/{tid}') or {}
            out = tr.get('output') or {}
            if not tr or not isinstance(tr.get('output'), dict):
                with lock:
                    fetch_failed[0] += 1
                    missing.append((rn, iid, tid))
                continue
            ao_raw = out.get('actual_outcome')
            try:
                ao = json.loads(ao_raw) if isinstance(ao_raw, str) else (ao_raw or {})
            except Exception:
                ao = {}
            ev = out.get('eval_result') or {}
            d1 = next(iter(ev.values()), {}) if isinstance(ev, dict) and ev else {}
            # 判定来源:有本地判分就用本地的(平台被 respond 拼接 bug 污染,
            # 且 --no-judge 跑的批次平台压根没判)。平台判定留在 plat_* 供对照。
            lj = (LOCAL_JUDGE or {}).get(f'{rn}|{iid}')
            judge = str((lj or d1).get('状态', '') or '')
            score = (lj or d1).get('分数')
            attr = str((lj or d1).get('失败归因', '') or '')[:40]
            row = {'iid': iid, 'tid': tid, 'run': rn, 'effort': effort_of(rn),
                   'err': bool(out.get('error')),
                   'envelope': bool(ao.get('respond')), 'n_tools': len(ao.get('tool') or []),
                   'judge': judge, 'score': score, 'attr': attr,
                   'src': 'local' if lj else 'platform',
                   'judge_source_sha256': LOCAL_JUDGE_SHA if lj else '',
                   'plat_judge': str(d1.get('状态', '')),
                   'lf_link': out.get('langfuse_trace_link', '')}
            with lock:
                cache[f'{rn}|{iid}'] = row
                n_done[0] += 1
                if n_done[0] % 500 == 0:
                    print(f'... {n_done[0]}/{len(todo)}', flush=True)

    ths = [threading.Thread(target=worker) for _ in range(16)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    write_cache(rows_f, cache)
    print(f'rows 总计 {len(cache)}')
    if fetch_failed[0]:
        by_run = Counter(rn for rn, _, _ in missing)
        detail = '; '.join(f'{rn.split(" - ")[0]}×{n}' for rn, n in by_run.most_common(8))
        if fetch_failed[0] > a.allow_missing:
            raise SystemExit(f'⛔ {fetch_failed[0]} 条 trace 拉取失败/尚未落库；缓存了成功部分，'
                             f'重跑后再收割。分布:{detail}\n'
                             f'   (确认这几条永远落不了库,就用 --allow-missing {fetch_failed[0]} 放行)')
        print(f'⚠️ {fetch_failed[0]} 条 trace 拉不到,已按 --allow-missing {a.allow_missing} 放行。'
              f'分布:{detail}', flush=True)
        for rn, iid, tid in missing[:20]:
            print(f'   漏:{rn.split(" - ")[0]} iid={iid} trace={tid}', flush=True)

    # ---- 按题聚合与三件套 ----
    lite = {}
    _lite = Path(a.lite) if a.lite else WORKDIR / 'remote_items_lite.jsonl'
    for l in (open(_lite) if _lite.exists() else []):
        x = json.loads(l)
        lite[x['id']] = x
    def teacher_of(run):
        """这条轨迹是哪个老师跑的。

        ⚠️ 按 run 名猜只对"一个批次里混了多个老师"的情况有意义(rq1 那样补做过
        opus)。**一个批次只用一个老师时必须传 --teacher** ——
        否则 run 名一个都匹配不上,全落到兜底值 `unknown-teacher`,下游按老师名过滤就全错。
        rq2 踩过:run 名是 rq2_Q2_...,老师其实是 deepseek-v4-flash-max,标错了。
        """
        r = (run or '').lower()
        if 'dsfull' in r or 'dsv4' in r: return 'deepseek-v4-max'
        if 'opus5' in r: return 'opus-5'
        if 'opus' in r: return 'opus-4-6'
        return TEACHER_DEFAULT

    # 真人邮箱标记/剔除。名单由 mail_safety_audit.py --dump-pii 产出(它扫的是**整条
    # trace 正文**,不只是收件人)。⚠️ 这里不能自己扫 —— rows.jsonl 只存了判定和链接、
    # **没有正文**,在这儿写正则等于写了个永远不触发的闸(差点这么发出去)。
    #
    # 两种处理,默认 mark:
    #   mark(默认)—— 轨迹**照常进教材**,只在 sft_pass / dpo_pairs 上打 `pii: true`。
    #                 数据一条不丢,风险留在明面上,由渲染步骤按批次授权
    #                 决定保留原文还是脱敏。raw_internal_only 会保留原文，仍保留 pii 标记。
    #   drop      —— 整条剔出教材(rq2 首版这么干,会丢 4.2%;后改回 mark)。
    #
    # 为什么会有真人邮箱:题面和种子我们洗干净了,但**测试账号 Drive 里有前人留下的
    # 真实通讯录**,模型 google_sheets_get 一读就抄进了轨迹。洗种子挡不住这种。
    pii_tids = {}
    if PII_FILE and Path(PII_FILE).exists():
        for l in open(PII_FILE):
            try:
                x = json.loads(l)
                pii_tids[x['trace']] = x.get('addrs') or []
            except Exception:
                pass
        print(f'真人邮箱名单 {len(pii_tids)} 条 trace(来自 mail_safety_audit --dump-pii),'
              f'处理方式 = {PII_MODE}')
    else:
        print('⚠️ 没给 --pii-traces —— 教材里可能混进真人邮箱且无人知晓,'
              '收割前先跑 mail_safety_audit.py --dump-pii')

    infra_tids = set()
    if not Path(a.infra_traces).exists():
        raise SystemExit(f'⛔ 缺账号隔离审计产物:{a.infra_traces}')
    for line in open(a.infra_traces):
        try:
            infra_tids.add(json.loads(line)['trace'])
        except Exception:
            raise SystemExit(f'⛔ 账号隔离审计产物损坏:{a.infra_traces}')
    print(f'账号环境异常名单 {len(infra_tids)} 条 trace，强制剔除')

    excluded_ids = set()
    if a.excluded_ids:
        excluded_path = Path(a.excluded_ids)
        if not excluded_path.exists():
            raise SystemExit(f'⛔ 静态排除名单不存在:{excluded_path}')
        excluded_ids = {line.strip() for line in excluded_path.read_text().splitlines()
                        if line.strip()}
    print(f'静态不可执行题 {len(excluded_ids)} 条，强制剔除所有 rollout')

    byq = defaultdict(list)
    pii_hit = []
    n_excl_run = 0
    for key, r in cache.items():
        if run_rejected(r.get('run')):
            n_excl_run += 1
            continue
        if not harvestable(r, infra_tids, excluded_ids):
            continue
        if r.get('tid') in pii_tids:
            pii_hit.append({'iid': r['iid'], 'run': r.get('run'), 'tid': r.get('tid'),
                            'addrs': pii_tids[r['tid']][:5],
                            'action': 'dropped' if PII_MODE == 'drop' else 'kept_flagged'})
            if PII_MODE == 'drop':
                continue
            r['pii'] = True
        r['teacher'] = teacher_of(r.get('run'))
        base = re.sub(r'__r\d+$', '', r['iid'])
        byq[base].append(r)
    if pii_hit:
        (OUT / 'pii_flagged.jsonl').write_text(
            '\n'.join(json.dumps(x, ensure_ascii=False) for x in pii_hit) + '\n')
        verb = '剔除' if PII_MODE == 'drop' else '保留并打标(渲染时按批次政策处理)'
        print(f'真人邮箱:{len(pii_hit)} 条轨迹 → {verb} · 明细 {OUT}/pii_flagged.jsonl')
    sftf = open(OUT / 'sft_pass.jsonl', 'w')
    dpof = open(OUT / 'dpo_pairs.jsonl', 'w')
    allfail = []
    verdicts_per_q = {}      # 题 → 拿到过几个判定;给下面那道「卷拉全了没」的闸用
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])  # 题数, 有PASS题, 混合题, 全败题, PASS轨迹数
    for base, rs in sorted(byq.items()):
        # ⚠️ 两种 id 形态都要认:
        #   老批次「一题 N 条」→ id 是 `xxx__r1`,base 去掉后缀,查 lite 要**补回** __r1
        #   新批次「一题一条」→ id 本身就是 base,直接查
        # 0809 踩过:换成一题一条后只查 base+'__r1',全查不到 → 覆盖表里臂全是 "?"
        li = lite.get(base) or lite.get(base + '__r1') or {}
        arm = (li.get('merged_from') or li.get('arm') or '?')
        ps = [r for r in rs if r['judge'] in PASS_SET]
        fs = [r for r in rs if r['judge'] and r['judge'] not in PASS_SET]
        verdicts_per_q[base] = len(ps) + len(fs)
        st = stats[arm]
        st[0] += 1; st[1] += bool(ps); st[2] += bool(ps and fs); st[3] += (not ps and len(rs) >= 3); st[4] += len(ps)
        if ps:
            best = stable_best(ps)
            # ⚠️ n_verdicts = 这道题**真的拿到过几个判定**(PASS/FAIL 都算)。
            # 这是可信度的直接度量:实测同题三次重跑 30.4% 会翻盘,
            # **只有 1 个判定的 PASS 有相当概率是判官抖出来的**。
            # 下游按 low_confidence 过滤或降权,别一视同仁。
            n_verdicts = len(ps) + len(fs)
            sftf.write(json.dumps({'question': base, 'arm': arm, 'n_rollouts': len(rs),
                                   'user_key': li.get('user_key'),
                                   'source_rule_key': li.get('rule_key'),
                                   'n_verdicts': n_verdicts,
                                   'low_confidence': n_verdicts < 2,
                                   # ⛔ 老版写死 `len(rs) >= 3`,那是 ×5 固定轮次留下的门槛。
                                   # 自适应轮次下多数题只跑 2 轮 → **这个标记一次都没触发过**。
                                   # 改成「至少 2 个判定且全是 PASS」。
                                   'all_pass': (not fs) and n_verdicts >= 2,
                                   # 选中轨迹里有身份数据 → 渲染时必须显式选 raw/mask
                                   'pii': bool(best.get('pii')),
                                   'pass_iids': [r['iid'] for r in ps], 'recommended': best['iid'],
                                   # ⚠️ iid **不是**一条轨迹的唯一键 —— 同一道题会被多个 run
                                   # 重跑(所以本脚本的缓存键是 `run|iid`)。下游只拿 pass_iids
                                   # 去 rows.jsonl 里查,会「后来者覆盖」查到**别的 run**那条,
                                   # 于是这里刚按档位筛掉的高档轨迹又被捡回去当老师(实测 157 条)。
                                   # 所以把 run 一起带出去,下游按 (run, iid) 配对。
                                   'pass_rollouts': [{'iid': r['iid'], 'run': r.get('run')} for r in ps],
                                   'rec_trace': best['tid'], 'rec_lf_link': best['lf_link'],
                                   'rec_teacher': best.get('teacher'), 'rec_run': best.get('run'),
                                   # ⭐ 判官给推荐轨迹打的 D1 分(5=完全达标,4=单点轻微瑕疵)。
                                   # ⛔ 0821 补:**原来这行不写 score**,而 render_sft 的
                                   # `--min-score` 闸把「取不到分数」当作不合格 ——
                                   # 于是那道闸会静默剔光 100% 的行、渲出 0 条教材。
                                   # 教训同 [[sft-render-whole-not-explode]]:
                                   # 闸的验收标准是「在真路径上会响」,不是「写了」。
                                   'score': best.get('score'),
                                   'scores_seen': sorted(
                                       {r.get('score') for r in ps if isinstance(r.get('score'), int)}),
                                   # ⭐ 这条轨迹是哪个思考档跑出来的
                                   'effort': best.get('effort', 'low'),
                                   'efforts_seen': sorted({r.get('effort', 'low') for r in rs}),
                                   'judge_source_sha256': best.get('judge_source_sha256', ''),
                                   'teachers': sorted({r.get('teacher') for r in ps})}, ensure_ascii=False) + '\n')
        if ps and fs:
            worst = stable_worst(fs)
            best = stable_best(ps)
            dpof.write(json.dumps({'question': base, 'arm': arm,
                                   'user_key': li.get('user_key'),
                                   'source_rule_key': li.get('rule_key'),
                                   'n_verdicts': len(ps) + len(fs),
                                   # ⚠️ 这道题的 chosen 轨迹**同时是 SFT 的正例**(有 PASS 就进 SFT)。
                                   # 先 SFT 再 DPO 没问题;**混在一起训就是重复计数**。
                                   'also_in_sft': True,
                                   # 正例/反例任一条带真人邮箱 → 渲染前必须脱敏
                                   'pii': bool(best.get('pii') or worst.get('pii')),
                                   'chosen': best['iid'], 'chosen_trace': best['tid'],
                                   # ⚠️ 渲染器要的是 **live trace 链接**,不是评测 trace id ——
                                   # 轨迹正文(思考/工具/信封)只在 live trace 里。
                                   # 少了这两个字段 render_dpo 一条都拉不到(踩过)
                                   'chosen_lf_link': best['lf_link'],
                                   'rejected_lf_link': worst['lf_link'],
                                   'rejected': worst['iid'], 'rejected_trace': worst['tid'],
                                   'rejected_attr': worst['attr'],
                                   'chosen_effort': best.get('effort', 'low'),
                                   'rejected_effort': worst.get('effort', 'low'),
                                   'n_pass': len(ps), 'n_fail': len(fs)}, ensure_ascii=False) + '\n')
        if not ps and len(rs) >= 3:
            allfail.append(f'{base}\t{arm}\t{len(rs)}rollouts')
    sftf.close(); dpof.close()
    (OUT / 'allfail_questions.txt').write_text('\n'.join(allfail) + '\n')
    partial_verdict_gate(verdicts_per_q, ALLOW_PARTIAL_VERDICTS)
    if n_excl_run:
        print(f'--exclude-run-re 在聚合处又拦下缓存里的 {n_excl_run} 条 rollout', flush=True)

    # ⛔ 选中老师的档位分布 —— 低档 SFT 要的是**实际档位 low** 的轨迹。
    # 混了高档不会报错、渲染也照常出样本,只是 prompt 里被写进 max 前缀,
    # 到低档推理时分布外。所以这行必须打出来给人看。
    eff = Counter()
    for l in open(OUT / 'sft_pass.jsonl'):
        eff[json.loads(l).get('effort', 'low')] += 1
    print('选中老师的档位分布:' + ' '.join(f'{k}={v}' for k, v in eff.most_common()), flush=True)
    if set(eff) - {'low'}:
        print(f'⚠️ 教材里混了非 low 档轨迹 {sum(v for k, v in eff.items() if k != "low")} 条 —— '
              f'低档 SFT 要么用 --exclude-run-re 把高档 run 踢掉,'
              f'要么 prepare 那步别用 --reasoning-effort source', flush=True)

    rep = [f'# 收割覆盖表({time.strftime("%F %T")} UTC)\n',
           '| 臂 | 已见题数 | 有PASS题 | 混合题(DPO可用) | 全败题(≥3尝试) | PASS轨迹 |',
           '|---|---|---|---|---|---|']
    for arm in sorted(stats):
        s = stats[arm]
        rep.append(f'| {arm} | {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} |')
    tot = [sum(s[i] for s in stats.values()) for i in range(5)]
    rep.append(f'| **合计** | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} |')
    txt = '\n'.join(rep)
    (OUT / 'coverage.md').write_text(txt)
    print('\n' + txt)


if __name__ == '__main__':
    main()
