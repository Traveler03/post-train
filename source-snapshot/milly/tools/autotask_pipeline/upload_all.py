#!/usr/bin/env python3
"""阶段4b:种子上 S3 + items 上 Langfuse(REST 直传,断点续传+校验)。

用法:
  python3 upload_all.py --seeds          # 只传种子
  python3 upload_all.py --items         # 只传 items
  python3 upload_all.py                 # 全部
凭证:S3 从 dataset-generator skill 的 .env 读;Langfuse 用评测环境共享 key。
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline_common import dataset_item_fingerprint

D = Path(__file__).resolve().parent.parent
SEEDS = D / 'out' / 'seeds'
ITEMS = D / 'out' / 'rq1_items.jsonl'
ENV_FILE = Path('/home/work/migoo_ai_public/haiyang/training/eval_framework/.claude/skills/dataset-generator/.env')

BASE = 'https://langfuse.us.migoo.shopee.io'
AUTH = os.environ.get('LANGFUSE_AUTH_B64', '')
DS = ''   # ⛔ 不设默认值:由 --dataset(必填)注入。
          #    原来写死的是 **rq1** 的卷 —— 直接 import 本模块调函数会把 rq3w 的题传进 rq1 卷。
S3_KEY_PREFIX = 'seeds/qa-autotask-rq1'


def load_env():
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            env[k.strip()] = v.strip().strip('"\'')
    return env


def _gate_no_templates(seeds_dir, items_file):
    """⛔ 上传前硬闸:题库和种子里**不许残留没渲染的模板占位符**。

    0815 实案:补题 146 道整卷废掉,原因是装配后**忘了跑 `bake_dates.py`**。
    平台原样把 `- now: {{ fmt_date(run_date, 'iso') }}` 发给模型 ——
    模型不知道"现在几点",时间敏感的族直接崩:
    晨报 27.2%(主批同族 68.8%)、晚间 37.5%,而不依赖时间的客户族 71.4% 毫发无损。

    **当时所有的闸都是绿的**:lint 通过(146 files, 0 errors)、种子闸通过、
    上传逐条对账通过。因为 lint 只看 `cases_raw`(那里**本来就该**有模板,
    等着后面渲染),而**没有任何一道闸看装配之后的成品**。这道闸就是补这个缺口。

    放在上传这一步是故意的:它是进平台前的最后一关,
    不管前面漏跑了装配、烘焙还是修模板,都拦得住。
    """
    import json as _j, re as _re, sys as _s
    from pathlib import Path as _P
    pat = _re.compile(r'\{\{|\{%')
    bad_items, n_items = [], 0
    for line in _P(items_file).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        n_items += 1
        o = _j.loads(line)
        if pat.search(_j.dumps(o, ensure_ascii=False)):
            bad_items.append((o.get('input') or {}).get('id') or '?')
    bad_seeds = [f.name for f in sorted(_P(seeds_dir).glob('*.json'))
                 if pat.search(f.read_text(encoding='utf-8'))]
    if bad_items or bad_seeds:
        print(f'⛔ 模板闸没过 —— 装配后的成品里还有没渲染的占位符,传上去整卷会废:', flush=True)
        if bad_items:
            print(f'   题库 {len(bad_items)}/{n_items} 条,例 {bad_items[:3]}', flush=True)
        if bad_seeds:
            print(f'   种子 {len(bad_seeds)} 份,例 {bad_seeds[:3]}', flush=True)
        print('   先跑:python3 scripts/fix_templates.py --items <题库> --seeds <种子目录>\n'
              '       python3 scripts/bake_dates.py     --items <题库> --seeds <种子目录>', flush=True)
        _s.exit(2)
    print(f'✅ 模板闸通过:题库 {n_items} 条、种子 {len(list(_P(seeds_dir).glob("*.json")))} 份零占位符',
          flush=True)


def _gate_seeds(seeds_dir, items_file):
    """⚠️ 上传前硬闸:拿**盘上原样**的种子喂注入器真函数,不预修。

    踩过两次:
      ① 重装配把修好的种子冲掉、没重修就传 → 中试 503 条全死在 seed inject failed;
      ② 用 fix_seeds_v2 --check 当闸 —— 它是"内存里先修一遍再验",
         报"全过"其实是"修完能过",盘上的数据照样是坏的。
    所以这里必须原样验。
    """
    import glob as _g, json as _j, os as _o, sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent))
    _o.environ.setdefault('PYTHONPATH', _o.path.expanduser('~/.local/lib/python3.12/site-packages'))
    from fix_seeds_v2 import build_emails, build_calendar_events, RUN_DT
    bad, n = {}, 0
    for f in sorted(_g.glob(str(_P(seeds_dir) / '*.json'))):
        sd = _j.load(open(f)); n += 1
        try:
            if sd.get('emails'):
                build_emails('migoo_testbeeai_48@shopee.com',
                             _j.loads(_j.dumps(sd['emails'])), 'shopee.com', run_date=RUN_DT)
        except Exception as e:
            bad.setdefault(f'emails:{type(e).__name__}', []).append(_P(f).name)
        try:
            if sd.get('calendar_events'):
                build_calendar_events(_j.loads(_j.dumps(sd['calendar_events'])), run_date=RUN_DT)
        except Exception as e:
            bad.setdefault(f'calendar:{type(e).__name__}', []).append(_P(f).name)
        for ct in sd.get('contacts') or []:
            if 'id' not in ct or not (ct.get('givenName') or ct.get('familyName')):
                bad.setdefault('contacts:形状', []).append(_P(f).name); break
        for l in sd.get('labels') or []:
            if not isinstance(l, dict) or 'id' not in l or 'name' not in l:
                bad.setdefault('labels:形状', []).append(_P(f).name); break
    if bad:
        print(f'⛔ 种子闸没过({n} 份里有问题的):', flush=True)
        for k, v in bad.items():
            print(f'   {k}: {len(v)} 份,例 {v[:3]}', flush=True)
        print('   先跑:python3 scripts/fix_seeds_v2.py --seeds <种子目录> --items <题库>', flush=True)
        sys.exit(2)
    print(f'✅ 种子闸通过:{n} 份原样喂真注入器,零问题', flush=True)


def upload_seeds():
    import boto3
    env = load_env()
    s3 = boto3.client('s3', endpoint_url=env['BEEAI_S3_ENDPOINT'],
                      aws_access_key_id=env['BEEAI_S3_ACCESS_KEY'],
                      aws_secret_access_key=env['BEEAI_S3_SECRET_KEY'])
    bucket = env['BEEAI_S3_BUCKET']
    files = sorted(SEEDS.glob('*.json'))
    print(f'uploading {len(files)} seeds -> s3://{bucket}/{S3_KEY_PREFIX}/', flush=True)
    ok = 0
    for i, fp in enumerate(files):
        key = f'{S3_KEY_PREFIX}/{fp.name}'
        s3.upload_file(str(fp), bucket, key)
        ok += 1
        if (i + 1) % 50 == 0:
            print(f'... {i+1}', flush=True)
    # 抽查 3 个:下载回来比 sha
    import random
    random.seed(1)
    for fp in random.sample(files, min(3, len(files))):
        key = f'{S3_KEY_PREFIX}/{fp.name}'
        body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        same = hashlib.sha256(body).hexdigest() == hashlib.sha256(fp.read_bytes()).hexdigest()
        print(f'verify {fp.name}: {"OK" if same else "MISMATCH"}', flush=True)
        if not same:
            sys.exit(1)
    print(f'seeds uploaded: {ok}', flush=True)


def call(method, path, body=None, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + path, method=method,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Authorization': 'Basic ' + AUTH, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404 and method == 'GET':
                return None
            # Retrying deterministic client conflicts only delays the same failure and
            # hides the useful status. In particular, dataset item IDs are globally
            # unique across datasets, not merely unique inside one dataset.
            if e.code in (400, 401, 403, 409, 422):
                detail = e.read().decode('utf-8', errors='replace')[:1000]
                raise RuntimeError(f'HTTP {e.code} {method} {path}: {detail}') from e
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))


def upload_items():
    items = [json.loads(l) for l in open(ITEMS)]
    print(f'待传 {len(items)} 条 -> {DS}', flush=True)
    ds = call('GET', '/api/public/v2/datasets/' + urllib.request.quote(DS, safe=''))
    if not ds:
        call('POST', '/api/public/datasets', {'name': DS, 'description':
             'rq 工厂 v1:真实线上 query 原文 × 全新种子世界,427 题 ×5(拒绝采样用,Hy3 low-think 收教材)。源=lsy v1_raw_best 0727;0801 建'})
        print('数据集已创建', flush=True)
    # 已存在的 ID 不再无脑跳过 —— 对 input/expectedOutput/metadata 做完整规范化指纹,
    # 不一致默认拒绝继续(防「重新装配后文档说重传了、远端其实还是旧内容」),
    # 显式 --upsert 才允许覆盖重传。
    def fetch_remote():
        out, page = {}, 1
        while True:
            r = call('GET', f'/api/public/dataset-items?datasetName={urllib.request.quote(DS, safe="")}&page={page}&limit=100')
            data = (r or {}).get('data') or []
            if not data:
                break
            for d in data:
                out[d.get('id')] = dataset_item_fingerprint(d)
            if page >= ((r.get('meta') or {}).get('totalPages') or 1):
                break
            page += 1
        return out

    remote = fetch_remote()
    fresh, stale = [], []
    for it in items:
        rid = it['input']['id']
        if rid not in remote:
            fresh.append(it)
        elif remote[rid] != dataset_item_fingerprint(it):
            stale.append(it)
    print(f'已存在 {len(remote)} 条;新条 {len(fresh)};内容漂移 {len(stale)}', flush=True)
    if stale and not UPSERT:
        for it in stale[:5]:
            print(f"  漂移: {it['input']['id']}", flush=True)
        print('❌ 远端已有同 ID 但内容不同。确认要覆盖就加 --upsert;否则先想清楚谁是对的。', flush=True)
        sys.exit(1)
    # Langfuse dataset item IDs are globally unique. A rebuild into a new dataset
    # can therefore collide even when that new dataset is empty. Probe every fresh
    # ID before starting writes so we fail before creating a partial dataset.
    def globally_exists(it):
        rid = it['input']['id']
        found = call('GET', '/api/public/dataset-items/' + urllib.parse.quote(rid, safe=''))
        return rid if found else None

    if fresh:
        with ThreadPoolExecutor(16) as ex:
            conflicts = [rid for rid in ex.map(globally_exists, fresh) if rid]
        if conflicts:
            for rid in conflicts[:5]:
                print(f'  全局 ID 冲突:{rid}', flush=True)
            print(f'❌ {len(conflicts)} 个 item ID 已被其他数据集占用。'
                  '从旧 cases 重建新版本时请给 assemble_items.py 传新的 --id-prefix。',
                  flush=True)
            sys.exit(1)
        print(f'✅ 全局 ID 预检通过:{len(fresh)} 个新 ID 均未占用', flush=True)
    # ⚠️ 串行发实测只有 1.7 条/秒,11,350 条要两小时。并发 16 → 十几分钟。
    todo = fresh + (stale if UPSERT else [])
    lock, done = threading.Lock(), [0]

    def put(it):
        call('POST', '/api/public/dataset-items', {
            'datasetName': DS, 'id': it['input']['id'],
            'input': it['input'], 'expectedOutput': it['expectedOutput'],
            'metadata': it['metadata']})
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(f'... {done[0]}/{len(todo)}', flush=True)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(put, todo))
    n = done[0]
    print(f'新传 {n} 条', flush=True)
    # 0811 评审整改:终验从「总数+抽3条」升级为「逐条内容对账」——
    # 每个 ID 必须存在且完整 payload 与本地一致,任一不符即失败。
    # 这同时兜住「--upsert 后端其实不支持覆盖」的情况:覆盖没生效会在这里现形。
    remote = fetch_remote()
    missing = [it['input']['id'] for it in items if it['input']['id'] not in remote]
    drift = [it['input']['id'] for it in items
             if it['input']['id'] in remote
             and remote[it['input']['id']] != dataset_item_fingerprint(it)]
    print(f'终验:远端 {len(remote)} 条;缺 {len(missing)};内容不一致 {len(drift)}', flush=True)
    if missing or drift:
        for x in (missing + drift)[:5]:
            print(f'  问题条: {x}', flush=True)
        print('❌ 终验失败', flush=True)
        sys.exit(1)
    print('✅ 上传校验通过(全量逐条对账)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', action='store_true')
    ap.add_argument('--items', action='store_true')
    # ⚠️ 批次路径必须显式传。写死 rq1 路径的脚本已经害我们静默检错批次一次(fix_seeds_v2)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--items-file', required=True)
    ap.add_argument('--dataset', required=True, help='本批数据集全名')
    ap.add_argument('--s3-prefix', required=True, help='如 seeds/qa-autotask-rq3')
    ap.add_argument('--upsert', action='store_true',
                    help='远端同 ID 内容不同时允许覆盖重传(默认拒绝继续)')
    a = ap.parse_args()
    SEEDS = Path(a.seeds_dir); ITEMS = Path(a.items_file)
    DS = a.dataset; S3_KEY_PREFIX = a.s3_prefix; UPSERT = a.upsert
    print(f'种子={SEEDS}\n题库={ITEMS}\n数据集={DS}\nS3前缀={S3_KEY_PREFIX}', flush=True)
    # ⛔ 模板闸放最前面:它拦的是「漏跑烘焙」,而只传题库(--items)那条路
    # 同样会踩,所以两条路都要过闸,不能塞进 _gate_seeds 里。
    _gate_no_templates(SEEDS, ITEMS)
    if a.seeds or not (a.seeds or a.items):
        _gate_seeds(SEEDS, ITEMS)
        upload_seeds()
    if a.items or not (a.seeds or a.items):
        upload_items()
