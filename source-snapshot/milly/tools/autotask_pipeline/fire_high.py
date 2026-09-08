#!/usr/bin/env python3
"""提交并守护一批 AutoTask rollout；首块结束后自动验模型与管线。

用法: fire_high.py --ids out/batch_ids.txt --model <精确部署名> ...
每轮一个 run 名 <batch>_<tag>_rN_bMM，状态写到显式 --state。
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

# 0818 工具链从 tools/autotask_pipeline/ 搬到 tools/autotask_pipeline/ 之后,
# 「脚本目录 = ROOT/scripts」这个假设不再成立 —— 用 HERE(脚本自己所在目录)。
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import submit_batch as sb

MODEL = 'dsv4-flash-online-autotask-983552-max'
STATE = ROOT / 'out' / 'high_state.json'
# 换老师只改这两处:--model 传平台的「被测模型」名,--tag 决定 run 名前缀(收割器按 run 名认老师)


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def resume_decision(entry, platform_present, run_items, expected):
    """Return skip/watch/submit/abort for a persisted chunk.

    A previous process can die after the remote task was accepted.  Blindly
    submitting that chunk again creates duplicate trajectories and makes the
    later SFT selector depend on an accidental restart.  Ambiguous partial
    states therefore fail closed and require an explicit recovery ID list.
    """
    status = (entry or {}).get('status')
    if status == 'done':
        return 'skip'
    if status == 'partial':
        return 'abort'
    if status == 'running':
        if platform_present or run_items >= expected * 0.98:
            return 'watch'
        return 'abort'
    # The process can die in the narrow window after the webhook accepted the
    # task but before local state was persisted.  Exact remote-name evidence is
    # stronger than the absent ledger: reclaim it, never submit a duplicate.
    if platform_present or run_items >= expected * 0.98:
        return 'watch'
    if run_items:
        return 'abort'
    return 'submit'


def receipt_task_id(receipt):
    data = receipt.get('data') if isinstance(receipt, dict) else None
    return (receipt.get('task_id') or receipt.get('taskId') or
            (data or {}).get('task_id') or (data or {}).get('taskId') or '')


def concurrent_autotask_conflicts(tasks, own_prefix, executor_email):
    """Return this executor's other active AutoTask jobs.

    AutoTask cases mutate shared sandbox accounts via cleanup + seed injection.
    Running synthesis beside a target evaluation can therefore erase the other
    run's world.  Same-batch chunks are allowed; unrelated jobs fail closed.
    """
    if isinstance(tasks, dict) and tasks.get('_err'):
        raise ValueError(f"task list unavailable: {tasks.get('_err')}")
    rows = tasks if isinstance(tasks, list) else (
        (tasks.get('data') or tasks.get('tasks') or []) if isinstance(tasks, dict) else None)
    if not isinstance(rows, list):
        raise ValueError(f'task list has unexpected shape: {type(tasks).__name__}')
    return [row for row in rows
            if row.get('status') in ('running', 'queued', 'pending')
            and str(row.get('dataset_name') or '').startswith('benchmark/AutoTask')
            and str(row.get('executor_user') or '') == str(executor_email or '')
            and not str(row.get('task_name') or '').startswith(own_prefix)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True)
    ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--conc', type=int, default=20)
    ap.add_argument('--chunk', type=int, default=300)
    ap.add_argument('--email', default='haiyang.xue@shopee.com')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--tag', default='high')
    ap.add_argument('--dataset', required=True, help='本批数据集全名')
    ap.add_argument('--batch', required=True, help='run 名前缀,必须跟批次一致')
    ap.add_argument('--no-judge', action='store_true',
                    help='⛔ 别用:线上版本会崩(wanted_image_vars),65%% 用例直接错。见 submit_batch 注释')
    ap.add_argument('--judge-model', default='gpt-5.5-2026-04-23',
                    help='平台侧判官模型。**autotask 管线下它的分就是最终分**,不再本地重判'
                         '(平台的 strip_reasoning 已把 respond 剥成干净信封,实测跟我们'
                         '自己抽的逐字节相同)。⛔ 别为省钱换 nano —— 它把 89%% 的 PASS 判成 FAIL')
    ap.add_argument('--judge-prompt', default='migoo-autotask-prompt')
    ap.add_argument('--judge-prompt-version', type=int, default=13,
                    help='平台 judge prompt 的精确版本；当前正式口径为 v13')
    ap.add_argument('--env', default='live', choices=['live', 'liveish', 'test'],
                    help='运行环境。⚠️ 决定哪些模型点得到 —— 自部署模型只在 live 注册,'
                         '在 liveish 点它会静默回落到默认模型。换环境务必重验')
    # 0811 评审整改(点8):必填 —— 空值曾是 rq2 第二次翻车(整批跑成聊天管线)的直接原因,
    # 已知致命坑不允许再当默认值。
    ap.add_argument('--app-name', required=True,
                    help='ADK app_name。本厂(AutoTask)只认 remind_agent_professional;'
                         '确要跑别的管线须同时传 --allow-other-app')
    ap.add_argument('--allow-other-app', action='store_true',
                    help='放行 remind_agent_professional 以外的 app_name(明知故犯开关)')
    ap.add_argument('--skip-verify', action='store_true',
                    help='跳过首块自动验模(明知故犯开关;正式批禁用)')
    ap.add_argument('--model-config-overrides', default='',
                    help='JSON 串,请求级运行时覆盖。不能把 reasoning_effort 字段存在'
                         '等价成实际档位；983552-max 直接用精确模型名且不传此项')
    ap.add_argument('--state', default='')
    ap.add_argument('--allow-drift', type=int, default=0,
                    help='容忍冻结锚点与今天差几天。默认 0 = 必须同一天;'
                         '给上一批补跑时才放宽(必须跟原批同锚点)')
    ap.add_argument('--skip-gold-check', action='store_true',
                    help='跳过 gold 粒度闸。⚠️ 只在复现老批次时用')
    ap.add_argument('--allow-bad-gold-frac', type=float, default=0.0,
                    help='容忍多少比例的题 gold 超标(默认 0)。实测代价约 15 分,见 preflight_gold.py')
    ap.add_argument('--skip-anchor-check', action='store_true',
                    help='⛔ 跳过时间闸。只有确知这批不冻结时才用')
    ap.add_argument('--run-anytime-contract', action='store_true',
                    help='用显式虚拟时钟契约替代真实墙钟闸；会拒绝实时天气/行情/新闻题，'
                         '不是无条件跳过时间检查')
    ap.add_argument('--wall-clock-tolerance-min', type=int, default=-1,
                    help='schedule shard 必传；最多允许晚于冻结锚点多少分钟')
    ap.add_argument('--early-tolerance-min', type=int, default=0,
                    help='最多允许早于冻结锚点多少分钟；默认 0')
    ap.add_argument('--verify-model-after', default='',
                    help='发车后自动核验实际跑的模型(传期望的模型名子串,如 deepseek-v4-flash)。'
                         '⛔ 强烈建议中试时传 —— rq2 整批 11,350 次跑错模型就是因为没这道闸')
    ap.add_argument('--verify-clock-after', action='store_true',
                    help='首块发完后核验「模型收到的 <msg_time> == 我们冻结的锚点」。\n'
                         '⭐ 锚点不等于发车当天时**强烈建议带上** —— 0819 实测平台钟会跟着\n'
                         '   锚点走(连星期几都对),但那是 n=1 的观测、线上随时可能变;\n'
                         '   带上这道闸,变了会当场停发余量,而不是整批数据事后才发现不可用。')
    ap.add_argument('--allow-concurrent-autotask', action='store_true',
                    help='明知同一 executor 还有其他 AutoTask cleanup/inject 任务仍发车')
    a = ap.parse_args()
    if a.dataset:
        sb.DATASET = a.dataset
    print(f'数据集={sb.DATASET}', flush=True)

    if not a.allow_concurrent_autotask:
        active_tasks, _ = sb.tasks_raw()
        try:
            conflicts = concurrent_autotask_conflicts(
                active_tasks, f'{a.batch}_', a.email)
        except ValueError as e:
            sys.exit(f'⛔ 无法可靠获取平台任务列表：{e}。为防并发污染，本批失败关闭。')
        if conflicts:
            examples = [str(row.get('task_name') or '') for row in conflicts[:5]]
            sys.exit(
                f'⛔ 同一 executor 有 {len(conflicts)} 个其他 AutoTask 正在运行：{examples}。'
                '这些任务可能复用假人账号并互相 cleanup/inject，本批失败关闭。'
            )

    # ⛔ 时间闸:锚点必须 = 开跑当天,否则模型会看到"几天后的邮件"。原委见 preflight_anchor.py
    drift_receipt = None     # 走 --skip-anchor-check / --run-anytime-contract 时不会有回执
    if a.run_anytime_contract and a.skip_anchor_check:
        sys.exit('⛔ --run-anytime-contract 与 --skip-anchor-check 不能同时使用')
    if a.run_anytime_contract:
        clock_cmd = [sys.executable,
                     str(HERE / 'virtual_clock_contract.py'),
                     '--dataset', sb.DATASET, '--ids', a.ids]
        if subprocess.run(clock_cmd).returncode != 0:
            sys.exit('⛔ 显式虚拟时钟契约没过，不发车')
    elif not a.skip_anchor_check:
        # 回执写在 state 旁边,和 effort_manifest.json 同目录 —— 下面折进账本
        drift_receipt = (Path(a.state).parent if a.state else ROOT / 'out') / 'anchor_drift.json'
        anchor_cmd = [sys.executable, str(HERE / 'preflight_anchor.py'),
                      '--dataset', sb.DATASET, '--ids', a.ids,
                      '--allow-drift', str(a.allow_drift),
                      '--receipt', str(drift_receipt)]
        if a.wall_clock_tolerance_min >= 0:
            anchor_cmd += ['--wall-clock-tolerance-min',
                           str(a.wall_clock_tolerance_min),
                           '--early-tolerance-min', str(a.early_tolerance_min)]
        rc = subprocess.run(anchor_cmd).returncode
        if rc != 0:
            sys.exit('⛔ 时间闸没过,不发车。要么重装配把锚点定成今天,要么 --allow-drift N 明确放行')
    # ⛔ gold 闸:判分标准太细一律不发车。原委见 preflight_gold.py。
    #    ⭐ 它必须挂在**发车**这道总闸门上,不能挂在某一步的出口 ——
    #    0819 实案:gold 粒度的修复是 regold_batch 这个**独立步骤**在做的,
    #    我漏跑了它,四卷全用长 gold 发出去,净损失实测 15.7 分,
    #    还差点把掉的分归因成「世界配方变难了」。挂在上游任何一步都拦不住"跳过那一步"。
    if not a.skip_gold_check:
        gold_cmd = [sys.executable, str(HERE / 'preflight_gold.py'),
                    '--dataset', sb.DATASET, '--ids', a.ids,
                    '--allow-bad-frac', str(a.allow_bad_gold_frac)]
        if subprocess.run(gold_cmd).returncode != 0:
            sys.exit('⛔ gold 闸没过,不发车。先跑 regold_batch.py,'
                     '或 --allow-bad-gold-frac N 明确放行(知道自己在放弃约 15 分)')
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    global STATE, ANCHOR_DRIFT
    # ⛔ 锚点偏差必须落进账本。CLAUDE.md 那条「档位没落进 run 档案,事后无法核」
    #    是同一个病:偏差只印在屏幕上,卷跑完就查不出来了。
    ANCHOR_DRIFT = None          # None = 没查过(--skip-anchor-check 之类),**别混同于 0**
    if drift_receipt is not None:
        try:
            ANCHOR_DRIFT = json.loads(Path(drift_receipt).read_text()).get('anchor_drift_days', 0)
        except Exception as exc:
            print(f'⚠️ 锚点回执读不到({exc}),账本里记 None')
    STATE = Path(a.state) if a.state else ROOT / 'out' / f'{a.tag}_state.json'
    st = state()
    chunks = [ids[i:i + a.chunk] for i in range(0, len(ids), a.chunk)]
    print(f'{a.tag} 第{a.round}轮:{len(ids)} 题 / {len(chunks)} 块 @conc{a.conc} '
          f'被测={a.model} 平台判官={a.judge_model} 环境={a.env} app={a.app_name or "assistant(默认)"}', flush=True)
    # 0811 评审整改(点8):app_name 收白名单;验模从「打印命令请人跑」改为
    # 首块收官后自动执行,不过闸就停发余量。
    if a.app_name != 'remind_agent_professional' and not a.allow_other_app:
        sys.exit(f'⛔ 本厂只认 remind_agent_professional,收到 {a.app_name!r}。'
                 f'确要跑别的管线加 --allow-other-app。')
    for bi, block in enumerate(chunks, 1):
        name = f'{a.batch}_{a.tag}_r{a.round}_b{bi:02d}'
        entry = st.get(name) or {}
        # 先用平台任务 + Langfuse run items 对本地状态做对账，再决定是否提交。
        # 这道闸专门防 tmux/session 中断后重跑脚本时重复发车。
        _, raw = sb.tasks_raw()
        n_existing, run_names = sb.run_items_count(name, sb.DATASET)
        action = resume_decision(entry, name in raw, n_existing, len(block))
        if action == 'skip':
            print(f'{name} 已完成,跳过')
            continue
        if action == 'abort':
            sys.exit(f'⛔ {name} 本地状态={entry.get("status")!r}, '
                     f'平台进行中={name in raw}, run_items={n_existing}/{len(block)} '
                     f'({run_names})。为防重复轨迹已失败关闭；请先对账，'
                     f'只为未完成 ID 新建 recovery 文件/新 tag。')

        task_id = entry.get('task_id', '')
        if action == 'watch':
            print(f'♻️ {name} 认领上次已提交任务，不重复发车：'
                  f'run_items={n_existing}/{len(block)}', flush=True)
            if not entry:
                st[name] = {'status': 'running', 'n': len(block),
                            'task_id': task_id, 'reclaimed_without_local_state': True,
                            't': time.strftime('%F %T')}
                save(st)
        else:
            for attempt in range(17):
                r = sb.submit(block, name, a.conc, dataset=sb.DATASET, email=a.email,
                              model=a.model, judge=not a.no_judge, judge_model=a.judge_model,
                              judge_prompt=a.judge_prompt,
                              judge_prompt_version=a.judge_prompt_version,
                              env=a.env, app_name=a.app_name,
                              model_config_overrides=(json.loads(a.model_config_overrides)
                                                      if a.model_config_overrides else None),
                              effort_manifest=STATE.parent / 'effort_manifest.json',
                              anchor_drift=ANCHOR_DRIFT)
                print(f'{name} 回执(第{attempt + 1}次):', json.dumps(r, ensure_ascii=False)[:200], flush=True)
                if not r.get('_err'):
                    task_id = receipt_task_id(r)
                    break
                time.sleep(20)
                _, raw = sb.tasks_raw()
                if name in raw:
                    print(f'⚠️ 回执失败但平台已有 {name},认领'); break
                if attempt == 16:
                    print('⛔ 提交连续失败,停'); sys.exit(2)
                time.sleep(420)
            st[name] = {'status': 'running', 'n': len(block), 'task_id': task_id,
                        't': time.strftime('%F %T')}
            save(st)
        verdict, n_last = sb.watch(name, len(block), task_id=task_id,
                                   dataset=sb.DATASET,
                                   log=str(STATE.parent / f'poll_{name}.jsonl'))
        st[name] = {'status': 'done' if n_last >= len(block) * 0.85 else 'partial',
                    'n': len(block), 'n_done': n_last, 'verdict': verdict, 't': time.strftime('%F %T')}
        save(st)
        print(f'{name}: {verdict} {n_last}/{len(block)}', flush=True)
        # 首块自动验模+验管线(0811,点8):expect 默认取 --model,--skip-verify 才跳过。
        # 不过闸立即停发,余下块不发 —— 防「整批跑错模型/管线」再次发生。
        if bi == 1 and not a.skip_verify:
            expect = a.verify_model_after or a.model
            vm = [sys.executable, str(Path(__file__).resolve().parent / 'verify_model.py'),
                  '--dataset', sb.DATASET, '--run', name, '--expect', expect,
                  '--expect-agent', a.app_name]
            if isinstance(a.model, str):
                vm += ['--expect-request', a.model]
            print(f'首块验模: {" ".join(vm)}', flush=True)
            if subprocess.run(vm).returncode != 0:
                sys.exit(f'⛔ 首块验模不过闸(expect={expect} / agent={a.app_name}),停发余量。')
        if a.verify_clock_after and bi == 1:
            vc = [sys.executable, str(Path(__file__).resolve().parent / 'verify_clock.py'),
                  '--dataset', sb.DATASET, '--run', name]
            print(f'首块验钟: {" ".join(vc)}', flush=True)
            if subprocess.run(vc).returncode != 0:
                sys.exit('⛔ 首块验钟不过闸(平台钟没跟上锚点),停发余量。')
    print(f'✅ {a.tag} 补做本轮结束', flush=True)


if __name__ == '__main__':
    main()
