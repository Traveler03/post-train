#!/usr/bin/env python3
# ruff: noqa: E501
"""Render a report focused on within-segment milestone reward decay."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

if __package__:
    from .render_turn_reward_misattribution import build_summary, load_trajectories
else:
    from render_turn_reward_misattribution import build_summary, load_trajectories


def _num(value: int | float) -> str:
    return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"


def _pct(value: float) -> str:
    return f"{value:.1%}"


def load_turn_audits(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    audits = payload.get("audits", payload) if isinstance(payload, dict) else payload
    if not isinstance(audits, list):
        raise ValueError("turn audit JSON must be a list or an object with an 'audits' list")
    return audits


def render_html(summary: dict[str, Any], title: str) -> str:
    overview = summary["overview"]
    segment = summary["segment_attribution"]
    impact = summary["milestone_impact"]
    updates = summary["updates"]
    configuration = summary["configuration"]
    backfilled = segment["backfilled_non_boundary_turns"]
    shaped = segment["shaped_turns"]
    backfilled_positive = segment["backfilled_local_sign"].get("positive", 0)
    unfinished_negative = segment["unshaped_local_sign"].get("negative", 0)
    backfill_reward_mean = segment["backfilled_process_reward_sum"] / backfilled if backfilled else 0.0
    positive_reward_sum = segment["boundary_process_reward_sum"] + segment["backfilled_process_reward_sum"]
    backfill_reward_share = (
        segment["backfilled_process_reward_sum"] / positive_reward_sum if positive_reward_sum else 0.0
    )
    distance_one_turns = next(
        (bucket["turn_rows"] for bucket in segment["distance_buckets"] if bucket["distance"] == 1),
        0,
    )
    audits = summary.get("confirmed_turn_audits", [])
    audited_turns = [turn for audit in audits for turn in audit.get("turns", [])]
    audited_backfilled_turns = [
        turn
        for audit in audits
        for turn in audit.get("turns", [])
        if int(turn["turn_index"]) < int(audit["milestone_turn"])
    ]
    confirmed_errors = [turn for turn in audited_turns if turn.get("classification") in {"tool_error", "harmful"}]
    rewarded_errors = [turn for turn in confirmed_errors if float(turn.get("local_advantage", 0.0)) > 0.0]
    confirmed_error_reward = sum(float(turn.get("process_reward", 0.0)) for turn in confirmed_errors)
    segment_length_rows = "".join(
        "<tr>"
        f"<td>{bucket['turn_count']} turn</td>"
        f"<td class='num'>{_num(bucket['segments'])}</td>"
        f"<td class='num'>{_pct(bucket['share'])}</td>"
        "</tr>"
        for bucket in segment["segment_length_buckets"]
    )

    audit_sections = "".join(
        "<section class='panel audit'><h2>真实轨迹：问题出在这里</h2>"
        f"<p><strong>{html.escape(audit['iid'])}</strong> · {html.escape(audit['query'])}</p>"
        "<table><thead><tr><th>Turn</th><th>实际动作</th><th>判断</th><th class='num'>Process reward</th>"
        "<th class='num'>A_local</th></tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td>turn {turn['turn_index']}</td>"
            f"<td>{html.escape(turn['action'])}</td>"
            f"<td class='{('red' if turn['classification'] in {'tool_error', 'harmful'} else 'green')}'>{html.escape(turn['label'])}</td>"
            f"<td class='num'>{float(turn['process_reward']):.3f}</td>"
            f"<td class='num'>{float(turn['local_advantage']):+.3f}</td>"
            "</tr>"
            for turn in audit.get("turns", [])
        )
        + "</tbody></table>"
        f"<div class='judge'><strong>Judge 证据：</strong>{html.escape(audit['judge_evidence'])}</div>"
        f"<p><strong>为什么这是错奖：</strong>{html.escape(audit['diagnosis'])}</p></section>"
        for audit in audits
    )

    step_rows = "".join(
        "<tr>"
        f"<td>Step {update['step']}</td>"
        f"<td class='num'>{_num(update['trajectories'])}</td>"
        f"<td class='num'>{_num(update['turn_rows'])}</td>"
        f"<td class='num blue'>{_num(update['milestone_boundary_turns'])}</td>"
        f"<td class='num amber'>{_num(update['backfilled_non_boundary_turns'])}</td>"
        f"<td class='num'>{_num(update['unfinished_segment_turns'])}</td>"
        f"<td class='num'>{update['process_reward_mean']:.3f}</td>"
        f"<td class='num'>{update['local_advantage_abs_mean']:.3f}</td>"
        "</tr>"
        for update in updates
    )
    distance_rows = "".join(
        "<tr>"
        f"<td>{'boundary' if bucket['distance'] == 0 else bucket['distance']}</td>"
        f"<td class='num'>{bucket['process_reward']:.4f}</td>"
        f"<td class='num'>{_num(bucket['turn_rows'])}</td>"
        f"<td class='num'>{_pct(bucket['turn_rows'] / (segment['milestone_boundary_turns'] if bucket['distance'] == 0 else backfilled))}</td>"
        "</tr>"
        for bucket in segment["distance_buckets"]
    )
    payload = json.dumps(summary, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#657087;--line:#dce2ec;--bg:#f3f6fa;--panel:#fff;--navy:#102b46;--red:#d94f4f;--red-soft:#fde8e7;--green:#238866;--green-soft:#e4f5ef;--amber:#b46b12;--amber-soft:#fff3dd;--blue:#3274b9;--blue-soft:#e8f1fb}}
*{{box-sizing:border-box;letter-spacing:0}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}}
header{{padding:30px max(20px,calc((100% - 1080px)/2));background:var(--navy);color:#fff;border-bottom:5px solid #ec7063}} h1{{margin:0;font-size:28px}} header p{{margin:6px 0 0;color:#d4e1eb}}
main{{max-width:1080px;margin:auto;padding:22px}} section{{margin-bottom:18px}} h2{{margin:0 0 10px;font-size:19px}} h3{{margin:0 0 7px;font-size:15px}}
.panel,.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px}} .panel{{padding:18px;overflow-x:auto}} .cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}} .card{{padding:15px}}
.answer{{padding:17px;border-radius:8px;background:var(--amber-soft);border:1px solid #ead09f;font-size:17px}} .answer strong{{color:#8b520d}}
.value{{display:block;margin:2px 0;font-size:29px;font-weight:760}} .muted{{color:var(--muted);font-size:13px}} .red{{color:var(--red)}} .green{{color:var(--green)}} .amber{{color:var(--amber)}} .blue{{color:var(--blue)}}
.flow{{display:grid;grid-template-columns:repeat(7,auto);align-items:center;gap:7px}} .node{{min-width:120px;padding:12px;border:1px solid var(--line);border-radius:7px;background:#fff;text-align:center}} .node b{{display:block;font-size:18px}} .arrow{{color:var(--red);font-size:21px;font-weight:800}}
.formula{{margin-top:12px;padding:11px 14px;border-left:4px solid var(--blue);background:var(--blue-soft)}} code{{font:13px ui-monospace,SFMono-Regular,Consolas,monospace}}
table{{width:100%;min-width:720px;border-collapse:collapse}} th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted);font-size:12px}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .verdict{{padding:14px;border-radius:8px}} .confirmed{{background:var(--green-soft);border:1px solid #acdccc}} .risk{{background:var(--red-soft);border:1px solid #f1b9b5}}
.bar-row{{display:grid;grid-template-columns:210px 1fr 110px;align-items:center;gap:10px;margin:10px 0}} .track{{height:22px;background:#edf1f6;border-radius:5px;overflow:hidden}} .fill{{height:100%}} .count{{font-weight:700;text-align:right}}
a{{color:var(--blue)}} footer{{color:var(--muted);font-size:12px;margin:20px 0}}
@media(max-width:780px){{.cards{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);text-align:center}}.bar-row{{grid-template-columns:1fr}}.count{{text-align:left}}}}
@media(max-width:480px){{.cards{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>只统计最新 run：{_num(overview["trajectories"])} 条 trajectory，{_num(len(updates))} 个 step，{_num(overview["turn_rows"])} 个 model turn。</p></header>
<main>
<section class="answer"><strong>真正的问题不是公式算错，而是 credit assignment 错了：</strong>系统只知道 turn 3 完成了 milestone，于是默认 turn 0～3 都有贡献。它没有判断中间某个 turn 是正确推进、冗余绕路还是工具错误。<code>gamma={configuration["gamma"]}</code> 又让错误 turn 几乎拿满奖励。</section>

<section class="cards">
<div class="card"><span class="muted">人工/证据确认的错误 turn</span><span class="value red">{_num(len(confirmed_errors))}</span><span class="muted">只计已审计案例，不外推</span></div>
<div class="card"><span class="muted">错误且 A_local 为正</span><span class="value red">{_num(len(rewarded_errors))}</span><span class="muted">训练方向明确错误</span></div>
<div class="card"><span class="muted">错误 turn 获得的 process reward</span><span class="value red">{confirmed_error_reward:.3f}</span><span class="muted">本次已确认案例合计</span></div>
<div class="card"><span class="muted">尚未逐 turn 审计</span><span class="value amber">{_num(backfilled - len(audited_backfilled_turns))}</span><span class="muted">不能直接叫错误 turn</span></div>
</section>

{audit_sections}

<section class="two">
<div class="verdict confirmed"><h3>实现层面</h3><p>衰减公式确实按 <code>10 × 0.95^distance</code> 执行，没有计算 bug。</p></div>
<div class="verdict risk"><h3>算法层面</h3><p>一个 milestone 完成后，整个 segment 被统一当作正贡献；缺少逐 turn 的质量或因果判断。这才是当前要修的问题。</p></div>
</section>

<section><h2>Segment 长度分布</h2><div class="cards">
<div class="card"><span class="muted">已完成 segment</span><span class="value blue">{_num(segment["completed_segments"])}</span><span class="muted">未完成尾段不计</span></div>
<div class="card"><span class="muted">平均长度</span><span class="value">{segment["mean_completed_segment_turns"]:.2f}</span><span class="muted">turn / segment，中位数 {_num(segment["median_completed_segment_turns"])} turn</span></div>
<div class="card"><span class="muted">只有 1 turn</span><span class="value amber">{_pct(segment["single_turn_segment_share"])}</span><span class="muted">{_num(segment["single_turn_segments"])} 个 segment</span></div>
<div class="card"><span class="muted">不超过 2 turns</span><span class="value">{_pct(segment["at_most_two_turn_segment_share"])}</span><span class="muted">{_num(segment["at_most_two_turn_segments"])} 个 segment</span></div>
</div></section>

<section class="two"><div class="panel"><h2>完整长度分布</h2><table><thead><tr><th>Segment 长度</th><th class="num">数量</th><th class="num">占比</th></tr></thead><tbody>{segment_length_rows}</tbody></table></div>
<div class="panel"><h2>这组数字的意义</h2><p><strong>{_pct(segment["single_turn_segment_share"])} 的 segment 只有 milestone 边界 turn 自己。</strong>这些 segment 没有前序 turn，段内衰减对它们不产生作用。</p><p>衰减只会影响剩余 <strong>{_pct(1.0 - segment["single_turn_segment_share"])}</strong> 的多 turn segment。当前错误归因风险也集中在这部分。</p><p>最长已完成 segment 为 <strong>{_num(segment["max_completed_segment_turns"])} turns</strong>。</p></div></section>

<section class="panel"><h2>当前计算方式</h2><div class="flow">
<div class="node">距 milestone 3 turn<b>8.574</b></div><div class="arrow">→</div>
<div class="node">距 milestone 2 turn<b>9.025</b></div><div class="arrow">→</div>
<div class="node">距 milestone 1 turn<b>9.500</b></div><div class="arrow">→</div>
<div class="node">完成 milestone<b>10.000</b></div>
</div><div class="formula"><code>process_reward(t) = 10 × 0.95^(milestone_turn - t)</code><br>这里只看 turn 与 milestone 的距离，没有单独判断这个 turn 是必要操作、无效绕路还是错误操作。</div></section>

<section class="cards">
<div class="card"><span class="muted">直接关闭 milestone</span><span class="value blue">{_num(segment["milestone_boundary_turns"])}</span><span class="muted">边界 reward 固定为 10</span></div>
<div class="card"><span class="muted">段内向前回填（风险集合）</span><span class="value amber">{_num(backfilled)}</span><span class="muted">占正 process reward turn 的 {_pct(backfilled / shaped)}</span></div>
<div class="card"><span class="muted">回填 turn 获得正 A_local</span><span class="value red">{_num(backfilled_positive)}</span><span class="muted">占回填 turn 的 {_pct(backfilled_positive / backfilled)}</span></div>
<div class="card"><span class="muted">未完成当前 segment</span><span class="value">{_num(segment["unshaped_turns"])}</span><span class="muted">其中 {_num(unfinished_negative)} 个得到负 A_local</span></div>
</section>

<section class="two"><div class="panel"><h2>回填距离分布</h2><table><thead><tr><th>距 milestone</th><th class="num">Process reward</th><th class="num">Turn 数</th><th class="num">同类占比</th></tr></thead><tbody>{distance_rows}</tbody></table></div>
<div class="panel"><h2>回填奖励有多强</h2>
<div class="bar-row"><span>Milestone 边界 reward 总和</span><div class="track"><div class="fill" style="width:{segment["boundary_process_reward_sum"] / positive_reward_sum:.1%};background:var(--blue)"></div></div><span class="count">{segment["boundary_process_reward_sum"]:.1f}</span></div>
<div class="bar-row"><span>非边界回填 reward 总和</span><div class="track"><div class="fill" style="width:{backfill_reward_share:.1%};background:var(--amber)"></div></div><span class="count amber">{segment["backfilled_process_reward_sum"]:.1f}</span></div>
<p>每个回填 turn 平均 process reward 为 <strong>{backfill_reward_mean:.3f}</strong>，接近边界满分 10。主要原因是 {_pct(distance_one_turns / backfilled)} 的回填发生在 milestone 前 1 个 turn。</p>
<p>最远回填距离为 <strong>{_num(segment["max_backfill_distance"])} turn</strong>。</p></div></section>

<section class="panel"><h2>每个训练 step</h2><table><thead><tr><th>Step</th><th class="num">Trajectory</th><th class="num">Turn</th><th class="num">边界</th><th class="num">回填</th><th class="num">未完成</th><th class="num">Process mean</th><th class="num">|A_local| mean</th></tr></thead><tbody>{step_rows}</tbody></table></section>

<section class="panel"><h2>训练信号的实际变化</h2>
<div class="bar-row"><span>Milestone turn 的负 advantage</span><div class="track"><div class="fill" style="width:{impact["event_turn_total_sign"].get("negative", 0) / segment["milestone_boundary_turns"]:.1%};background:var(--red)"></div></div><span class="count">{_num(impact["event_turn_global_sign"].get("negative", 0))} → {_num(impact["event_turn_total_sign"].get("negative", 0))}</span></div>
<div class="bar-row"><span>回填 turn 的负 advantage</span><div class="track"><div class="fill" style="width:{segment["backfilled_total_sign"].get("negative", 0) / backfilled:.1%};background:var(--red)"></div></div><span class="count">{_num(segment["backfilled_global_sign"].get("negative", 0))} → {_num(segment["backfilled_total_sign"].get("negative", 0))}</span></div>
<div class="bar-row"><span>未完成 turn 的负 advantage</span><div class="track"><div class="fill" style="width:{segment["unshaped_total_sign"].get("negative", 0) / max(segment["unshaped_turns"], 1):.1%};background:var(--red)"></div></div><span class="count">{_num(segment["unshaped_global_sign"].get("negative", 0))} → {_num(segment["unshaped_total_sign"].get("negative", 0))}</span></div>
<p>箭头左侧是只看 <code>A_global</code>，右侧是叠加 <code>A_local</code> 后。局部奖励明显强化已完成 segment，并惩罚未完成 segment。</p></section>

<section class="two">
<div class="verdict confirmed"><h3>已确认</h3><p>至少 {_num(len(confirmed_errors))} 个真实错误 turn 被正向训练；这不是推测，工具结果和 milestone judge 都能验证。</p></div>
<div class="verdict risk"><h3>还不能下结论</h3><p>{_num(backfilled)} 是待审计风险集合，不是错误数。当前只审计了 {_num(len(audits))} 条轨迹，不能用它估算整体错误率。</p></div>
</section>

<section class="panel"><h2>统计口径</h2><p>这份报告只使用最新一次 `milestone_v4_resume_s1_seedfix` 的 336 条 rewarded trajectory，没有拼接前一段 64 条训练。该 run 仍是 LoRA；当前目录中没有更新的全参段内衰减训练轨迹。</p><p><a href="overview.png">查看专用总览图</a> · <a href="summary.json">查看完整 JSON</a></p></section>
<footer>Generated {html.escape(summary["generated_at"])}.</footer>
<script id="report-data" type="application/json">{payload}</script>
</main></body></html>"""


def render_png(summary: dict[str, Any], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    segment = summary["segment_attribution"]
    updates = summary["updates"]
    distances = segment["distance_buckets"]
    audited_turns = [
        turn for audit in summary.get("confirmed_turn_audits", []) for turn in audit.get("turns", [])
    ]
    confirmed_errors = [
        turn for turn in audited_turns if turn.get("classification") in {"tool_error", "harmful"}
    ]
    confirmed_error_reward = sum(float(turn.get("process_reward", 0.0)) for turn in confirmed_errors)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    figure = plt.figure(figsize=(19, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    axes = {
        "step": figure.add_subplot(grid[0, 0]),
        "length": figure.add_subplot(grid[0, 1]),
        "distance": figure.add_subplot(grid[0, 2]),
        "advantage": figure.add_subplot(grid[1, 0]),
        "diagnosis": figure.add_subplot(grid[1, 1:]),
    }
    figure.patch.set_facecolor("#f3f6fa")
    figure.suptitle(title, fontsize=20, fontweight="bold", color="#172033")

    axis = axes["step"]
    labels = [f"S{update['step']}" for update in updates]
    boundary = [update["milestone_boundary_turns"] for update in updates]
    backfilled = [update["backfilled_non_boundary_turns"] for update in updates]
    unfinished = [update["unfinished_segment_turns"] for update in updates]
    axis.bar(labels, boundary, color="#3274b9", label="Milestone boundary")
    axis.bar(labels, backfilled, bottom=boundary, color="#d68b22", label="Backfilled prefix")
    axis.bar(
        labels,
        unfinished,
        bottom=[left + middle for left, middle in zip(boundary, backfilled, strict=True)],
        color="#a6b1bf",
        label="Unfinished segment",
    )
    axis.set_title("Turn attribution by optimizer step")
    axis.set_ylabel("Model-call turns")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes["length"]
    length_buckets = segment["segment_length_buckets"]
    length_x = [bucket["turn_count"] for bucket in length_buckets]
    length_counts = [bucket["segments"] for bucket in length_buckets]
    axis.bar(length_x, length_counts, color=["#d68b22" if value == 1 else "#3274b9" for value in length_x])
    axis.set_title("Completed segment length distribution")
    axis.set_xlabel("Model-call turns per segment")
    axis.set_ylabel("Completed segments")
    axis.set_xticks(length_x)
    axis.text(
        0.98,
        0.94,
        f"mean {segment['mean_completed_segment_turns']:.2f}\n"
        f"1-turn {segment['single_turn_segment_share']:.1%}\n"
        f"<=2 turns {segment['at_most_two_turn_segment_share']:.1%}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#172033",
    )
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes["distance"]
    x = [bucket["distance"] for bucket in distances]
    counts = [bucket["turn_rows"] for bucket in distances]
    rewards = [bucket["process_reward"] for bucket in distances]
    axis.bar(x, counts, color=["#3274b9" if distance == 0 else "#d68b22" for distance in x])
    axis.set_title("Backfill distance and reward strength")
    axis.set_xlabel("Turns before milestone (0 = boundary)")
    axis.set_ylabel("Turn count")
    reward_axis = axis.twinx()
    reward_axis.plot(x, rewards, color="#d94f4f", marker="o", linewidth=2)
    reward_axis.set_ylabel("Process reward", color="#d94f4f")
    reward_axis.set_ylim(0, 10.8)
    axis.spines["top"].set_visible(False)
    reward_axis.spines["top"].set_visible(False)

    axis = axes["advantage"]
    categories = ["Boundary", "Backfilled", "Unfinished"]
    sign_maps = [
        segment["event_local_sign"],
        segment["backfilled_local_sign"],
        segment["unshaped_local_sign"],
    ]
    totals = [sum(values.values()) for values in sign_maps]
    positive = [values.get("positive", 0) / total for values, total in zip(sign_maps, totals, strict=True)]
    zero = [values.get("zero", 0) / total for values, total in zip(sign_maps, totals, strict=True)]
    negative = [values.get("negative", 0) / total for values, total in zip(sign_maps, totals, strict=True)]
    axis.barh(categories, positive, color="#238866", label="Positive A_local")
    axis.barh(categories, zero, left=positive, color="#a6b1bf", label="Zero A_local")
    axis.barh(
        categories,
        negative,
        left=[left + middle for left, middle in zip(positive, zero, strict=True)],
        color="#d94f4f",
        label="Negative A_local",
    )
    axis.set_xlim(0, 1)
    axis.set_title("Local advantage sign by turn type")
    axis.set_xlabel("Share of turns")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    axis.spines[["top", "right", "left"]].set_visible(False)

    axis = axes["diagnosis"]
    axis.axis("off")
    axis.text(0.02, 0.94, "Confirmed diagnosis", fontsize=15, fontweight="bold", va="top")
    axis.text(
        0.03,
        0.78,
        f"{len(confirmed_errors):,} proven erroneous turns receive positive training signal\n"
        f"{confirmed_error_reward:.3f} process reward assigned to those errors\n"
        f"{segment['backfilled_non_boundary_turns']:,} backfilled turns remain the audit pool\n"
        f"{segment['backfilled_local_sign'].get('positive', 0):,} of them receive positive A_local",
        fontsize=14,
        linespacing=1.65,
        va="top",
        color="#172033",
    )
    axis.text(
        0.03,
        0.28,
        "The decay formula is correct. Credit assignment is not: completing a\n"
        "milestone labels the whole segment as useful, including failed tool calls.",
        fontsize=11,
        color="#657087",
        va="top",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", nargs="+", help="Reward-backed trace directories from one training run")
    parser.add_argument("--output-dir", default="artifacts/latest_segment_decay_turn_report")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--turn-audits", type=Path, help="Optional JSON containing evidence-backed turn audits")
    parser.add_argument("--title", default="最新一次训练 · Segment 内奖励衰减")
    parser.add_argument("--png-title", default="Latest Training | Within-Segment Reward Decay")
    args = parser.parse_args()

    trajectories, source_info = load_trajectories(Path(value) for value in args.trace_dirs)
    summary = build_summary(
        trajectories,
        gamma=args.gamma,
        local_weight=args.local_weight,
        source_info=source_info,
    )
    summary["confirmed_turn_audits"] = load_turn_audits(args.turn_audits)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(render_html(summary, args.title), encoding="utf-8")
    render_png(summary, output_dir / "overview.png", args.png_title)
    print(f"Rendered segment-decay report from {len(trajectories)} trajectories to {output_dir}")


if __name__ == "__main__":
    main()
