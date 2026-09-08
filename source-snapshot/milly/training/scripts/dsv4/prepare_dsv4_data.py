#!/usr/bin/env python3
"""把线上日志语料渲染 + 切词成 DeepSeek-V4-Flash 能训的样子。

输入:/home/work/migoo_ai_public/lsy/autotask/data/v1/train/v1_clean_best.jsonl.gz
      (1,871 条真实线上定时任务执行记录,**不含思考** —— 就是"nothink 那份")

⛔ **必须是 clean 那份,别用 v1_raw_best**(2026-08-06 查实,两份差异很大):
   同目录下还有个 `v1_raw_best.jsonl.gz`,trace_id 一一对得上、消息条数完全一样,
   但 **2164 条信封消息的正文不同** —— raw 那份把「一段旁白」和「信封 JSON」
   挤在同一条 assistant 里,clean 把旁白删了只留信封:

       raw  : Now let me filter by the criteria. Emails after 08:00 …(旁白 526 字符)
              {"content":"🌙 **Penutup Hari** …               ← 信封跟在旁白后面
       clean: {"content":"🌙 **Penutup Hari** …               ← 只有信封

   受影响 1274/1871 条样本,其中 **1144 条那段旁白会进 loss**(是样本最后一条
   assistant)。用 raw 训出来的模型会学成「先写 500 字旁白再吐 JSON」,
   直接打穿信封解析(见 eval/docs/11_JSON合规率.md)。
   8-04 实际训练用的是 clean 那份,是对的;是这里的默认值一直写错。

输出:<out-dir>/
        train.bin        int32,所有样本 token 首尾相接
        train.mask.bin   uint8,1 = 这个 token 进 loss
        train.idx.npy    (N,2) 的 [起始偏移, 长度]
        train.jsonl      渲染后的中间格式(可读,便于排查)
        meta.json        统计与自检结果
      (格式和同目录 dsv4_sft_data.py 完全一致)

⛔ 为什么不能直接用 tokenize_hy3.py:
   DeepSeek-V4 **没有 chat_template**(tokenizer_config.json 里没这个字段),
   它自带一个 Python 编码器 `encoding/encoding_dsv4.py`,而且工具调用用的是
   DSML 标记(<｜tool▁calls▁begin｜>...)不是 JSON。所以必须走官方编码器。

用法:
    python3 prepare_dsv4_data.py --out-dir /本地盘/dsv4_data
    python3 prepare_dsv4_data.py --out-dir ... --teacher-only opus --judge-strict
    python3 prepare_dsv4_data.py --out-dir ... --limit 20      # 先拿 20 条试

⚠️ 这份语料里有**真实用户的邮箱、日历、姓名**。产出只能留在内网、只能喂给
   我们自己部署的模型。别往任何外部 API 送。
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

# ⛔ 必须是 clean 那份,别改成 v1_raw_best —— 原因见文件头的说明(旁白会混进信封)
DEFAULT_SRC = "/home/work/migoo_ai_public/lsy/autotask/data/v1/train/v1_clean_best.jsonl.gz"
DEFAULT_MODEL = "/home/work/migoo_ai_public/posttrain/post-train/models/DeepSeek-V4-Flash-0731"

# 2026-08-06 实测的指纹。对不上说明上游换了文件,先搞清楚换了什么再往下跑
KNOWN_SRC_SHA256 = {
    "a058c1c7c595fbecf34ff492813426d2bc704fb08a0596f7d0fd989a052d97e9": "v1_clean_best.jsonl.gz(✅ 该用这份)",
    "500491f902649306d867439f88f3cf558fe8927fcc3e657932eb55d16a9a1faa": "v1_raw_best.jsonl.gz(⛔ 旁白没删,别用)",
    # v2:2,163 条,信封多了 reason 字段,无旁白问题(2026-08-06 摸底:全部以 { 开头)。
    # 已知瑕疵(闸会自动丢):7 条 assistant 背靠背、2 条调用悬空、1 条参数解不出。
    "fdb6387a3d27ded05252cb4401de5a18d540a01383d7b30b105f8bfca8e0b3ab": "v2_clean.jsonl.gz(✅ v2 语料)",
}


def _sha256(path, limit=None):
    """算文件的 SHA256。limit 非空时只算前 limit 字节(大文件用)。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() >= limit:
                break
    return h.hexdigest()


def provenance(src, model):
    """记清楚「这批数据是谁、用什么、在什么代码下生出来的」。

    为什么要这个:2026-08-06 排查「文档命令复现不出已训数据」时,全靠 meta.json
    里恰好记了源文件**路径**才查清 —— 但路径会变、文件会被原地覆盖,不足以定位。
    现在连内容指纹和脚本的 git commit 一起记下来。
    """
    src_sha = _sha256(src)
    out = {
        "源文件": src,
        "源文件 SHA256": src_sha,
        "源文件认得出吗": KNOWN_SRC_SHA256.get(src_sha, "⚠️ 不在已知清单里(上游换文件了?)"),
        "源文件大小": os.path.getsize(src),
        "模型目录": model,
    }
    # 决定切词结果的那几个文件(整个模型目录 159G,不可能全哈希)
    fp = {}
    for rel in ("config.json", "tokenizer.json", "tokenizer_config.json",
                "encoding/encoding_dsv4.py"):
        p = os.path.join(model, rel)
        if os.path.exists(p):
            fp[rel] = _sha256(p)[:16]
    out["分词/编码器指纹"] = fp
    # 转换脚本自己的版本
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", __file__],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        out["转换脚本 git commit"] = (rev or "?") + ("(有未提交改动)" if dirty else "")
    except Exception as e:
        out["转换脚本 git commit"] = f"取不到: {e}"
    out["转换脚本 SHA256"] = _sha256(os.path.abspath(__file__))[:16]
    return out


# ──────────────────────────────────────────────────────────────────────
# 1. 渲染:我们的日志格式 → DeepSeek 编码器认的 OpenAI 格式
# ──────────────────────────────────────────────────────────────────────
# 我们的日志里角色是 system / user / assistant / tool_call / tool_response,
# 而编码器要的是 OpenAI 那套:assistant 带 tool_calls 列表、工具结果用 role="tool"。
#
# 监督掩码的约定(**看清楚,反直觉**):
#   消息带 "loss": False  → 不监督(那是更早一次触发的历史)
#   消息**没有** loss 字段 → 监督
# 实测 1,871 条全都是「最后一条 = 被监督的 assistant + 信封」,这正是我们要教的东西。

def to_openai(messages, terminal_contract="autotask"):
    """返回 (openai_messages, sup_flags, 错误原因)。成功时错误原因是 None。
    sup_flags[i] 说明第 i 条要不要进 loss。

    ⛔ 这里有过一个**会毁掉模型**的 bug(2026-08-03 修):

    线上日志里模型一次出手是「一段文本 + 若干工具调用」,存成了相邻的
    `assistant` + `tool_call`。老代码把它们**拆成两条独立的 assistant 消息**,
    而官方编码器给每条 assistant 都补一个 eos,于是渲染成了:

        </think>我来查一下邮件<｜end▁of▁sentence｜>     ← 假的结束符,而且进了 loss
        <｜DSML｜tool_calls>...                          ← 工具调用被切到下一条

    也就是把模型训成「说完计划就停机,一个工具都不调」。实测 72% 的样本中招,
    共 3062 个假结束符。**而且 loss 曲线完全看不出来**(假 eos 是高置信度目标)。

    现在:文本和它后面紧跟的 tool_call 合并成**一条** assistant,只在末尾收一个 eos。
    合并后「eos 个数 == assistant 消息数」这条地基断言依然成立。
    """
    out, sup = [], []
    i, n = 0, len(messages)
    call_seq = 0                              # 全局递增,保证 tool_call_id 不重号

    def take_tool_calls(j):
        """吃掉从 j 开始连续的 tool_call,返回 (calls, 各条监督标记的集合, 新下标)。"""
        nonlocal call_seq
        calls, flags = [], set()
        while j < n and messages[j]["role"] == "tool_call":
            mm = messages[j]
            flags.add("loss" not in mm)
            try:
                c = json.loads(mm.get("content") or "{}")
            except Exception:
                return None, None, None       # 解不出来的整条丢掉,不带病渲染
            calls.append({
                "id": f"call_{call_seq}",
                "type": "function",
                "function": {"name": c.get("name", ""),
                             "arguments": json.dumps(c.get("arguments", {}),
                                                     ensure_ascii=False)},
            })
            call_seq += 1
            j += 1
        return calls, flags, j

    # 等着被 tool_response 认领的 call id(按调用顺序)
    pending_ids = []

    while i < n:
        m = messages[i]
        role, content = m["role"], (m.get("content") or "")
        supervised = "loss" not in m          # 没写 loss 字段 = 要监督

        if role in ("system", "user"):
            out.append({"role": role, "content": content})
            sup.append(False)                 # 输入永远不监督
            i += 1

        elif role == "assistant":
            i += 1
            # 紧跟着的 tool_call 属于同一次出手 → 合并进这一条
            if i < n and messages[i]["role"] == "tool_call":
                calls, flags, i = take_tool_calls(i)
                if calls is None:
                    return None, None, "工具参数解不出"
                # 同一次出手的文本和调用必须同为监督或同为历史 —— 不一致时合并
                # 会悄悄改标签,宁可丢样本(实测全语料 0 例,真出现就是上游变了)
                if len(flags | {supervised}) > 1:
                    return None, None, "同一次出手里 loss 标记不一致"
                a = {"role": "assistant", "content": content, "tool_calls": calls}
                if m.get("reasoning_content"):
                    a["reasoning_content"] = m["reasoning_content"]
                out.append(a)
                sup.append(supervised or any(flags))
                pending_ids.extend(c["id"] for c in calls)
            else:
                a = {"role": "assistant", "content": content}
                if m.get("reasoning_content"):
                    a["reasoning_content"] = m["reasoning_content"]
                out.append(a)
                sup.append(supervised)

        elif role == "tool_call":
            # 前面没有文本、直接就是工具调用的情况(全语料 3,222 组)——
            # 自立一条 assistant,同样收一个结束符,数动作数时别把它落下
            calls, flags, i = take_tool_calls(i)
            if calls is None:
                return None, None, "工具参数解不出"
            if len(flags) > 1:
                return None, None, "同一次出手里 loss 标记不一致"
            out.append({"role": "assistant", "content": "", "tool_calls": calls})
            sup.append(any(flags))
            pending_ids.extend(c["id"] for c in calls)

        elif role == "tool_response":
            # ⛔ 老代码这里写的是 f"call_{len(out)}" —— 数的是**输出消息总条数**,
            #    和上面调用侧的编号是两套互不相干的计数器。编码器
            #    sort_tool_results_by_call_order 按 id 查调用次序,对不上就当 0,
            #    实测有 5 条样本的工具返回被排序打乱、张冠李戴。
            #    现在按调用顺序逐个认领,严格对应。
            if not pending_ids:
                return None, None, "工具返回比调用还多"   # 结构不对,丢
            out.append({"role": "tool", "content": content,
                        "tool_call_id": pending_ids.pop(0)})
            sup.append(False)
            i += 1
        else:
            return None, None, f"没见过的角色:{role}"   # 丢掉并计数
    if pending_ids and terminal_contract == "open":
        # Media-agent records may end at the submitted action; the external runner owns its result.
        pass
    elif pending_ids:
        # ── 例外:末尾单个「移交」调用是合法终局,不是日志坏了 ──────────────
        # 2026-08-10 加。线上 auto-task 的一条记录 = 一个 agent_run;
        # 模型调 transfer_to_agent 把活交出去,这一轮到此为止 —— 控制权不回来,
        # 下游 agent 干的事属于【另一条独立记录】,所以本条【本来就不会有】
        # 对应的 tool_response。实测(v3 数据 8,392 条判决):
        #   professional → sandbox_runner              400 / 400 转出后本 trace 内无新 agent_run
        #   lite/compact → remind_agent_professional    87 /  87 同理
        # 这类记录在 v3 最终教材 2,112 条里占 719 条(34.0%),全部是「末尾恰好
        # 一个悬空调用、工具名 transfer_to_agent」。
        #
        # ⛔ 为什么必须放行:megatron_bridge/build_cotmix64.py 记着直接教训 ——
        #   「①的漏斗把转交行全剔了(以 tool_call 结尾无信封,不满足合成契约)。
        #     屏蔽转交的历史教训是 93%→53.5%,所以必须补回来。」
        #   把 34% 的样本当成坏数据丢掉,等于重蹈那次事故。
        #
        # 判据写窄:只放行【末尾恰好一个】悬空调用、且是 transfer_to_agent。
        # 其余悬空(多个、或别的工具名)仍按日志截断丢弃 —— 那才是真的结构坏了。
        last = out[-1] if out else {}
        tcs = last.get("tool_calls") or []
        if (len(pending_ids) == 1 and len(tcs) == 1
                and tcs[0].get("function", {}).get("name") == "transfer_to_agent"):
            pass                                  # 合法终局,放行
        else:
            return None, None, "样本结束仍有未认领的工具调用"
    return out, sup, None


# ──────────────────────────────────────────────────────────────────────
# 1.5 转换保真闸:bug①② 的事后检查
# ──────────────────────────────────────────────────────────────────────
# 文档 §3.4 曾承认:修复前那版 to_openai 接回来,400/400 全过(旧的)验收 ——
# 两个会毁模型的 bug 只靠代码写法本身保证,没有事后检查。下面就是那个事后检查。
# 改 to_openai 之前先跑 `python3 prepare_dsv4_data.py --self-test`。

def expected_actions(messages):
    """原始日志的「原子动作数」= assistant 条数 + 不紧跟 assistant 的裸 tool_call 组数。

    ⛔ 口径注意:**不是**只数 assistant。语料里有 3,222 组「没有前置文本、直接
    调工具」的出手(占 1,098/1,871 条样本),转换后各自成一条 assistant、收一个
    结束符。早先文档提议的「结束符数 == 原始 assistant 数」就折在这 —— 会在 59%
    的正确样本上误报。正确口径 2026-08-06 实测:11,073 + 3,222 = 14,295,与实际
    训练 bin 里的结束符总数分毫不差。
    """
    n_asst = n_bare = 0
    prev = None
    for m in messages:
        r = m["role"]
        if r == "assistant":
            n_asst += 1
        elif r == "tool_call" and prev not in ("assistant", "tool_call"):
            n_bare += 1
        prev = r
    return n_asst, n_bare


R_SPLIT = "动作数对不上(bug① 指纹:一次出手被劈开或凭空合并)"
R_ROUNDTRIP = "工具往返对不上(bug② 指纹:名字/参数/返回被改坏或认领错序)"


def action_count_mismatch(raw_msgs, oa_msgs):
    """挡 bug①:渲染出的 assistant 条数必须等于原始原子动作数,劈开一次就多一条。"""
    n_asst, n_bare = expected_actions(raw_msgs)
    n_conv = sum(1 for m in oa_msgs if m["role"] == "assistant")
    if n_conv != n_asst + n_bare:
        return f"渲染 {n_conv} ≠ assistant {n_asst} + 裸工具组 {n_bare}"
    return None


def roundtrip_check(raw_msgs, oa_msgs):
    """挡 bug②:把渲染结果逐条对回原始日志,内容级比对。

    查三件事:① 工具名和参数(深度比较,键序无关)顺序一致;② 工具返回的
    内容顺序一致;③ 每条返回认领的 tool_call_id 必须正好是「按调用顺序轮到的
    那一个」—— 老代码调用侧和返回侧两套计数器的 bug 在这一步现形。
    """
    raw_calls, raw_resp = [], []
    for m in raw_msgs:
        if m["role"] == "tool_call":
            try:
                c = json.loads(m.get("content") or "{}")
            except Exception:
                return "工具参数解不出"        # 正常到不了这:to_openai 已拦
            raw_calls.append((c.get("name", ""),
                              json.dumps(c.get("arguments", {}), sort_keys=True,
                                         ensure_ascii=False)))
        elif m["role"] == "tool_response":
            raw_resp.append(m.get("content") or "")
    conv_calls, call_ids, conv_resp, claimed = [], [], [], []
    for m in oa_msgs:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc["function"]
                conv_calls.append((f["name"],
                                   json.dumps(json.loads(f["arguments"]), sort_keys=True,
                                              ensure_ascii=False)))
                call_ids.append(tc["id"])
        elif m["role"] == "tool":
            conv_resp.append(m.get("content") or "")
            claimed.append(m.get("tool_call_id"))
    if conv_calls != raw_calls:
        return "工具名/参数和原始对不上"
    if conv_resp != raw_resp:
        return "工具返回内容和原始对不上"
    if claimed != call_ids[:len(claimed)]:
        return "工具返回没按调用顺序认领"
    return None


def _self_test():
    """把历史 bug 的行为各回放一遍,证明新闸抓得住。不碰数据、不加载模型,一秒出结果。"""
    def call(name, args):
        return {"role": "tool_call",
                "content": json.dumps({"name": name, "arguments": args}), "loss": False}
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "我查一下", "loss": False},
        call("search", {"q": 1}),
        {"role": "tool_response", "content": "r1", "loss": False},
        call("send", {"x": 2}),                    # 裸工具组(没有前置文本)
        {"role": "tool_response", "content": "r2", "loss": False},
        {"role": "assistant", "content": '{"content":"…","pass":true}'},
    ]
    oa, _, err = to_openai(msgs)
    assert err is None and action_count_mismatch(msgs, oa) is None \
        and roundtrip_check(msgs, oa) is None, "正常样本不该被拦"

    # bug①:老 to_openai 把「文本 + 工具调用」劈成两条 assistant,各收一个 eos
    split = []
    for m in oa:
        if m["role"] == "assistant" and m.get("tool_calls") and m.get("content"):
            split.append({"role": "assistant", "content": m["content"]})
            split.append({"role": "assistant", "content": "", "tool_calls": m["tool_calls"]})
        else:
            split.append(m)
    assert action_count_mismatch(msgs, split), "劈开必须被动作数闸抓住"

    # bug②:老代码返回侧用 f"call_{len(out)}" 另起一套计数器,和调用侧对不上
    bad = json.loads(json.dumps(oa))
    for k, m in enumerate(x for x in bad if x["role"] == "tool"):
        m["tool_call_id"] = f"call_{100 + k}"
    assert roundtrip_check(msgs, bad), "错认领必须被往返闸抓住"

    # 结构异常:调用悬在末尾没人认领(日志被截断的指纹)
    _, _, err = to_openai(msgs[:4] + [msgs[-1]])
    assert err == "样本结束仍有未认领的工具调用", f"未认领调用必须被拦,实际:{err}"

    # 结构异常:同一次出手里文本和调用的 loss 标记不一致(合并会悄悄改标签)
    mixed = [dict(m) for m in msgs]
    mixed[3].pop("loss")                           # 调用要监督,前面的文本却是历史
    _, _, err = to_openai(mixed)
    assert err == "同一次出手里 loss 标记不一致", f"混合标记必须被拦,实际:{err}"

    # 思考通道:assistant 上的 reasoning_content 必须原样透传(rq2 加思考批用)。
    # 纯文本条和「文本+工具调用」合并条都要带;丢了思考 = 思考批白训。
    think = [dict(m) for m in msgs]
    think[2] = {**think[2], "reasoning_content": "先想想"}
    think[-1] = {**think[-1], "reasoning_content": "组装信封"}
    oa2, _, err = to_openai(think)
    assert err is None, f"带思考的样本不该被拦:{err}"
    asst = [m for m in oa2 if m["role"] == "assistant"]
    assert asst[0].get("reasoning_content") == "先想想" and asst[0].get("tool_calls"), \
        "合并条丢了 reasoning_content"
    assert asst[-1].get("reasoning_content") == "组装信封", "收尾条丢了 reasoning_content"
    assert not asst[1].get("reasoning_content"), "裸工具组不该凭空长出思考"

    print("self-test: 5/5 —— bug①(劈开)、bug②(错认领)、未认领调用、混合标记、思考透传,全部抓得住")


# ──────────────────────────────────────────────────────────────────────
# 2. 切词 + 打监督掩码
# ──────────────────────────────────────────────────────────────────────
# ⛔ 走过的弯路(别再试):**逐前缀编码不行**。
#    `encode_messages` 内部会 merge_tool_messages + sort_tool_results_by_call_order,
#    把工具结果合并/重排。所以"编到第 k 条"并不是"编到第 k+1 条"的前缀,
#    实测 20 条里 18 条前缀断言失败。
#
# ✅ 实际用的办法:**按 EOS 锚点切段**(和我们 Hy3 那套同思路,实测 12/12 通过)。
#    - 每条 assistant 消息**必定**以一个 eos_token 收尾 → EOS 个数 == assistant 消息数
#      (12/12 样本严格成立,是本方法的地基)
#    - `<｜Assistant｜>` 标记**只在前一条不是 assistant 时才出现**,所以不能只靠它数段
#    - 于是第 k 段 = 从「上一段结束」和「本段前最后一个 Assistant 标记」中靠后的那个,
#      到第 k 个 EOS(含)
#
# 为什么要这么小心:掩码错位会把 user 的文本/工具返回当成训练目标,而且**从 loss
# 曲线上完全看不出来**。我们在 Hy3 上踩过这个坑,所以宁可丢样本也不带病监督。

def wrap_tools(tools):
    """我们的日志里工具是裸 schema `{name, description, parameters}`,
    而编码器 `tools_from_openai_format` 要的是 OpenAI 外层包装
    `{"type":"function","function":{...}}`(它会剥掉这层)。这里补上。"""
    if not tools:
        return None
    out = []
    for t in tools:
        out.append(t if "function" in t else {"type": "function", "function": t})
    return out


def build_sample(enc, tok, oa_msgs, sup_flags, tools, thinking_mode, ids_A, id_EOS,
                 reasoning_effort=None, truncated=False,
                 terminal_contract="autotask"):
    """整条渲一次 + 多段掩码。chat 模式一直走这条;thinking 模式 0819 起也能走
    (`--render whole`),见 build_sample_whole。

    reasoning_effort:⛔ 0819 补。老代码不传,编码器就按默认 low 渲 —— 这份语料
      全是 low 所以当时没错,但做严格 high 对照臂会**静默**渲成 low。
    truncated:这条是「渲到最后一个装得下的轮为止」的截断样本,末段合法地
      落在工具调用上(不是信封),所以放行末段闸。别在非截断路径上传 True。
    """
    sub = [dict(m) for m in oa_msgs]
    tools = wrap_tools(tools)
    if tools:
        sub[0] = {**sub[0], "tools": tools}           # 工具清单挂在第一条上
    # drop_thinking=False:线上服务在上下文里**保留历史轮思考**(rq1 实测,
    # 末轮输入带全部往轮 thought),训练必须同形状。另外 drop_thinking=True 会把
    # 历史轮渲染成空思考(光秃一个 </think>),若中间轮被监督,等于教模型
    # "工具轮不用想"。chat 模式没有 reasoning_content,这个参数无副作用。
    text = enc.encode_messages(sub, thinking_mode=thinking_mode,
                               drop_thinking=False,
                               reasoning_effort=reasoning_effort,
                               add_default_bos_token=True)
    ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int32)

    eos = np.nonzero(ids == id_EOS)[0]
    mk = np.nonzero(ids == ids_A)[0]
    asst = [(i, s) for i, (m, s) in enumerate(zip(oa_msgs, sup_flags))
            if m["role"] == "assistant"]

    # 地基断言:EOS 个数必须等于 assistant 消息数。不等 = 渲染假设破了,整条丢掉。
    if len(eos) != len(asst):
        return None, None, f"EOS 数({len(eos)})对不上 assistant 消息数({len(asst)})"

    mask = np.zeros(ids.shape[0], dtype=np.uint8)
    prev = 0
    last_span = None
    for (msg_index, supervised), e in zip(asst, eos):
        cand = mk[(mk >= prev) & (mk < e)]
        start = int(cand[-1]) + 1 if len(cand) else prev
        end = int(e) + 1                              # 含 EOS,让模型学会收尾

        # ⛔ 段内除了段尾那个,不许再出现结束符。
        #    这条是用来自动抓「文本和工具调用被拆成两条、各补一个 eos」那个 bug 的
        #    —— 那种情况下段里会多出结束符,而模型会被训成说一半就停机。
        #    (老代码那条「段内还有 Assistant 标记」的断言,按 start 的定义
        #     数学上永远不可能触发,是死代码,已经换成这条真能响的。)
        if bool((ids[start:end - 1] == id_EOS).any()):
            return None, None, "监督段内有多余的结束符(文本和工具调用没合并?)"
        if bool((ids[start:end] == ids_A).any()):
            return None, None, "切段越界(段内还有 Assistant 标记)"

        if supervised:
            mask[start:end] = 1
            # 每个被监督的段都要长得像样:要么是信封,要么是完整的工具调用块。
            # 老代码只查最后一段,中间那些工具调用段全裸奔 —— 上面那个假结束符
            # 的 bug 正好全藏在中间段里。
            seg = tok.decode(ids[start:end])
            # ⛔ 0819 修:老代码这里是 `if not (is_env or is_call or
            #   len(seg.strip()) > 0)` —— 只要段非空就放行,**这道闸从来没响过**。
            #   后果是那句 tick 前的罐头应答(「System initialized… I am ready」)
            #   会被算进 loss:它是 ADK 塞进输入里的上下文,不是我们的输出。
            #   爆破路径写的是严格版,所以整条/爆破两条路以前对不上(实测每 20
            #   个任务多 560 个 loss token)。真正该排除它的地方是 sup_flags,
            #   见 mask_context_turns();这里把闸恢复成真会响的。
            is_env = '"content"' in seg and '"pass"' in seg
            source_message = oa_msgs[msg_index]
            open_output = bool((source_message.get("content") or "").strip()
                               or source_message.get("tool_calls"))
            if not (is_env or _seg_is_call(enc, seg)
                    or (terminal_contract == "open" and open_output)):
                return None, None, "监督段既不是信封也不是工具调用块"
        prev = end
        last_span = (start, end)

    # 掩码连续段级别的复查:两条 assistant **背靠背**时(相邻 assistant 之间编码器
    # 不加任何标记),两个监督段在 token 上无缝相接,前一段的段尾 EOS 就落到了
    # 合并段的中间 —— 训练目标变成「吐出结束符后接着说」,而推理到 EOS 就停,
    # 后半段结构上永远到不了。上面按 EOS 锚点的逐段检查看不见这种情况(EOS 恰好
    # 是段界),必须按掩码连续性再查一遍。v2 语料实测 7/2163 条,整条丢。
    # (同时兜底 bug①:万一动作数闸失效,被劈开的段在这里也会现形。)
    sup_pos = np.nonzero(mask == 1)[0]
    if len(sup_pos):
        run_start = prev_p = int(sup_pos[0])
        runs = []
        for p in sup_pos[1:]:
            p = int(p)
            if p != prev_p + 1:
                runs.append((run_start, prev_p)); run_start = p
            prev_p = p
        runs.append((run_start, prev_p))
        for s, e in runs:                      # e 含;段尾那个 EOS 是合法的
            if bool((ids[s:e] == id_EOS).any()):
                return None, None, "监督区有多余结束符(assistant 背靠背?)"

    if mask.sum() == 0:
        return None, None, "没有任何 token 进 loss"
    if not asst[-1][1]:
        return None, None, "最后一条 assistant 没被监督(末条信封必须进 loss)"
    if terminal_contract == "open":
        # Generic/media agents may end with natural language or any submitted tool action.
        return ids, mask, None
    # 末段必须是信封 —— 这是这批数据要教的东西,不是就说明对齐错了。
    # 例外见下:以「移交」收尾的样本没有信封,那是【合法终局】不是对齐错。
    tail = tok.decode(ids[last_span[0]:last_span[1]])
    if truncated and _seg_is_call(enc, tail):
        return ids, mask, None          # 截断样本:末段落在工具调用上是合法的
    if '"content"' not in tail or '"pass"' not in tail:
        # ── 例外:末段是 transfer_to_agent 工具调用 ──────────────────────────
        # 2026-08-10 加。线上一条记录 = 一个 agent_run;模型调 transfer_to_agent
        # 把活交出去,这一轮就结束了 —— 控制权不回来(实测 pro→sandbox 400/400、
        # lite/compact→pro 87/87 转出后本 trace 内无新 agent_run),下游 agent
        # 干的事是【另一条独立记录】。所以这类样本【本来就不该有信封】,
        # 它要教的正是"该交出去的时候交出去"这个动作本身。
        #
        # ⛔ 为什么必须放行:megatron_bridge/build_cotmix64.py 记着直接教训 ——
        #   「①的漏斗把转交行全剔了(以 tool_call 结尾无信封,不满足合成契约)。
        #     屏蔽转交的历史教训是 93%→53.5%,所以必须补回来。」
        #   v3 最终教材 2,112 条里这类占 719 条(34.0%),丢掉就是重蹈那次事故。
        #
        # 判据写窄:末段必须是【工具调用段】且里面出现 transfer_to_agent。
        # 其他"末段不是信封"仍然拦 —— 那才是真的对齐错了。
        if not (_seg_is_call(enc, tail) and "transfer_to_agent" in tail):
            return None, None, "末段解码出来不是信封"
    return ids, mask, None


def _seg_is_call(enc, seg):
    if hasattr(enc, "TOOL_CALLS_BEGIN") and enc.TOOL_CALLS_BEGIN in seg:
        return True
    return "tool▁calls▁begin" in seg or "DSML｜tool_calls" in seg


# ══════════════════════════════════════════════════════════════════════════════
# 整条渲染(--render whole)。爆破那条路留着只为复现旧实验,见 build_samples_thinking
# 的注释:两条路的梯度**逐 token 相同**(0819 实测:前缀 24/24 逐 token 一致、
# 掩码位置 20/20 重合、loss 按全局 token 归一),而爆破多花 4.19 倍算力
# (369M 槽位 vs 88M)。所以新实验一律走整条。
# ══════════════════════════════════════════════════════════════════════════════

def last_user_index(oa_msgs):
    return max((i for i, m in enumerate(oa_msgs) if m.get("role") == "user"), default=-1)


def context_assistant_turns(oa_msgs):
    """排在最后一条 user 之前的 assistant 轮 —— 它们是**输入**不是我们的输出。

    线上一条记录是四段消息:`系统 / 题面 / assistant 那句罐头应答 / 用户 tick`。
    那句应答是 ADK 塞进来的,最后一条 user 就是 tick,所以 tick 之后的每一轮才是
    我们的模型真正生成的东西。

    ⭐ **判据是结构,不是字符串。** 别写成 `content.startswith("System initialized")`
    —— 平台一改措辞那种闸就静默失效,而且失效方向是「多训了不该训的东西」,不报错、
    事后也看不出来。用结构判据的话:改措辞无影响;不再发这条 → 自动排除 0 条;
    发两条 → 自动排除两条。结构**本身**变了由 structure_issues() 当场拦下。
    """
    lu = last_user_index(oa_msgs)
    return [i for i, m in enumerate(oa_msgs)
            if m.get("role") == "assistant" and i < lu]


def structure_issues(oa_msgs):
    """闸 A:消息结构还是不是我们假设的那个形状。不是就**当场停机**,别静默按老假设跑。

    v7 语料实测:1281/1281 个任务都恰好 1 条 tick 前的 assistant 轮,且它无思考、
    无工具调用。平台哪天让那条应答带上工具调用,它就会被当成正经监督轮训进去 ——
    这道闸就是拦这个的。
    """
    out = []
    ctx = context_assistant_turns(oa_msgs)
    if len(ctx) > 1:
        out.append(f"tick 前有 {len(ctx)} 条 assistant 轮(一直是 1 条)")
    for i in ctx:
        m = oa_msgs[i]
        if m.get("reasoning_content"):
            out.append(f"tick 前第 {i} 条 assistant 带思考(一直是没有的)")
        if m.get("tool_calls"):
            out.append(f"tick 前第 {i} 条 assistant 带工具调用(一直是没有的)")
    lu = last_user_index(oa_msgs)
    if not any(m.get("role") == "assistant" for m in oa_msgs[lu + 1:]):
        out.append("最后一条 user 之后没有 assistant 轮(tick 不是最后一条 user?)")
    return out


def mask_context_turns(oa_msgs, sup_flags):
    """把 tick 前的 assistant 轮显式标成不监督。

    ⚠️ 爆破路径是**碰巧**排掉它的(靠「既不是信封也不是工具调用块」那道形状闸),
    碰巧对的东西平台一改就翻。所以不管走哪条渲染路,这条排除都要显式写出来。
    """
    out = list(sup_flags)
    for i in context_assistant_turns(oa_msgs):
        out[i] = False
    return out


def mask_runs(mask):
    """掩码里的连续监督段,返回 [(起, 止)) 列表。"""
    pos = np.nonzero(np.asarray(mask) == 1)[0]
    if not len(pos):
        return []
    runs, start, prev = [], int(pos[0]), int(pos[0])
    for q in pos[1:]:
        q = int(q)
        if q != prev + 1:
            runs.append((start, prev + 1)); start = q
        prev = q
    runs.append((start, prev + 1))
    return runs


def build_sample_whole(enc, tok, oa_msgs, sup_flags, tools, ids_A, id_EOS,
                       reasoning_effort=None, max_len=98304, thinking_mode="thinking"):
    """整条渲一次;超窗就**渲到最后一个装得下的轮为止**。

    返回 (ids, mask, err, 用到第几条消息为止)。

    ⛔ 超窗必须截断而不是整条丢掉:v7 语料 51 个任务(4.0%)整条超过 98304
    (最长 368,848),整条丢掉会连带损失 382 个本来学得到的轮。截断之后**丢 0 个任务**,
    丢掉的轮恰好就是爆破本来也丢掉的那 110 个 —— 两条路保住的轮数都是 7538。
    """
    sup = mask_context_turns(oa_msgs, sup_flags)
    aidx = [j for j, m in enumerate(oa_msgs) if m.get("role") == "assistant"]
    ids, mask, err = build_sample(enc, tok, oa_msgs, sup, tools, thinking_mode,
                                  ids_A, id_EOS, reasoning_effort=reasoning_effort)
    if err is None and ids.shape[0] <= max_len:
        return ids, mask, None, len(oa_msgs) - 1
    if err is not None:
        return None, None, err, None
    # 超窗:从后往前找第一个装得下的 assistant 轮
    for j in reversed(aidx):
        if not sup[j]:
            continue
        cut = oa_msgs[: j + 1]
        ids, mask, err = build_sample(enc, tok, cut, sup[: j + 1], tools,
                                      thinking_mode, ids_A, id_EOS,
                                      reasoning_effort=reasoning_effort, truncated=True)
        if err is None and ids.shape[0] <= max_len:
            return ids, mask, None, j
    return None, None, f"整条超长(>{max_len})且没有装得下的截断点", None


def _render_equivalence_proof(enc, tok, oa_msgs, sup_flags, tools, ids_A, id_EOS,
                              reasoning_effort, max_len):
    """闸 B:整条渲染 == 逐轮爆破,逐 token + 掩码位置都要对上。

    这道闸把 0819 那个一次性证明变成**每次跑数据都自动复查**:
    编码器以后改了行为(比如哪天渲染不再是"流式"的、加消息会回头改前面),
    这里会当场响,而不是训完拿分数猜。

    只在第一条形态够用的样本上跑一次(≥2 个监督轮才验得出前缀关系)。
    """
    ex = build_samples_thinking(enc, tok, oa_msgs, sup_flags, tools, ids_A, id_EOS,
                                reasoning_effort)
    ex_ok = [(x[0], x[1]) for x in ex
             if x[3] is None and x[0].shape[0] <= max_len]
    if len(ex_ok) < 2:
        return None                                   # 形态不够,换下一条
    wids, wmask, werr, _ = build_sample_whole(
        enc, tok, oa_msgs, sup_flags, tools, ids_A, id_EOS,
        reasoning_effort=reasoning_effort, max_len=max_len)
    if werr:
        sys.exit(f"⛔ 等价性自证失败:整条渲染这条样本被闸拦下 -> {werr}")
    spans = []
    for ids, mk in ex_ok:
        n = ids.shape[0]
        if not np.array_equal(wids[:n], ids):
            first = int(np.nonzero(wids[:n] != ids)[0][0])
            sys.exit(f"⛔ 等价性自证失败:整条渲染的前 {n} 个 token 和爆破样本不一致"
                     f"(首处分歧 @{first})。渲染不再是「流式」的了 —— 加消息会回头改"
                     f"前面的 token,整条渲染与爆破**不再等价**,禁止起训。")
        p = np.nonzero(mk == 1)[0]
        spans.append((int(p[0]), int(p[-1]) + 1))
    runs = mask_runs(wmask)
    if set(spans) != set(runs):
        sys.exit(f"⛔ 等价性自证失败:监督位置对不上。\n"
                 f"   爆破 {sorted(spans)}\n   整条 {sorted(runs)}\n"
                 f"   多出来的段 {sorted(set(runs) - set(spans))}、"
                 f"少掉的段 {sorted(set(spans) - set(runs))}")
    return True


def build_samples_thinking(enc, tok, oa_msgs, sup_flags, tools, ids_A, id_EOS,
                           reasoning_effort="low"):
    """思考模式:逐监督轮爆破成多条样本。

    每个被监督的 assistant 轮出一条样本:前缀 = 到它为止的消息,
    loss 只盖最后一段(该轮的思考 + 正文/工具调用 + EOS,思考进 loss)。

    ⛔ **0819 更正:原来这里写的理由是错的。** 原文说「我们的 vLLM 服务渲染上下文时
    丢历史轮思考,所以整条渲染会与服务相悖,逐轮爆破是唯一自洽解」。三层都不成立:

    1. **编码器那个开关是空转的。** `encoding_dsv4.encode_messages` 里:
           if any(m.get("tools") for m in full_messages):
               effective_drop_thinking = False
       我们的样本永远带工具 schema(27 个),所以传 `drop_thinking=True` **不起作用**,
       历史思考一直被保留。实测四种组合:带 tools 时 True/False 都保留 2/2,
       不带 tools 时 True 才丢。
    2. **线上也没有丢。** 1600 份 worker 落盘请求(ADK 发来的原件)零例外:
           最后一条 user 之后的 assistant 轮   304/304 = 100% 带思考
           最后一条 user 之前的那一轮            0/155 = 0%
       真实结构是四段消息(system / user / assistant 短应答 / user tick),
       **最后一条 user 就是那个 tick,所以 agent 循环内每一轮的思考线上全都保留**。
       丢的只有 tick 之前那个短应答轮 —— 而它本来就没有思考。
    3. **我们的数据也一致。** train.jsonl 全量 6257 条:被监督轮带思考 99.8%,
       前缀里循环内的历史轮带思考 15430/15454 = 99.8%,tick 前那轮 0/6257
       (它本来就没有)。⇒ 训推没有不一致。

    ⇒ 所以逐轮爆破**不是必需的**,它的代价是共享前缀被重复编码:
    332,138,424 token vs 整条渲一次的 76,608,004 token = **4.3×**
    (loss token 一个不少:8,928,285 个,占 2.69%)。

    ⭐ **0819 下午:等价性已经验完了,整条渲染是 `build_sample_whole`,新实验一律走那条。**
    这个函数留着只为复现旧实验(`--render explode`)。三层证据:

      前缀      6 个任务 24 个轮次,逐字符 + 逐 token 比对 **24/24 一致,0 处分歧**
      掩码位置  20 个任务,监督区绝对下标 **20/20 完全重合**
      归一化    `train_dsv4.py:181 calculate_per_token_loss = True`,按全局监督 token 归一

    因果注意力下第 k 轮只依赖它前面的 token ⇒ 三条合起来,两种渲染的梯度是同一个东西。
    打包也不拦路:`pack_dsv4_windows.py` 只管整条首尾相接,一条样本内部掩码断成几段
    它不看;`cu_seqlens` 保证窗口内互不可见、`position_ids` 每段重置。
    实测算力:爆破 3749 窗口 / 369M 槽位 → 整条 896 窗口 / 88M 槽位 = **省 4.19 倍**。
    这个证明现在由 `_render_equivalence_proof`(闸 B)每次跑数据自动复查一遍。

    ⭐ 真正要守的不变量由 `_thinking_explode_proof` 看着(见那儿):
    **不许有「最后一条 user 之前、却带思考」的 assistant 轮** ——
    那种轮线上会被丢掉,训练里却会留下,才是真的训推不一致。

    返回 [(ids, mask, 轮号, err)];err 非空的条目由调用方计数丢弃。"""
    out = []
    tools_w = wrap_tools(tools)
    for j, (m, supervised) in enumerate(zip(oa_msgs, sup_flags)):
        if m["role"] != "assistant" or not supervised:
            continue
        sub = [dict(x) for x in oa_msgs[: j + 1]]
        if tools_w:
            sub[0] = {**sub[0], "tools": tools_w}
        text = enc.encode_messages(sub, thinking_mode="thinking",
                                   drop_thinking=True,
                                   reasoning_effort=reasoning_effort,
                                   add_default_bos_token=True)
        ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"],
                         dtype=np.int32)
        eos = np.nonzero(ids == id_EOS)[0]
        n_asst = sum(1 for x in sub if x["role"] == "assistant")
        if len(eos) != n_asst:
            out.append((None, None, j,
                        f"EOS 数({len(eos)})对不上 assistant 数({n_asst})"))
            continue
        e = int(eos[-1])
        prev = int(eos[-2]) + 1 if len(eos) >= 2 else 0
        mk = np.nonzero(ids == ids_A)[0]
        cand = mk[(mk >= prev) & (mk < e)]
        start = int(cand[-1]) + 1 if len(cand) else prev
        end = e + 1                                   # 含 EOS,学会收尾
        if bool((ids[start:end - 1] == id_EOS).any()):
            out.append((None, None, j, "监督段内有多余的结束符"))
            continue
        seg = tok.decode(ids[start:end])
        is_env = '"content"' in seg and '"pass"' in seg
        if not (is_env or _seg_is_call(enc, seg)):
            out.append((None, None, j, "监督段既不是信封也不是工具调用块"))
            continue
        mask = np.zeros(ids.shape[0], dtype=np.uint8)
        mask[start:end] = 1
        out.append((ids, mask, j, None))
    return out


def _thinking_explode_proof(enc, tok, oa_msgs, tools=None, reasoning_effort="low"):
    """首条真实样本的一次性自证:渲染出来的前缀和线上服务给的形状一致。

    ⛔ **0819 重写。** 老版有两个毛病,合起来让它变成一道**永远不会响的闸**:

    1. **它调编码器时不传 tools**,而真实渲染路径(`build_samples_thinking`)
       是 `sub[0] = {**sub[0], "tools": tools_w}`。编码器里
       `if any(m.get("tools") ...): effective_drop_thinking = False` ——
       所以闸走的是「无 tools → 真会丢思考」那一支,数据走的是「带 tools → 不丢」那一支。
       **闸验证的是一条我们不走的路径。**
    2. 它的断言是「前缀里不许出现历史思考」。那条断言本身就基于一个错前提
       (以为线上会丢历史思考)。实测 1600 份 ADK 原件:agent 循环内 304/304
       **全都带思考**。所以老断言即使跑在真路径上也是**反的**。

    现在守的是真正会出事的那条不变量:

      ⭐ **不许有「排在最后一条 user 之前、却带思考」的 assistant 轮。**
        线上对这种轮会丢掉思考(实测 0/155 带思考),训练里却会保留 ——
        那才是真的训推不一致。v7 语料里这种轮是 0/6257(它们是 tick 前的短应答,
        本来就没思考),所以现状是安全的;换语料时这条会立刻响。

    另外两条顺带验:目标轮的思考必须进渲染文本(思考要进 loss);
    循环内的历史思考必须**在**渲染文本里(和线上一致,不许被悄悄丢掉)。
    """
    last_user = max((i for i, m in enumerate(oa_msgs)
                     if m.get("role") == "user"), default=-1)
    # 真路径同款渲染:tools 必须注入,否则验的不是我们跑的那一支
    sub = [dict(x) for x in oa_msgs]
    tools_w = wrap_tools(tools)
    if tools_w:
        sub[0] = {**sub[0], "tools": tools_w}
    tgt = oa_msgs[-1].get("reasoning_content")
    inloop = [m.get("reasoning_content") for i, m in enumerate(oa_msgs[:-1])
              if m.get("role") == "assistant" and m.get("reasoning_content")
              and i > last_user]
    if not tgt or not inloop:
        return None                                    # 这条形态不够,换下一条

    # ⭐ 主闸:线上会丢的那种轮,我们的语料里一条都不许有
    bad = [i for i, m in enumerate(oa_msgs)
           if m.get("role") == "assistant" and m.get("reasoning_content")
           and i < last_user]
    if bad:
        sys.exit(f"⛔ 自证失败:第 {bad} 条 assistant 轮排在最后一条 user 之前"
                 f"却带思考。线上这种轮的思考会被丢掉(实测 0/155),"
                 f"训练里却会保留 —— 训推不一致,禁止起训。")

    text = enc.encode_messages(sub, thinking_mode="thinking", drop_thinking=True,
                               reasoning_effort=reasoning_effort,
                               add_default_bos_token=True)
    if tgt[:64] not in text:
        sys.exit("⛔ 自证失败:目标轮的思考没有出现在渲染文本里(思考没进监督段)")
    missing = [h[:40] for h in inloop if h[:64] not in text]
    if missing:
        sys.exit(f"⛔ 自证失败:循环内的历史思考被丢了 {len(missing)} 段"
                 f"(编码器语义变了?线上是 100% 保留的,丢了就和服务不一致)")
    return True


def resolve_reasoning_effort(row, requested):
    """Resolve the actual DSv4 prompt effort used during tokenization.

    ``source`` preserves mixed-effort corpora by reading the rollout ledger
    value carried into each rendered SFT row.  Explicit low/high/max forces one
    controlled training arm.  Invalid or no-think labels in a thinking corpus
    are rejected instead of silently falling back to the encoder's low default.
    """
    effort = requested
    if requested == "source":
        meta = row.get("meta") or {}
        effort = (row.get("effort") or row.get("reasoning_effort")
                  or meta.get("effort") or meta.get("reasoning_effort") or "low")
    if effort not in {"low", "high", "max"}:
        raise ValueError(f"thinking 样本的 reasoning_effort 非法:{effort!r}")
    return effort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir")
    ap.add_argument("--self-test", action="store_true",
                    help="不碰数据,回放历史 bug①② 验证保真闸抓得住(改 to_openai 前必跑)")
    ap.add_argument("--max-len", type=int, default=98304,
                    help="超过这个长度的样本丢掉。实测语料真实最大 83,747 token,"
                         "所以 98304 一条都不丢(和 SEQ_LEN 保持一致)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teacher-only", default="",
                    help="只留某个老师,如 opus(匹配 meta.model 含该串)")
    ap.add_argument("--judge-strict", action="store_true",
                    help="只留**两个判官都判 ok** 的(默认 ok+minor 都要)")
    ap.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"],
                    help="chat=不思考(这份语料本来就没思考,默认就用它);"
                         "thinking=思考模式,逐监督轮爆破(见 build_samples_thinking)")
    ap.add_argument("--render", choices=["whole", "explode"],
                    help="thinking 模式**必填**,不给默认值。\n"
                         "  whole  = 一个任务渲一条,掩码盖住这条里所有要学的段。\n"
                         "           新实验一律用这个:梯度和 explode 逐 token 相同,\n"
                         "           算力省 4.19 倍(369M 槽位 -> 88M)。\n"
                         "  explode= 逐监督轮各渲一条(0819 之前的老做法)。留着只为\n"
                         "           复现旧实验 —— 同一份前缀被重复编码 4.9 遍。\n"
                         "⛔ 为什么不给默认值:默认 whole 会让人复现旧实验时静默拿到新渲染,\n"
                         "   默认 explode 会让新实验白烧 4 倍算力。会骗人的默认值比没有默认值更糟\n"
                         "   (harvest_select 那个 hy3-low 兜底值的教训)。\n"
                         "chat 模式忽略此参数 —— 它从来就是整条渲染。")
    ap.add_argument("--allow-structure-outliers", type=int, default=0,
                    help="闸 A 的逃生口:允许多少条消息结构不合假设的样本。默认 0(fail-closed)")
    ap.add_argument("--reasoning-effort", default="source",
                    choices=["source", "low", "high", "max"],
                    help="thinking 模式真正写入 DSv4 prompt 的档位。source=逐行读取"
                         " effort/reasoning_effort；显式 high 可构造严格 high 对照臂。"
                         "旧代码未透传此值，实际总是编码器默认 low。")
    ap.add_argument("--src-format", default="v", choices=["v", "openai"],
                    help="v=v系语料(过 to_openai 转换+保真闸);"
                         "openai=已是 OpenAI messages 的语料(如 rq2e 思考批),"
                         "直接编码,监督所有 assistant 轮")
    ap.add_argument("--terminal-contract", default="autotask",
                    choices=["autotask", "open"],
                    help="autotask requires an envelope/transfer tail; open allows media-agent text or tool tails")
    ap.add_argument("--val-count", type=int, default=32,
                    help="val 的目标条数。按 rule_head 整组切,所以实际条数会略多于这个数")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="切 train/val 用的随机种子(只影响哪些任务组进 val)")
    ap.add_argument("--allow-narration", action="store_true",
                    help="放行 v1_raw_best 那种「旁白和信封挤在一条里」的语料。"
                         "⛔ 只有明确想训旁白时才加,常规 SFT 别用(见文件头)")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    if not a.out_dir:
        ap.error("--out-dir 必须给(除非 --self-test)")
    if a.thinking_mode == "thinking" and not a.render:
        ap.error("thinking 模式必须显式给 --render whole|explode(见该参数的说明)。"
                 "新实验用 whole。")
    render_mode = a.render or "whole"        # chat 模式本来就是整条渲染
    # ⛔ 同一个目录里混两种渲染 = 事后谁也说不清这份数据是怎么来的
    _old_meta = os.path.join(a.out_dir, "meta.json")
    if os.path.exists(_old_meta):
        try:
            _prev = json.load(open(_old_meta)).get("render")
        except Exception:
            _prev = None
        if _prev and _prev != render_mode:
            sys.exit(f"⛔ {a.out_dir} 里已有一份 render={_prev} 的数据,这次是 "
                     f"{render_mode}。两种渲染混在一个目录里事后查不清,换个 --out-dir。")

    # ── 闸 0:源文件认不认得。防的是「照文档敲命令,结果喂进了 raw 那份」
    prov = provenance(a.src, a.model)
    print(f"[出处] {os.path.basename(a.src)}  sha256={prov['源文件 SHA256'][:16]}…  "
          f"{prov['源文件认得出吗']}", file=sys.stderr)
    if "旁白没删" in prov["源文件认得出吗"] and not a.allow_narration:
        sys.exit("⛔ 这是 v1_raw_best —— 它有 2164 条信封消息前面挂着旁白,1144 条会进 loss,\n"
                 "   训出来的模型会「先写 500 字旁白再吐 JSON」,打穿信封解析。\n"
                 f"   要的应该是:{DEFAULT_SRC}\n"
                 "   确实想训旁白就加 --allow-narration。")

    sys.path.insert(0, os.path.join(a.model, "encoding"))
    import encoding_dsv4 as enc
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids_A = tok.convert_tokens_to_ids(enc.ASSISTANT_SP_TOKEN)
    id_EOS = tok.convert_tokens_to_ids(enc.eos_token)

    os.makedirs(a.out_dir, exist_ok=True)
    # 按 gzip 魔数(1f 8b)判压没压,别按文件名后缀 —— 基准语料是 .jsonl.gz,
    # 而 render_sft.py 出的是**没压缩的 .jsonl**,写死 gzip.open 会当场炸
    # (BadGzipFile,0815 踩了)。两种都收,省得每次先想起来 gzip 一下。
    with open(a.src, "rb") as _f:
        _gz = _f.read(2) == b"\x1f\x8b"
    with (gzip.open(a.src, "rt") if _gz else open(a.src, "rt")) as _fh:
        rows = [json.loads(l) for l in _fh if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    drop = {}
    def _d(why):
        drop[why] = drop.get(why, 0) + 1

    kept_ids, kept_mask, kept_json, kept_group = [], [], [], []
    rendered_efforts = {}
    proof_done = False
    equiv_done = False
    n_truncated = 0
    structure_bad = []
    for d in rows:
        meta = d.get("meta") or {}

        if a.src_format == "openai":
            # 语料已是 OpenAI messages:不做转换。保留每条 assistant 的
            # `loss: false`，这样 media-zh-reasoning-v1 可以把历史/上下文
            # assistant 轮保留在输入里而不误进监督；没有 loss 字段仍按旧约定监督。
            oa = d["messages"]
            sup = [
                m.get("role") == "assistant" and m.get("loss") is not False
                for m in oa
            ]
            group = str(d.get("question") or d.get("iid") or meta.get("rule_head")
                        or len(kept_ids))
            meta_out = {k: d.get(k) for k in
                        ("question", "iid", "teacher", "effort", "run", "trace")}
        else:
            if a.teacher_only and a.teacher_only not in (meta.get("model") or ""):
                _d("老师不符"); continue
            if a.judge_strict:
                j = meta.get("judge") or {}
                if not j or any((v or {}).get("severity") != "ok" for v in j.values()):
                    _d("判官非全 ok"); continue

            oa, sup, conv_err = to_openai(
                d["messages"], terminal_contract=a.terminal_contract)
            if oa is None:
                _d(f"渲染失败:{conv_err}"); continue

            # 转换保真闸(§1.5):bug①② 从「只靠代码写法保证」变成有事后检查
            if action_count_mismatch(d["messages"], oa):
                _d(R_SPLIT); continue
            if roundtrip_check(d["messages"], oa):
                _d(R_ROUNDTRIP); continue
            group = str(meta.get("rule_head") or meta.get("trace_id") or len(kept_ids))
            meta_out = {k: meta.get(k) for k in
                        ("trace_id", "model", "rule_lang", "n_triggers", "rule_head")}

        if a.thinking_mode == "thinking":
            try:
                row_effort = resolve_reasoning_effort(d, a.reasoning_effort)
            except ValueError as error:
                _d(str(error)); continue
            rendered_efforts[row_effort] = rendered_efforts.get(row_effort, 0) + 1
            meta_out = {**meta_out, "rendered_reasoning_effort": row_effort}
            # ── 闸 A:消息结构还是不是我们假设的那个形状 ──────────────────
            _iss = structure_issues(oa)
            if _iss:
                structure_bad.append((group, _iss))
                _d("结构不合假设:" + _iss[0]); continue
            if not proof_done:
                proof_done = bool(_thinking_explode_proof(
                    enc, tok, oa, d.get("tools"), row_effort))
                if proof_done:
                    print("[自证] 渲染形状与线上一致:tick 前无带思考的轮、"
                          "循环内历史思考保留、目标思考在监督段 ✅", file=sys.stderr)
            # ── 闸 B:整条渲染 == 逐轮爆破(只在第一条形态够用的样本上跑一次)──
            if render_mode == "whole" and not equiv_done:
                equiv_done = bool(_render_equivalence_proof(
                    enc, tok, oa, sup, d.get("tools"), ids_A, id_EOS,
                    row_effort, a.max_len))
                if equiv_done:
                    print("[自证] 整条渲染与逐轮爆破逐 token 等价、监督位置重合 ✅",
                          file=sys.stderr)
            if render_mode == "whole":
                ids, mask, err, upto = build_sample_whole(
                    enc, tok, oa, sup, d.get("tools"), ids_A, id_EOS,
                    reasoning_effort=row_effort, max_len=a.max_len)
                if err:
                    _d(f"整条:{err}"); continue
                if upto < len(oa) - 1:
                    n_truncated += 1
                sup_used = mask_context_turns(oa, sup)[: upto + 1]
                kept_ids.append(ids); kept_mask.append(mask)
                kept_group.append(group)
                kept_json.append({"messages": oa[: upto + 1], "sup": sup_used,
                                  "tools": d.get("tools"), "meta": meta_out})
            else:
                for ids, mask, jturn, err in build_samples_thinking(
                        enc, tok, oa, sup, d.get("tools"), ids_A, id_EOS,
                        row_effort):
                    if err:
                        _d(f"爆破:{err}"); continue
                    if ids.shape[0] > a.max_len:
                        _d(f"超长(>{a.max_len})"); continue
                    kept_ids.append(ids); kept_mask.append(mask)
                    kept_group.append(group)
                    kept_json.append({"messages": oa[:jturn + 1], "sup_turn": jturn,
                                      "tools": d.get("tools"), "meta": meta_out})
        else:
            ids, mask, err = build_sample(enc, tok, oa, mask_context_turns(oa, sup),
                                          d.get("tools"), a.thinking_mode, ids_A, id_EOS,
                                          terminal_contract=a.terminal_contract)
            if err:
                _d(err); continue
            if ids.shape[0] > a.max_len:
                _d(f"超长(>{a.max_len})"); continue

            kept_ids.append(ids); kept_mask.append(mask)
            kept_group.append(group)
            kept_json.append({"messages": oa, "sup": sup, "tools": d.get("tools"),
                              "meta": meta_out})

    if len(structure_bad) > a.allow_structure_outliers:
        import collections as _c
        why = _c.Counter(i for _, iss in structure_bad for i in iss)
        sys.exit(f"⛔ 闸 A:{len(structure_bad)} 条样本的消息结构不合假设"
                 f"(允许 {a.allow_structure_outliers} 条)。\n"
                 + "\n".join(f"   {v:5d} × {k}" for k, v in why.most_common(8))
                 + "\n   平台可能改了那四段消息的结构。**别直接放行** —— 先确认"
                   "「tick 前那条 assistant 是不是还该排除在 loss 之外」,"
                   "确认过再用 --allow-structure-outliers。")
    if not kept_ids:
        sys.exit(f"⛔ 一条都没留下。丢弃原因:{drop}")

    # ── 落盘。**格式必须和同目录 dsv4_sft_data.py 完全一致**,这样能直接喂给那份
    #    已经跑过几十次的数据提供器,不用新写一个(新写 = 新 bug)。
    #      {split}.bin        int32 memmap,所有样本的 token 首尾相接
    #      {split}.mask.bin   uint8 memmap,同长,1=这个 token 进 loss
    #      {split}.idx.npy    (N,2) 的 [起始偏移, 长度]
    # ⛔ train/val 必须**按任务整组切**,不能按文件顺序切尾巴。
    #    同一个定时任务(rule_head)会被触发很多次,每次都是一条样本。按顺序切的话
    #    同一个任务的不同次触发会被劈到两边 —— 实测老切法下 32 条 val 里有 31 条的
    #    系统提示词在 train 里逐字出现过,val loss 会偏乐观,不能拿来判早停。
    fingerprints = {}                      # 落盘产物的 SHA,起训闸 4.5 核对用
    n_val_target = min(a.val_count, max(0, len(kept_ids) // 10))
    rng = np.random.default_rng(a.split_seed)
    groups = {}
    for i, g in enumerate(kept_group):
        groups.setdefault(g, []).append(i)
    gnames = sorted(groups)
    rng.shuffle(gnames)
    val_idx, n_val = set(), 0
    for g in gnames:                       # 整组整组地放进 val,直到够数
        if n_val >= n_val_target:
            break
        val_idx.update(groups[g]); n_val += len(groups[g])
    train_pos = [i for i in range(len(kept_ids)) if i not in val_idx]
    val_pos = sorted(val_idx)
    parts = {"train": train_pos, "val": val_pos}
    leak = len({kept_group[i] for i in train_pos} & {kept_group[i] for i in val_pos})
    stats = {}
    for split, pos in parts.items():
        if not pos:
            continue
        ii, mm = [kept_ids[i] for i in pos], [kept_mask[i] for i in pos]
        lens_arr = np.asarray([x.shape[0] for x in ii], dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(lens_arr)[:-1]])
        np.concatenate(ii).astype(np.int32).tofile(f"{a.out_dir}/{split}.bin")
        np.concatenate(mm).astype(np.uint8).tofile(f"{a.out_dir}/{split}.mask.bin")
        np.save(f"{a.out_dir}/{split}.idx.npy",
                np.stack([offs, lens_arr], axis=1).astype(np.int64))
        # 产物指纹:防「本地盘把文件静默截成 0 字节」(文档 §1.1 的真实事故)。
        # run_dsv4.sh 的闸 4.5 起训前会核长度 + 这个 SHA。
        for name in (f"{split}.bin", f"{split}.mask.bin", f"{split}.idx.npy"):
            fingerprints[name] = _sha256(f"{a.out_dir}/{name}")[:16]
        off = np.concatenate([offs, [int(offs[-1] + lens_arr[-1])]])
        lens = [int(x.shape[0]) for x in ii]
        sup = [int(x.sum()) for x in mm]
        stats[split] = {
            "样本数": len(ii),
            "任务(rule_head)组数": len({kept_group[i] for i in pos}),
            "token 总数": int(off[-1]),
            "长度 p50/p90/max": [int(np.percentile(lens, 50)), int(np.percentile(lens, 90)), max(lens)],
            # ⚠️ 这个比例天然很低(1%~3%)。这份语料输入极长(系统提示 1.1万字 +
            # 带 3 万字技能说明的 user + 一堆工具返回),而要学的输出就几百 token。
            # 低不代表错;真正说明对齐没错的是「末段解码出来是信封」那道逐条断言。
            "进 loss 的 token 占比": round(sum(sup) / max(off[-1], 1), 4),
            "每样本进 loss 的 token 数 p50/max": [int(np.percentile(sup, 50)), max(sup)],
        }

    # train 和 val 都落一份可读中间格式 —— 排查时直接 grep,不用反解 bin
    for split, pos in parts.items():
        if not pos:
            continue
        with open(f"{a.out_dir}/{split}.jsonl", "w") as f:
            for i in pos:
                f.write(json.dumps(kept_json[i], ensure_ascii=False) + "\n")

    meta_out = {
        "出处": prov,          # 闸 0 已经算过,不重复哈希
        "源文件": a.src, "模型": a.model, "thinking_mode": a.thinking_mode,
        "render": render_mode,
        "整条超窗被截断的任务数": n_truncated,
        "结构不合假设的样本数": len(structure_bad),
        "reasoning_effort 参数": a.reasoning_effort,
        "实际渲染档位分布": rendered_efforts,
        "src_format": a.src_format,
        "terminal_contract": a.terminal_contract,
        "读入": len(rows), "留下": len(kept_ids), "丢弃原因": drop,
        "切分": {"方式": "按 rule_head 整组切", "seed": a.split_seed,
                 "train/val 共享的任务组数(必须为 0)": leak},
        "各 split": stats,
        "产物指纹": fingerprints,
    }
    json.dump(meta_out, open(f"{a.out_dir}/meta.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(meta_out, ensure_ascii=False, indent=2))

    # ── 验收闸:任何一条不过就别开训
    tr = stats.get("train", {})

    # 直接在落盘数据上复查一遍「非段尾结束符」。上面 build_sample 里已经逐条拦过了,
    # 这里是独立的第二道 —— 万一切段逻辑以后被改坏,这道还能兜住。
    n_fake, n_fake_rows = 0, 0
    for i in parts["train"]:
        idsv, mkv = kept_ids[i], kept_mask[i]
        pos = np.nonzero(mkv == 1)[0]
        if not len(pos):
            continue
        segs, st, prv = [], pos[0], pos[0]
        for p in pos[1:]:
            if p != prv + 1:
                segs.append((st, prv)); st = p
            prv = p
        segs.append((st, prv))
        f = sum(int((idsv[s:e] == id_EOS).sum()) for s, e in segs)   # 不含段尾那个
        if f:
            n_fake += f; n_fake_rows += 1

    print("\n=== 验收 ===")
    bad = 0
    for name, ok, detail in [
        ("留存率 ≥80%", len(kept_ids) >= 0.8 * len(rows), f"{len(kept_ids)}/{len(rows)}"),
        # 占比只做兜底(防"全 0"和"全监督"两种崩法),真正的对齐验收在逐条断言里
        ("有 loss 的 token 占比在 0.5%~40%",
         0.005 <= tr.get("进 loss 的 token 占比", 0) <= 0.40,
         str(tr.get("进 loss 的 token 占比"))),
        ("每样本进 loss 的 token 数中位 ≥100",
         tr.get("每样本进 loss 的 token 数 p50/max", [0, 0])[0] >= 100,
         str(tr.get("每样本进 loss 的 token 数 p50/max"))),
        ("最长样本 ≤ max_len", tr.get("长度 p50/p90/max", [0, 0, 10**9])[2] <= a.max_len,
         str(tr.get("长度 p50/p90/max"))),
        # train/val 不许共享任务:同一个定时任务的不同次触发被劈两边 = val loss 偏乐观
        ("train/val 没有任务泄漏", leak == 0, f"共享 {leak} 个任务组"),
        # 落盘数据里不许有「非段尾的结束符」—— 兜底;单靠它抓不到 bug①(多出来的
        # 结束符自己会成为段边界),真正挡 bug①② 的是下面三条保真闸
        ("监督区里没有多余的结束符", n_fake == 0, f"{n_fake} 个(分布在 {n_fake_rows} 条样本里)"),
        # 转换保真闸(§1.5):中招 = to_openai 被改坏了,一条都不能有
        ("渲染动作数 == 原始 assistant + 裸工具组(挡 bug①)",
         drop.get(R_SPLIT, 0) == 0, f"违例 {drop.get(R_SPLIT, 0)} 条"),
        ("工具名/参数/返回逐条对得上原始且按序认领(挡 bug②)",
         drop.get(R_ROUNDTRIP, 0) == 0, f"违例 {drop.get(R_ROUNDTRIP, 0)} 条"),
        # 结构异常是上游语料的瑕疵,正确处理就是丢掉个别样本 —— 所以按**比例**兜底:
        # 零星几条(v2 实测 7+2+0)是常态,超过 0.5% 才说明上游批量变了或转换坏了
        ("结构异常丢弃 ≤0.5%(未认领调用 / 标记不一致 / assistant 背靠背)",
         sum(v for k, v in drop.items()
             if ("未认领" in k) or ("标记不一致" in k) or ("背靠背" in k)) <= 0.005 * len(rows),
         str({k: v for k, v in drop.items()
              if ("未认领" in k) or ("标记不一致" in k) or ("背靠背" in k)} or "0 条")),
    ]:
        print(("  ✅ " if ok else "  ⛔ ") + name + "  —— " + detail)
        bad += (not ok)
    print("\n" + ("✅ 可以进下一步(起训)" if not bad else "⛔ 有闸没过,先查清楚再训"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
