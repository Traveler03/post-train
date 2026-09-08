#!/usr/bin/env python3
"""渲染模式的回归测试:整条渲染(--render whole)与逐轮爆破的等价性 + 两道闸会不会响。

⭐ **不加载真模型。** 这里用一个"假编码器"(FakeEnc/FakeTok):它满足 build_sample
依赖的全部结构约定(每条 assistant 一个 A 标记 + 一个结束符 E、信封段里有 "content"
和 "pass"、工具调用段里有 tool▁calls▁begin),于是掩码机器、截断逻辑、两道闸都能
在毫秒级测到。真编码器上的等价性由 `--render whole` 跑数据时的闸 B 每次复查
(0819 实测:前缀 24/24 逐 token 一致、掩码位置 20/20 重合)。

⛔ 这份测试的重点不是"闸存在",是**"闸会响"** —— `_thinking_explode_proof` 那道闸
写了但走的是我们不走的路径,于是从来没响过。所以每道闸都配一个注入用例。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare_dsv4_data as P                                    # noqa: E402

ENV = '"content" "pass"'                     # 信封段;不含 A/E 大写字母
CALL = 'tool▁calls▁begin x'                  # 工具调用段
ACK = 'ack ready'                            # tick 前那句罐头应答


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return ''.join(chr(int(i)) for i in ids)

    def convert_tokens_to_ids(self, t):
        return ord(t)


class FakeEnc:
    ASSISTANT_SP_TOKEN = 'A'
    eos_token = 'E'
    TOOL_CALLS_BEGIN = 'tool▁calls▁begin'
    seen_efforts = []

    @classmethod
    def encode_messages(cls, messages, thinking_mode, drop_thinking=True,
                        reasoning_effort=None, add_default_bos_token=True, context=None):
        cls.seen_efforts.append(reasoning_effort)
        out = 'B' if add_default_bos_token else ''
        for m in messages:
            role = m.get('role')
            if role == 'system':
                out += 'S' * 3
            elif role == 'user':
                out += 'U' * 3
            elif role == 'tool':
                out += 'O' * 3
            elif role == 'assistant':
                out += 'A'
                if m.get('reasoning_content'):
                    out += 't' * 2
                out += m.get('content') or ''
                out += 'E'
        return out


def msg(role, content='', **kw):
    return {'role': role, 'content': content, **kw}


def convo(n_loop=3, ack=ACK, ack_kw=None, tail_env=True):
    """system / 题面 / 罐头应答 / tick,然后 n_loop 轮循环(前 n-1 轮调工具、末轮交信封)。"""
    out = [msg('system', ''), msg('user', '')]
    if ack is not None:
        out.append(msg('assistant', ack, **(ack_kw or {})))
    out.append(msg('user', ''))                                   # tick
    for k in range(n_loop):
        last = (k == n_loop - 1)
        body = ENV if (last and tail_env) else CALL
        out.append(msg('assistant', body, reasoning_content='think'))
        if not last:
            out.append(msg('tool', ''))
    return out


def sup_all(oa):
    return [m['role'] == 'assistant' for m in oa]


def build(oa, max_len=10 ** 6, effort='low'):
    return P.build_sample_whole(FakeEnc, FakeTok(), oa, sup_all(oa), None,
                               ord('A'), ord('E'), reasoning_effort=effort,
                               max_len=max_len)


class ContextTurnTest(unittest.TestCase):
    """tick 前的 assistant 轮 = ADK 塞进来的输入,不是我们的输出,不该进 loss。"""

    def test_normal_shape_has_exactly_one_context_turn(self):
        oa = convo()
        self.assertEqual(P.context_assistant_turns(oa), [2])
        self.assertEqual(P.structure_issues(oa), [])

    def test_wording_change_does_not_break_it(self):
        """⭐ 用户 0819 的担心:平台改了那句罐头应答怎么办。判据是结构不是字符串。"""
        for ack in ('Sistema inicializado. Estou pronto.',
                    '系统已就绪',
                    '',
                    'x' * 500):
            oa = convo(ack=ack)
            self.assertEqual(P.context_assistant_turns(oa), [2], ack[:20])
            self.assertEqual(P.structure_issues(oa), [], ack[:20])

    def test_ack_removed_excludes_nothing(self):
        oa = convo(ack=None)
        self.assertEqual(P.context_assistant_turns(oa), [])
        self.assertEqual(P.structure_issues(oa), [])

    def test_mask_context_turns_only_touches_context(self):
        oa = convo()
        got = P.mask_context_turns(oa, sup_all(oa))
        self.assertFalse(got[2])                                  # 罐头应答
        self.assertEqual([i for i, x in enumerate(got) if x],
                         [i for i, m in enumerate(oa)
                          if m['role'] == 'assistant' and i > 3])


class StructureGateTest(unittest.TestCase):
    """闸 A:结构变了要当场停机,不许静默按老假设跑。"""

    def test_two_context_turns_flagged(self):
        oa = convo()
        oa.insert(3, msg('assistant', 'second ack'))
        self.assertIn('tick 前有 2 条 assistant 轮', P.structure_issues(oa)[0])

    def test_context_turn_with_tool_calls_flagged(self):
        oa = convo(ack_kw={'tool_calls': [{'function': {'name': 'x', 'arguments': '{}'}}]})
        self.assertTrue(any('带工具调用' in x for x in P.structure_issues(oa)))

    def test_context_turn_with_thinking_flagged(self):
        oa = convo(ack_kw={'reasoning_content': 'hmm'})
        self.assertTrue(any('带思考' in x for x in P.structure_issues(oa)))

    def test_no_assistant_after_tick_flagged(self):
        oa = [msg('system'), msg('user'), msg('assistant', ACK), msg('user')]
        self.assertTrue(any('之后没有 assistant' in x for x in P.structure_issues(oa)))


class MaskRunsTest(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(P.mask_runs([0, 1, 1, 0, 1, 0]), [(1, 3), (4, 5)])
        self.assertEqual(P.mask_runs([0, 0]), [])
        self.assertEqual(P.mask_runs([1]), [(0, 1)])


class WholeRenderTest(unittest.TestCase):
    def test_ack_is_not_in_loss(self):
        oa = convo()
        ids, mask, err, upto = build(oa)
        self.assertIsNone(err)
        self.assertEqual(upto, len(oa) - 1)
        text = FakeTok().decode(ids)
        # 罐头应答那段在渲染文本里(它是上下文),但一个 token 都不进 loss
        self.assertIn(ACK, text)
        covered = ''.join(FakeTok().decode(ids[s:e]) for s, e in P.mask_runs(mask))
        self.assertNotIn(ACK, covered)
        self.assertEqual(len(P.mask_runs(mask)), 3)               # 三个循环内轮次

    def test_matches_explode_token_for_token(self):
        oa = convo(n_loop=4)
        wids, wmask, err, _ = build(oa)
        self.assertIsNone(err)
        ex = P.build_samples_thinking(FakeEnc, FakeTok(), oa, sup_all(oa), None,
                                     ord('A'), ord('E'), 'low')
        ok = [(i, m) for i, m, _j, e in ex if e is None]
        self.assertEqual(len(ok), 4)                              # 罐头应答被形状闸排掉
        spans = []
        for ids, mk in ok:
            np.testing.assert_array_equal(wids[:len(ids)], ids)
            pos = np.nonzero(mk == 1)[0]
            spans.append((int(pos[0]), int(pos[-1]) + 1))
        self.assertEqual(sorted(spans), sorted(P.mask_runs(wmask)))
        self.assertEqual(int(wmask.sum()), sum(int(m.sum()) for _, m in ok))

    def test_truncates_to_last_fitting_turn_instead_of_dropping(self):
        """超窗的任务不能整个丢掉 —— v7 有 51 个(4.0%),整条丢会连带损失 382 个轮。"""
        oa = convo(n_loop=5)
        full = len(build(oa)[0])
        ids, mask, err, upto = build(oa, max_len=full - 10)
        self.assertIsNone(err)
        self.assertLess(upto, len(oa) - 1)
        self.assertLessEqual(len(ids), full - 10)
        self.assertGreater(int(mask.sum()), 0)
        # 截断点落在工具调用轮上:末段闸必须放行,否则整条被丢
        self.assertIn(FakeEnc.TOOL_CALLS_BEGIN,
                      FakeTok().decode(ids[slice(*P.mask_runs(mask)[-1])]))

    def test_effort_reaches_the_encoder(self):
        """⛔ 老代码 build_sample 不传 reasoning_effort,做 high 对照臂会静默渲成 low。"""
        FakeEnc.seen_efforts = []
        build(convo(), effort='high')
        self.assertIn('high', FakeEnc.seen_efforts)
        self.assertNotIn('low', FakeEnc.seen_efforts)


class EquivalenceGateFiresTest(unittest.TestCase):
    """闸 B:注入三种真实会发生的退化,每种都必须停机。闸不响 = 没有闸。"""

    def _proof(self, oa):
        return P._render_equivalence_proof(FakeEnc, FakeTok(), oa, sup_all(oa), None,
                                          ord('A'), ord('E'), 'low', 10 ** 6)

    def test_passes_when_healthy(self):
        self.assertTrue(self._proof(convo(n_loop=3)))

    def test_fires_when_context_turn_leaks_into_loss(self):
        oa = convo(n_loop=3)
        orig, P.mask_context_turns = P.mask_context_turns, lambda m, s: list(s)
        try:
            with self.assertRaises(SystemExit) as cm:
                self._proof(oa)
            self.assertIn('自证失败', str(cm.exception))
        finally:
            P.mask_context_turns = orig

    def test_fires_when_mask_drifts(self):
        oa = convo(n_loop=3)
        orig = P.build_sample

        def drifted(*a, **k):
            ids, mask, err = orig(*a, **k)
            if err is None:
                mask = mask.copy(); mask[0] = 1
            return ids, mask, err

        P.build_sample = drifted
        try:
            with self.assertRaises(SystemExit) as cm:
                self._proof(oa)
            self.assertIn('监督位置对不上', str(cm.exception))
        finally:
            P.build_sample = orig

    def test_fires_when_render_stops_being_streaming(self):
        """编码器哪天改成「加消息会回头改前面的 token」,整条与爆破就不再等价。"""
        oa = convo(n_loop=3)
        n = len(oa)
        orig = FakeEnc.encode_messages.__func__

        def bad(cls, messages, *a, **k):
            text = orig(cls, messages, *a, **k)
            return ('Z' + text) if len(messages) == n else text

        FakeEnc.encode_messages = classmethod(bad)
        try:
            with self.assertRaises(SystemExit) as cm:
                self._proof(oa)
            self.assertIn('不再是「流式」', str(cm.exception))
        finally:
            FakeEnc.encode_messages = classmethod(orig)


class CliContractTest(unittest.TestCase):
    def test_render_has_no_default_in_thinking_mode(self):
        with open(P.__file__, encoding='utf-8') as f:
            src = f.read().split('"""', 2)[-1]
        self.assertIn('"--render", choices=["whole", "explode"]', src)
        self.assertIn('thinking 模式必须显式给 --render', src)
        # 会骗人的默认值比没有默认值更糟:这里不许出现 default=
        head = src.split('"--render"', 1)[1].split('ap.add_argument', 1)[0]
        self.assertNotIn('default=', head)

    def test_meta_records_render_mode(self):
        with open(P.__file__, encoding='utf-8') as f:
            src = f.read()
        self.assertIn('"render": render_mode', src)
        self.assertIn('两种渲染混在一个目录里', src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
