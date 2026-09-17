"""Offline tests using synthetic records and the published aggregates only."""

import hashlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from audit_sft_interactions import audit_rollouts, audit_train, histogram, stats


class MemoryJSONL:
    """Minimal Path substitute so fixtures never touch private files."""

    name = "synthetic.jsonl"

    def __init__(self, rows):
        self.data = b"\n" + b"".join(
            json.dumps(row).encode("utf-8") + b"\n" for row in rows
        )

    def open(self, mode):
        assert mode == "rb"
        return io.BytesIO(self.data)

    def stat(self):
        return SimpleNamespace(st_size=len(self.data))


class InteractionAuditTests(unittest.TestCase):
    def test_quantiles_and_histogram(self):
        self.assertEqual(stats([9, 1, 5, 3]), {
            "min": 1, "p50": 3, "p90": 5, "max": 9,
            "total": 18, "mean": 4.5,
        })
        self.assertEqual(histogram([2, 1, 2]), [
            {"value": 1, "records": 1}, {"value": 2, "records": 2},
        ])
        with self.assertRaises(ValueError):
            stats([])

    def test_batched_calls_and_unsupervised_initializer(self):
        fixture = MemoryJSONL([{
            "messages": [
                {"role": "system"},
                {"role": "user"},
                {"role": "assistant"},
                {"role": "user"},
                {"role": "assistant", "tool_calls": [{}, {}]},
                {"role": "tool"},
                {"role": "tool"},
                {"role": "assistant"},
            ],
            "sup": [False, False, False, False, True, False, False, True],
        }])
        report = audit_train(fixture)
        for key, expected in {
            "messages": 8, "assistant_messages": 3,
            "supervised_assistant_turns": 2, "tool_call_turns": 1,
            "tool_calls": 2, "tool_result_messages": 2, "user_role_messages": 2,
        }.items():
            self.assertEqual(report["statistics"][key]["total"], expected)
        self.assertEqual(report["unsupervised_assistant_messages"], 1)
        self.assertEqual(report["records_with_at_least_two_supervised_turns"], 1)
        self.assertEqual(report["source"]["bytes"], len(fixture.data))
        self.assertEqual(report["source"]["sha256"], hashlib.sha256(fixture.data).hexdigest())

    def test_invalid_training_records(self):
        invalid = [
            {"messages": [], "sup": [False]},
            {"messages": [{"role": "assistant"}], "sup": [1]},
            {"messages": [{"role": "user"}], "sup": [True]},
            {"messages": [{"role": "unknown"}], "sup": [False]},
            {"messages": [{"role": "assistant", "tool_calls": None}], "sup": [True]},
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                audit_train(MemoryJSONL([row]))

    def test_rollout_counts(self):
        fixture = MemoryJSONL([
            {"n_call_llm_this_turn": 1}, {"n_call_llm_this_turn": 17},
        ])
        report = audit_rollouts(fixture)
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["recorded_model_calls"]["total"], 18)
        for count in (True, -1, "2", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                audit_rollouts(MemoryJSONL([{"n_call_llm_this_turn": count}]))


class PublishedInteractionTests(unittest.TestCase):
    def test_published_statistics_match_prior_audit(self):
        base = Path(__file__).resolve().parents[1] / "docs/source-materials/artifacts/pro-v7"
        train = json.loads((base / "interaction-stats.json").read_text())["train"]
        old = json.loads((base / "training-run/train-structure.json").read_text())
        self.assertEqual(train["records"], old["records"])
        self.assertEqual(train["source"]["sha256"], old["sha256"])
        self.assertEqual(train["source"]["bytes"], old["bytes"])
        self.assertEqual(train["role_counts"], old["role_counts"])
        self.assertEqual(train["unsupervised_assistant_messages"], old["assistant_messages_not_directly_supervised"])
        for name, old_name in (
            ("messages", "messages_per_record"),
            ("supervised_assistant_turns", "supervised_messages_per_record"),
        ):
            for key, value in old[old_name].items():
                self.assertEqual(train["statistics"][name][key], value)
        for name, bins in train["histograms"].items():
            self.assertEqual(sum(item["records"] for item in bins), train["records"])
            self.assertEqual(
                sum(item["value"] * item["records"] for item in bins),
                train["statistics"][name]["total"],
            )

    def test_outline_internal_consistency(self):
        base = Path(__file__).resolve().parents[1] / "docs/source-materials/artifacts/pro-v7"
        outline = json.loads((base / "trajectory-example-outline.json").read_text())
        messages, counts = outline["messages"], outline["counts"]
        self.assertTrue(outline["not_a_training_record"])
        self.assertEqual([item["index"] for item in messages], list(range(counts["messages"])))
        self.assertEqual([item["index"] for item in messages if item["sup"]], outline["supervised_message_indices"])
        self.assertEqual(sum(item["sup"] for item in messages), counts["supervised_assistant_turns"])
        self.assertTrue(all(item["role"] == "assistant" for item in messages if item["sup"]))
        self.assertEqual(sum(len(item.get("tool_names", [])) for item in messages), counts["tool_calls"])
        self.assertEqual(sum(bool(item.get("tool_names")) for item in messages), counts["tool_call_turns"])
        self.assertEqual(sum(item["role"] == "tool" for item in messages), counts["tool_result_messages"])
        self.assertEqual(sum(item["role"] == "user" for item in messages), counts["user_role_messages"])


if __name__ == "__main__":
    unittest.main()
