"""Consistency checks on published aggregates; no private inputs required."""
import csv
import json
import statistics
import unittest
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[2] / "docs/pro/spike-study/artifacts"


def read_json(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (ARTIFACTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class PublishedArtifactTests(unittest.TestCase):
    def test_data_partition_and_counts(self):
        data = read_json("data_summary.json")
        self.assertEqual(data["subset_check"], {
            "strict_row_subset": True, "added_rows": 0, "removed_rows": 92})
        for key in ("records", "tool_calls_total"):
            self.assertEqual(data["before"][key], data["after"][key] + data["removed"][key])
        for arm in ("before", "after", "removed"):
            row = data[arm]
            self.assertEqual(sum(row["drive_docs_cooccurrence"].values()), row["records"])
            self.assertEqual(sum(row["tool_call_count_bins"].values()), row["records"])
            self.assertEqual(sum(v["calls"] for v in row["families"].values()), row["tool_calls_total"])
        for family in data["before"]["families"]:
            for key in ("records", "calls"):
                self.assertEqual(data["before"]["families"][family][key],
                                 data["after"]["families"][family][key] + data["removed"]["families"][family][key])
        for arm in ("before", "after"):
            pipeline = data["pipeline"][arm]
            self.assertEqual(pipeline["input"], data[arm]["records"])
            self.assertEqual(pipeline["input"] - pipeline["kept"], 15)
            self.assertEqual(sum(v["样本数"] for v in pipeline["splits"].values()), pipeline["kept"])

    def test_removed_list_is_hashes_and_numbers_only(self):
        rows = read_csv("removed_fingerprints.csv")
        self.assertEqual(len(rows), 92)
        self.assertEqual(len({r["canonical_sha256"] for r in rows}), 92)
        for row in rows:
            self.assertEqual(set(row), {"source_row", "canonical_sha256", "assistant_md5", "tool_calls"})
            self.assertRegex(row["canonical_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["assistant_md5"], r"^[0-9a-f]{32}$")
            self.assertTrue(1 <= int(row["source_row"]) <= 1149)
            self.assertGreaterEqual(int(row["tool_calls"]), 0)

    def test_training_csv_matches_summary(self):
        summary = read_json("training_summary.json")
        for arm in ("before", "after"):
            values = summary[arm]
            rows = read_csv(f"{arm}_training_metrics.csv")
            self.assertEqual([int(r["step"]) for r in rows], list(range(1, values["steps"] + 1)))
            norm = [float(r["grad_norm"]) for r in rows]
            loss = [float(r["loss"]) for r in rows]
            self.assertAlmostEqual(statistics.mean(loss), values["loss_mean"])
            self.assertEqual(max(norm), values["grad_norm_max"])
            self.assertEqual(sum(v > 10 for v in norm), values["grad_norm_gt_10_steps"])
            self.assertEqual(max(int(r["nan"]) for r in rows), 0)
            self.assertEqual(max(int(r["skipped"]) for r in rows), 0)
            for row in rows:
                self.assertAlmostEqual(float(row["epoch"]), int(row["step"]) * 8 / values["train_samples"])

    def test_early_pair_aggregation(self):
        data = read_json("early_evaluation.json")
        self.assertEqual([(r["pc7_step"], r["pc8_step"]) for r in data["rows"]],
                         [(27, 25), (54, 50), (81, 75), (108, 100)])
        for row in data["rows"]:
            result = row["combined"]
            for key in ("common_items", "original_pass", "filtered_pass", "gains", "losses"):
                self.assertEqual(result[key], sum(v[key] for v in row["volumes"].values()))
            self.assertEqual(result["filtered_pass"] - result["original_pass"], result["gains"] - result["losses"])
            self.assertAlmostEqual(result["delta_pp"], 100 * (result["gains"] - result["losses"]) / result["common_items"])
            self.assertGreater(result["holm_p_four_early_pairs"], .05)
            self.assertLess(row["pc8_epoch"], .8)
        self.assertAlmostEqual(data["mean_delta_pp"], statistics.mean(r["combined"]["delta_pp"] for r in data["rows"]))
        self.assertAlmostEqual(data["mean_delta_pp"], 4.34201535, places=7)

    def test_meta_log_binding(self):
        binding = read_json("binding_summary.json")
        data = read_json("data_summary.json")
        training = read_json("training_summary.json")
        for arm in ("before", "after"):
            item = binding[arm]
            self.assertEqual(item["source_sha256_in_meta"], data["sources"][arm]["sha256"])
            self.assertEqual(item["launch_data_basename"], item["logged_data_basename"])
            self.assertEqual(item["logged_train_samples"], training[arm]["train_samples"])
            self.assertEqual(item["logged_output_basename"], training[arm]["source_directory"])


if __name__ == "__main__":
    unittest.main()
