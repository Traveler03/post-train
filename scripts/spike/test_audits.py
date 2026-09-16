"""Small offline tests for the published audit methodology."""
import unittest
from audit_data import feature, summarize, quantiles
from audit_eval import paired


class DataAuditTests(unittest.TestCase):
    def test_only_actual_calls_count(self):
        row = {"messages": [
            {"role": "user", "content": "google_docs_read is mentioned, not called"},
            {"role": "tool_call", "content": '{"name":"proxy_tool","arguments":{"tool_name":"google_drive_list"}}'},
            {"role": "tool_response", "content": "google_docs_write schema text"},
            {"role": "assistant", "content": "done"}],
            "tools": [{"name": "google_slides_create"}]}
        result = feature(row, 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(dict(result["families"]), {"Drive": 1})

    def test_same_user_prefix_is_not_identity(self):
        first = {"messages": [{"role": "user", "content": "shared skills"}, {"role": "assistant", "content": "one"}], "tools": []}
        second = {"messages": [{"role": "user", "content": "shared skills"}, {"role": "assistant", "content": "two"}], "tools": []}
        a, b = feature(first, 1), feature(second, 2)
        self.assertEqual(a["first_user_sha256"], b["first_user_sha256"])
        self.assertNotEqual(a["assistant_md5"], b["assistant_md5"])
        self.assertNotEqual(a["canonical_sha256"], b["canonical_sha256"])

    def test_canonical_hash_ignores_key_order(self):
        a = feature({"messages": [], "tools": []}, 1)
        b = feature({"tools": [], "messages": []}, 2)
        self.assertEqual(a["canonical_sha256"], b["canonical_sha256"])

    def test_quantiles(self):
        self.assertEqual(quantiles([1, 2, 3, 4]), {"p50": 2.5, "p90": 3.7, "max": 4})

    def test_zero_call_record_is_preserved(self):
        summary = summarize([feature({"messages": [], "tools": []}, 1)])
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["drive_docs_cooccurrence"], {"Neither": 1})


class PairedTests(unittest.TestCase):
    def test_common_ids_not_independent_denominators(self):
        a = {"a": "PASS", "b": "FAIL", "c": "PASS", "only-a": "FAIL"}
        b = {"a": "PASS", "b": "PASS", "c": "FAIL", "only-b": "PASS"}
        result = paired(a, b)
        self.assertEqual(result["common_items"], 3)
        self.assertEqual((result["gains"], result["losses"]), (1, 1))
        self.assertEqual(result["delta_pp"], 0)
        self.assertEqual(result["mcnemar_exact_p"], 1)

    def test_direction_is_filtered_minus_original(self):
        result = paired({"a": "FAIL"}, {"a": "PASS"})
        self.assertEqual(result["delta_pp"], 100)

    def test_no_overlap_fails(self):
        with self.assertRaises(ValueError):
            paired({"a": "PASS"}, {"b": "PASS"})


if __name__ == "__main__":
    unittest.main()
