#!/usr/bin/env python3
"""Execute a manifest-defined local-LoRA Pro evaluation matrix."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval/scripts"))
import pull_pro_bench as pull  # noqa: E402
import submit_pro_bench as submitter  # noqa: E402
from retrying_eval_runner import RetrySettings, RetryingEvalRunner  # noqa: E402


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class MatrixRunner:
    def __init__(self, manifest_path: Path, dry_run: bool = False) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.dry_run = dry_run
        self.run_root = self.manifest_path.parent
        self.state_path = self.run_root / "state.json"
        self.log_path = self.run_root / "controller.log"
        self.state_lock = threading.Lock()
        self.worker: subprocess.Popen[bytes] | None = None
        self.worker_log = None
        self._validate_manifest()

        self.run_root.mkdir(parents=True, exist_ok=True)
        self.lock_handle = (self.run_root / "state.lock").open("w")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有评测控制器运行: {self.run_root}") from exc
        self.state = self._read_state()
        retry = self.manifest["retry"]
        self.task_runner = RetryingEvalRunner(
            submit_request=lambda body: self.qa_request("POST", "/api/webhook", body),
            list_tasks=lambda: self.qa_request("GET", "/api/tasks"),
            trace_get=pull.get,
            log=self.log,
            scan_dir=self.run_root / "scans",
            settings=RetrySettings(
                case_retries=int(retry["case_retries"]),
                judge_wait=int(retry["judge_wait_seconds"]),
                poll_interval=int(retry.get("poll_interval_seconds", 30)),
                task_timeout=int(retry.get("task_timeout_seconds", 43200)),
                trace_workers=int(retry.get("trace_workers", 8)),
            ),
        )

    def _validate_manifest(self) -> None:
        if self.manifest.get("schema_version") != 1:
            raise ValueError("manifest.schema_version must be 1")
        if self.manifest.get("live_smoke") is not False:
            raise ValueError("manifest must explicitly set live_smoke=false")
        if not self.manifest.get("checkpoints") or not self.manifest.get("collections"):
            raise ValueError("manifest checkpoints/collections cannot be empty")
        if int(self.manifest.get("rounds", 0)) < 1:
            raise ValueError("manifest.rounds must be >= 1")
        for collection in self.manifest["collections"]:
            if collection.get("family") != "pro":
                raise ValueError("checkpoint matrix currently supports Pro collections")
            item_file = Path(collection["item_file"])
            if not item_file.is_absolute():
                item_file = REPO / item_file
            ids = self.read_ids(item_file)
            if len(ids) != int(collection["case_count"]):
                raise ValueError(
                    f"{collection['dataset_name']} count {len(ids)} != "
                    f"{collection['case_count']}"
                )
        for checkpoint in self.manifest["checkpoints"]:
            adapter = Path(checkpoint["adapter_path"])
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"missing adapter_config.json: {adapter}")
            model_file = adapter / "adapter_model.safetensors"
            if not model_file.is_file() or model_file.stat().st_size == 0:
                raise FileNotFoundError(f"missing/empty adapter weights: {model_file}")

    @staticmethod
    def read_ids(path: Path) -> list[str]:
        ids = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("id"):
                    raise ValueError(f"item without id: {path}")
                ids.append(str(row["id"]))
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate item ids: {path}")
        return ids

    def _read_state(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"run_id": self.manifest["run_id"], "started_at": timestamp(), "jobs": {}}

    def save_state(self) -> None:
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    def log(self, message: str) -> None:
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def qa_request(self, method: str, path: str, payload: dict | None = None) -> object:
        token = os.environ.get("MIGOO_EVAL_WEBHOOK_TOKEN") or submitter.TOKEN
        url = f"{submitter.QA}{path}"
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"token": token})
        request = urllib.request.Request(
            url,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        return json.loads(raw) if raw else {}

    def vllm_request(self, path: str, payload: dict | None = None) -> object:
        serving = self.manifest["serving"]
        request = urllib.request.Request(
            f"http://{serving['host']}:{serving['port']}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"_raw": raw.decode(errors="replace")}

    def load_adapter(self, checkpoint: dict) -> None:
        model = checkpoint["model_name"]
        models = self.vllm_request("/v1/models")
        loaded = {row["id"] for row in models.get("data", [])}
        if model not in loaded:
            self.vllm_request(
                "/v1/load_lora_adapter",
                {"lora_name": model, "lora_path": checkpoint["adapter_path"]},
            )
        models = self.vllm_request("/v1/models")
        if model not in {row["id"] for row in models.get("data", [])}:
            raise RuntimeError(f"adapter not visible after load: {model}")
        self.log(f"adapter ready model={model} path={checkpoint['adapter_path']}")

    def unload_adapter(self, checkpoint: dict) -> None:
        try:
            self.vllm_request(
                "/v1/unload_lora_adapter", {"lora_name": checkpoint["model_name"]}
            )
            self.log(f"adapter unloaded model={checkpoint['model_name']}")
        except Exception as exc:
            self.log(f"⚠️ adapter unload failed: {type(exc).__name__}: {exc}")

    def start_worker(self, checkpoint: dict) -> None:
        serving = self.manifest["serving"]
        token = (REPO / "relay/.hub_token").read_text(encoding="utf-8").strip()
        worker_dir = self.run_root / checkpoint["label"]
        worker_dir.mkdir(parents=True, exist_ok=True)
        self.worker_log = (worker_dir / "worker.log").open("ab", buffering=0)
        extra_body = self.manifest["model"]["extra_body"]
        command = [
            serving["worker_python"], str(REPO / "relay/worker.py"),
            "--transport", "http", "--relay-url", serving["relay_url"],
            "--token", token, "--target", f"http://{serving['host']}:{serving['port']}/v1",
            "--model", checkpoint["lane"], "--upstream-model", checkpoint["model_name"],
            "--extra-body", json.dumps(extra_body, separators=(",", ":")),
            "--worker-id", checkpoint["worker_id"],
            "--concurrency", str(serving["worker_concurrency"]),
            "--request-timeout", str(serving.get("request_timeout_seconds", 900)),
            "--dump-dir", str(worker_dir / "dumps"),
        ]
        self.worker = subprocess.Popen(
            command, cwd=REPO, stdin=subprocess.DEVNULL,
            stdout=self.worker_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.log(f"worker started pid={self.worker.pid} lane={checkpoint['lane']}")
        deadline = time.time() + 90
        status_url = serving["hub_status_url"]
        while time.time() < deadline:
            if self.worker.poll() is not None:
                raise RuntimeError(f"worker exited rc={self.worker.returncode}")
            try:
                raw = urllib.request.urlopen(status_url, timeout=10).read()
                models = (json.loads(raw)["results"].get("model") or "").split(",")
                count = sum(item == checkpoint["lane"] for item in models)
                if count == 1:
                    self.log(f"hub lane ready and unique: {checkpoint['lane']}")
                    return
                if count > 1:
                    raise RuntimeError(f"duplicate workers on lane: {checkpoint['lane']}")
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError(f"hub did not register lane: {checkpoint['lane']}")

    def stop_worker(self) -> None:
        if self.worker is None:
            return
        process, self.worker = self.worker, None
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self.worker_log:
            self.worker_log.close()
            self.worker_log = None
        self.log(f"worker stopped pid={process.pid} rc={process.returncode}")

    def job_key(self, checkpoint: dict, round_no: int, collection: dict) -> str:
        return f"{checkpoint['label']}/r{round_no}/{collection['id']}"

    def build_submission(
        self, checkpoint: dict, round_no: int, collection: dict,
        attempt: int, ids: list[str] | None,
    ) -> tuple[str, dict]:
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        name = (
            f"{self.manifest['run_id']}_{checkpoint['label']}_{collection['id']}_"
            f"{self.manifest['model']['tier']}_r{round_no}{suffix}"
        )
        payload = {
            "任务名": name,
            "任务描述": f"manifest={self.manifest_path}; checkpoint={checkpoint['label']}",
            "被测模型": {"assistant_professional": "ais-relay"},
            "评估模型": collection.get("judge_model", "gpt-5.5-2026-04-23"),
            "评估prompt": collection["judge_prompt"],
            "评估prompt版本": collection["judge_version"],
            "运行环境": collection["environment"],
            "region": collection["region"],
            "链路模式": collection["chain"],
            "指定用例ID": ids,
            "model_config_overrides": {
                "ais-relay": {
                    "extraHeaders": {"X-Ais-Lane": checkpoint["lane"]},
                    "extraBody": self.manifest["model"]["extra_body"],
                }
            },
            "最大并发数": self.manifest["task_concurrency"],
        }
        return name, {
            "datasetName": collection["dataset_name"],
            "userEmail": self.manifest["user_email"],
            "payload": payload,
        }

    def run_job(self, checkpoint: dict, round_no: int, collection: dict) -> None:
        key = self.job_key(checkpoint, round_no, collection)
        item_file = Path(collection["item_file"])
        if not item_file.is_absolute():
            item_file = REPO / item_file
        ids = self.read_ids(item_file)

        def is_completed(job_key: str) -> bool:
            with self.state_lock:
                return self.state["jobs"].get(job_key, {}).get("status") == "completed"

        def load_progress(job_key: str) -> dict | None:
            with self.state_lock:
                record = self.state["jobs"].get(job_key)
                return dict(record) if record else None

        def mark_progress(job_key: str, record: dict) -> None:
            with self.state_lock:
                self.state["jobs"][job_key] = record
                self.save_state()

        def mark_completed(job_key: str, record: dict) -> None:
            with self.state_lock:
                self.state["jobs"][job_key] = record
                self.save_state()
            for run_name in record["runs"]:
                try:
                    pull.pull_run(collection["dataset_name"], run_name)
                except Exception as exc:
                    self.log(f"⚠️ result pull failed run={run_name}: {exc}")

        self.task_runner.run_job(
            job_key=key, label=collection["id"], dataset=collection["dataset_name"],
            expected_ids=ids,
            payload_factory=lambda attempt, pending: self.build_submission(
                checkpoint, round_no, collection, attempt, pending
            ),
            is_completed=is_completed, mark_completed=mark_completed,
            load_progress=load_progress, mark_progress=mark_progress,
            result_metadata={
                "checkpoint": checkpoint["label"], "round": round_no,
                "collection": collection["id"], "tier": self.manifest["model"]["tier"],
            },
            finished_at=timestamp,
        )

    def print_dry_run(self) -> None:
        for checkpoint in self.manifest["checkpoints"]:
            for round_no in range(1, int(self.manifest["rounds"]) + 1):
                for collection in self.manifest["collections"]:
                    item_file = Path(collection["item_file"])
                    if not item_file.is_absolute():
                        item_file = REPO / item_file
                    ids = self.read_ids(item_file)
                    name, body = self.build_submission(
                        checkpoint, round_no, collection, 0, ids
                    )
                    preview = json.loads(json.dumps(body, ensure_ascii=False))
                    preview["payload"]["指定用例ID"] = f"[{len(ids)} ids]"
                    print(json.dumps({"name": name, "body": preview}, ensure_ascii=False))

    def run(self) -> None:
        if self.dry_run:
            self.print_dry_run()
            return
        for checkpoint in self.manifest["checkpoints"]:
            self.load_adapter(checkpoint)
            self.start_worker(checkpoint)
            try:
                for round_no in range(1, int(self.manifest["rounds"]) + 1):
                    self.log(f"start checkpoint={checkpoint['label']} round={round_no}")
                    with ThreadPoolExecutor(
                        max_workers=int(self.manifest.get("dataset_parallelism", 3))
                    ) as pool:
                        futures = [
                            pool.submit(self.run_job, checkpoint, round_no, collection)
                            for collection in self.manifest["collections"]
                        ]
                        for future in as_completed(futures):
                            future.result()
            finally:
                self.stop_worker()
                self.unload_adapter(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runner = MatrixRunner(args.manifest, dry_run=args.dry_run)
    try:
        runner.run()
    finally:
        runner.stop_worker()
        runner.save_state()


if __name__ == "__main__":
    main()
