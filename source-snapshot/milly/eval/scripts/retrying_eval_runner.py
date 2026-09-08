#!/usr/bin/env python3
"""Reusable QA-platform task lifecycle with execution-failure retries.

Model provisioning, payload contents, dataset selection, and repetition count
belong to the caller.  This module owns the invariant lifecycle shared by all
of them: submit, wait, allow judge persistence, inspect traces, retry only
execution failures, and persist a resumable terminal record.
"""

from __future__ import annotations

import heapq
import json
import threading
import time
import urllib.parse
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Iterator, Mapping, Sequence


JsonDict = dict[str, object]
LogFn = Callable[[str], None]
PayloadFactory = Callable[[int, list[str] | None], tuple[str, JsonDict]]
ExecutionSlot = Callable[[int], ContextManager[None]]


class PriorityExecutionGate:
    """Serialize QA tasks while allowing older-checkpoint retries to jump ahead."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._sequence = 0
        self._waiters: list[tuple[tuple[int, ...], int]] = []

    @contextmanager
    def slot(self, priority: tuple[int, ...]) -> Iterator[None]:
        with self._condition:
            token = (priority, self._sequence)
            self._sequence += 1
            heapq.heappush(self._waiters, token)
            while self._active or self._waiters[0] != token:
                self._condition.wait()
            heapq.heappop(self._waiters)
            self._active = True
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


@dataclass(frozen=True)
class RetrySettings:
    case_retries: int = 2
    judge_wait: int = 480
    poll_interval: int = 30
    result_poll_interval: int = 30
    task_timeout: int = 12 * 60 * 60
    trace_workers: int = 8

    def __post_init__(self) -> None:
        if self.case_retries < 0:
            raise ValueError("case_retries must be >= 0")
        waits = (
            self.judge_wait,
            self.poll_interval,
            self.result_poll_interval,
            self.task_timeout,
        )
        if min(waits) < 0:
            raise ValueError("wait and timeout values must be >= 0")
        if self.judge_wait and not self.result_poll_interval:
            raise ValueError(
                "result_poll_interval must be > 0 when judge_wait is enabled"
            )
        if self.trace_workers < 1:
            raise ValueError("trace_workers must be >= 1")


class RetryingEvalRunner:
    """Run one dataset job to a resumable terminal state.

    The injected functions keep credentials, transport, state layout, naming,
    and logging outside this reusable core.
    """

    def __init__(
        self,
        *,
        submit_request: Callable[[JsonDict], object],
        list_tasks: Callable[[], object],
        trace_get: Callable[[str], JsonDict],
        log: LogFn,
        scan_dir: Path,
        settings: RetrySettings,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.submit_request = submit_request
        self.list_tasks = list_tasks
        self.trace_get = trace_get
        self.log = log
        self.scan_dir = scan_dir
        self.settings = settings
        self.sleep = sleep
        self.clock = clock

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(
            char if char.isalnum() or char in "._-" else "_" for char in value
        )[:140]

    @staticmethod
    def _score_string(score: Mapping[str, object]) -> str | None:
        value = score.get("stringValue")
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    def submit(self, payload: JsonDict, requested_name: str, item_count: object) -> str:
        response = self.submit_request(payload)
        if not isinstance(response, dict) or not response.get("task_id"):
            raise RuntimeError(f"提交未返回 task_id: {response}")
        task_id = str(response["task_id"])
        self.log(f"已提交 {requested_name} task={task_id} ids={item_count}")
        return task_id

    def wait_task(
        self,
        task_id: str,
        requested_name: str,
        on_update: Callable[[str, str], None] | None = None,
    ) -> str:
        run_name = requested_name
        deadline = self.clock() + self.settings.task_timeout
        while self.clock() < deadline:
            try:
                tasks = self.list_tasks()
                if not isinstance(tasks, list):
                    raise TypeError(f"任务列表不是 list: {type(tasks).__name__}")
            except Exception as exc:
                self.log(
                    f"任务查询失败，{self.settings.poll_interval} 秒后重试: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.sleep(self.settings.poll_interval)
                continue
            active = next(
                (item for item in tasks if item.get("task_id") == task_id), None
            )
            if active is None:
                self.log(f"任务结束 task={task_id} run={run_name}")
                return run_name
            run_name = str(active.get("task_name") or run_name)
            status = str(active.get("status") or "")
            if on_update is not None:
                on_update(run_name, status)
            if status.lower() in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(
                    f"任务终态失败 task={task_id} status={status} run={run_name}"
                )
            self.log(
                f"任务 {run_name}: {status} {active.get('completed_cases', 0)}/"
                f"{active.get('total_cases', 0)}"
            )
            self.sleep(self.settings.poll_interval)
        raise TimeoutError(f"任务超过 {self.settings.task_timeout}s: {requested_name}")

    def failed_ids(
        self, dataset: str, run_name: str, expected_ids: Sequence[str], label: str
    ) -> list[str]:
        path = f"/api/public/datasets/{urllib.parse.quote(dataset, safe='')}/runs/"
        run = self.trace_get(path + urllib.parse.quote(run_name, safe=""))
        run_items = run.get("datasetRunItems") or []
        expected = set(expected_ids)
        seen: set[str] = set()
        failures: dict[str, str] = {}

        def inspect(item: Mapping[str, object]) -> tuple[str, str | None]:
            item_id = str(item.get("datasetItemId") or item.get("itemId") or "")
            if not item_id:
                return "", "missing dataset item id"
            trace_id = str(item.get("traceId") or "")
            if not trace_id:
                return item_id, "missing trace"
            try:
                trace = self.trace_get(
                    f"/api/public/traces/{urllib.parse.quote(trace_id, safe='')}"
                )
            except Exception as exc:
                return item_id, f"trace fetch: {type(exc).__name__}"
            output = trace.get("output")
            if (
                not isinstance(output, dict)
                or not str(output.get("actual_outcome") or "").strip()
            ):
                return item_id, "empty actual_outcome"
            for score in trace.get("scores") or []:
                if score.get("name") == "回复出错":
                    value = self._score_string(score)
                    if value is not None and value.lower() != "success":
                        return item_id, f"回复出错={value}"
            return item_id, None

        with ThreadPoolExecutor(max_workers=self.settings.trace_workers) as pool:
            for item_id, reason in pool.map(inspect, run_items):
                if not item_id:
                    continue
                seen.add(item_id)
                if reason:
                    failures[item_id] = reason
        for item_id in expected - seen:
            failures[item_id] = "missing from dataset run"

        self.scan_dir.mkdir(parents=True, exist_ok=True)
        scan_path = self.scan_dir / f"{label}__{self._safe(run_name)}.scan.json"
        scan_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "run_name": run_name,
                    "expected": len(expected),
                    "seen": len(seen),
                    "failed": failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if failures:
            sample = dict(list(failures.items())[:3])
            self.log(
                f"{run_name}: 执行失败 {len(failures)}/{len(expected)}，原因={sample}"
            )
        else:
            self.log(f"{run_name}: {len(expected)} 条均有有效执行结果")
        return sorted(failures)

    def result_readiness(
        self, dataset: str, run_name: str, expected_ids: Sequence[str]
    ) -> tuple[bool, list[str]]:
        """Return whether expected traces and execution scores are persisted."""

        path = f"/api/public/datasets/{urllib.parse.quote(dataset, safe='')}/runs/"
        try:
            run = self.trace_get(path + urllib.parse.quote(run_name, safe=""))
        except Exception:
            return False, list(expected_ids)
        items = {
            str(item.get("datasetItemId") or item.get("itemId") or ""): item
            for item in run.get("datasetRunItems") or []
        }

        def ready(item_id: str) -> tuple[str, bool]:
            item = items.get(item_id)
            if not item or not item.get("traceId"):
                return item_id, False
            try:
                trace = self.trace_get(
                    f"/api/public/traces/"
                    f"{urllib.parse.quote(str(item['traceId']), safe='')}"
                )
            except Exception:
                return item_id, False
            output = trace.get("output")
            if (
                not isinstance(output, dict)
                or not str(output.get("actual_outcome") or "").strip()
            ):
                return item_id, False
            execution_score_ready = any(
                score.get("name") == "回复出错"
                and self._score_string(score) is not None
                for score in trace.get("scores") or []
            )
            return item_id, execution_score_ready

        pending: list[str] = []
        with ThreadPoolExecutor(max_workers=self.settings.trace_workers) as pool:
            for item_id, is_ready in pool.map(ready, expected_ids):
                if not is_ready:
                    pending.append(item_id)
        return not pending, sorted(pending)

    def wait_for_results(
        self,
        dataset: str,
        run_name: str,
        expected_ids: Sequence[str],
        label: str,
        *,
        ready_deadline: float | None = None,
    ) -> list[str]:
        """Poll persistence without occupying a model-execution slot."""

        if not self.settings.judge_wait:
            return self.failed_ids(dataset, run_name, expected_ids, label)
        deadline = (
            ready_deadline
            if ready_deadline is not None
            else self.clock() + self.settings.judge_wait
        )
        self.log(
            f"后台等待判定落档，最长 {max(0, int(deadline - self.clock()))}s: "
            f"{run_name}"
        )
        last_pending = -1
        while self.clock() < deadline:
            is_ready, pending = self.result_readiness(dataset, run_name, expected_ids)
            if is_ready:
                self.log(f"判定已落档: {run_name}")
                return self.failed_ids(dataset, run_name, expected_ids, label)
            if len(pending) != last_pending:
                self.log(f"判定落档中 pending={len(pending)}: {run_name}")
                last_pending = len(pending)
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            self.sleep(min(self.settings.result_poll_interval, remaining))
        self.log(f"判定落档达到等待上限，按当前结果扫描: {run_name}")
        return self.failed_ids(dataset, run_name, expected_ids, label)

    def run_job(
        self,
        *,
        job_key: str,
        label: str,
        dataset: str,
        expected_ids: Sequence[str],
        payload_factory: PayloadFactory,
        is_completed: Callable[[str], bool],
        mark_completed: Callable[[str, JsonDict], None],
        load_progress: Callable[[str], JsonDict | None] | None = None,
        mark_progress: Callable[[str, JsonDict], None] | None = None,
        result_metadata: Mapping[str, object] | None = None,
        finished_at: Callable[[], str] | None = None,
        execution_slot: ExecutionSlot | None = None,
        on_attempt_executed: Callable[[int], None] | None = None,
    ) -> None:
        if is_completed(job_key):
            self.log(f"跳过已完成任务 {job_key}")
            return

        saved = load_progress(job_key) if load_progress is not None else None
        saved_status = str(saved.get("status") or "") if saved else ""
        resume = saved_status == "running"
        resume_finalizing = saved_status == "finalizing"
        pending = (
            list(saved.get("pending_ids") or expected_ids)
            if saved
            else list(expected_ids)
        )
        runs = list(saved.get("runs") or []) if saved else []
        first_attempt = int(saved.get("attempt", 0)) if saved else 0
        last_failed: list[str] = []

        for attempt in range(first_attempt, self.settings.case_retries + 1):
            retry_ids = pending
            ready_deadline: float | None = None
            if resume_finalizing and attempt == first_attempt:
                actual_name = str(
                    saved.get("actual_name") or saved.get("requested_name") or ""
                )
                if not actual_name:
                    raise RuntimeError(f"落档恢复记录缺少 run name: {job_key}")
                ready_deadline = float(
                    saved.get("ready_deadline_epoch")
                    or self.clock() + self.settings.judge_wait
                )
                self.log(f"恢复落档检查 {job_key} attempt={attempt} run={actual_name}")
            else:
                slot = execution_slot(attempt) if execution_slot else nullcontext()
                with slot:
                    if resume and attempt == first_attempt:
                        task_id = str(saved["task_id"])
                        requested_name = str(
                            saved.get("actual_name") or saved["requested_name"]
                        )
                        self.log(
                            f"恢复在途任务 {job_key} task={task_id} attempt={attempt}"
                        )
                    else:
                        requested_name, payload = payload_factory(attempt, retry_ids)
                        task_id = self.submit(payload, requested_name, len(retry_ids))
                        if mark_progress is not None:
                            mark_progress(
                                job_key,
                                {
                                    "status": "running",
                                    "attempt": attempt,
                                    "task_id": task_id,
                                    "requested_name": requested_name,
                                    "actual_name": requested_name,
                                    "pending_ids": retry_ids,
                                    "runs": runs,
                                    **dict(result_metadata or {}),
                                    "label": label,
                                    "dataset": dataset,
                                },
                            )

                    def update(actual_name: str, task_status: str) -> None:
                        if mark_progress is not None:
                            mark_progress(
                                job_key,
                                {
                                    "status": "running",
                                    "attempt": attempt,
                                    "task_id": task_id,
                                    "requested_name": requested_name,
                                    "actual_name": actual_name,
                                    "task_status": task_status,
                                    "pending_ids": retry_ids,
                                    "runs": runs,
                                    **dict(result_metadata or {}),
                                    "label": label,
                                    "dataset": dataset,
                                },
                            )

                    actual_name = self.wait_task(task_id, requested_name, update)
                resume = False
                if actual_name not in runs:
                    runs.append(actual_name)
                ready_deadline = self.clock() + self.settings.judge_wait
                if mark_progress is not None:
                    mark_progress(
                        job_key,
                        {
                            "status": "finalizing",
                            "attempt": attempt,
                            "requested_name": requested_name,
                            "actual_name": actual_name,
                            "pending_ids": retry_ids,
                            "runs": runs,
                            "ready_deadline_epoch": ready_deadline,
                            **dict(result_metadata or {}),
                            "label": label,
                            "dataset": dataset,
                        },
                    )
                if on_attempt_executed is not None:
                    on_attempt_executed(attempt)

            resume_finalizing = False
            last_failed = self.wait_for_results(
                dataset,
                actual_name,
                retry_ids,
                label,
                ready_deadline=ready_deadline,
            )
            if not last_failed:
                break
            if attempt < self.settings.case_retries:
                pending = last_failed
                self.log(f"{job_key}: 第 {attempt + 1} 次补跑 {len(pending)} 条")
                if mark_progress is not None:
                    mark_progress(
                        job_key,
                        {
                            "status": "queued_retry",
                            "attempt": attempt + 1,
                            "pending_ids": pending,
                            "runs": runs,
                            **dict(result_metadata or {}),
                            "label": label,
                            "dataset": dataset,
                        },
                    )

        record: JsonDict = {
            "status": "completed",
            **dict(result_metadata or {}),
            "label": label,
            "dataset": dataset,
            "runs": runs,
            "remaining_execution_failures": last_failed,
        }
        if finished_at is not None:
            record["finished_at"] = finished_at()
        mark_completed(job_key, record)
