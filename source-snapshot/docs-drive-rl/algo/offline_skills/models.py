"""Typed records shared by the offline skill generation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def skill_name_for_family(family: str) -> str:
    """Convert a taxonomy id into a valid, stable Skill package name."""

    return family.replace(".", "-").replace("_", "-")


@dataclass(frozen=True)
class ToolEvent:
    index: int
    name: str
    args: JsonObject
    response: Any
    source: str
    successful: bool

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass
class RolloutRecord:
    domain: str
    case_id: str
    iid: str
    request_id: str
    trace_id: str
    query: str
    answer: str
    metadata: JsonObject
    seed_summary: JsonObject
    reward: JsonObject
    events: list[ToolEvent]
    rollout_path: Path
    rollout_line: int

    @property
    def benchmark_pass(self) -> bool:
        return float(self.reward.get("benchmark_overall_pass") or 0.0) == 1.0

    @property
    def score(self) -> float:
        return float(self.reward.get("score") or 0.0)


@dataclass
class CaseRecord:
    domain: str
    case_id: str
    iid: str
    query: str
    metadata: JsonObject
    seed_summary: JsonObject
    dataset_position: int
    rollouts: list[RolloutRecord] = field(default_factory=list)


@dataclass(frozen=True)
class CaseClassification:
    family: str
    domain: str
    operation: str
    outcome: str
    cardinality: str
    complexity: str
    expected_operations: tuple[str, ...]
    signals: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        value = asdict(self)
        value["expected_operations"] = list(self.expected_operations)
        value["signals"] = list(self.signals)
        return value


@dataclass
class Corpus:
    domain: str
    trace_dir: Path
    dataset_path: Path
    cases: list[CaseRecord]
    matched_rollouts: int
    unmatched_rollouts: int
    reward_rows: int
    duplicate_reward_ids: list[str]


@dataclass(frozen=True)
class StateCheck:
    kind: str
    tool_patterns: tuple[str, ...] = ()
    min_count: int = 1
    description: str = ""

    def to_dict(self) -> JsonObject:
        return {
            "kind": self.kind,
            "tool_patterns": list(self.tool_patterns),
            "min_count": self.min_count,
            "description": self.description,
        }


@dataclass(frozen=True)
class StateBlueprint:
    id: str
    required: str
    applies_when: str
    depends_on: tuple[str, ...]
    expansion: str
    checks: tuple[StateCheck, ...]
    required_operations: tuple[str, ...] = ()

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "required": self.required,
            "applies_when": self.applies_when,
            "depends_on": list(self.depends_on),
            "expansion": self.expansion,
            "required_operations": list(self.required_operations),
            "deterministic_checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class FamilyBlueprint:
    family: str
    purpose: str
    states: tuple[StateBlueprint, ...]

    def to_dict(self) -> JsonObject:
        return {
            "family": self.family,
            "purpose": self.purpose,
            "states": [state.to_dict() for state in self.states],
        }
