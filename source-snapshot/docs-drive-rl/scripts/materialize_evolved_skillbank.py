#!/usr/bin/env python3
"""Materialize evolved family proposals into runtime case instances."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo.offline_skills.authorization import infer_instance_authorization_policy  # noqa: E402
from algo.offline_skills.evolution import (  # noqa: E402
    candidate_states_fingerprint,
    case_component_subtypes,
    compose_active_candidate_states,
)

JsonObject = dict[str, Any]


def _read_json(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[JsonObject]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            values.append(value)
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    path.write_text(text, encoding="utf-8")


def _load_proposals(evolution_dir: Path) -> dict[str, JsonObject]:
    proposals = {}
    for path in sorted((evolution_dir / "proposals").glob("*.json")):
        wrapper = _read_json(path)
        proposal = wrapper.get("proposal", wrapper)
        if not isinstance(proposal, dict) or not proposal.get("family"):
            raise ValueError(f"{path} has no family proposal")
        proposals[str(proposal["family"])] = proposal
    if not proposals:
        raise FileNotFoundError(f"no proposals found under {evolution_dir / 'proposals'}")
    return proposals


def _confirmation_roots(nodes: list[JsonObject]) -> list[str]:
    """Find the first node that requires later confirmation on each active branch."""

    roots = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        semantic_id = str(node.get("template_state_id") or node_id).lower()
        objective = str(node.get("objective") or "").lower()
        if "confirm" in semantic_id and any(
            marker in semantic_id for marker in ("receive", "later", "explicit")
        ):
            roots.append(node_id)
        elif "confirmation" in objective and any(
            marker in objective for marker in ("receive", "later", "subsequent")
        ):
            roots.append(node_id)
    if roots:
        return sorted(set(roots))
    return sorted(
        str(node["id"])
        for node in nodes
        if str(node.get("kind") or "") == "action"
    )


def _materialize_instance(
    instance: JsonObject,
    proposals: dict[str, JsonObject],
) -> tuple[JsonObject, bool]:
    component_subtypes = case_component_subtypes(instance)
    if not component_subtypes or not set(component_subtypes).issubset(proposals):
        return deepcopy(instance), False

    nodes = compose_active_candidate_states(proposals, component_subtypes)
    for node in nodes:
        node["required"] = True
        node["binding"] = deepcopy(instance.get("bindings") or {})

    source_policy = infer_instance_authorization_policy(instance)
    materialized = deepcopy(instance)
    materialized["state_graph"] = nodes
    materialized["evolved_component_subtypes"] = {
        family: list(subtypes) for family, subtypes in component_subtypes.items()
    }
    materialized["evolved_state_graph"] = True

    policy = source_policy
    if isinstance(policy, dict):
        roots = _confirmation_roots(nodes)
        if not roots:
            raise ValueError(
                f"mutation instance {instance.get('iid')!r} has no confirmation or action boundary"
            )
        policy = deepcopy(policy)
        policy["execution_state_ids"] = roots
        materialized["authorization_policy"] = policy
    return materialized, True


def materialize(source_dir: Path, evolution_dir: Path, output_dir: Path) -> JsonObject:
    source_dir = source_dir.expanduser().resolve()
    evolution_dir = evolution_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == source_dir:
        raise ValueError("output directory must differ from source SkillBank")

    proposals = _load_proposals(evolution_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)

    materialized_count = 0
    fallback_count = 0
    for path in sorted((source_dir / "case_instances").glob("*.jsonl")):
        output_values = []
        for instance in _read_jsonl(path):
            value, evolved = _materialize_instance(instance, proposals)
            output_values.append(value)
            materialized_count += int(evolved)
            fallback_count += int(not evolved)
        _write_jsonl(output_dir / "case_instances" / path.name, output_values)

    proposal_fingerprint = candidate_states_fingerprint(
        [{"family": family, "proposal": proposals[family]} for family in sorted(proposals)]
    )
    manifest = _read_json(output_dir / "manifest.json")
    manifest["evolution_runtime"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_skillbank": str(source_dir),
        "source_evolution": str(evolution_dir),
        "proposal_fingerprint": proposal_fingerprint,
        "proposal_families": sorted(proposals),
        "materialized_cases": materialized_count,
        "fallback_cases": fallback_count,
        "sealed": False,
        "purpose": "experimental_training",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest["evolution_runtime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-skillbank", type=Path, required=True)
    parser.add_argument("--evolution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.source_skillbank, args.evolution_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
