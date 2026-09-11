"""End-to-end offline SkillBank generation and rollout replay."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .blueprints import BLUEPRINTS, TEMPLATE_VERSION, VERIFIER_VERSION, get_blueprint
from .classifier import classify_case
from .evidence import build_experience_card, select_prompt_cards, verify_extraction_candidate
from .instances import compact_case_catalog, instantiate_case
from .io import file_sha256, load_corpus
from .models import (
    CaseClassification,
    CaseRecord,
    Corpus,
    FamilyBlueprint,
    JsonObject,
    skill_name_for_family,
)
from .prompts import (
    SYSTEM_INSTRUCTIONS,
    build_family_prompt,
    skill_package_schema,
    validate_generated_package,
)
from .render import build_state_template, render_judge_rubric, render_report, render_skill_md
from .replay import build_validation_report, replay_rollout
from .responses_client import DirectResponsesClient, ResponsesProviderConfig
from .storage import write_json, write_jsonl, write_text


@dataclass(frozen=True)
class GenerationOptions:
    docs_trace_dir: Path
    drive_trace_dir: Path
    docs_dataset: Path
    drive_dataset: Path
    output_dir: Path
    api_config: Path = Path("~/.codex/config.toml")
    model: str | None = None
    reasoning_effort: str | None = None
    api_timeout_s: float = 300.0
    api_max_attempts: int = 3
    api_concurrency: int = 3
    max_output_tokens: int = 6000
    positive_examples_per_family: int = 8
    negative_examples_per_family: int = 4
    use_api: bool = True
    force: bool = False


PackageGenerator = Callable[[FamilyBlueprint, str], tuple[JsonObject, JsonObject]]


def _case_key(case: CaseRecord) -> str:
    return f"{case.domain}:{case.case_id}"


def _git_metadata(root: Path) -> JsonObject:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _fallback_package(blueprint: FamilyBlueprint) -> JsonObject:
    """Deterministic package used by unit tests and explicit --no-api runs."""

    name = skill_name_for_family(blueprint.family)
    annotations = [
        {
            "state_id": state.id,
            "objective": state.applies_when,
            "checklist": [check.description for check in state.checks],
            "accepted_evidence": [check.description for check in state.checks],
            "hard_failures": ["Claims completion when the required state is not evidenced."],
            "semantic_checks": ["Judge the achieved state from tool results and the final response together."],
        }
        for state in blueprint.states
    ]
    return {
        "skill": {
            "name": name,
            "description": f"Handle {blueprint.family} tasks using evidence-backed task states.",
            "purpose": blueprint.purpose,
            "workflow": [
                {
                    "id": state.id,
                    "instruction": state.applies_when,
                    "applies_when": state.applies_when,
                }
                for state in blueprint.states
            ],
            "decision_rules": ["Use retrieved results rather than tool names or unsupported assumptions."],
            "stop_conditions": ["Stop before an unsupported or unauthorized mutation."],
            "failure_recovery": ["Report the blocker and request the minimum missing input."],
        },
        "state_guidance": {
            "summary": blueprint.purpose,
            "state_annotations": annotations,
        },
        "judge_rubric": {
            "overview": "Judge task-state completion from observable evidence.",
            "pass_rule": "All required states pass and no hard failure occurs.",
            "criteria": [
                {
                    "id": "evidence",
                    "name": "Evidence",
                    "question": "Are all required states supported by observable results?",
                    "pass_condition": "Each required state has consistent tool-result or response evidence.",
                    "fail_condition": "A required state is missing, contradicted, or supported only by a claim.",
                    "severity": "hard",
                }
            ],
            "hard_failures": ["Unsupported completion claim", "Unauthorized mutation"],
        },
    }


def _responses_generator(options: GenerationOptions) -> tuple[PackageGenerator, JsonObject]:
    config = ResponsesProviderConfig.from_toml(
        options.api_config,
        model_override=options.model,
        reasoning_effort_override=options.reasoning_effort,
    )

    def generate(blueprint: FamilyBlueprint, prompt: str) -> tuple[JsonObject, JsonObject]:
        correction = ""
        validation_errors: list[str] = []
        for semantic_attempt in range(1, 3):
            client = DirectResponsesClient(
                config,
                timeout_s=options.api_timeout_s,
                max_attempts=options.api_max_attempts,
            )
            response = client.generate_json(
                instructions=SYSTEM_INSTRUCTIONS,
                prompt=prompt + correction,
                schema_name="offline_task_skill_package",
                schema=skill_package_schema(),
                max_output_tokens=options.max_output_tokens,
            )
            try:
                validate_generated_package(response.content, blueprint)
            except ValueError as exc:
                validation_errors.append(str(exc))
                correction = (
                    "\n\n上一次输出未通过本地校验："
                    + str(exc)
                    + "。请重新生成完整 JSON，并严格满足 required_skill_name 和 state_id 顺序。"
                )
                continue
            return response.content, {
                "response_id": response.response_id,
                "model": response.model,
                "status": response.status,
                "usage": response.usage,
                "elapsed_s": response.elapsed_s,
                "transport_attempts": response.attempts,
                "semantic_attempt": semantic_attempt,
                "prior_validation_errors": validation_errors,
            }
        raise RuntimeError(
            f"Responses API output for {blueprint.family} failed semantic validation twice: {validation_errors}"
        )

    return generate, config.public_metadata()


def _prepare_output(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    if manifest.exists() and not force:
        raise FileExistsError(f"{path} already contains a generated manifest; pass --force to overwrite known files")


def _source_manifest(corpora: list[Corpus]) -> JsonObject:
    result: JsonObject = {}
    for corpus in corpora:
        result[corpus.domain] = {
            "trace_dir": str(corpus.trace_dir),
            "dataset_path": str(corpus.dataset_path),
            "dataset_sha256": file_sha256(corpus.dataset_path),
            "rollouts": {
                "path": str(corpus.trace_dir / "rollouts.jsonl"),
                "sha256": file_sha256(corpus.trace_dir / "rollouts.jsonl"),
            },
            "rewards": {
                "path": str(corpus.trace_dir / "rewards.jsonl"),
                "sha256": file_sha256(corpus.trace_dir / "rewards.jsonl"),
            },
            "matched_rollouts": corpus.matched_rollouts,
            "unmatched_rollouts_without_reward": corpus.unmatched_rollouts,
            "reward_rows": corpus.reward_rows,
            "duplicate_reward_ids": corpus.duplicate_reward_ids,
        }
    return result


def _write_family_tests(
    path: Path,
    family_cases: list[CaseRecord],
    classifications: dict[str, CaseClassification],
    instances: dict[str, JsonObject],
) -> None:
    rows: list[JsonObject] = []
    for case in family_cases:
        classification = classifications[_case_key(case)]
        verified: list[str] = []
        negatives: list[str] = []
        for rollout in case.rollouts:
            accepted, _, _ = verify_extraction_candidate(rollout, classification)
            (verified if accepted else negatives).append(rollout.request_id)
        instance = instances[_case_key(case)]
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "expected_family": classification.family,
                "expected_operations": list(classification.expected_operations),
                "verified_success_request_ids": verified,
                "regression_negative_request_ids": negatives,
                "required_state_ids": [state["id"] for state in instance["state_graph"] if state["required"]],
            }
        )
    write_jsonl(path, rows)


def run_generation(
    options: GenerationOptions,
    *,
    package_generator: PackageGenerator | None = None,
) -> JsonObject:
    root = Path(__file__).resolve().parents[2]
    output_dir = options.output_dir.expanduser().resolve()
    _prepare_output(output_dir, options.force)

    corpora = [
        load_corpus(options.docs_trace_dir, options.docs_dataset, "docs"),
        load_corpus(options.drive_trace_dir, options.drive_dataset, "drive"),
    ]
    cases = [case for corpus in corpora for case in corpus.cases]
    classifications = {_case_key(case): classify_case(case) for case in cases}
    cards: list[JsonObject] = []
    card_source: dict[str, JsonObject] = {}
    for case in cases:
        classification = classifications[_case_key(case)]
        for rollout in case.rollouts:
            card = build_experience_card(rollout, classification)
            cards.append(card)
            card_source[str(card["card_id"])] = {
                "card_id": card["card_id"],
                "domain": case.domain,
                "case_id": case.case_id,
                "request_id": rollout.request_id,
                "trace_id": rollout.trace_id,
                "rollout_path": str(rollout.rollout_path),
                "rollout_line": rollout.rollout_line,
            }

    cases_by_family: dict[str, list[CaseRecord]] = defaultdict(list)
    cards_by_family: dict[str, list[JsonObject]] = defaultdict(list)
    for case in cases:
        cases_by_family[classifications[_case_key(case)].family].append(case)
    for card in cards:
        cards_by_family[str(card["family"])].append(card)
    families = [family for family in BLUEPRINTS if cases_by_family.get(family)]

    write_jsonl(output_dir / "experience_cards.jsonl", cards)
    write_jsonl(output_dir / "provenance.jsonl", card_source.values())
    write_jsonl(
        output_dir / "case_catalog.jsonl",
        [compact_case_catalog(case, classifications[_case_key(case)]) for case in cases],
    )

    if package_generator is None:
        if options.use_api:
            package_generator, provider_metadata = _responses_generator(options)
        else:
            package_generator = lambda blueprint, _prompt: (
                _fallback_package(blueprint),
                {"mode": "deterministic_fallback"},
            )
            provider_metadata = {"mode": "disabled"}
    else:
        provider_metadata = {"mode": "injected_generator"}

    prompts: dict[str, str] = {}
    prompt_cards: dict[str, list[JsonObject]] = {}
    for family in families:
        prompt_cards[family] = select_prompt_cards(
            cards_by_family[family],
            positive_limit=options.positive_examples_per_family,
            negative_limit=options.negative_examples_per_family,
        )
        prompts[family] = build_family_prompt(
            blueprint=get_blueprint(family),
            case_catalog=[
                compact_case_catalog(case, classifications[_case_key(case)]) for case in cases_by_family[family]
            ],
            experience_cards=prompt_cards[family],
        )
        write_json(
            output_dir / "prompts" / f"{family}.json",
            {
                "schema": "migoo-offline-skill-prompt-v1",
                "family": family,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": json.loads(prompts[family]),
                "structured_output_schema": skill_package_schema(),
            },
        )

    generated_packages: dict[str, JsonObject] = {}
    generation_metadata: dict[str, JsonObject] = {}
    workers = max(1, min(options.api_concurrency, len(families)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skill-generation") as executor:
        futures = {
            executor.submit(package_generator, get_blueprint(family), prompts[family]): family for family in families
        }
        for future in as_completed(futures):
            family = futures[future]
            generated, metadata = future.result()
            validate_generated_package(generated, get_blueprint(family))
            generated_packages[family] = generated
            generation_metadata[family] = metadata
            print(f"generated {family}", flush=True)

    instances: dict[str, JsonObject] = {}
    for family in families:
        blueprint = get_blueprint(family)
        generated = generated_packages[family]
        skill_dir = output_dir / "skills" / skill_name_for_family(family)
        write_text(skill_dir / "SKILL.md", render_skill_md(generated, blueprint=blueprint))
        write_json(
            skill_dir / "state_template.json",
            build_state_template(
                blueprint,
                generated,
                template_version=TEMPLATE_VERSION,
                verifier_version=VERIFIER_VERSION,
            ),
        )
        write_text(
            skill_dir / "judge_rubric.md",
            render_judge_rubric(generated, verifier_version=VERIFIER_VERSION, blueprint=blueprint),
        )
        write_json(
            output_dir / "generations" / f"{family}.json",
            {
                "family": family,
                "content": generated,
                "metadata": generation_metadata[family],
            },
        )
        family_instances: list[JsonObject] = []
        for case in cases_by_family[family]:
            instance = instantiate_case(
                case,
                classifications[_case_key(case)],
                blueprint,
                generated,
                verifier_version=VERIFIER_VERSION,
                template_version=TEMPLATE_VERSION,
                generated_packages=generated_packages,
            )
            instances[_case_key(case)] = instance
            family_instances.append(instance)
        write_jsonl(output_dir / "case_instances" / f"{family}.jsonl", family_instances)
        _write_family_tests(
            skill_dir / "tests.jsonl",
            cases_by_family[family],
            classifications,
            instances,
        )

    replay_results = [
        replay_rollout(instances[_case_key(case)], rollout) for case in cases for rollout in case.rollouts
    ]
    compact_cases = [
        {"case_id": case.case_id, "domain": case.domain, "family": classifications[_case_key(case)].family}
        for case in cases
    ]
    validation = build_validation_report(replay_results, cases=compact_cases)
    write_jsonl(output_dir / "validation_results.jsonl", replay_results)
    write_json(output_dir / "validation_report.json", validation)

    family_stats: JsonObject = {}
    for family in families:
        family_cards = cards_by_family[family]
        family_stats[family] = {
            "cases": len(cases_by_family[family]),
            "rollouts": len(family_cards),
            "benchmark_passes": sum(card["reward"]["overall_pass"] for card in family_cards),
            "verified_successes": sum(card["label"] == "verified_success" for card in family_cards),
            "verified_success_cases": len(
                {card["case_id"] for card in family_cards if card["label"] == "verified_success"}
            ),
            "regression_negatives": sum(card["label"] == "regression_negative" for card in family_cards),
            "prompt_positive_examples": sum(card["label"] == "verified_success" for card in prompt_cards[family]),
            "prompt_negative_examples": sum(card["label"] == "regression_negative" for card in prompt_cards[family]),
            "skill_path": f"skills/{skill_name_for_family(family)}",
        }

    manifest = {
        "schema": "migoo-offline-skillbank-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "template_version": TEMPLATE_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "classification": {
            "method": "rules",
            "dimensions": ["domain", "operation", "outcome", "cardinality", "complexity"],
            "kmeans_used": False,
            "counts": dict(sorted(Counter(value.family for value in classifications.values()).items())),
        },
        "generation": {
            "provider": provider_metadata,
            "family_metadata": generation_metadata,
        },
        "sources": _source_manifest(corpora),
        "git": _git_metadata(root),
        "families": family_stats,
        "totals": {
            "cases": len(cases),
            "matched_rollouts": sum(corpus.matched_rollouts for corpus in corpora),
            "unmatched_rollouts": sum(corpus.unmatched_rollouts for corpus in corpora),
            "verified_successes": sum(card["label"] == "verified_success" for card in cards),
            "regression_negatives": sum(card["label"] == "regression_negative" for card in cards),
        },
        "artifacts": {
            "experience_cards": "experience_cards.jsonl",
            "case_catalog": "case_catalog.jsonl",
            "case_instances": "case_instances/",
            "prompts": "prompts/",
            "skills": "skills/",
            "provenance": "provenance.jsonl",
            "validation_report": "validation_report.json",
            "validation_results": "validation_results.jsonl",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_text(output_dir / "REPORT.md", render_report(manifest, validation))
    return manifest
