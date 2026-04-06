from __future__ import annotations

import argparse
import json
from collections import defaultdict

from crisis_verify_env import CrisisVerifyEnv
from crisis_verify_env.models import Action, ActionType, Verdict


def choose_verdict(discovered: list[str]) -> Verdict:
    text = " ".join(discovered).lower()
    if (
        "official denial" in text
        or "media forensics" in text
        or "no trusted confirmation" in text
        or "archive video match" in text
        or "brand spoof indicators" in text
        or "negotiation status" in text
    ):
        return Verdict.FAKE
    if (
        "timeline mismatch" in text
        or "operations update" in text
        or "inventory" in text
        or "grid operator" in text
        or "caption exaggeration" in text
        or "field reports" in text
        or "causal leap" in text
    ):
        return Verdict.MISLEADING
    if "official emergency post" in text or "local confirmation" in text:
        return Verdict.REAL
    return Verdict.INSUFFICIENT


def run_episode(env: CrisisVerifyEnv, task_id: str) -> dict[str, object]:
    initial = env.reset(task_id=task_id)
    ordered_actions = [
        Action(action_type=ActionType.INSPECT_SOURCE),
        Action(action_type=ActionType.CHECK_TIMELINE),
        Action(action_type=ActionType.CROSSCHECK_TRUSTED_REPORTS),
        Action(action_type=ActionType.ANALYZE_MEDIA),
        Action(action_type=ActionType.SCAN_LANGUAGE),
    ]

    result = None
    for action in ordered_actions:
        result = env.step(action)
        if result.done:
            break

    if result is None:
        raise RuntimeError("Baseline took no steps.")

    verdict = choose_verdict(result.observation.discovered_evidence)
    result = env.step(Action(action_type=ActionType.SUBMIT_VERDICT, verdict=verdict, confidence=0.72))
    return {
        "task_id": task_id,
        "difficulty": initial.difficulty.value,
        "predicted_verdict": verdict.value,
        "reward": result.reward,
        "done_reason": result.info.done_reason,
        "discovered_evidence": result.observation.discovered_evidence,
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    for item in results:
        by_difficulty[str(item["difficulty"])].append(float(item["reward"]))

    difficulty_summary = {
        difficulty: {
            "count": len(scores),
            "average_reward": round(sum(scores) / len(scores), 3),
        }
        for difficulty, scores in sorted(by_difficulty.items())
    }
    overall = [float(item["reward"]) for item in results]
    return {
        "task_count": len(results),
        "average_reward": round(sum(overall) / len(overall), 3),
        "difficulty_summary": difficulty_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the heuristic baseline on CrisisVerifyEnv.")
    parser.add_argument("--task-id", help="Run only one task id.")
    args = parser.parse_args()

    env = CrisisVerifyEnv(seed=7)
    task_ids = [args.task_id] if args.task_id else [item["id"] for item in env.available_tasks()]
    results = [run_episode(env, task_id) for task_id in task_ids]
    print(json.dumps({"summary": summarize(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
