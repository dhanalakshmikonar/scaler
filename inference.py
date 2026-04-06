from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional local dependency for offline use
    OpenAI = None

from crisis_verify_env import CrisisVerifyEnv
from crisis_verify_env.models import Action, ActionType, Verdict


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def log_line(stage: str, payload: dict[str, object]) -> None:
    print(f"{stage} {json.dumps(payload, ensure_ascii=True)}", flush=True)


def build_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


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


def run_episode(task_id: str | None = None) -> dict[str, object]:
    env = CrisisVerifyEnv(seed=7)
    observation = env.reset(task_id=task_id)
    log_line(
        "START",
        {
            "task_id": observation.scenario_id,
            "difficulty": observation.difficulty.value,
            "model_name": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "hf_token_present": bool(HF_TOKEN),
            "local_image_name": LOCAL_IMAGE_NAME,
        },
    )

    ordered_actions = [
        Action(action_type=ActionType.INSPECT_SOURCE),
        Action(action_type=ActionType.CHECK_TIMELINE),
        Action(action_type=ActionType.CROSSCHECK_TRUSTED_REPORTS),
        Action(action_type=ActionType.ANALYZE_MEDIA),
        Action(action_type=ActionType.SCAN_LANGUAGE),
    ]

    step_index = 0
    latest = None
    for action in ordered_actions:
        step_index += 1
        latest = env.step(action)
        log_line(
            "STEP",
            {
                "step": step_index,
                "action": action.action_type.value,
                "reward": latest.reward,
                "done": latest.done,
                "discovered_evidence": latest.observation.discovered_evidence,
            },
        )
        if latest.done:
            break

    if latest is None:
        raise RuntimeError("Episode produced no intermediate steps.")

    verdict = choose_verdict(latest.observation.discovered_evidence)
    final_result = env.step(
        Action(action_type=ActionType.SUBMIT_VERDICT, verdict=verdict, confidence=0.72)
    )
    payload = {
        "task_id": final_result.observation.scenario_id,
        "difficulty": final_result.observation.difficulty.value,
        "predicted_verdict": verdict.value,
        "reward": final_result.reward,
        "done_reason": final_result.info.done_reason,
        "discovered_evidence": final_result.observation.discovered_evidence,
        "reward_breakdown": asdict(final_result.info.reward_breakdown),
    }
    log_line("END", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Checklist-compatible inference entrypoint.")
    parser.add_argument("--task-id", help="Specific task id to run")
    args = parser.parse_args()

    _ = build_openai_client()
    result = run_episode(task_id=args.task_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
