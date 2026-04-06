from __future__ import annotations

from collections import defaultdict
from random import Random

from .grader import grade_final_submission
from .models import Action, ActionType, Difficulty, EnvironmentState, Observation, RewardBreakdown, Scenario, StepInfo, StepResult, dataclass_to_dict
from .tasks import get_scenarios


class CrisisVerifyEnv:
    def __init__(self, seed: int = 42) -> None:
        self._rng = Random(seed)
        self._scenarios = get_scenarios()
        self._scenario_lookup = {scenario.id: scenario for scenario in self._scenarios}
        self._action_index = self._build_action_index(self._scenarios)
        self._state: EnvironmentState | None = None

    @staticmethod
    def _build_action_index(scenarios: list[Scenario]) -> dict[str, dict[ActionType, list[str]]]:
        index: dict[str, dict[ActionType, list[str]]] = {}
        for scenario in scenarios:
            action_map: dict[ActionType, list[str]] = defaultdict(list)
            for item in scenario.evidence:
                for action_type in item.useful_actions:
                    action_map[action_type].append(item.id)
            index[scenario.id] = dict(action_map)
        return index

    def available_tasks(self) -> list[dict[str, str]]:
        return [{"id": s.id, "difficulty": s.difficulty.value, "claim": s.claim} for s in self._scenarios]

    def reset(self, task_id: str | None = None, difficulty: Difficulty | str | None = None) -> Observation:
        scenario = self._choose_scenario(task_id, difficulty)
        self._state = EnvironmentState(
            scenario=scenario,
            discovered_evidence_ids=[],
            action_history=[],
            steps_taken=0,
            done=False,
        )
        return self.state()

    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        scenario = self._state.scenario
        discovered_titles = [item.title for item in scenario.evidence if item.id in self._state.discovered_evidence_ids]
        visible_evidence = [item.title for item in scenario.evidence[:2]]
        return Observation(
            scenario_id=scenario.id,
            difficulty=scenario.difficulty,
            claim=scenario.claim,
            source_name=scenario.source_name,
            source_type=scenario.source_type,
            region=scenario.region,
            timestamp=scenario.timestamp,
            visible_evidence=visible_evidence,
            discovered_evidence=discovered_titles,
            action_history=list(self._state.action_history),
            steps_taken=self._state.steps_taken,
            steps_remaining=max(0, scenario.max_steps - self._state.steps_taken),
        )

    def state_dict(self) -> dict:
        return dataclass_to_dict(self.state())

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        if self._state.done:
            raise RuntimeError("Episode already completed. Call reset() to start another task.")

        self._state.steps_taken += 1
        if action.action_type == ActionType.SUBMIT_VERDICT:
            reward, explanation = self._handle_submit(action)
        else:
            reward, explanation = self._handle_investigation(action.action_type)

        self._state.action_history.append(action.action_type.value)
        done_reason = "continue"

        if not self._state.done and self._state.steps_taken >= self._state.scenario.max_steps:
            self._state.done = True
            reward -= 0.1
            explanation = "Maximum investigation steps reached before a final verdict."
            done_reason = "max_steps_reached"
        elif self._state.done:
            done_reason = "submitted"

        reward_breakdown = grade_final_submission(self._state) if self._state.done else RewardBreakdown(total=reward)
        final_reward = reward_breakdown.total if self._state.done else reward
        return StepResult(
            observation=self.state(),
            reward=final_reward,
            done=self._state.done,
            info=StepInfo(
                explanation=explanation,
                reward_breakdown=reward_breakdown,
                done_reason=done_reason,
            ),
        )

    def step_dict(self, action: Action) -> dict:
        return dataclass_to_dict(self.step(action))

    def debug_state(self) -> dict:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return dataclass_to_dict(self._state)

    def _handle_investigation(self, action_type: ActionType) -> tuple[float, str]:
        assert self._state is not None
        scenario = self._state.scenario
        matched_ids = self._action_index[scenario.id].get(action_type, [])
        new_ids = [item_id for item_id in matched_ids if item_id not in self._state.discovered_evidence_ids]
        if not new_ids:
            return -0.05, f"{action_type.value} produced little useful signal for this claim."

        self._state.discovered_evidence_ids.extend(new_ids)
        titles = [item.title for item in scenario.evidence if item.id in new_ids]
        reward = min(0.18 * len(new_ids), 0.3)
        return reward, f"{action_type.value} revealed: {', '.join(titles)}."

    def _handle_submit(self, action: Action) -> tuple[float, str]:
        assert self._state is not None
        if action.verdict is None:
            raise ValueError("submit_verdict action requires a verdict.")
        self._state.done = True
        self._state.submitted_verdict = action.verdict
        self._state.submitted_confidence = action.confidence or 0.5
        if action.verdict == self._state.scenario.verdict:
            return 0.5, "Submitted the correct final verdict."
        return -0.45, "Submitted an incorrect final verdict."

    def _choose_scenario(self, task_id: str | None, difficulty: Difficulty | str | None) -> Scenario:
        if task_id is not None:
            if task_id not in self._scenario_lookup:
                raise KeyError(f"Unknown task_id: {task_id}")
            return self._scenario_lookup[task_id]

        candidates = self._scenarios
        if difficulty is not None:
            selected = difficulty if isinstance(difficulty, Difficulty) else Difficulty(difficulty)
            candidates = [scenario for scenario in self._scenarios if scenario.difficulty == selected]
        return self._rng.choice(candidates)
