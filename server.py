from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from crisis_verify_env import CrisisVerifyEnv
from crisis_verify_env.models import Action, Difficulty, StepResult


app = FastAPI(title="CrisisVerifyEnv", version="0.1.0")
ENV = CrisisVerifyEnv(seed=7)


class ResetRequest(BaseModel):
    task_id: str | None = None
    difficulty: str | None = None


class StepRequest(BaseModel):
    action_type: str
    argument: str | None = None
    verdict: str | None = None
    confidence: float | None = None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable_encoder(asdict(value))
    if isinstance(value, Enum):
        return value.value
    return jsonable_encoder(value)


def normalize_result(result: StepResult) -> dict[str, Any]:
    return {
        "observation": to_jsonable(result.observation),
        "reward": result.reward,
        "done": result.done,
        "info": to_jsonable(result.info),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <html>
      <head><title>CrisisVerifyEnv</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.5;">
        <h1>CrisisVerifyEnv API</h1>
        <p>This deployment exposes the OpenEnv-style API for automated checks.</p>
        <ul>
          <li><code>POST /reset</code></li>
          <li><code>POST /step</code></li>
          <li><code>GET /state</code></li>
          <li><code>GET /tasks</code></li>
          <li><code>GET /health</code></li>
        </ul>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {"tasks": ENV.available_tasks()}


@app.post("/reset")
def reset(request: ResetRequest) -> dict[str, Any]:
    try:
        difficulty = request.difficulty
        if difficulty is not None:
            difficulty = Difficulty(difficulty).value
        observation = ENV.reset(task_id=request.task_id, difficulty=difficulty)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"observation": to_jsonable(observation)}


@app.get("/state")
def state() -> dict[str, Any]:
    try:
        return {"observation": to_jsonable(ENV.state())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    try:
        action = Action(
            action_type=request.action_type,
            argument=request.argument,
            verdict=request.verdict,
            confidence=request.confidence,
        )
        result = ENV.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return normalize_result(result)

