# CrisisVerify AI

CrisisVerify AI is a hackathon-ready offline verification demo for crisis misinformation. A user can paste a claim, add source context, and get a verdict with confidence, key signals, and verification guidance. Under the hood, the app uses `CrisisVerifyEnv`, an OpenEnv-style offline environment for crisis misinformation workflows.

## Why this project

During wars, disasters, and political crises, false or misleading claims can spread faster than human fact-checkers can respond. This environment turns that workflow into a deterministic agent task with partial rewards, task difficulty levels, and reproducible scoring.

## App experience

The demo app lets a judge or user:

- paste a headline or claim
- choose the source type
- indicate whether media is attached
- receive a verdict such as `Likely True`, `Likely False`, `Mixed / Needs Context`, or `Unverified`
- see confidence, matched signals, and suggested checks

Launch the local demo app:

```bash
streamlit run app.py
```

Main app file:

- `app.py`: hackathon-facing prediction UI
- `server.py`: API server for OpenEnv-style automated checks and deployment

## Environment concept

Each episode gives the agent:

- a crisis-related claim
- source metadata
- region and timestamp context
- a hidden pool of evidence recoverable through actions

The environment currently ships with `12` deterministic scenarios spanning reused media, spoofed alerts, inflated casualty or displacement claims, deepfakes, fake ceasefire bulletins, and misleading satellite-image narratives.

The agent can choose actions such as:

- inspect source credibility
- check timeline consistency
- cross-check trusted reports
- analyze media
- scan emotional or propagandistic language
- request more context
- submit a final verdict

Supported verdicts:

- `real`
- `misleading`
- `fake`
- `insufficient_evidence`

## Difficulty design

- `easy`: clear evidence, reused media, official denial or confirmation
- `medium`: partial truths, mixed reporting, uncertain scale claims
- `hard`: deepfakes, manipulated framing, multi-step evidence synthesis

## Reward design

The reward function provides partial progress signals:

- positive reward when an action discovers relevant evidence
- small penalty for unproductive actions
- efficiency penalty for long investigations
- strong reward for a correct final verdict
- confidence bonus or penalty depending on whether confidence matches correctness

## Project structure

- `app.py`: Streamlit prediction app
- `crisis_verify_env/predictor.py`: lightweight app-side prediction and explanation layer
- `crisis_verify_env/models.py`: typed models for observations, actions, rewards, and state
- `crisis_verify_env/tasks.py`: deterministic scenario bank across easy, medium, and hard tasks
- `crisis_verify_env/env.py`: environment implementation with `reset()`, `step()`, and `state()`
- `crisis_verify_env/grader.py`: deterministic final grading logic
- `baseline/run.py`: reproducible heuristic baseline
- `openenv.yaml`: environment metadata
- `Dockerfile`: containerized execution

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Safer on Windows if `streamlit` is not on PATH:

```bash
python -m streamlit run app.py
```

## Run the API server

```bash
uvicorn server:app --host 0.0.0.0 --port 8501
```

Important deployment note:

- the Hugging Face / automated checker deployment should point to `server.py`
- the Streamlit UI is for local demo use

## Run baseline

```bash
python -m baseline.run
```

The baseline now prints both per-task outputs and an aggregate summary by difficulty.

Run a single task:

```bash
python -m baseline.run --task-id easy_reused_image_bridge
```

## Checklist-compatible inference entrypoint

This repository also includes `inference.py` to better match automated checklist expectations.

Environment variables used there:

- `API_BASE_URL` with a default of `https://api.openai.com/v1`
- `MODEL_NAME` with a default of `gpt-4.1-mini`
- `HF_TOKEN` with no default
- optional `LOCAL_IMAGE_NAME`

Run it like this:

```bash
python inference.py --task-id easy_reused_image_bridge
```

The script prints structured `START`, `STEP`, and `END` logs and then emits the final JSON result.

## Submission notes

This repository is intentionally scoped for a hackathon build:

- user-facing prediction app for demo day
- deterministic, curated scenarios instead of live scraping
- an environment-first design that matches OpenEnv expectations
- a simple but reproducible baseline policy
- a clean path to Docker and Hugging Face deployment

For demos, the environment also exposes:

- `state_dict()` for serialized observation output
- `step_dict(action)` for serialized transition output
- `debug_state()` for internal inspection during development

## Next recommended improvements

- add more scenarios per region and misinformation type
- attach richer rationale scoring
- add multimodal artifact references for images and video clips
- plug in a stronger LLM-based baseline agent
