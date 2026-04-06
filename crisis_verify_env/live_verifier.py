from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib import error, request


class FactCheckConfigurationError(RuntimeError):
    pass


@dataclass
class SourceReference:
    title: str
    url: str


@dataclass
class LiveFactCheckResult:
    verdict_label: str
    confidence: float
    risk_level: str
    explanation: str
    signals: list[str]
    suggested_checks: list[str]
    sources: list[SourceReference]
    checked_on: str
    backend: str


def verify_general_claim(claim: str, source_type: str, include_media: bool) -> LiveFactCheckResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise FactCheckConfigurationError(
            "OPENAI_API_KEY is missing. Add it to your environment before using live fact check mode."
        )

    model = os.getenv("CRISISVERIFY_FACT_MODEL", "o4-mini")
    today = date.today().isoformat()
    prompt = f"""
You are a careful fact-checking assistant with live web access.
Today's date is {today}.

Task:
- Verify this claim using current web results.
- Be conservative with breaking news and high-stakes claims.
- If evidence is mixed or still developing, do not overstate certainty.
- Distinguish between likely true, likely false, mixed/context-needed, and unverified.

Claim: {claim}
Source type provided by user: {source_type}
Media attached: {"yes" if include_media else "no"}

Return ONLY valid JSON with this exact schema:
{{
  "verdict_label": "Likely True | Likely False | Mixed / Needs Context | Unverified",
  "confidence": 0.0,
  "risk_level": "Low | Moderate | High | Breaking / unstable",
  "explanation": "short paragraph",
  "signals": ["3 to 5 short bullets"],
  "suggested_checks": ["2 to 4 short bullets"]
}}
""".strip()

    payload = {
        "model": model,
        "reasoning": {"effort": "low"},
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "input": prompt,
    }

    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenAI API: {exc.reason}") from exc

    text = _extract_output_text(raw)
    parsed = _parse_json_block(text)
    sources = _extract_sources(raw)

    verdict_label = str(parsed.get("verdict_label", "Unverified"))
    confidence = float(parsed.get("confidence", 0.35))
    explanation = str(parsed.get("explanation", "The live verifier could not produce a stable explanation."))
    signals = [str(item) for item in parsed.get("signals", [])][:5]
    suggested_checks = [str(item) for item in parsed.get("suggested_checks", [])][:4]
    risk_level = str(parsed.get("risk_level", "Breaking / unstable"))

    return LiveFactCheckResult(
        verdict_label=verdict_label,
        confidence=max(0.0, min(1.0, confidence)),
        risk_level=risk_level,
        explanation=explanation,
        signals=signals or ["No structured signals returned"],
        suggested_checks=suggested_checks or ["Cross-check against additional reliable coverage"],
        sources=sources,
        checked_on=today,
        backend=f"OpenAI Responses API ({model}) with web_search",
    )


def _extract_output_text(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("output_text"), str) and raw["output_text"].strip():
        return raw["output_text"]

    parts: list[str] = []
    for item in raw.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def _parse_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse model output as JSON: {text[:400]}") from exc


def _extract_sources(raw: Any) -> list[SourceReference]:
    found: list[SourceReference] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("sources"), list):
                for source in node["sources"]:
                    if not isinstance(source, dict):
                        continue
                    url = str(source.get("url", "")).strip()
                    title = str(source.get("title", url)).strip()
                    if url and url not in seen:
                        seen.add(url)
                        found.append(SourceReference(title=title or url, url=url))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw)
    return found[:6]
