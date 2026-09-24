"""Jev-only Choice transport for OpenRouter and Vercel; no model fallback."""

import json
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import httpx

PROVIDERS = {
    "openrouter": (
        "https://openrouter.ai/api/alpha/decisions",
        "typesafe/jev-1.13",
        "OPENROUTER_API_KEY",
    ),
    "vercel": ("https://ai-gateway.vercel.sh/v1/evaluate", "typesafe-ai/jev", "AI_GATEWAY_API_KEY"),
}


def load_credential(provider, path=None):
    name = PROVIDERS[provider][2]
    value = os.environ.get(name)
    if not value and path and Path(path).is_file():
        match = re.search(
            r"(?m)^\s*(?:export\s+)?" + name + r"\s*=\s*[\"']?([^\s\"'#]+)", Path(path).read_text()
        )
        value = match.group(1) if match else None
    if not value:
        raise ValueError(f"Missing {name}; set it in the environment or key file")
    return value


def validate_answers(answers, questions):
    if set(answers) != set(questions):
        raise ValueError("Mismatched answer keys")
    for name, question in questions.items():
        answer = answers[name]
        if answer["choice"] not in question["criteria"]:
            raise ValueError("Invalid choice")
        probabilities = answer["probabilities"]
        if set(probabilities) != set(question["criteria"]):
            raise ValueError("Mismatched probabilities")
        if any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or not 0 <= p <= 1
            for p in probabilities.values()
        ):
            raise ValueError("Invalid probabilities")
        if abs(sum(probabilities.values()) - 1) > 0.03:
            raise ValueError("Invalid probability sum")


class JevGateway:
    def __init__(self, provider, key, timeout=3.0, transport=None, journal=None):
        self.provider = provider
        self.journal = journal
        self.endpoint, self.model, _ = PROVIDERS[provider]
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def evaluate(self, state, questions):
        started = time.perf_counter()
        # Planner memories and detour progress continue changing after this call.
        # Store the actual input, not a reference to future experiment state.
        state, questions = deepcopy(state), deepcopy(questions)
        result = {
            "provider": self.provider,
            "requested_model": self.model,
            "state": state,
            "questions": questions,
        }
        try:
            response = self.client.post(
                self.endpoint, json={"model": self.model, "state": state, "questions": questions}
            )
            result["http_status"] = response.status_code
            if response.status_code != 200:
                result["error"] = f"http_{response.status_code}"
            else:
                data = response.json()
                result.update(
                    model=data.get("model"),
                    usage=data.get("usage", {}),
                    provider_metadata=data.get("providerMetadata", {}),
                )
                if self.provider == "vercel":
                    metadata = result["provider_metadata"].get("gateway", {})
                    if metadata.get("cost") is not None:
                        result["usage"] = {**result["usage"], "cost": float(metadata["cost"])}
                if result["model"] and "jev" not in result["model"].lower():
                    raise ValueError("Unexpected model")
                validate_answers(data["answers"], questions)
                result["answers"] = data["answers"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            result["error"] = type(exc).__name__
        result["latency_seconds"] = time.perf_counter() - started
        if self.journal is not None:
            with Path(self.journal).open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        return result

    def choose(self, state, instructions, criteria):
        result = self.evaluate(
            state,
            {"selection": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        )
        if not result.get("error"):
            result["choice"] = result["answers"]["selection"]["choice"]
        return result
