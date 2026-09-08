from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from .skill_planning import SkillPlanLLMOutput
from .task_understanding import TaskParseLLMOutput
from .prompts import SKILL_PLANNING_PROMPT, TASK_UNDERSTANDING_PROMPT, VISION_GROUNDING_PROMPT, prompt_payload
from .vision_grounding import VisionLLMOutput
from .budget import StageGenerationConfig


class QwenProviderError(RuntimeError):
    pass


class QwenHTTPProvider:
    """OpenAI-compatible fixed-call provider; retries are intentionally disabled."""

    provider_id = "qwen_http"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0,
                 stage_max_completion_tokens: dict[str, int] | None = None,
                 stage_generation: dict[str, StageGenerationConfig] | None = None,
                 use_structured_output: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.calls: list[dict[str, Any]] = []
        self.stage_max_completion_tokens = stage_max_completion_tokens or {
            "task_understanding": 256, "vision_grounding": 384, "skill_planning": 512,
        }
        self.stage_generation = stage_generation or {}
        self.use_structured_output = use_structured_output

    def _call(self, stage: str, prompt: str, user_content: str | list[dict[str, Any]]) -> str:
        response = None
        try:
            config = self.stage_generation.get(stage)
            payload = {"model": self.model, "temperature": config.temperature if config else 0, "messages": [
                {"role": "system", "content": prompt}, {"role": "user", "content": user_content}
            ]}
            max_tokens = config.max_completion_tokens if config else self.stage_max_completion_tokens.get(stage)
            if max_tokens is not None:
                payload["max_completion_tokens"] = max_tokens
            # OpenAI-compatible Qwen servers that implement guided JSON accept
            # this parameter.  It is deliberately optional for older servers.
            if self.use_structured_output:
                payload["response_format"] = {"type": "json_object"}
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list): content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            if not isinstance(content, str) or not content.strip():
                raise QwenProviderError(f"{stage}: empty model content")
            extracted = _extract_json(content)
            self.calls.append({"stage": stage, "status": "succeeded", "model": self.model, "prompt_tokens": body.get("usage", {}).get("prompt_tokens"), "completion_tokens": body.get("usage", {}).get("completion_tokens")})
            return extracted
        except QwenProviderError as exc:
            self.calls.append({"stage": stage, "status": "failed", "model": self.model, "http_status": getattr(response, "status_code", None), "error": str(exc)})
            raise
        except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            status = getattr(response, "status_code", None)
            self.calls.append({"stage": stage, "status": "failed", "model": self.model, "http_status": status, "error": str(exc)})
            raise QwenProviderError(f"{stage}: {type(exc).__name__}: {exc}") from exc

    def understand(self, request):
        content = prompt_payload({"instruction": request.instruction})
        value = json.loads(self._call("task_understanding", TASK_UNDERSTANDING_PROMPT, content))
        # Accept the prototype's echoed instruction for compatibility, but do
        # not expose it as part of the minimal parser contract.
        if isinstance(value, dict):
            value.pop("instruction", None)
        return TaskParseLLMOutput.model_validate(value)

    def detect(self, request):
        image = Path(request.rgb_path)
        if not image.is_file(): raise QwenProviderError(f"vision_grounding: RGB not found: {image}")
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        content = [{"type": "text", "text": prompt_payload({"entities": [item.model_dump(mode="json") for item in request.entities]})}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"}}]
        return VisionLLMOutput.model_validate(json.loads(self._call("vision_grounding", VISION_GROUNDING_PROMPT, content)))

    def plan(self, request):
        content = prompt_payload({"operations": [item.model_dump(mode="json") for item in request.context.operations], "goals": request.context.goals, "skills": request.skill_catalog})
        return SkillPlanLLMOutput.model_validate(json.loads(self._call("skill_planning", SKILL_PLANNING_PROMPT, content)))


def _extract_json(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text[3:].strip()
        if text.startswith("json"): text = text[4:].lstrip()
        if text.endswith("```"): text = text[:-3].rstrip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{": continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict): return json.dumps(value, ensure_ascii=False)
    raise QwenProviderError("model output does not contain a JSON object")
