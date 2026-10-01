"""POST /v1/ask — an image+text model for the questions a detector can't answer.

"Is there anything in front of me I'd hit?", "Are these the same plant, just
turned?", "Which side of this plant hasn't been photographed?" — judgement
calls. They are slow (1-10 s) and the robot is slow on purpose, so they fit.

Providers are tried in order (VS_VLM_PROVIDERS, default "gemini,ollama"):

  gemini — Google's API (VS_GEMINI_MODEL, default gemini-3.6-flash). Best
           answers, costs per call, needs the internet and GEMINI_API_KEY.
  ollama — a local model through `ollama serve` (VS_OLLAMA_MODEL, default
           qwen3-vl:8b, which Ollama ships 4-bit quantised). Private, free,
           works offline; slower and a little less sharp.

Stdlib HTTP only (urllib), so this adds no dependency to the service. Called in
a worker thread, never on the event loop, and never through the GPU runner —
a 10 s VLM call must not block detection.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request

from .settings import Settings

log = logging.getLogger(__name__)


class VLMUnavailable(RuntimeError):
    pass


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class VLM:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.providers = [p.strip() for p in cfg.vlm_providers.split(",") if p.strip()]
        if "gemini" in self.providers and not cfg.gemini_api_key:
            log.warning("VLM: gemini listed but GEMINI_API_KEY is not set — skipping it")
            self.providers.remove("gemini")

    def available(self) -> list[str]:
        return list(self.providers)

    def ask(self, images: list[bytes], question: str, json_mode: bool = False) -> dict:
        errors = []
        for provider in self.providers:
            t0 = time.perf_counter()
            try:
                answer, model = getattr(self, f"_{provider}")(images, question, json_mode)
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
                log.warning("VLM %s failed: %s", provider, exc)
                errors.append(f"{provider}: {exc}")
                continue
            out = {"answer": answer.strip(), "provider": provider, "model": model,
                   "ms": round((time.perf_counter() - t0) * 1000, 1), "parsed": None}
            if json_mode:
                try:
                    out["parsed"] = json.loads(_strip_fence(answer))
                except ValueError:
                    pass
            return out
        raise VLMUnavailable("; ".join(errors) or "no VLM provider configured")

    def _gemini(self, images, question, json_mode):
        parts = [{"text": question}] + [
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(i).decode()}}
            for i in images]
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.2}}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        model = self.cfg.gemini_model
        r = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            body, {"x-goog-api-key": self.cfg.gemini_api_key}, self.cfg.vlm_timeout_s)
        text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
        return text, model

    def _ollama(self, images, question, json_mode):
        body = {"model": self.cfg.ollama_model, "stream": False, "options": {"temperature": 0.2},
                "messages": [{"role": "user", "content": question,
                              "images": [base64.b64encode(i).decode() for i in images]}]}
        if json_mode:
            body["format"] = "json"
        r = _post_json(self.cfg.ollama_url.rstrip("/") + "/api/chat", body, {}, self.cfg.vlm_timeout_s)
        return r["message"]["content"], self.cfg.ollama_model


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t
