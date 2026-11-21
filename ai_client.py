"""Thin client for DeepSeek and OpenAI-compatible chat endpoints.

Notes:
 - Endpoint: {base_url}/chat/completions  (POST, JSON)
 - Designed to be dependency-light: only requires `requests`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import requests

# Always log to file (no env var required). Also echo to stderr when AI_DEBUG=1.
_LOG_PATH = Path(__file__).parent / "ai_debug.log"
_LOG_LOCK = Lock()
_ECHO_STDERR = os.environ.get("AI_DEBUG") == "1"


def _log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    with _LOG_LOCK:
        try:
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except OSError:
            pass
    if _ECHO_STDERR:
        print(line, file=sys.stderr, flush=True)


# Mark a fresh app run in the log
_log("=" * 60)
_log(f"ai_client loaded from {Path(__file__)}")

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
_DEEPSEEK_THINKING_MODELS = {"deepseek-v4-flash", "deepseek-flash", "deepseek-v4-pro"}


class AIError(RuntimeError):
    """Raised when the AI call fails after all retries."""


@dataclass
class AIConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int = 120
    max_retries: int = 3
    backoff: float = 2.0

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AIConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            api_key=data["api_key"],
            base_url=data["base_url"].rstrip("/"),
            model=data["model"],
            timeout=int(data.get("request_timeout_seconds", 120)),
            max_retries=int(data.get("max_retries", 3)),
            backoff=float(data.get("retry_backoff_seconds", 2.0)),
        )

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(
            json.dumps(
                {
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                    "model": self.model,
                    "request_timeout_seconds": self.timeout,
                    "max_retries": self.max_retries,
                    "retry_backoff_seconds": self.backoff,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class AIClient:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.load()

    # --------------------------- low-level call ---------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Call /chat/completions and return assistant message content (string).

        Raises AIError on persistent failure. JSON-mode requests should pass
        ``response_format={"type": "json_object"}`` and instruct the model to
        emit JSON in the prompt.
        """
        url = f"{self.config.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            # Stream by default: long generations otherwise risk the server
            # closing the connection before the full body arrives, which raises
            # ChunkedEncodingError("Response ended prematurely").
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        attempts = max(1, self.config.max_retries)
        last_error = "模型未返回正式答案。"
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            _log(
                f"-> POST {url}  model={self.config.model}  "
                f"msgs={len(messages)}  max_tokens={payload.get('max_tokens')}  "
                f"json_mode={response_format is not None}  stream=True  attempt={attempt}"
            )
            try:
                with requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                    stream=True,
                ) as resp:
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                        if resp.status_code not in (429, 500, 502, 503, 504):
                            raise AIError(last_error)
                    else:
                        resp.encoding = "utf-8"
                        content, finish_reason = self._consume_stream(resp)
                        _log(
                            f"<- stream done in {time.time()-t0:.1f}s, "
                            f"content_len={len(content)} finish_reason={finish_reason}"
                        )
                        if finish_reason == "length":
                            last_error = "模型输出达到长度上限，答案被截断。"
                            if max_tokens is not None and payload["max_tokens"] == max_tokens:
                                payload["max_tokens"] = max_tokens * 2
                        elif finish_reason in ("aborted", "insufficient_system_resource"):
                            last_error = f"模型生成中断（{finish_reason}）。"
                        elif finish_reason not in (None, "stop"):
                            raise AIError(f"模型未能完成回答（{finish_reason}）。")
                        elif content.strip():
                            return content
                        else:
                            last_error = "模型未返回正式答案。"
            except requests.exceptions.RequestException as e:
                last_error = f"连接模型接口失败（{type(e).__name__}）：{e}"

            _log(f"!! attempt {attempt}/{attempts}: {last_error}")
            if attempt < attempts:
                time.sleep(self.config.backoff * attempt)
        raise AIError(f"已尝试 {attempts} 次。{last_error}")

    @staticmethod
    def _consume_stream(resp: "requests.Response") -> tuple[str, str | None]:
        """Read the final answer and completion status from an SSE stream."""
        content_parts: list[str] = []
        finish_reason = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                choice = obj["choices"][0]
            except (KeyError, IndexError, TypeError):
                continue
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                content_parts.append(piece)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        return "".join(content_parts), finish_reason

    # ---------------------------- helpers --------------------------------

    def ping(self) -> tuple[bool, str]:
        """Minimal connectivity test. Returns (ok, message).

        Note: thinking/reasoning models spend tokens on internal reasoning
        before producing visible content, so we leave headroom in max_tokens.
        """
        try:
            out = self.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a connectivity probe. Reply with exactly "
                            "the two characters: OK"
                        ),
                    },
                    {"role": "user", "content": "ping"},
                ],
                temperature=0.0,
                max_tokens=30000,
            )
            return True, f"Got: {out.strip()[:120]!r}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> Any:
        """Like ``chat`` but expects a JSON-encoded response and parses it.

        Strips Markdown code fences if present (some proxies add them even when
        ``response_format`` is set).
        """
        extra = None
        if disable_thinking and self.config.model in _DEEPSEEK_THINKING_MODELS:
            extra = {"thinking": {"type": "disabled"}}
        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra=extra,
        )
        return _parse_json_loose(raw)


def _parse_json_loose(raw: str) -> Any:
    """Parse JSON tolerating ```json ... ``` fences and stray prose."""
    s = raw.strip()
    if s.startswith("```"):
        # strip first fence line + trailing fence
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
        # If the first fence was ```json, that 'json' got stripped above
    # Try plain parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try to find the outermost {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = s.find(opener), s.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    _log(f"!! invalid JSON response, content_len={len(raw)}")
    raise AIError("模型答案格式异常，无法读取结果。请重试。")


if __name__ == "__main__":
    # Smoke test: just attempt a ping.
    client = AIClient()
    ok, msg = client.ping()
    print(("OK" if ok else "FAIL"), msg)
