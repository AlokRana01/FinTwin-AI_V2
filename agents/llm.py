"""
agents/llm.py
=============
Provider-neutral LLM Abstraction & Controlled AI Explanation layer for FinTwin AI.

Architectural Guarantees:
- Output-only language generation: LLM output is strictly non-authoritative for financial math.
- Provider-neutral protocol: isolates specialist agents from external provider SDKs and formats.
- Safe configuration & credential masking: never exposes API keys in errors or logs.
- Bounded retries & timeouts: guarantees non-blocking execution with predictable error handling.
- Zero autonomous tools/loops: LLM has zero execution authority over tools, database, or filesystem.
"""

from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Sequence, Union


# ── Exception Hierarchy ───────────────────────────────────────────────────────

class LLMError(Exception):
    """Base exception for LLM abstraction failures."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when provider configuration or API keys are missing/invalid."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication with provider fails (e.g. 401/403)."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an external LLM request exceeds the timeout threshold."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits (e.g. 429) are exceeded."""
    pass


class LLMProviderError(LLMError):
    """Raised on upstream provider failure or unparseable payload."""
    pass


class LLMEmptyResponseError(LLMError):
    """Raised when a provider returns an empty or whitespace-only response."""
    pass


# ── Structured Response ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMResponse:
    """
    Standardized immutable result container for LLM generation requests.
    """
    success: bool
    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    error: Optional[str] = None
    tokens_used: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "tokens_used": self.tokens_used,
        }


# ── Abstract Provider Protocol ────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract base interface for LLM provider adapters.
    """
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'groq', 'gemini', 'mock')."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model identifier for this provider."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Executes a synchronous completion request against the provider API.
        """
        ...


# ── Provider 1: Groq Adapter ──────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """
    HTTP client adapter for Groq Cloud chat completions API.
    """
    DEFAULT_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not self._api_key:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                        self._api_key = str(st.secrets["GROQ_API_KEY"]).strip()
                except Exception:
                    pass
        self._model = model or self.DEFAULT_MODEL
        self._endpoint_url = endpoint_url or self.DEFAULT_URL

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def default_model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise LLMConfigurationError("Groq API key is not configured. Set GROQ_API_KEY environment variable.")

        target_model = model or self._model
        start_t = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": max(0.0, min(temperature, 2.0)),
            "max_tokens": max(1, max_tokens),
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "FinTwinAI-LLMService/1.0",
        }

        req = urllib.request.Request(
            self._endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
                exec_ms = (time.perf_counter() - start_t) * 1000

                if status_code != 200:
                    raise LLMProviderError(f"Groq API returned HTTP {status_code}")

                data = json.loads(raw_body)
                choices = data.get("choices", [])
                if not choices:
                    raise LLMEmptyResponseError("Groq API returned choices list.")

                content = choices[0].get("message", {}).get("content", "").strip()
                if not content:
                    raise LLMEmptyResponseError("Groq API returned an empty text completion.")

                total_tokens = data.get("usage", {}).get("total_tokens")

                return LLMResponse(
                    success=True,
                    text=content,
                    provider="groq",
                    model=target_model,
                    latency_ms=exec_ms,
                    tokens_used=total_tokens,
                )

        except urllib.error.HTTPError as e:
            exec_ms = (time.perf_counter() - start_t) * 1000
            if e.code in (401, 403):
                raise LLMAuthenticationError(f"Groq authentication failed (HTTP {e.code}). Verify API key.") from e
            elif e.code == 429:
                raise LLMRateLimitError("Groq rate limit exceeded (HTTP 429).") from e
            elif e.code >= 500:
                raise LLMProviderError(f"Groq server error (HTTP {e.code}).") from e
            else:
                raise LLMProviderError(f"Groq HTTP error {e.code}: {e.reason}") from e

        except urllib.error.URLError as e:
            exec_ms = (time.perf_counter() - start_t) * 1000
            if "timed out" in str(e.reason).lower() or isinstance(e.reason, TimeoutError):
                raise LLMTimeoutError(f"Groq request timed out after {timeout}s.") from e
            raise LLMProviderError(f"Groq network connection error: {e.reason}") from e

        except TimeoutError as e:
            raise LLMTimeoutError(f"Groq request timed out after {timeout}s.") from e


# ── Provider 2: Gemini Adapter ────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """
    HTTP client adapter for Google Gemini Generative Language API.
    """
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not self._api_key:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                        self._api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
                except Exception:
                    pass
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url or self.DEFAULT_BASE_URL

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise LLMConfigurationError("Gemini API key is not configured. Set GEMINI_API_KEY environment variable.")

        target_model = model or self._model
        start_t = time.perf_counter()

        endpoint = f"{self._base_url}/{target_model}:generateContent?key={self._api_key}"

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": max(0.0, min(temperature, 2.0)),
                "maxOutputTokens": max(1, max_tokens),
            },
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FinTwinAI-LLMService/1.0",
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
                exec_ms = (time.perf_counter() - start_t) * 1000

                if status_code != 200:
                    raise LLMProviderError(f"Gemini API returned HTTP {status_code}")

                data = json.loads(raw_body)
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMEmptyResponseError("Gemini API returned no candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise LLMEmptyResponseError("Gemini candidate contains no parts.")

                content = parts[0].get("text", "").strip()
                if not content:
                    raise LLMEmptyResponseError("Gemini candidate returned empty text.")

                usage = data.get("usageMetadata", {})
                total_tokens = usage.get("totalTokenCount")

                return LLMResponse(
                    success=True,
                    text=content,
                    provider="gemini",
                    model=target_model,
                    latency_ms=exec_ms,
                    tokens_used=total_tokens,
                )

        except urllib.error.HTTPError as e:
            exec_ms = (time.perf_counter() - start_t) * 1000
            if e.code in (400, 401, 403):
                raise LLMAuthenticationError(f"Gemini authentication failed (HTTP {e.code}). Verify API key.") from e
            elif e.code == 429:
                raise LLMRateLimitError("Gemini rate limit exceeded (HTTP 429).") from e
            elif e.code >= 500:
                raise LLMProviderError(f"Gemini server error (HTTP {e.code}).") from e
            else:
                raise LLMProviderError(f"Gemini HTTP error {e.code}: {e.reason}") from e

        except urllib.error.URLError as e:
            exec_ms = (time.perf_counter() - start_t) * 1000
            if "timed out" in str(e.reason).lower() or isinstance(e.reason, TimeoutError):
                raise LLMTimeoutError(f"Gemini request timed out after {timeout}s.") from e
            raise LLMProviderError(f"Gemini network connection error: {e.reason}") from e

        except TimeoutError as e:
            raise LLMTimeoutError(f"Gemini request timed out after {timeout}s.") from e


# ── Provider 3: Mock Provider (Testing / Offline) ─────────────────────────────

class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for unit testing without external API calls or network access.
    """
    def __init__(
        self,
        fixed_response: str = "This is a deterministic mock LLM explanation.",
        provider_name: str = "mock",
        default_model: str = "mock-v1",
        latency_ms: float = 15.0,
        should_fail: bool = False,
        failure_exception: Optional[Exception] = None,
    ):
        self._response = fixed_response
        self._provider_name = provider_name
        self._default_model = default_model
        self._latency_ms = latency_ms
        self._should_fail = should_fail
        self._failure_exception = failure_exception or LLMProviderError("Mock provider simulated failure")
        self.call_history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._default_model

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "model": model or self._default_model,
        })

        if self._should_fail:
            raise self._failure_exception

        return LLMResponse(
            success=True,
            text=self._response,
            provider=self._provider_name,
            model=model or self._default_model,
            latency_ms=self._latency_ms,
            tokens_used=len(prompt.split()) + len(self._response.split()),
        )


# ── LLM Service Manager ───────────────────────────────────────────────────────

class LLMService:
    """
    Central service manager for LLM orchestration, provider registration,
    fallback routing, and bounded retry policy.
    """
    def __init__(
        self,
        providers: Optional[Dict[str, LLMProvider]] = None,
        default_provider: str = "groq",
        max_retries: int = 1,
        default_timeout: float = 30.0,
    ):
        self._providers: Dict[str, LLMProvider] = providers or {}
        self._default_provider_name = default_provider
        self._max_retries = max(0, min(max_retries, 3))
        self._default_timeout = default_timeout

        if not self._providers:
            self._register_default_providers()

    def _register_default_providers(self):
        """Registers default Groq and Gemini providers using environment variables."""
        groq_key = os.environ.get("GROQ_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

        self._providers["groq"] = GroqProvider(api_key=groq_key)
        self._providers["gemini"] = GeminiProvider(api_key=gemini_key)

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Registers or replaces a provider instance."""
        self._providers[name.lower()] = provider

    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        """Retrieves a provider by name or returns the default provider."""
        p_name = (name or self._default_provider_name).lower()
        if p_name not in self._providers:
            raise LLMConfigurationError(f"Provider '{p_name}' is not registered. Available: {list(self._providers.keys())}")
        return self._providers[p_name]

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: Optional[float] = None,
        fallback_providers: Optional[Sequence[str]] = None,
    ) -> LLMResponse:
        """
        Executes an LLM completion request with bounded retries and provider waterfall.
        """
        if not prompt or not prompt.strip():
            raise LLMEmptyResponseError("Cannot generate completion for empty prompt.")

        primary_provider_name = (provider or self._default_provider_name).lower()
        call_timeout = timeout if timeout is not None else self._default_timeout

        # Build execution order: primary -> explicit fallbacks
        waterfall = [primary_provider_name]
        if fallback_providers:
            for fb in fallback_providers:
                fb_lower = fb.lower()
                if fb_lower not in waterfall:
                    waterfall.append(fb_lower)

        last_error: Optional[Exception] = None

        for current_provider_name in waterfall:
            try:
                prov = self.get_provider(current_provider_name)
            except LLMConfigurationError as e:
                last_error = e
                continue

            # Bounded retry loop for current provider
            for attempt in range(self._max_retries + 1):
                try:
                    return prov.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=call_timeout,
                        model=model,
                    )
                except (LLMConfigurationError, LLMAuthenticationError):
                    # Do not retry fatal auth/config errors
                    raise
                except (LLMTimeoutError, LLMRateLimitError, LLMProviderError, LLMEmptyResponseError) as e:
                    last_error = e
                    if attempt < self._max_retries:
                        time.sleep(0.1 * (attempt + 1))  # Minor backoff
                        continue
                    break

        # If all providers/retries failed, return or raise structured error
        if last_error:
            raise last_error

        raise LLMProviderError("All providers in waterfall failed to return a response.")

    def explain_finding(
        self,
        finding_title: str,
        finding_detail: str,
        *,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Convenience helper to format and explain an already-computed specialist finding.
        Enforces that input findings are provided explicitly by deterministic specialists.
        """
        sys = system_prompt or (
            "You are FinTwin AI's financial explanation assistant. "
            "Explain the following authoritative financial finding in clear, empathetic language. "
            "Do NOT recalculate or invent numbers. Use only the provided finding metrics."
        )
        prompt = f"Finding: {finding_title}\nDetails: {finding_detail}\n\nPlease explain what this means for the user's financial journey."
        return self.generate(prompt=prompt, system_prompt=sys, provider=provider)


# Global default service instance
DEFAULT_LLM_SERVICE = LLMService()
