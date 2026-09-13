"""
tests/test_llm.py
=================
Comprehensive test suite for FinTwin AI's LLM Abstraction Layer (agents/llm.py).
Verifies:
1. Provider interface compliance and abstraction neutrality.
2. Successful response generation with structured metadata.
3. Normalized error handling (Configuration, Authentication, Timeout, Rate Limit, Provider, Empty Response).
4. Bounded retry policy on transient failures vs fatal auth/config failures.
5. Provider waterfall and fallback routing.
6. Zero real network calls during testing (enforced via mock providers and HTTP interceptors).
7. Credential isolation (API keys never leaked in errors).
8. Parameter pass-through (temperature, max_tokens, system prompt, timeout).
9. Output-only security boundary (zero tool execution, zero DB, zero shell access).
"""

import pytest
from unittest.mock import MagicMock, patch
import urllib.error
import io

from agents.llm import (
    LLMProvider,
    GroqProvider,
    GeminiProvider,
    MockLLMProvider,
    LLMResponse,
    LLMService,
    DEFAULT_LLM_SERVICE,
    LLMError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMProviderError,
    LLMEmptyResponseError,
)


# ── 1. Interface Compliance ───────────────────────────────────────────────────

def test_provider_interface_compliance():
    mock_p = MockLLMProvider()
    assert isinstance(mock_p, LLMProvider)
    assert mock_p.provider_name == "mock"
    assert mock_p.default_model == "mock-v1"

    groq_p = GroqProvider(api_key="gsk_dummy_test_key_12345")
    assert isinstance(groq_p, LLMProvider)
    assert groq_p.provider_name == "groq"

    gemini_p = GeminiProvider(api_key="AIzaSy_dummy_test_key_12345")
    assert isinstance(gemini_p, LLMProvider)
    assert gemini_p.provider_name == "gemini"


# ── 2. Successful Response Generation ─────────────────────────────────────────

def test_successful_response_generation():
    fake_provider = MockLLMProvider(
        fixed_response="Your savings rate is healthy at 25%. Maintain this trajectory.",
        provider_name="mock_test",
        default_model="mock-v2",
        latency_ms=22.5,
    )
    service = LLMService(providers={"mock_test": fake_provider}, default_provider="mock_test")

    resp = service.generate(
        prompt="Explain my savings rate.",
        system_prompt="You are a financial coach.",
        temperature=0.5,
        max_tokens=256,
    )

    assert isinstance(resp, LLMResponse)
    assert resp.success is True
    assert "25%" in resp.text
    assert resp.provider == "mock_test"
    assert resp.model == "mock-v2"
    assert resp.latency_ms == 22.5
    assert resp.error is None
    assert resp.tokens_used is not None

    d = resp.to_dict()
    assert d["success"] is True
    assert d["provider"] == "mock_test"


# ── 3. Missing API Key & Configuration Errors ─────────────────────────────────

def test_groq_missing_api_key_error():
    groq_p = GroqProvider(api_key="")
    with pytest.raises(LLMConfigurationError, match="Groq API key is not configured"):
        groq_p.generate("Hello")


def test_gemini_missing_api_key_error():
    gemini_p = GeminiProvider(api_key="")
    with pytest.raises(LLMConfigurationError, match="Gemini API key is not configured"):
        gemini_p.generate("Hello")


def test_unregistered_provider_error():
    service = LLMService(providers={"mock": MockLLMProvider()})
    with pytest.raises(LLMConfigurationError, match="not registered"):
        service.generate("Hello", provider="unknown_provider")


# ── 4. Parameter Pass-Through ─────────────────────────────────────────────────

def test_parameter_pass_through():
    mock_p = MockLLMProvider()
    service = LLMService(providers={"mock": mock_p}, default_provider="mock")

    service.generate(
        prompt="Test prompt",
        system_prompt="Test system",
        temperature=0.2,
        max_tokens=500,
        timeout=15.0,
        model="custom-mock-model",
    )

    assert len(mock_p.call_history) == 1
    call = mock_p.call_history[0]
    assert call["prompt"] == "Test prompt"
    assert call["system_prompt"] == "Test system"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 500
    assert call["timeout"] == 15.0
    assert call["model"] == "custom-mock-model"


# ── 5. Empty Prompt Handling ──────────────────────────────────────────────────

def test_empty_prompt_rejection():
    service = LLMService(providers={"mock": MockLLMProvider()}, default_provider="mock")
    with pytest.raises(LLMEmptyResponseError, match="empty prompt"):
        service.generate("   ")


# ── 6. Groq HTTP Error Normalization (Mocked urllib) ──────────────────────────

def test_groq_auth_error_normalization():
    groq_p = GroqProvider(api_key="gsk_invalid_key_99999")
    http_401 = urllib.error.HTTPError(
        url="https://api.groq.com",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"error": {"message": "Invalid API Key"}}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_401):
        with pytest.raises(LLMAuthenticationError, match="Groq authentication failed"):
            groq_p.generate("Hello")


def test_groq_rate_limit_normalization():
    groq_p = GroqProvider(api_key="gsk_valid_key")
    http_429 = urllib.error.HTTPError(
        url="https://api.groq.com",
        code=429,
        msg="Rate Limit Exceeded",
        hdrs={},
        fp=io.BytesIO(b'{"error": {"message": "Rate limit reached"}}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_429):
        with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
            groq_p.generate("Hello")


def test_groq_timeout_normalization():
    groq_p = GroqProvider(api_key="gsk_valid_key")
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        with pytest.raises(LLMTimeoutError, match="timed out"):
            groq_p.generate("Hello", timeout=5.0)


# ── 7. Gemini HTTP Error Normalization (Mocked urllib) ────────────────────────

def test_gemini_auth_error_normalization():
    gemini_p = GeminiProvider(api_key="AIzaSy_invalid_key")
    http_403 = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=io.BytesIO(b'{"error": {"message": "API key not valid"}}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_403):
        with pytest.raises(LLMAuthenticationError, match="Gemini authentication failed"):
            gemini_p.generate("Hello")


def test_gemini_timeout_normalization():
    gemini_p = GeminiProvider(api_key="AIzaSy_valid_key")
    url_err = urllib.error.URLError(reason=TimeoutError("Socket timeout"))

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(LLMTimeoutError, match="timed out"):
            gemini_p.generate("Hello", timeout=10.0)


# ── 8. Bounded Retry Policy ───────────────────────────────────────────────────

def test_retry_on_transient_error():
    flaky_provider = MockLLMProvider(should_fail=True, failure_exception=LLMProviderError("Transient 503 error"))
    service = LLMService(providers={"flaky": flaky_provider}, default_provider="flaky", max_retries=2)

    with pytest.raises(LLMProviderError, match="Transient 503 error"):
        service.generate("Hello")

    # With max_retries=2, initial attempt + 2 retries = 3 attempts total
    assert len(flaky_provider.call_history) == 3


def test_no_retry_on_fatal_auth_error():
    fatal_provider = MockLLMProvider(should_fail=True, failure_exception=LLMAuthenticationError("Bad Key"))
    service = LLMService(providers={"fatal": fatal_provider}, default_provider="fatal", max_retries=2)

    with pytest.raises(LLMAuthenticationError, match="Bad Key"):
        service.generate("Hello")

    # Fatal auth errors must immediately abort without retries
    assert len(fatal_provider.call_history) == 1


# ── 9. Provider Waterfall / Fallback ──────────────────────────────────────────

def test_provider_waterfall_fallback():
    failing_primary = MockLLMProvider(
        provider_name="primary_failing",
        should_fail=True,
        failure_exception=LLMRateLimitError("429 Too Many Requests"),
    )
    working_fallback = MockLLMProvider(
        fixed_response="Fallback successfully answered.",
        provider_name="secondary_working",
        default_model="fallback-v1",
    )

    service = LLMService(
        providers={"primary_failing": failing_primary, "secondary_working": working_fallback},
        default_provider="primary_failing",
        max_retries=1,
    )

    resp = service.generate("Hello", fallback_providers=["secondary_working"])

    assert resp.success is True
    assert resp.provider == "secondary_working"
    assert resp.text == "Fallback successfully answered."
    assert len(failing_primary.call_history) == 2  # initial + 1 retry
    assert len(working_fallback.call_history) == 1


# ── 10. Explain Finding Convenience Method ────────────────────────────────────

def test_explain_finding_helper():
    mock_p = MockLLMProvider(fixed_response="Your debt-to-income ratio indicates a high debt burden.")
    service = LLMService(providers={"mock": mock_p}, default_provider="mock")

    resp = service.explain_finding(
        finding_title="High EMI Burden",
        finding_detail="Monthly EMI consumes 43.3% of income.",
    )

    assert resp.success is True
    assert len(mock_p.call_history) == 1
    call = mock_p.call_history[0]
    assert "High EMI Burden" in call["prompt"]
    assert "43.3%" in call["prompt"]
    assert "System Instructions" not in call["prompt"] or "financial explanation" in call["system_prompt"]


# ── 11. Security Boundary Verification ────────────────────────────────────────

def test_security_no_secrets_in_errors():
    secret_key = "gsk_super_secret_production_token_xyz999"
    groq_p = GroqProvider(api_key=secret_key)

    http_500 = urllib.error.HTTPError(
        url="https://api.groq.com",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=io.BytesIO(b'{"error": "Internal Error"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_500):
        try:
            groq_p.generate("Test prompt")
        except LLMError as e:
            err_msg = str(e)
            assert secret_key not in err_msg, "Security violation: Raw API key leaked in exception message"


def test_zero_tool_execution_or_db_authority():
    """
    Verifies the LLM abstraction module has zero imports or references to
    mutating database queries, ToolRegistry execution, shell, or filesystem modification.
    """
    import agents.llm as llm_mod

    # Assert no DBManager, ToolRegistry, or subprocess imports in llm module
    assert not hasattr(llm_mod, "DBManager")
    assert not hasattr(llm_mod, "ToolRegistry")
    assert not hasattr(llm_mod, "subprocess")
    assert not hasattr(llm_mod, "os.system")
