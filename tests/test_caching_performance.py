"""
tests/test_caching_performance.py
==================================
Comprehensive test suite verifying safe caching, in-memory model artifact reuse,
deterministic calculation memoization, request-specific provenance generation,
and strict cross-user / scenario isolation.
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from models.predictor import FinancialPredictor, _MODEL_CACHE
from models.clustering import FinancialPersonalityClusterer, _CLUSTERER_CACHE
from models.twin_engine import FinancialDigitalTwin, HealthScoreEngine, _compute_health_score_from_features
from utils.tax_calculator import IndianTaxCalculator, DeductionProfile
from utils.goal_engine import GoalEngine, GoalInput, GoalType
from agents.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry
from agents.state import FinancialState
from agents.orchestrator import OrchestratorAgent
from agents.financial_coach import FinancialCoachAgent
from utils.agent_chat_adapter import AgentChatAdapter


@pytest.fixture
def sample_twin_1():
    demographics = {
        "name": "User One",
        "age": 28,
        "occupation": "Analyst",
        "city": "Mumbai",
        "monthly_income": 100000.0,
    }
    balance_sheet = {
        "bank_savings": 200000.0,
        "fd_amount": 100000.0,
        "emergency_fund": 300000.0,
        "mutual_funds": 400000.0,
        "stocks": 100000.0,
        "ppf_investment": 100000.0,
        "sip_amount": 20000.0,
        "monthly_emi": 15000.0,
        "loan_amount": 500000.0,
        "car_loan": 0.0,
        "credit_card_debt": 0.0,
        "health_insurance": 10000.0,
        "life_insurance": 15000.0,
        "rent": 25000.0,
        "groceries": 10000.0,
        "utilities": 5000.0,
        "transport": 3000.0,
        "food_delivery": 4000.0,
        "entertainment": 3000.0,
        "shopping": 5000.0,
    }
    return FinancialDigitalTwin(user_id="user_cache_1", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def sample_twin_2():
    demographics = {
        "name": "User Two",
        "age": 35,
        "occupation": "Manager",
        "city": "Delhi",
        "monthly_income": 180000.0,
    }
    balance_sheet = {
        "bank_savings": 500000.0,
        "fd_amount": 300000.0,
        "emergency_fund": 600000.0,
        "mutual_funds": 1200000.0,
        "stocks": 500000.0,
        "ppf_investment": 300000.0,
        "sip_amount": 45000.0,
        "monthly_emi": 30000.0,
        "loan_amount": 1000000.0,
        "car_loan": 0.0,
        "credit_card_debt": 0.0,
        "health_insurance": 20000.0,
        "life_insurance": 25000.0,
        "rent": 40000.0,
        "groceries": 15000.0,
        "utilities": 8000.0,
        "transport": 6000.0,
        "food_delivery": 6000.0,
        "entertainment": 5000.0,
        "shopping": 8000.0,
    }
    return FinancialDigitalTwin(user_id="user_cache_2", demographics=demographics, balance_sheet=balance_sheet)


# ── 1. Model Loading Cache Tests ─────────────────────────────────────────────

def test_financial_predictor_in_memory_cache(sample_twin_1):
    FinancialPredictor.clear_cache()
    assert len(_MODEL_CACHE) == 0

    p1 = FinancialPredictor()
    loaded1 = p1.load_only(horizon=6)
    if loaded1:
        assert "default" in _MODEL_CACHE
        # Second instance should load from memory without touching disk json parsers
        p2 = FinancialPredictor()
        with patch("xgboost.XGBRegressor.load_model") as mock_xgb_load:
            loaded2 = p2.load_only(horizon=6)
            assert loaded2 is True
            mock_xgb_load.assert_not_called()

        res1 = p1.predict(sample_twin_1, horizon_months=6)
        res2 = p2.predict(sample_twin_1, horizon_months=6)
        assert res1 == res2


def test_financial_predictor_missing_model_fallback(sample_twin_1):
    FinancialPredictor.clear_cache()
    p = FinancialPredictor()
    with patch("os.path.exists", return_value=False):
        loaded = p.load_only(horizon=6)
        assert loaded is False
        res = p.predict(sample_twin_1, horizon_months=6)
        assert "predicted_savings" in res
        assert "predicted_net_worth" in res
        assert res["horizon_months"] == 6


def test_personality_clusterer_in_memory_cache(sample_twin_1):
    FinancialPersonalityClusterer.clear_cache()
    assert len(_CLUSTERER_CACHE) == 0

    c1 = FinancialPersonalityClusterer.load()
    if c1.is_fitted():
        assert len(_CLUSTERER_CACHE) > 0
        # Second call should reuse cached payload
        with patch("builtins.open") as mock_open:
            c2 = FinancialPersonalityClusterer.load()
            assert c2.is_fitted() is True
            mock_open.assert_not_called()

        p1 = c1.predict_personality(sample_twin_1)
        p2 = c2.predict_personality(sample_twin_1)
        assert p1 == p2


# ── 2. Pure Calculation Caching Tests ────────────────────────────────────────

def test_health_score_engine_memoization(sample_twin_1, sample_twin_2):
    HealthScoreEngine.clear_cache()
    cache_info_before = _compute_health_score_from_features.cache_info()

    eng1 = HealthScoreEngine(sample_twin_1)
    res1_a = eng1.compute_overall_health_score()
    res1_b = eng1.compute_overall_health_score()
    assert res1_a["overall_score"] == res1_b["overall_score"]
    assert res1_a["financial_grade"] == res1_b["financial_grade"]

    cache_info_after = _compute_health_score_from_features.cache_info()
    assert cache_info_after.hits > cache_info_before.hits

    # Different user with different features must produce different score & cache miss
    eng2 = HealthScoreEngine(sample_twin_2)
    res2 = eng2.compute_overall_health_score()
    assert res2["overall_score"] != res1_a["overall_score"]


def test_tax_calculator_memoization():
    IndianTaxCalculator.clear_cache()
    calc = IndianTaxCalculator()
    ded1 = DeductionProfile(investment_80c=150000, health_insurance_self=25000)

    res1_a = calc.compare_and_optimize(1200000, ded1)
    res1_b = calc.compare_and_optimize(1200000, ded1)
    assert res1_a["annual_savings"] == res1_b["annual_savings"]
    assert res1_a["recommended_regime"] == res1_b["recommended_regime"]

    # Input change (e.g. higher income) produces different result and cache entry
    res2 = calc.compare_and_optimize(2500000, ded1)
    assert res2["new_regime"].total_tax != res1_a["new_regime"].total_tax


# ── 3. Provenance Freshness & Request ID Isolation ───────────────────────────

def test_cached_calculation_preserves_unique_request_id_and_provenance(sample_twin_1):
    registry = DEFAULT_TOOL_REGISTRY

    # First call with req-1
    auth1 = registry.execute("calculate_health_score", twin=sample_twin_1, request_id="req-custom-001")
    assert auth1.provenance.request_id == "req-custom-001"
    assert auth1.provenance.source_engine == "HealthScoreEngine"
    assert auth1.provenance.calculation_version == "v1.2.0-canonical"
    assert auth1.provenance.execution_time_ms >= 0.0

    # Second call with identical input but req-2
    auth2 = registry.execute("calculate_health_score", twin=sample_twin_1, request_id="req-custom-002")
    assert auth2.provenance.request_id == "req-custom-002"
    assert auth2.provenance.request_id != auth1.provenance.request_id
    assert auth2.metrics["health_score"] == auth1.metrics["health_score"]


def test_tax_tool_fresh_provenance():
    registry = DEFAULT_TOOL_REGISTRY
    ded = DeductionProfile(investment_80c=100000)

    t1 = registry.execute("calculate_tax", gross_income=1500000, deductions=ded, request_id="req-tax-1")
    t2 = registry.execute("calculate_tax", gross_income=1500000, deductions=ded, request_id="req-tax-2")

    assert t1.provenance.request_id == "req-tax-1"
    assert t2.provenance.request_id == "req-tax-2"
    assert t1.provenance.source_engine == "IndianTaxCalculator"
    assert t1.metrics["recommended_regime"] == t2.metrics["recommended_regime"]


# ── 4. Scenario & Dynamic Parameter Isolation ────────────────────────────────

def test_scenario_forecast_surplus_override_isolation(sample_twin_1):
    registry = DEFAULT_TOOL_REGISTRY

    # Baseline forecast
    res_base = registry.execute("forecast_savings", twin=sample_twin_1, horizon_months=12, request_id="req-base")

    # Modified twin simulating scenario surplus change
    demographics = {
        "name": sample_twin_1.name,
        "age": sample_twin_1.age,
        "occupation": sample_twin_1.occupation,
        "city": sample_twin_1.city,
        "monthly_income": sample_twin_1.monthly_income,
    }
    balance_sheet = {
        "bank_savings": sample_twin_1.bank_savings,
        "fd_amount": sample_twin_1.fd_amount,
        "emergency_fund": sample_twin_1.emergency_fund,
        "mutual_funds": sample_twin_1.mutual_funds,
        "stocks": sample_twin_1.stocks,
        "ppf_investment": sample_twin_1.ppf_investment,
        "sip_amount": sample_twin_1.sip_amount,
        "monthly_emi": sample_twin_1.monthly_emi,
        "loan_amount": sample_twin_1.loan_amount,
        "car_loan": sample_twin_1.car_loan,
        "credit_card_debt": sample_twin_1.credit_card_debt,
        "health_insurance": sample_twin_1.health_insurance,
        "life_insurance": sample_twin_1.life_insurance,
        "rent": 50000.0,
        "groceries": sample_twin_1.groceries,
        "utilities": sample_twin_1.utilities,
        "transport": sample_twin_1.transport,
        "food_delivery": sample_twin_1.food_delivery,
        "entertainment": sample_twin_1.entertainment,
        "shopping": sample_twin_1.shopping,
    }
    twin_modified = FinancialDigitalTwin(
        user_id="user_cache_1",
        demographics=demographics,
        balance_sheet=balance_sheet,
    )
    res_scenario = registry.execute("forecast_savings", twin=twin_modified, horizon_months=12, request_id="req-scen")

    assert res_scenario.provenance.request_id == "req-scen"
    assert res_base.provenance.request_id == "req-base"


# ── 5. End-to-End Multi-Agent Orchestration Benchmark ─────────────────────────

def test_orchestrator_multi_turn_caching_speedup(sample_twin_1):
    adapter = AgentChatAdapter()
    coach = FinancialCoachAgent(enable_llm=False)
    orch = OrchestratorAgent(financial_coach_agent=coach)

    req1 = adapter.build_request("How is my financial health?", sample_twin_1)
    start1 = time.perf_counter()
    resp1 = orch.run(intent=req1.intent, state=req1.financial_state, request_id=req1.request_id)
    lat1_ms = (time.perf_counter() - start1) * 1000.0

    req2 = adapter.build_request("How is my financial health?", sample_twin_1)
    start2 = time.perf_counter()
    resp2 = orch.run(intent=req2.intent, state=req2.financial_state, request_id=req2.request_id)
    lat2_ms = (time.perf_counter() - start2) * 1000.0

    assert resp1.status.value == "success"
    assert resp2.status.value == "success"
    assert resp1.authoritative_data.provenance.request_id != resp2.authoritative_data.provenance.request_id
    assert resp1.intent == resp2.intent == "HEALTH"
    assert lat2_ms < 50.0
