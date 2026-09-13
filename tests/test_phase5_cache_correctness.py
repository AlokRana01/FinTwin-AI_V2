"""
tests/test_phase5_cache_correctness.py
======================================
Exhaustive verification test suite for Phase 5.2 cache correctness, concurrency,
mutability safety, cross-user isolation, scenario isolation, and numerical equivalence.
"""

import concurrent.futures
import copy
import time
import pytest
from unittest.mock import patch

from models.predictor import FinancialPredictor, _MODEL_CACHE
from models.clustering import FinancialPersonalityClusterer, _CLUSTERER_CACHE
from config import PERSONALITY_LABELS
from models.twin_engine import FinancialDigitalTwin, HealthScoreEngine, _compute_health_score_from_features
from utils.tax_calculator import IndianTaxCalculator, DeductionProfile
from utils.goal_engine import GoalEngine, GoalInput, GoalType
from agents.tool_registry import DEFAULT_TOOL_REGISTRY
from agents.schemas import ComputationType


@pytest.fixture
def twin_user_a():
    demographics = {
        "name": "User Alpha",
        "age": 29,
        "occupation": "Product Manager",
        "city": "Bengaluru",
        "monthly_income": 150000.0,
    }
    balance_sheet = {
        "bank_savings": 350000.0,
        "fd_amount": 200000.0,
        "emergency_fund": 450000.0,
        "mutual_funds": 800000.0,
        "stocks": 300000.0,
        "ppf_investment": 150000.0,
        "sip_amount": 35000.0,
        "monthly_emi": 25000.0,
        "loan_amount": 750000.0,
        "car_loan": 0.0,
        "credit_card_debt": 0.0,
        "health_insurance": 15000.0,
        "life_insurance": 20000.0,
        "rent": 35000.0,
        "groceries": 15000.0,
        "utilities": 7000.0,
        "transport": 5000.0,
        "food_delivery": 6000.0,
        "entertainment": 5000.0,
        "shopping": 7000.0,
    }
    return FinancialDigitalTwin(user_id="user_alpha", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def twin_user_b():
    demographics = {
        "name": "User Beta",
        "age": 34,
        "occupation": "Tech Lead",
        "city": "Hyderabad",
        "monthly_income": 220000.0,
    }
    balance_sheet = {
        "bank_savings": 600000.0,
        "fd_amount": 400000.0,
        "emergency_fund": 700000.0,
        "mutual_funds": 1500000.0,
        "stocks": 700000.0,
        "ppf_investment": 250000.0,
        "sip_amount": 60000.0,
        "monthly_emi": 40000.0,
        "loan_amount": 1500000.0,
        "car_loan": 0.0,
        "credit_card_debt": 0.0,
        "health_insurance": 25000.0,
        "life_insurance": 30000.0,
        "rent": 45000.0,
        "groceries": 20000.0,
        "utilities": 10000.0,
        "transport": 8000.0,
        "food_delivery": 8000.0,
        "entertainment": 7000.0,
        "shopping": 10000.0,
    }
    return FinancialDigitalTwin(user_id="user_beta", demographics=demographics, balance_sheet=balance_sheet)


# ── 1. Model Cache Correctness ───────────────────────────────────────────────

def test_model_cache_cold_vs_warm_and_numerical_parity(twin_user_a):
    FinancialPredictor.clear_cache()

    # Cold load
    p_cold = FinancialPredictor()
    t_start_cold = time.perf_counter()
    loaded_cold = p_cold.load_only(horizon=6)
    t_cold_ms = (time.perf_counter() - t_start_cold) * 1000.0

    # Warm load
    p_warm = FinancialPredictor()
    t_start_warm = time.perf_counter()
    loaded_warm = p_warm.load_only(horizon=6)
    t_warm_ms = (time.perf_counter() - t_start_warm) * 1000.0

    if loaded_cold:
        assert loaded_warm is True
        # Warm load must be dramatically faster than cold load
        assert t_warm_ms < t_cold_ms or t_warm_ms < 1.0

        # Numerical prediction output must match exactly
        pred_cold = p_cold.predict(twin_user_a, horizon_months=6)
        pred_warm = p_warm.predict(twin_user_a, horizon_months=6)
        assert pred_cold["predicted_savings"] == pred_warm["predicted_savings"]
        assert pred_cold["predicted_net_worth"] == pred_warm["predicted_net_worth"]


# ── 2. Personality Cache Correctness ─────────────────────────────────────────

def test_personality_cache_canonical_labels_and_parity(twin_user_a, twin_user_b):
    FinancialPersonalityClusterer.clear_cache()

    c1 = FinancialPersonalityClusterer.load()
    c2 = FinancialPersonalityClusterer.load()

    label_a1 = c1.predict_personality(twin_user_a)
    label_a2 = c2.predict_personality(twin_user_a)
    label_b = c2.predict_personality(twin_user_b)

    assert label_a1 == label_a2
    assert label_a1 in PERSONALITY_LABELS
    assert label_b in PERSONALITY_LABELS


# ── 3. Health Score Memoization: Cases A through H ───────────────────────────

def test_health_score_memoization_cases_a_through_h(twin_user_a):
    HealthScoreEngine.clear_cache()
    base_engine = HealthScoreEngine(twin_user_a)
    base_res = base_engine.compute_overall_health_score()
    base_score = base_res["overall_score"]

    # Case A: Same inputs twice -> cache hit & exact same result
    cache_info_1 = _compute_health_score_from_features.cache_info()
    res_a2 = base_engine.compute_overall_health_score()
    cache_info_2 = _compute_health_score_from_features.cache_info()
    assert res_a2["overall_score"] == base_score
    assert cache_info_2.hits > cache_info_1.hits

    # Case B: Change income -> recalculates
    twin_b = copy.deepcopy(twin_user_a)
    twin_b.monthly_income = 250000.0
    twin_b._recompute_derived()
    score_b = HealthScoreEngine(twin_b).compute_overall_health_score()["overall_score"]
    assert score_b != base_score

    # Case C: Change basic expenses -> recalculates
    twin_c = copy.deepcopy(twin_user_a)
    twin_c.rent = 80000.0
    twin_c._recompute_derived()
    score_c = HealthScoreEngine(twin_c).compute_overall_health_score()["overall_score"]
    assert score_c != base_score

    # Case D: Change EMI -> recalculates
    twin_d = copy.deepcopy(twin_user_a)
    twin_d.monthly_emi = 60000.0
    twin_d._recompute_derived()
    score_d = HealthScoreEngine(twin_d).compute_overall_health_score()["overall_score"]
    assert score_d != base_score

    # Case E: Change investments (SIP) -> recalculates
    twin_e = copy.deepcopy(twin_user_a)
    twin_e.sip_amount = 5000.0
    twin_e._recompute_derived()
    score_e = HealthScoreEngine(twin_e).compute_overall_health_score()["overall_score"]
    assert score_e != base_score

    # Case F: Change emergency fund -> recalculates
    twin_f = copy.deepcopy(twin_user_a)
    twin_f.emergency_fund = 50000.0
    twin_f._recompute_derived()
    score_f = HealthScoreEngine(twin_f).compute_overall_health_score()["overall_score"]
    assert score_f != base_score

    # Case G: Change life insurance -> recalculates
    twin_g = copy.deepcopy(twin_user_a)
    twin_g.life_insurance = 18000000.0  # Full 120x cover
    twin_g._recompute_derived()
    score_g = HealthScoreEngine(twin_g).compute_overall_health_score()["overall_score"]
    assert score_g != base_score

    # Case H: Change loan amount (debt) -> recalculates
    twin_h = copy.deepcopy(twin_user_a)
    twin_h.loan_amount = 5000000.0
    twin_h._recompute_derived()
    score_h = HealthScoreEngine(twin_h).compute_overall_health_score()["overall_score"]
    assert score_h != base_score


# ── 4. Tax Memoization Verification ──────────────────────────────────────────

def test_tax_memoization_deduction_invalidation():
    IndianTaxCalculator.clear_cache()
    calc = IndianTaxCalculator()

    ded_base = DeductionProfile(investment_80c=150000, health_insurance_self=25000)
    res_base = calc.compare_and_optimize(1500000, ded_base)

    # Invalidate 80D parents
    ded_parents = DeductionProfile(investment_80c=150000, health_insurance_self=25000, health_insurance_parents=50000)
    res_parents = calc.compare_and_optimize(1500000, ded_parents)
    assert res_parents["old_regime"].total_tax < res_base["old_regime"].total_tax

    # Invalidate 80CCD(1B)
    ded_nps = DeductionProfile(investment_80c=150000, health_insurance_self=25000, nps_80ccd1b=50000)
    res_nps = calc.compare_and_optimize(1500000, ded_nps)
    assert res_nps["old_regime"].total_tax < res_base["old_regime"].total_tax

    # Invalidate Sec 24b Home Loan
    ded_hl = DeductionProfile(investment_80c=150000, health_insurance_self=25000, home_loan_interest=200000)
    res_hl = calc.compare_and_optimize(1500000, ded_hl)
    assert res_hl["old_regime"].total_tax < res_base["old_regime"].total_tax


# ── 5. Scenario Isolation Verification ───────────────────────────────────────

def test_scenario_isolation_car_and_sip_deltas(twin_user_a):
    registry = DEFAULT_TOOL_REGISTRY

    # 1. Baseline
    auth_base = registry.execute("calculate_health_score", twin=twin_user_a, request_id="req-base")

    # 2. Scenario A: Car purchase 8 Lakh (loan + EMI added)
    twin_car_8l = copy.deepcopy(twin_user_a)
    twin_car_8l.car_loan = 800000.0
    twin_car_8l.loan_amount += 800000.0
    twin_car_8l.monthly_emi += 18000.0
    twin_car_8l._recompute_derived()
    auth_car_8l = registry.execute("calculate_health_score", twin=twin_car_8l, request_id="req-car8")

    # 3. Scenario B: Car purchase 10 Lakh (loan + EMI added)
    twin_car_10l = copy.deepcopy(twin_user_a)
    twin_car_10l.car_loan = 1000000.0
    twin_car_10l.loan_amount += 1000000.0
    twin_car_10l.monthly_emi += 22500.0
    twin_car_10l._recompute_derived()
    auth_car_10l = registry.execute("calculate_health_score", twin=twin_car_10l, request_id="req-car10")

    # 4. Scenario A again: Car purchase 8 Lakh
    auth_car_8l_again = registry.execute("calculate_health_score", twin=twin_car_8l, request_id="req-car8-again")

    # Assertions
    assert auth_car_8l.metrics["health_score"] != auth_base.metrics["health_score"]
    assert auth_car_10l.metrics["health_score"] != auth_car_8l.metrics["health_score"]
    assert auth_car_8l_again.metrics["health_score"] == auth_car_8l.metrics["health_score"]
    assert auth_car_8l_again.provenance.request_id == "req-car8-again"
    assert auth_car_8l.provenance.request_id == "req-car8"


# ── 6. Provenance & Request ID Freshness ─────────────────────────────────────

def test_provenance_fresh_request_ids_on_cached_hits(twin_user_a):
    registry = DEFAULT_TOOL_REGISTRY

    req_ids = [f"req-audit-test-{i}" for i in range(10)]
    provenance_records = []

    for rid in req_ids:
        data = registry.execute("calculate_health_score", twin=twin_user_a, request_id=rid)
        provenance_records.append(data.provenance)

    # Every single execution must carry its OWN request_id
    returned_req_ids = [p.request_id for p in provenance_records]
    assert returned_req_ids == req_ids
    assert len(set(returned_req_ids)) == 10
    assert all(p.source_engine == "HealthScoreEngine" for p in provenance_records)
    assert all(p.calculation_version == "v1.2.0-canonical" for p in provenance_records)


# ── 7. Mutability & Defensive Copy Safety ────────────────────────────────────

def test_cached_dictionary_mutation_immunity(twin_user_a):
    engine = HealthScoreEngine(twin_user_a)

    # 1. Fetch result and mutate the returned dictionary
    res1 = engine.compute_overall_health_score()
    res1["overall_score"] = -999.0
    res1["components"]["savings_rate"]["score"] = -999.0

    # 2. Fetch second result from cache
    res2 = engine.compute_overall_health_score()

    # The cached structure inside the engine must NOT have been corrupted
    assert res2["overall_score"] != -999.0
    assert res2["components"]["savings_rate"]["score"] != -999.0


def test_tax_cached_dictionary_mutation_immunity():
    calc = IndianTaxCalculator()
    ded = DeductionProfile(investment_80c=150000)

    res1 = calc.compare_and_optimize(1200000, ded)
    res1["tips"].append("POISONED TIP")
    res1["annual_savings"] = -1.0

    res2 = calc.compare_and_optimize(1200000, ded)
    assert "POISONED TIP" not in res2["tips"]
    assert res2["annual_savings"] != -1.0


# ── 8. Concurrency & Multi-Threaded Safety ───────────────────────────────────

def test_multithreaded_concurrent_cache_access(twin_user_a, twin_user_b):
    def worker(twin, req_id):
        reg = DEFAULT_TOOL_REGISTRY
        h = reg.execute("calculate_health_score", twin=twin, request_id=req_id)
        t = reg.execute("calculate_tax", gross_income=twin.total_income * 12, request_id=req_id)
        p = reg.execute("classify_financial_personality", twin=twin, request_id=req_id)
        return h.provenance.request_id, t.provenance.request_id, p.provenance.request_id

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(40):
            twin = twin_user_a if i % 2 == 0 else twin_user_b
            rid = f"req-concurrent-{i}"
            tasks.append(executor.submit(worker, twin, rid))

        results = [t.result() for t in concurrent.futures.as_completed(tasks)]

    assert len(results) == 40
    for h_rid, t_rid, p_rid in results:
        assert h_rid == t_rid == p_rid
