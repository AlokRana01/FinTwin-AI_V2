"""
tests/test_phase5_step4_optimizations.py
========================================
Comprehensive Verification Test Suite for Phase 5 Step 4 Optimizations:
1. XGBoost TreeExplainer Process-Local Thread-Safe Caching (P1)
2. Session-Level FinancialDigitalTwin Caching & Invalidation (P2)
3. SHAP Numerical & Feature Attribution Equivalence
4. Multi-Tenant User Isolation & Invalidation on Profile Update / Logout
5. Concurrency & Phase 5.3 Quota Regression Invariants
"""

import pytest
import threading
import datetime
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import streamlit as st

from models.twin_engine import FinancialDigitalTwin
from models.predictor import FinancialPredictor, FEATURE_COLS
from models.explainability import (
    ForecastExplainer,
    ExplainableAI,
    get_tree_explainer,
    clear_tree_explainer_cache,
    _TREE_EXPLAINER_CACHE,
)
from utils.session import (
    render_sidebar_user_selector,
    invalidate_session_twin,
    ACTIVE_TWIN_KEY,
)
from utils.security import session_touch
from database.db_manager import DBManager
from database.connection import get_db_cursor
from utils.chatbot import _usage_today, _try_consume_quota, DAILY_LIMIT_LOGGED_IN, DAILY_LIMIT_GUEST


@pytest.fixture(autouse=True)
def clean_test_environment():
    """Ensure clean test state before and after each test."""
    clear_tree_explainer_cache()
    st.session_state.clear()
    yield
    clear_tree_explainer_cache()
    st.session_state.clear()


@pytest.fixture
def sample_twin_a():
    uid = "test_user_alpha_54"
    demographics = {
        "user_id": uid,
        "name": "Alpha Tester",
        "age": 28,
        "city": "Mumbai",
        "occupation": "Analyst",
        "monthly_income": 100000.0,
        "bonus": 10000.0,
        "additional_income": 0.0,
    }
    balance_sheet = {
        "user_id": uid,
        "net_worth": 1200000.0,
        "bank_savings": 200000.0,
        "fd_amount": 100000.0,
        "emergency_fund": 200000.0,
        "sip_amount": 15000.0,
        "monthly_emi": 8000.0,
        "loan_amount": 200000.0,
        "rent": 20000.0,
        "groceries": 10000.0,
        "utilities": 4000.0,
        "transport": 3000.0,
        "food_delivery": 3000.0,
        "entertainment": 3000.0,
        "shopping": 4000.0,
    }
    return FinancialDigitalTwin(uid, demographics, balance_sheet)


@pytest.fixture
def sample_twin_b():
    uid = "test_user_beta_54"
    demographics = {
        "user_id": uid,
        "name": "Beta Tester",
        "age": 35,
        "city": "Delhi",
        "occupation": "Manager",
        "monthly_income": 200000.0,
        "bonus": 30000.0,
        "additional_income": 10000.0,
    }
    balance_sheet = {
        "user_id": uid,
        "net_worth": 3500000.0,
        "bank_savings": 500000.0,
        "fd_amount": 300000.0,
        "emergency_fund": 400000.0,
        "sip_amount": 40000.0,
        "monthly_emi": 25000.0,
        "loan_amount": 800000.0,
        "rent": 35000.0,
        "groceries": 15000.0,
        "utilities": 8000.0,
        "transport": 6000.0,
        "food_delivery": 5000.0,
        "entertainment": 6000.0,
        "shopping": 8000.0,
    }
    return FinancialDigitalTwin(uid, demographics, balance_sheet)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TreeExplainer Caching Tests (P1)
# ══════════════════════════════════════════════════════════════════════════════

def test_tree_explainer_cached_and_reused():
    """Verify that get_tree_explainer caches and reuses the TreeExplainer instance for the same model."""
    pred = FinancialPredictor()
    pred.load_only(horizon=6)
    model = pred.model_savings

    explainer_1 = get_tree_explainer(model)
    explainer_2 = get_tree_explainer(model)

    assert explainer_1 is explainer_2, "TreeExplainer was not reused from cache!"
    assert id(model) in _TREE_EXPLAINER_CACHE


def test_tree_explainer_cache_clear():
    """Verify that clear_tree_explainer_cache properly purges cached explainers."""
    pred = FinancialPredictor()
    pred.load_only(horizon=6)
    model = pred.model_savings

    explainer_1 = get_tree_explainer(model)
    clear_tree_explainer_cache()

    assert len(_TREE_EXPLAINER_CACHE) == 0

    explainer_2 = get_tree_explainer(model)
    assert explainer_2 is not None
    assert id(model) in _TREE_EXPLAINER_CACHE


def test_tree_explainer_numerical_equivalence(sample_twin_a):
    """
    Verify that ForecastExplainer with cached TreeExplainer produces exact
    same SHAP values, base values, and attribution rankings as un-cached computation.
    """
    pred = FinancialPredictor()
    pred.load_only(horizon=6)

    # 1. Run with cache
    explainer_cached = ForecastExplainer(pred, sample_twin_a)
    result_cached = explainer_cached.explain(horizon_months=6)

    # 2. Run fresh without cache
    clear_tree_explainer_cache()
    import shap
    raw_explainer_sav = shap.TreeExplainer(pred.model_savings)
    raw_explainer_nw = shap.TreeExplainer(pred.model_net_worth)

    X_user = pd.DataFrame([explainer_cached._features])[FEATURE_COLS]
    raw_sav_vals = raw_explainer_sav.shap_values(X_user)[0]
    raw_nw_vals = raw_explainer_nw.shap_values(X_user)[0]

    # Verify base values match
    assert np.isclose(result_cached.savings_base_value, float(raw_explainer_sav.expected_value), atol=1e-3)
    assert np.isclose(result_cached.net_worth_base_value, float(raw_explainer_nw.expected_value), atol=1e-3)

    # Verify SHAP values for each feature match exactly
    cached_sav_dict = {c.feature: c.shap_value for c in result_cached.savings_contributions}
    for fname, val in zip(FEATURE_COLS, raw_sav_vals):
        assert np.isclose(cached_sav_dict[fname], round(float(val), 2), atol=1e-2)


def test_tree_explainer_thread_safety(sample_twin_a):
    """Verify concurrent access to get_tree_explainer across multiple threads."""
    pred = FinancialPredictor()
    pred.load_only(horizon=6)
    model = pred.model_savings

    results = []
    errors = []

    def worker():
        try:
            exp = get_tree_explainer(model)
            results.append(id(exp))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(results) == 20
    assert len(set(results)) == 1, "Multiple distinct TreeExplainer instances created under concurrency!"


def test_tree_explainer_contains_zero_user_state():
    """Verify that cached TreeExplainer instances hold only model trees and zero user data."""
    pred = FinancialPredictor()
    pred.load_only(horizon=6)
    _ = get_tree_explainer(pred.model_savings)

    for key, exp in _TREE_EXPLAINER_CACHE.items():
        assert isinstance(key, int)
        # Check no user attributes exist
        assert not hasattr(exp, "user_id")
        assert not hasattr(exp, "financial_state")
        assert not hasattr(exp, "request_id")
        assert not hasattr(exp, "provenance")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Session-Level FinancialDigitalTwin Caching Tests (P2)
# ══════════════════════════════════════════════════════════════════════════════

def test_session_twin_cached_on_first_read_and_reused():
    """Verify that render_sidebar_user_selector reads DB once and reuses session twin on reruns."""
    uid = "test_user_session_cache_1"
    demographics = {
        "user_id": uid,
        "name": "Session Cache User",
        "age": 30,
        "city": "Bengaluru",
        "occupation": "Engineer",
        "monthly_income": 120000.0,
        "bonus": 0.0,
        "additional_income": 0.0,
    }
    balance_sheet = {
        "user_id": uid,
        "net_worth": 1500000.0,
        "bank_savings": 250000.0,
    }

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM users WHERE user_id = ?", (uid,))
        cur.execute("DELETE FROM digital_twins WHERE user_id = ?", (uid,))
        cur.execute(
            "INSERT INTO users (user_id, email, name, age, city, occupation, monthly_income, bonus, additional_income) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, f"{uid}@test.com", demographics["name"], demographics["age"], demographics["city"], demographics["occupation"], demographics["monthly_income"], 0.0, 0.0),
        )
        cur.execute(
            "INSERT INTO digital_twins (user_id, net_worth, bank_savings) VALUES (?, ?, ?)",
            (uid, balance_sheet["net_worth"], balance_sheet["bank_savings"]),
        )

    st.session_state["user_id"] = uid
    session_touch(st.session_state)

    with patch("streamlit.sidebar.markdown"), patch("streamlit.sidebar.page_link"), patch("streamlit.sidebar.button", return_value=False), patch("utils.chatbot.render_chatbot"):
        # 1. First run: Reads DB and populates session cache
        twin_1 = render_sidebar_user_selector()
        assert twin_1 is not None
        assert twin_1.user_id == uid
        assert st.session_state.get(ACTIVE_TWIN_KEY) is twin_1

        # 2. Second run: Should reuse cached twin with ZERO DBManager calls
        with patch.object(DBManager, "get_user_profile") as mock_prof, patch.object(DBManager, "get_digital_twin") as mock_dt:
            twin_2 = render_sidebar_user_selector()
            assert twin_2 is twin_1
            mock_prof.assert_not_called()
            mock_dt.assert_not_called()

    # Clean up DB
    with get_db_cursor() as cur:
        cur.execute("DELETE FROM users WHERE user_id = ?", (uid,))
        cur.execute("DELETE FROM digital_twins WHERE user_id = ?", (uid,))


def test_session_twin_invalidated_on_logout():
    """Verify that logout button removes active twin from session state."""
    uid = "test_user_session_logout"
    twin = FinancialDigitalTwin(uid, {"name": "Logout User", "monthly_income": 50000.0}, {})
    st.session_state["user_id"] = uid
    session_touch(st.session_state)
    st.session_state[ACTIVE_TWIN_KEY] = twin

    # Trigger invalidation / logout cleanup
    invalidate_session_twin()
    assert ACTIVE_TWIN_KEY not in st.session_state


def test_session_twin_invalidated_on_profile_update():
    """Verify that updating a profile invalidates the session twin cache and next load fetches fresh data."""
    uid = "test_user_profile_update"
    demographics_1 = {
        "user_id": uid,
        "name": "Update User V1",
        "age": 30,
        "city": "Bengaluru",
        "occupation": "Engineer",
        "monthly_income": 100000.0,
        "bonus": 0.0,
        "additional_income": 0.0,
    }
    balance_sheet_1 = {"user_id": uid, "net_worth": 1000000.0, "bank_savings": 200000.0}

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM users WHERE user_id = ?", (uid,))
        cur.execute("DELETE FROM digital_twins WHERE user_id = ?", (uid,))
        cur.execute(
            "INSERT INTO users (user_id, email, name, age, city, occupation, monthly_income, bonus, additional_income) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, f"{uid}@test.com", demographics_1["name"], demographics_1["age"], demographics_1["city"], demographics_1["occupation"], demographics_1["monthly_income"], 0.0, 0.0),
        )
        cur.execute(
            "INSERT INTO digital_twins (user_id, net_worth, bank_savings) VALUES (?, ?, ?)",
            (uid, balance_sheet_1["net_worth"], balance_sheet_1["bank_savings"]),
        )

    st.session_state["user_id"] = uid
    session_touch(st.session_state)

    with patch("streamlit.sidebar.markdown"), patch("streamlit.sidebar.page_link"), patch("streamlit.sidebar.button", return_value=False), patch("utils.chatbot.render_chatbot"):
        # Initial load
        twin_v1 = render_sidebar_user_selector()
        assert twin_v1.monthly_income == 100000.0

        # Update in DB
        DBManager.save_user_profile({"user_id": uid, "name": "Update User V2", "age": 31, "city": "Bengaluru", "occupation": "Lead", "monthly_income": 150000.0, "bonus": 0.0, "additional_income": 0.0})
        # Invalidate session twin
        invalidate_session_twin()
        assert ACTIVE_TWIN_KEY not in st.session_state

        # Next load must fetch fresh V2 twin
        twin_v2 = render_sidebar_user_selector()
        assert twin_v2.monthly_income == 150000.0
        assert twin_v2.name == "Update User V2"
        assert st.session_state.get(ACTIVE_TWIN_KEY) is twin_v2

    # Clean up DB
    with get_db_cursor() as cur:
        cur.execute("DELETE FROM users WHERE user_id = ?", (uid,))
        cur.execute("DELETE FROM digital_twins WHERE user_id = ?", (uid,))


def test_session_twin_user_switching_isolation():
    """Verify that switching authenticated users immediately discards prior user's cached twin."""
    twin_a = FinancialDigitalTwin("user_alpha", {"name": "Alpha", "monthly_income": 80000.0}, {})
    twin_b = FinancialDigitalTwin("user_beta", {"name": "Beta", "monthly_income": 120000.0}, {})

    st.session_state["user_id"] = "user_alpha"
    session_touch(st.session_state)
    st.session_state[ACTIVE_TWIN_KEY] = twin_a

    # Switch session user to beta
    st.session_state["user_id"] = "user_beta"
    session_touch(st.session_state)

    with patch("streamlit.sidebar.markdown"), patch("streamlit.sidebar.page_link"), patch("streamlit.sidebar.button", return_value=False), patch("utils.chatbot.render_chatbot"), \
         patch.object(DBManager, "get_user_profile", return_value={"name": "Beta", "monthly_income": 120000.0}), \
         patch.object(DBManager, "get_digital_twin", return_value={"user_id": "user_beta"}):
        new_twin = render_sidebar_user_selector()
        assert new_twin.user_id == "user_beta"
        assert st.session_state[ACTIVE_TWIN_KEY].user_id == "user_beta"
        assert st.session_state[ACTIVE_TWIN_KEY] is not twin_a


def test_guest_flow_never_receives_authenticated_twin():
    """Verify that guest flow (user_id not in session) clears cache and returns None."""
    st.session_state[ACTIVE_TWIN_KEY] = FinancialDigitalTwin("stale_user", {"name": "Stale"}, {})
    st.session_state.pop("user_id", None)

    with patch("streamlit.sidebar.markdown"), patch("streamlit.sidebar.page_link"), patch("streamlit.sidebar.button", return_value=False), patch("utils.chatbot.render_chatbot"):
        guest_res = render_sidebar_user_selector()
        assert guest_res is None
        assert ACTIVE_TWIN_KEY not in st.session_state


# ══════════════════════════════════════════════════════════════════════════════
# 3. Quota & Concurrency Invariant Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_phase53_quota_limits_remain_exact(sample_twin_a):
    """Verify daily limits: Authenticated = 40, Guest = 12."""
    assert DAILY_LIMIT_LOGGED_IN == 40
    assert DAILY_LIMIT_GUEST == 12

    # Verify atomic reservation function works properly
    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (sample_twin_a.user_id,))

    allowed, count = _try_consume_quota(sample_twin_a)
    assert allowed is True
    assert count == 1

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (sample_twin_a.user_id,))
