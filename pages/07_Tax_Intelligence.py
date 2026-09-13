"""
Streamlit Page: Tax Intelligence  (Module 8)
========================================
Full Indian Tax Intelligence dashboard — Old vs New Regime optimization,
deduction analysis, waterfall charts, regime recommendation, and actionable tax-saving strategies.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.session import render_sidebar_user_selector
from utils.tax_calculator import IndianTaxCalculator, DeductionProfile
from utils.pdf_report import FinTwinPDFReport

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tax Intelligence — FinTwin AI",
    page_icon="assets/logos/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom Scoped Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tab Bar Spacing */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    margin-bottom: 1.75rem;
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    border-radius: 8px;
}

/* Page Header */
.tax-header-box {
    margin-bottom: 2.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.tax-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.03em;
}
.tax-subtitle {
    color: #94A3B8;
    font-size: 0.96rem;
    margin: 0;
    line-height: 1.55;
}

/* Section Header */
.tax-section-header {
    margin: 2.25rem 0 1.25rem 0;
}
.tax-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.tax-section-desc {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 0 0 1.15rem 0;
    line-height: 1.5;
}

/* Recommendation Card */
.tax-rec-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, #111827 100%);
    border: 1.5px solid rgba(16, 185, 129, 0.45);
    border-radius: 14px;
    padding: 1.6rem 1.85rem;
    margin: 1.5rem 0 2rem 0;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
}
.tax-rec-card-blue {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, #111827 100%);
    border: 1.5px solid rgba(79, 140, 255, 0.45);
    border-radius: 14px;
    padding: 1.6rem 1.85rem;
    margin: 1.5rem 0 2rem 0;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
}

/* Metric Display Boxes */
.tax-metric-box {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    min-height: 98px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.tax-metric-box:hover {
    border-color: rgba(79, 140, 255, 0.4);
    transform: translateY(-1px);
}
.tax-metric-box-featured {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(79, 140, 255, 0.45);
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    min-height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
}
.tax-metric-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    color: #94A3B8;
    margin-bottom: 0.45rem;
}
.tax-metric-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.25;
    margin-bottom: 0.3rem;
}
.tax-metric-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}

/* Narrative / Callout Box */
.tax-narrative-box {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 1.35rem 1.6rem;
    margin: 1.25rem 0 2.25rem 0;
    color: #E2E8F0;
    font-size: 0.94rem;
    line-height: 1.65;
}

/* Tip Accent Box */
.tax-tip-box {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #3B82F6;
    border-radius: 10px;
    padding: 1rem 1.35rem;
    margin-bottom: 0.85rem;
    color: #E2E8F0;
    font-size: 0.92rem;
    line-height: 1.55;
}

/* Headroom Pill */
.tax-headroom-banner {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1.25rem 0;
    color: #F8FAFC;
    font-size: 0.92rem;
    line-height: 1.55;
}
.tax-headroom-banner-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1.25rem 0;
    color: #F8FAFC;
    font-size: 0.92rem;
    line-height: 1.55;
}

/* Expanders Spacing */
div[data-testid="stExpander"] {
    margin-top: 1.25rem !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    background: #111827 !important;
}

/* Bordered Action & Report Banners */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.15rem !important;
}

/* Download Button Polish */
div[data-testid="stDownloadButton"] {
    display: flex;
    align-items: center;
    justify-content: center;
}
div[data-testid="stDownloadButton"] > button {
    background: #4F8CFF !important;
    color: #FFFFFF !important;
    border: 1px solid #4F8CFF !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.25rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 4px 14px rgba(79, 140, 255, 0.3) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #3B7BF6 !important;
    border-color: #3B7BF6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79, 140, 255, 0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar user selector ──────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tax-header-box">
    <h1 class="tax-title">Tax Intelligence</h1>
    <p class="tax-subtitle">Understand your tax liability, compare tax regimes, and identify potential savings for FY 2025-26 (AY 2026-27).</p>
</div>
""", unsafe_allow_html=True)

if not twin:
    st.markdown("""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; text-align: center; padding: 2.5rem 1.5rem; max-width: 680px; margin: 2rem auto; box-shadow: 0 4px 14px rgba(0,0,0,0.2);">
        <i class="fa-solid fa-user-gear" style="color: #4F8CFF; font-size: 2.2rem; margin-bottom: 0.85rem; display: inline-block;"></i>
        <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.5rem 0;">
            Complete Your Financial Profile
        </h2>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5; margin: 0 0 1.5rem 0;">
            Add your financial information to load your personalized tax profile and compute optimal regime recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1.4, 1])
    with c_btn2:
        st.page_link("pages/01_Digital_Twin.py", label="Complete Financial Profile →", icon=":material/arrow_forward:", use_container_width=True)
    st.stop()

# ── Prefill helper from twin profile ──────────────────────────────────────────
def _prefill(attr: str, default: float) -> float:
    if twin:
        return float(getattr(twin, attr, default))
    return default

annual_income_default = _prefill("monthly_income", 100000) * 12
nps_default           = _prefill("nps_investment", 0) / 12 * 12  # yearly
insurance_default     = _prefill("health_insurance", 0)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: INCOME & DEDUCTIONS (TAX PROFILE)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="tax-section-header" style="margin-top: 0.5rem;">
    <h2 class="tax-section-title"><i class="fa-solid fa-sliders" style="color: #4F8CFF; font-size: 1.1rem;"></i> Income & Deductions Profile</h2>
    <p class="tax-section-desc">Adjust your gross compensation and eligible tax deductions to evaluate both regimes in real time.</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("""
    <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; margin-bottom: 0.75rem;">
        Income Profile
    </div>
    """, unsafe_allow_html=True)
    
    col_inc, col_meta = st.columns([2.2, 1.8])
    with col_inc:
        annual_income = st.number_input(
            "Gross Annual Income (₹)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=float(round(annual_income_default / 50000) * 50000),
            step=50000.0,
            help="Total gross CTC or annual salary before standard deduction or exemptions.",
            key="tax_annual_income"
        )
    with col_meta:
        monthly_approx = annual_income / 12
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.85rem 1.1rem; margin-top: 1.7rem;">
            <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600;">Monthly Gross Equivalent</div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">
                ₹{monthly_approx:,.0f}<span style="font-size: 0.8rem; color: #64748B; font-weight: 500;"> /month</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; margin: 1.5rem 0 0.75rem 0; padding-top: 1.25rem; border-top: 1px solid rgba(255, 255, 255, 0.08);">
        Old Regime Deductions
    </div>
    """, unsafe_allow_html=True)

    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        invest_80c = st.number_input(
            "80C — PPF / ELSS / EPF / LIC (₹)",
            min_value=0.0,
            max_value=150000.0,
            value=min(float(_prefill("ppf_investment", 0)), 150000.0),
            step=5000.0,
            help="Total 80C investments — statutory cap ₹1,50,000.",
            key="tax_80c"
        )

        nps_extra = st.number_input(
            "80CCD(1B) — NPS Extra (₹)",
            min_value=0.0,
            max_value=50000.0,
            value=min(float(nps_default), 50000.0),
            step=5000.0,
            help="Additional NPS contribution over the 80C limit. Max ₹50,000.",
            key="tax_80ccd1b"
        )

    with dc2:
        health_ins_self = st.number_input(
            "80D — Health Ins. Self + Family (₹)",
            min_value=0.0,
            max_value=25000.0,
            value=min(float(insurance_default), 25000.0) if insurance_default <= 500000 else 25000.0,
            step=1000.0,
            help="Medical insurance premium for yourself and family. Max ₹25,000.",
            key="tax_80d_self"
        )

        health_ins_parents = st.number_input(
            "80D — Health Ins. Parents (₹)",
            min_value=0.0,
            max_value=50000.0,
            value=0.0,
            step=1000.0,
            help="Medical insurance premium for parents. Max ₹50,000.",
            key="tax_80d_parents"
        )

    with dc3:
        home_loan_interest = st.number_input(
            "Sec 24(b) — Home Loan Interest (₹)",
            min_value=0.0,
            max_value=200000.0,
            value=0.0,
            step=10000.0,
            help="Annual home loan interest for self-occupied property. Max ₹2,00,000.",
            key="tax_24b"
        )

        other_deductions = st.number_input(
            "Other Deductions (80E, 80G, 80TTA…) (₹)",
            min_value=0.0,
            max_value=500000.0,
            value=0.0,
            step=5000.0,
            help="Education loan interest (80E), donations (80G), savings interest (80TTA), etc.",
            key="tax_other"
        )

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
deductions = DeductionProfile(
    investment_80c           = invest_80c,
    health_insurance_self    = health_ins_self,
    health_insurance_parents = health_ins_parents,
    nps_80ccd1b              = nps_extra,
    home_loan_interest       = home_loan_interest,
    other_deductions         = other_deductions,
)

calc   = IndianTaxCalculator()
result = calc.compare_and_optimize(annual_income, deductions)
old    = result["old_regime"]
new    = result["new_regime"]
rec    = result["recommended_regime"]
savings = result["annual_savings"]
monthly_savings = result["monthly_savings"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PRIMARY RESULT — RECOMMENDED REGIME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="tax-section-header">
    <h2 class="tax-section-title"><i class="fa-solid fa-award" style="color: #10B981; font-size: 1.15rem;"></i> Optimal Regime Recommendation</h2>
    <p class="tax-section-desc">Recommended tax framework based on your current deductions and gross income profile.</p>
</div>
""", unsafe_allow_html=True)

rec_card_class = "tax-rec-card" if rec == "Old Regime" else "tax-rec-card-blue"
badge_color    = "#10B981" if rec == "Old Regime" else "#4F8CFF"
active_icon    = "fa-solid fa-circle-check"

rec_html = f"""<div class="{rec_card_class}"><div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 1.25rem;"><div style="display: flex; align-items: center; gap: 10px;"><span style="background: {badge_color}22; border: 1px solid {badge_color}; color: {badge_color}; border-radius: 8px; padding: 4px 14px; font-weight: 700; font-size: 0.88rem; display: inline-flex; align-items: center; gap: 6px;"><i class="{active_icon}"></i> Recommended</span><span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.55rem; font-weight: 800; color: #F8FAFC;">{rec}</span></div><div style="text-align: right;"><div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700;">Estimated Annual Savings</div><div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.65rem; font-weight: 800; color: {badge_color};">₹{savings:,.0f}</div></div></div><p style="color: #CBD5E1; font-size: 0.96rem; line-height: 1.6; margin: 0 0 1.25rem 0;">{result['recommendation_text']}</p><div style="display: flex; flex-wrap: wrap; gap: 16px; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.9rem; color: #94A3B8;"><span style="display: inline-flex; align-items: center; gap: 6px;"><i class="fa-solid fa-wallet" style="color: {badge_color};"></i> Monthly Tax Savings: <strong style="color: #F8FAFC;">₹{monthly_savings:,.0f}</strong></span><span style="color: #64748B;">•</span><span style="display: inline-flex; align-items: center; gap: 6px;"><i class="fa-solid fa-percent" style="color: {badge_color};"></i> Effective Tax Rate: <strong style="color: #F8FAFC;">{min(old.effective_rate, new.effective_rate):.2f}%</strong></span></div></div>"""
st.markdown(rec_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: REGIME COMPARISON (SIDE BY SIDE)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="tax-section-header">
    <h2 class="tax-section-title"><i class="fa-solid fa-scale-balanced" style="color: #4F8CFF; font-size: 1.1rem;"></i> Regime Comparison Overview</h2>
    <p class="tax-section-desc">Key metrics evaluated side-by-side between the Old and New tax regimes.</p>
</div>
""", unsafe_allow_html=True)

old_badge_html = '<span style="background: rgba(16,185,129,0.2); border: 1px solid #10B981; color: #10B981; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; font-weight: 700; margin-left: 8px;">Optimal</span>' if rec == "Old Regime" else ""
new_badge_html = '<span style="background: rgba(79,140,255,0.2); border: 1px solid #4F8CFF; color: #4F8CFF; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; font-weight: 700; margin-left: 8px;">Optimal</span>' if rec == "New Regime" else ""

old_card_border = "1.5px solid rgba(16,185,129,0.4)" if rec == "Old Regime" else "1px solid rgba(255,255,255,0.08)"
new_card_border = "1.5px solid rgba(79,140,255,0.4)" if rec == "New Regime" else "1px solid rgba(255,255,255,0.08)"

old_card_html = f"""<div style="background: #111827; border: {old_card_border}; border-radius: 14px; padding: 1.4rem 1.6rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 14px rgba(0,0,0,0.2);"><div><div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.12rem; font-weight: 700; color: #F59E0B; margin-bottom: 1.2rem; display: flex; align-items: center;"><i class="fa-solid fa-clock-rotate-left" style="margin-right: 8px;"></i> Old Tax Regime {old_badge_html}</div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;"><div class="tax-metric-box"><div class="tax-metric-label">Taxable Income</div><div class="tax-metric-val" style="font-size: 1.35rem;">₹{old.taxable_income:,.0f}</div><div class="tax-metric-sub">After all deductions</div></div><div class="tax-metric-box"><div class="tax-metric-label">Total Tax Payable</div><div class="tax-metric-val" style="font-size: 1.35rem; color: {'#10B981' if rec == 'Old Regime' else '#EF4444'};">₹{old.total_tax:,.0f}</div><div class="tax-metric-sub">Includes 4% cess</div></div><div class="tax-metric-box"><div class="tax-metric-label">Effective Rate</div><div class="tax-metric-val" style="font-size: 1.35rem;">{old.effective_rate:.2f}%</div><div class="tax-metric-sub">Of gross income</div></div><div class="tax-metric-box"><div class="tax-metric-label">Monthly Take-Home</div><div class="tax-metric-val" style="font-size: 1.35rem; color: #F8FAFC;">₹{old.take_home_monthly:,.0f}</div><div class="tax-metric-sub">Post-tax salary</div></div></div></div><div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08);"><span style="font-size: 0.86rem; color: #94A3B8;">Total Deductions Claimed:</span><span style="font-size: 0.92rem; font-weight: 700; color: #F8FAFC;">₹{old.standard_deduction + old.total_deductions:,.0f}</span></div></div>"""

new_card_html = f"""<div style="background: #111827; border: {new_card_border}; border-radius: 14px; padding: 1.4rem 1.6rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 14px rgba(0,0,0,0.2);"><div><div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.12rem; font-weight: 700; color: #4F8CFF; margin-bottom: 1.2rem; display: flex; align-items: center;"><i class="fa-solid fa-bolt" style="margin-right: 8px;"></i> New Tax Regime {new_badge_html}</div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;"><div class="tax-metric-box"><div class="tax-metric-label">Taxable Income</div><div class="tax-metric-val" style="font-size: 1.35rem;">₹{new.taxable_income:,.0f}</div><div class="tax-metric-sub">Standard deduction only</div></div><div class="tax-metric-box"><div class="tax-metric-label">Total Tax Payable</div><div class="tax-metric-val" style="font-size: 1.35rem; color: {'#10B981' if rec == 'New Regime' else '#EF4444'};">₹{new.total_tax:,.0f}</div><div class="tax-metric-sub">Includes 4% cess</div></div><div class="tax-metric-box"><div class="tax-metric-label">Effective Rate</div><div class="tax-metric-val" style="font-size: 1.35rem;">{new.effective_rate:.2f}%</div><div class="tax-metric-sub">Of gross income</div></div><div class="tax-metric-box"><div class="tax-metric-label">Monthly Take-Home</div><div class="tax-metric-val" style="font-size: 1.35rem; color: #F8FAFC;">₹{new.take_home_monthly:,.0f}</div><div class="tax-metric-sub">Post-tax salary</div></div></div></div><div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08);"><span style="font-size: 0.86rem; color: #94A3B8;">Standard Deduction:</span><span style="font-size: 0.92rem; font-weight: 700; color: #F8FAFC;">₹{new.standard_deduction:,.0f}</span></div></div>"""

comp_grid_html = f"""<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin: 1.25rem 0 2rem 0;">{old_card_html}{new_card_html}</div>"""
st.markdown(comp_grid_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: IN-DEPTH VISUAL & REGIME ANALYSIS TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/query_stats: Visual Breakdown & Summary",
    ":material/history: Old Regime Deep Dive",
    ":material/fiber_new: New Regime Deep Dive",
    ":material/lightbulb: Tax Saving Strategies",
])

# ─── Tab 1: Visual Breakdown & Summary ────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="tax-section-header" style="margin-top: 0.5rem;">
        <h3 class="tax-section-title">Tax Computation Waterfalls</h3>
        <p class="tax-section-desc">Trace the step-by-step path from gross earnings to net tax liability under each regime.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ── Waterfall Chart: Old Regime ──
    with col_a:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F59E0B; margin-bottom: 0.5rem;'>Old Regime — Tax Waterfall</div>", unsafe_allow_html=True)
        total_ded_old = old.standard_deduction + old.total_deductions

        waterfall_labels = ["Gross Income", "Std Deduction", "Deductions",
                            "Taxable Inc", "Base Tax", "87A Rebate",
                            "Surcharge", "Cess (4%)", "Total Tax"]

        fig_wf_old = go.Figure(go.Waterfall(
            orientation = "v",
            measure     = ["absolute", "relative", "relative", "total",
                           "absolute", "relative", "relative", "relative", "total"],
            x           = waterfall_labels,
            y           = [
                annual_income,
                -old.standard_deduction,
                -old.total_deductions,
                old.taxable_income,
                old.base_tax,
                -old.rebate_87a,
                old.surcharge,
                old.cess,
                old.total_tax,
            ],
            connector   = {"line": {"color": "rgba(255, 255, 255, 0.15)"}},
            increasing  = {"marker": {"color": "#EF4444"}},
            decreasing  = {"marker": {"color": "#10B981"}},
            totals      = {"marker": {"color": "#F59E0B"}},
            text        = [f"₹{abs(v):,.0f}" for v in [
                annual_income, old.standard_deduction, old.total_deductions,
                old.taxable_income, old.base_tax, old.rebate_87a,
                old.surcharge, old.cess, old.total_tax,
            ]],
            textposition = "outside",
        ))
        fig_wf_old.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
            height        = 430,
            showlegend    = False,
            margin        = dict(l=10, r=10, t=35, b=20),
            yaxis         = {"tickformat": "₹,.0f", "gridcolor": "rgba(255, 255, 255, 0.06)", "zerolinecolor": "rgba(255, 255, 255, 0.1)"},
            xaxis         = {"tickangle": -25, "gridcolor": "rgba(0,0,0,0)"},
        )
        st.plotly_chart(fig_wf_old, use_container_width=True)

    # ── Waterfall Chart: New Regime ──
    with col_b:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #4F8CFF; margin-bottom: 0.5rem;'>New Regime — Tax Waterfall</div>", unsafe_allow_html=True)
        fig_wf_new = go.Figure(go.Waterfall(
            orientation = "v",
            measure     = ["absolute", "relative", "total",
                           "absolute", "relative", "relative", "relative", "total"],
            x           = ["Gross Income", "Std Deduction", "Taxable Inc",
                           "Base Tax", "87A Rebate", "Surcharge", "Cess (4%)", "Total Tax"],
            y           = [
                annual_income,
                -new.standard_deduction,
                new.taxable_income,
                new.base_tax,
                -new.rebate_87a,
                new.surcharge,
                new.cess,
                new.total_tax,
            ],
            connector   = {"line": {"color": "rgba(255, 255, 255, 0.15)"}},
            increasing  = {"marker": {"color": "#EF4444"}},
            decreasing  = {"marker": {"color": "#10B981"}},
            totals      = {"marker": {"color": "#4F8CFF"}},
            text        = [f"₹{abs(v):,.0f}" for v in [
                annual_income, new.standard_deduction, new.taxable_income,
                new.base_tax, new.rebate_87a, new.surcharge, new.cess, new.total_tax,
            ]],
            textposition = "outside",
        ))
        fig_wf_new.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
            height        = 430,
            showlegend    = False,
            margin        = dict(l=10, r=10, t=35, b=20),
            yaxis         = {"tickformat": "₹,.0f", "gridcolor": "rgba(255, 255, 255, 0.06)", "zerolinecolor": "rgba(255, 255, 255, 0.1)"},
            xaxis         = {"tickangle": -25, "gridcolor": "rgba(0,0,0,0)"},
        )
        st.plotly_chart(fig_wf_new, use_container_width=True)

    # ── Side-by-Side Bar Comparison ──
    st.markdown("""
    <div class="tax-section-header">
        <h3 class="tax-section-title">Tax Component Comparison</h3>
        <p class="tax-section-desc">Compare taxable income, deductions, and tax obligations across both regimes.</p>
    </div>
    """, unsafe_allow_html=True)

    components = ["Gross Income", "Total Deductions", "Taxable Income", "Base Tax", "Total Tax"]
    old_vals   = [annual_income,
                  old.standard_deduction + old.total_deductions,
                  old.taxable_income, old.base_tax, old.total_tax]
    new_vals   = [annual_income,
                  new.standard_deduction,
                  new.taxable_income, new.base_tax, new.total_tax]

    fig_bar = go.Figure(data=[
        go.Bar(name="Old Regime", x=components, y=old_vals,
               marker_color="#F59E0B",
               text=[f"₹{v:,.0f}" for v in old_vals],
               textposition="outside"),
        go.Bar(name="New Regime", x=components, y=new_vals,
               marker_color="#4F8CFF",
               text=[f"₹{v:,.0f}" for v in new_vals],
               textposition="outside"),
    ])
    fig_bar.update_layout(
        barmode       = "group",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
        height        = 400,
        legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin        = dict(l=10, r=10, t=35, b=20),
        yaxis         = {"gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
        xaxis         = {"gridcolor": "rgba(0,0,0,0)"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Summary Table ──
    st.markdown("""
    <div class="tax-section-header">
        <h3 class="tax-section-title">Full Comparison Summary</h3>
        <p class="tax-section-desc">Comprehensive itemized comparison of the two tax computation systems.</p>
    </div>
    """, unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        "Component": [
            "Gross Annual Income",
            "Standard Deduction",
            "Additional Deductions (80C, 80D, etc.)",
            "Total Deductions",
            "Taxable Income",
            "Base Tax (before rebate)",
            "Section 87A Rebate",
            "Surcharge",
            "Health & Education Cess (4%)",
            "Total Tax Payable",
            "Effective Tax Rate",
            "Monthly Take-Home",
        ],
        "Old Regime": [
            f"₹{annual_income:,.0f}",
            f"₹{old.standard_deduction:,.0f}",
            f"₹{old.total_deductions:,.0f}",
            f"₹{old.standard_deduction + old.total_deductions:,.0f}",
            f"₹{old.taxable_income:,.0f}",
            f"₹{old.base_tax:,.0f}",
            f"₹{old.rebate_87a:,.0f}",
            f"₹{old.surcharge:,.0f}",
            f"₹{old.cess:,.0f}",
            f"₹{old.total_tax:,.0f}",
            f"{old.effective_rate:.2f}%",
            f"₹{old.take_home_monthly:,.0f}",
        ],
        "New Regime": [
            f"₹{annual_income:,.0f}",
            f"₹{new.standard_deduction:,.0f}",
            "₹0 (not applicable)",
            f"₹{new.standard_deduction:,.0f}",
            f"₹{new.taxable_income:,.0f}",
            f"₹{new.base_tax:,.0f}",
            f"₹{new.rebate_87a:,.0f}",
            f"₹{new.surcharge:,.0f}",
            f"₹{new.cess:,.0f}",
            f"₹{new.total_tax:,.0f}",
            f"{new.effective_rate:.2f}%",
            f"₹{new.take_home_monthly:,.0f}",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ─── Tab 2: Old Regime Details ────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="tax-section-header" style="margin-top: 0.5rem;">
        <h3 class="tax-section-title">Old Regime — Deductions & Slab Breakdown</h3>
        <p class="tax-section-desc">Itemized deductions claimed and progressive slab calculation under the Old Tax Regime.</p>
    </div>
    """, unsafe_allow_html=True)

    # Deductions table
    st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.6rem;'>Deductions Applied</div>", unsafe_allow_html=True)
    ded_rows = []
    total_claimed = 0.0
    for k, v in old.deduction_breakdown.items():
        ded_rows.append({"Deduction Head": k, "Amount (₹)": f"₹{v:,.0f}"})
        total_claimed += v
    ded_rows.append({"Deduction Head": "Total Deductions Claimed", "Amount (₹)": f"₹{total_claimed:,.0f}"})
    st.dataframe(pd.DataFrame(ded_rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="tax-narrative-box">
        <strong>Taxable Income Formula:</strong><br/>
        ₹{annual_income:,.0f} (Gross Income) − ₹{total_claimed:,.0f} (Total Deductions) = <strong style="color: #F8FAFC;">₹{old.taxable_income:,.0f}</strong> Taxable Base.
    </div>
    """, unsafe_allow_html=True)

    # Slab breakdown
    st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.6rem;'>Old Regime Tax Slab Breakdown</div>", unsafe_allow_html=True)
    if old.slab_breakdown:
        slab_df = pd.DataFrame(old.slab_breakdown)
        st.dataframe(slab_df.style.format({"Taxable Amount": "₹{:,.0f}", "Tax": "₹{:,.0f}"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No tax payable (taxable income within zero-rate threshold).")

    st.markdown("""
    <div class="tax-section-header" style="margin-top: 1.5rem;">
        <h4 class="tax-section-title">Tax Computation Details</h4>
    </div>
    """, unsafe_allow_html=True)
    
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    with oc1:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">Base Tax</div>
            <div class="tax-metric-val" style="font-size: 1.25rem;">₹{old.base_tax:,.0f}</div>
            <div class="tax-metric-sub">Before rebate</div>
        </div>
        """, unsafe_allow_html=True)
    with oc2:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">87A Rebate</div>
            <div class="tax-metric-val" style="font-size: 1.25rem; color: #10B981;">₹{old.rebate_87a:,.0f}</div>
            <div class="tax-metric-sub">Tax relief</div>
        </div>
        """, unsafe_allow_html=True)
    with oc3:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">Surcharge</div>
            <div class="tax-metric-val" style="font-size: 1.25rem;">₹{old.surcharge:,.0f}</div>
            <div class="tax-metric-sub">High income tier</div>
        </div>
        """, unsafe_allow_html=True)
    with oc4:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">Cess (4%)</div>
            <div class="tax-metric-val" style="font-size: 1.25rem;">₹{old.cess:,.0f}</div>
            <div class="tax-metric-sub">Health & Education</div>
        </div>
        """, unsafe_allow_html=True)
    with oc5:
        st.markdown(f"""
        <div class="tax-metric-box-featured">
            <div class="tax-metric-label" style="color: #60A5FA;">Total Tax</div>
            <div class="tax-metric-val" style="font-size: 1.35rem; color: #F59E0B;">₹{old.total_tax:,.0f}</div>
            <div class="tax-metric-sub">Effective: {old.effective_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Tab 3: New Regime Details ────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="tax-section-header" style="margin-top: 0.5rem;">
        <h3 class="tax-section-title">New Regime — FY 2025-26 Slabs & Analysis</h3>
        <p class="tax-section-desc">Default tax framework featuring simplified slabs and standard deduction of ₹75,000.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tax-narrative-box">
        Under the New Tax Regime, you receive a standard deduction of <strong>₹75,000</strong>. 
        Other exemptions (80C, 80D, HRA, Sec 24b) are not available, but the slab brackets are substantially wider with lower tax rates.
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.6rem;'>Statutory Slab Rates (FY 2025-26)</div>", unsafe_allow_html=True)
        slab_ref = pd.DataFrame([
            {"Income Range": "₹0 – ₹4,00,000",           "Tax Rate": "0%"},
            {"Income Range": "₹4,00,001 – ₹8,00,000",    "Tax Rate": "5%"},
            {"Income Range": "₹8,00,001 – ₹12,00,000",   "Tax Rate": "10%"},
            {"Income Range": "₹12,00,001 – ₹16,00,000",  "Tax Rate": "15%"},
            {"Income Range": "₹16,00,001 – ₹20,00,000",  "Tax Rate": "20%"},
            {"Income Range": "₹20,00,001 – ₹24,00,000",  "Tax Rate": "25%"},
            {"Income Range": "Above ₹24,00,000",          "Tax Rate": "30%"},
        ])
        st.dataframe(slab_ref, use_container_width=True, hide_index=True)

    with col_s2:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.6rem;'>Your Tax Slab Breakdown</div>", unsafe_allow_html=True)
        if new.slab_breakdown:
            slab_df = pd.DataFrame(new.slab_breakdown)
            st.dataframe(slab_df.style.format({"Taxable Amount": "₹{:,.0f}", "Tax": "₹{:,.0f}"}),
                         use_container_width=True, hide_index=True)
        else:
            st.success("Zero tax payable under the New Regime! (Income covered under rebate threshold)")

    st.markdown("""
    <div class="tax-section-header" style="margin-top: 1.5rem;">
        <h4 class="tax-section-title">Tax Computation Details</h4>
    </div>
    """, unsafe_allow_html=True)

    nc1, nc2, nc3, nc4, nc5 = st.columns(5)
    with nc1:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">Std Deduction</div>
            <div class="tax-metric-val" style="font-size: 1.25rem;">₹{new.standard_deduction:,.0f}</div>
            <div class="tax-metric-sub">Sec 16 default</div>
        </div>
        """, unsafe_allow_html=True)
    with nc2:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">Taxable Base</div>
            <div class="tax-metric-val" style="font-size: 1.25rem;">₹{new.taxable_income:,.0f}</div>
            <div class="tax-metric-sub">Net taxable</div>
        </div>
        """, unsafe_allow_html=True)
    with nc3:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">87A Rebate</div>
            <div class="tax-metric-val" style="font-size: 1.25rem; color: #10B981;">₹{new.rebate_87a:,.0f}</div>
            <div class="tax-metric-sub">Up to ₹12L limit</div>
        </div>
        """, unsafe_allow_html=True)
    with nc4:
        st.markdown(f"""
        <div class="tax-metric-box">
            <div class="tax-metric-label">Cess (4%)</div>
            <div class="tax-metric-val" style="font-size: 1.25rem;">₹{new.cess:,.0f}</div>
            <div class="tax-metric-sub">Health & Education</div>
        </div>
        """, unsafe_allow_html=True)
    with nc5:
        st.markdown(f"""
        <div class="tax-metric-box-featured">
            <div class="tax-metric-label" style="color: #60A5FA;">Total Tax</div>
            <div class="tax-metric-val" style="font-size: 1.35rem; color: #4F8CFF;">₹{new.total_tax:,.0f}</div>
            <div class="tax-metric-sub">Effective: {new.effective_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Tab 4: Tax Saving Strategies ─────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="tax-section-header" style="margin-top: 0.5rem;">
        <h3 class="tax-section-title">Personalised Tax Optimization Strategies</h3>
        <p class="tax-section-desc">Actionable insights to minimize tax liability and utilize remaining statutory headroom.</p>
    </div>
    """, unsafe_allow_html=True)

    for i, tip in enumerate(result["tips"], 1):
        st.markdown(f"""
        <div class="tax-tip-box">
            <strong style="color: #60A5FA;">Strategy {i}:</strong> {tip}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tax-section-header" style="margin-top: 2rem;">
        <h3 class="tax-section-title">Deduction Headroom Analysis</h3>
        <p class="tax-section-desc">Untapped statutory deduction capacity under the Old Tax Regime.</p>
    </div>
    """, unsafe_allow_html=True)

    headroom = {
        "80C (max ₹1.5L)":          max(150000 - deductions.effective_80c, 0),
        "80D Self (max ₹25K)":      max(25000  - deductions.effective_80d + deductions.effective_80d - min(health_ins_self, 25000), 0),
        "80CCD(1B) NPS (max ₹50K)": max(50000  - deductions.effective_80ccd1b, 0),
        "Sec 24(b) (max ₹2L)":      max(200000 - deductions.effective_24b, 0),
    }
    headroom_df = pd.DataFrame(
        [{"Deduction Section": k, "Unused Headroom (₹)": f"₹{v:,.0f}"} for k, v in headroom.items()]
    )
    st.dataframe(headroom_df, use_container_width=True, hide_index=True)

    total_headroom = sum(headroom.values())
    if total_headroom > 0:
        potential_saving = total_headroom * 0.30 * (1 + 0.04)
        st.markdown(f"""
        <div class="tax-headroom-banner">
            <strong style="color: #F59E0B;"><i class="fa-solid fa-triangle-exclamation"></i> Untapped Deduction Headroom: ₹{total_headroom:,.0f}</strong><br/>
            Maximizing these available limits could reduce your tax liability by up to approximately <strong>₹{potential_saving:,.0f}</strong> under the Old Regime.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tax-headroom-banner-success">
            <strong style="color: #10B981;"><i class="fa-solid fa-circle-check"></i> Statutory Deductions Fully Utilized</strong><br/>
            You have maximized all major deduction ceilings under the Old Tax Regime.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TAX REPORT DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="tax-section-header" style="margin-top: 2.25rem;">
    <h3 class="tax-section-title"><i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.15rem;"></i> Export Tax Report</h3>
    <p class="tax-section-desc">Download an export-ready PDF summary with side-by-side regime comparison, slab details, and optimization strategies.</p>
</div>
""", unsafe_allow_html=True)

_uid = twin.user_id if twin else "guest"
try:
    _pdf_tax = FinTwinPDFReport(
        report_title="Tax Intelligence & Optimisation Report",
        user_id=_uid,
        report_id=f"TX-{_uid}",
        subtitle="Old Regime vs New Regime Analysis — FY 2025-26 / AY 2026-27",
    )
    _pdf_tax.add_cover_page(
        user_name=getattr(twin, "name", "") or "",
        report_type="Tax Intelligence & Optimisation Report",
    )
    _pdf_tax.add_kpi_summary_row([
        {"label": "Recommended Regime", "value": str(rec), "status": "positive"},
        {"label": "Annual Tax Saving", "value": f"Rs. {savings:,.0f}", "status": "positive"},
        {"label": "Old Regime Tax", "value": f"Rs. {old.total_tax:,.0f}", "status": "neutral"},
        {"label": "New Regime Tax", "value": f"Rs. {new.total_tax:,.0f}", "status": "neutral"},
    ])
    _pdf_tax.add_callout(
        f"Switch to the {rec} to save Rs. {savings:,.0f} per year (Rs. {result['monthly_savings']:,.0f}/month).",
        style="success", title="Tax Optimisation"
    )

    _pdf_tax.add_section_divider("Regime Comparison")
    _pdf_tax.add_key_value_grid({
        "Recommended Regime": rec,
        "Annual Savings": savings,
        "Monthly Savings": result["monthly_savings"],
        "Taxable Income (Old)": old.taxable_income,
        "Total Tax (Old)": old.total_tax,
        "Take-Home/mo (Old)": old.take_home_monthly,
        "Taxable Income (New)": new.taxable_income,
        "Total Tax (New)": new.total_tax,
        "Take-Home/mo (New)": new.take_home_monthly,
    })
    try:
        _pdf_tax.add_plotly_figure(fig_bar, caption="Old vs New Regime Comparison")
    except NameError:
        pass
    try:
        _pdf_tax.add_plotly_figure(fig_wf_old, caption="Old Regime Tax Waterfall")
        _pdf_tax.add_plotly_figure(fig_wf_new, caption="New Regime Tax Waterfall")
    except NameError:
        pass

    _pdf_tax.add_section_divider("Deduction Headroom")
    try:
        _pdf_tax.add_dataframe_table(headroom_df)
    except NameError:
        pass

    tips = result.get("tips", [])
    if tips:
        _pdf_tax.add_section_divider("Personalised Tax Saving Tips")
        for tip in tips:
            _pdf_tax.add_callout(str(tip), style="info")

    pdf_tax_bytes = _pdf_tax.build()
    pdf_tax_fn = _pdf_tax.suggested_filename("tax_intelligence_report")
except Exception:
    pdf_tax_bytes = None
    pdf_tax_fn = "tax_intelligence_report.pdf"

with st.container(border=True):
    c_rep1, c_rep2 = st.columns([2.2, 1.1], vertical_alignment="center")
    with c_rep1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; padding: 4px 0;">
            <div style="background: rgba(79, 140, 255, 0.15); border: 1px solid rgba(79, 140, 255, 0.3); border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.3rem;"></i>
            </div>
            <div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;">
                    Download Official Tax Summary (PDF)
                </div>
                <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                    Comprehensive summary of taxable income, deductions claimed, effective rates, and regime comparison.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_rep2:
        if pdf_tax_bytes:
            st.download_button(
                label="Download PDF Report",
                icon=":material/download:",
                data=pdf_tax_bytes,
                file_name=pdf_tax_fn,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="btn_download_tax_report"
            )
        else:
            st.error("Report is currently unavailable.")
