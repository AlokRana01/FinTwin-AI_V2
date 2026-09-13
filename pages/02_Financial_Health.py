"""
Streamlit Page: Financial Health
Computes, explains, and breaks down the 0-100 score, accompanied by dynamic simulations,
factor diagnostics, peer comparisons, and PDF reporting.
"""

import streamlit as st
import pandas as pd
from html import escape as _he
from utils.session import render_sidebar_user_selector
from utils.visualizer import PlotlyVisualizer
from utils.pdf_report import FinTwinPDFReport
from models.twin_engine import FinancialDigitalTwin, HealthScoreEngine
from database.db_manager import DBManager

st.set_page_config(
    page_title="Financial Health — FinTwin AI",
    page_icon="assets/logos/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar User Selector ──────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

# ── Custom Page CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.fh-header-box {
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.fh-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.fh-subtitle {
    color: #94A3B8;
    font-size: 0.95rem;
    margin: 0;
}

/* Hero Score Card */
.fh-hero-card {
    background: linear-gradient(135deg, rgba(31, 41, 55, 0.75) 0%, rgba(17, 24, 39, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
.fh-score-display {
    font-family: 'Outfit', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
}
.fh-score-max {
    font-size: 1.25rem;
    color: #64748B;
    font-weight: 600;
}
.fh-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}

/* Key Insights Card */
.fh-insights-box {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    height: 100%;
}
.fh-insights-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.fh-insight-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.86rem;
}
.fh-insight-item:last-child {
    border-bottom: none;
}

/* What This Means Narrative */
.fh-narrative-box {
    background: rgba(31, 41, 55, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid #4F8CFF;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.75rem;
}
.fh-narrative-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.fh-narrative-text {
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.55;
    margin: 0;
}

/* Factor Grid & Cards */
.fh-factor-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.75rem;
}
@media (max-width: 1000px) {
    .fh-factor-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 600px) {
    .fh-factor-grid {
        grid-template-columns: 1fr;
    }
}
.fh-factor-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 140px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.fh-factor-card:hover {
    transform: translateY(-2px);
    border-color: rgba(79, 140, 255, 0.25);
}
.fh-factor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}
.fh-factor-name {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.fh-factor-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}
.fh-factor-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}
.fh-factor-status {
    font-size: 0.78rem;
    font-weight: 700;
}
.fh-factor-score {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94A3B8;
}

/* Simulator Result Box */
.fh-sim-result-card {
    background: linear-gradient(135deg, rgba(31, 41, 55, 0.75) 0%, rgba(17, 24, 39, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}

/* Export Card */
.fh-export-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1.25rem;
}
</style>
""", unsafe_allow_html=True)

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fh-header-box">
    <h1 class="fh-title">Financial Health</h1>
    <p class="fh-subtitle">Understand what is driving your financial health score and where you can improve.</p>
</div>
""", unsafe_allow_html=True)

if not twin:
    st.info("Log in or register in the sidebar to view and simulate your Financial Health Score.")
    st.stop()

# ── Compute Core Score Data ────────────────────────────────────────────────────
engine = HealthScoreEngine(twin)
score_data = engine.compute_overall_health_score()
overall_score = float(score_data.get("overall_score", 0.0))
financial_grade = score_data.get("financial_grade", "N/A")
components = score_data.get("components", {})

# Classification helper
def get_classification(score_val: float):
    if score_val >= 80:
        return "Excellent", "#10B981", "fa-solid fa-circle-check"
    elif score_val >= 65:
        return "Good", "#3B82F6", "fa-solid fa-circle-check"
    elif score_val >= 50:
        return "Moderate", "#F59E0B", "fa-solid fa-triangle-exclamation"
    else:
        return "Needs Attention", "#EF4444", "fa-solid fa-circle-exclamation"

overall_label, overall_color, overall_icon = get_classification(overall_score)

# Icon dictionary for factors
_factor_icons = {
    "savings_rate": "fa-solid fa-piggy-bank",
    "emi_burden": "fa-solid fa-credit-card",
    "emergency_fund": "fa-solid fa-shield-halved",
    "investment_ratio": "fa-solid fa-chart-line",
    "insurance_adequacy": "fa-solid fa-shield",
    "debt_level": "fa-solid fa-landmark",
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_simulator, tab_peers = st.tabs([
    ":material/monitor_heart: Health Score & Diagnostics",
    ":material/tune: Financial Score Simulator",
    ":material/group: Peer Benchmarking",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Health Score & Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    # ── 1. Hero Score & Key Insights ──
    col_hero, col_insights = st.columns([1.1, 1.15])

    with col_hero:
        st.markdown(f"""<div class="fh-hero-card">
<div style="font-size: 0.78rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;">Overall Financial Health Score</div>
<div style="display: flex; align-items: baseline; gap: 6px;">
<span class="fh-score-display" style="color: {overall_color};">{overall_score:.1f}</span>
<span class="fh-score-max">/ 100</span>
</div>
<div style="display: flex; align-items: center; gap: 8px; margin-top: 0.75rem; flex-wrap: wrap;">
<div class="fh-hero-badge" style="background: {overall_color}22; border: 1px solid {overall_color}55; color: {overall_color};">
<i class="{overall_icon}"></i> {overall_label}
</div>
<div class="fh-hero-badge" style="background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.12); color: #F8FAFC;">
Grade {financial_grade}
</div>
</div>
<p style="color: #CBD5E1; font-size: 0.88rem; margin: 1rem 0 0 0; line-height: 1.5;">
{'Your financial health is strong overall, with solid resilience across major financial commitments.' if overall_score >= 80 else ('Your financial health is stable, with balanced fundamentals and targeted opportunities to optimize.' if overall_score >= 60 else 'Your financial profile indicates important areas for optimization to enhance stability.')}
</p>
</div>""", unsafe_allow_html=True)

        fig_gauge = PlotlyVisualizer.plot_health_score_gauge(overall_score)
        st.plotly_chart(fig_gauge, use_container_width=True, key="fh_overview_gauge")

    with col_insights:
        st.markdown("""<div class="fh-insights-box">
<div class="fh-insights-title">
<i class="fa-solid fa-list-check" style="color: #4F8CFF;"></i>
<span>Key Insights</span>
</div>""", unsafe_allow_html=True)

        for key, val in components.items():
            f_name = key.replace("_", " ").title()
            f_score = val.get("score", 0.0)
            f_label, f_color, f_icon = get_classification(f_score)
            f_icon_class = _factor_icons.get(key, "fa-solid fa-circle-info")
            st.markdown(f"""<div class="fh-insight-item">
<div style="display: flex; align-items: center; gap: 10px;">
<i class="{f_icon_class}" style="color: #94A3B8; font-size: 0.9rem; width: 16px; text-align: center;"></i>
<span style="color: #F8FAFC; font-weight: 500;">{f_name}</span>
</div>
<span style="color: {f_color}; font-weight: 700; font-size: 0.82rem;">{f_label} &middot; {f_score:.0f}/100</span>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── 2. "What This Means" Narrative ──
    # Identify top strength and top area of improvement
    sorted_factors = sorted(components.items(), key=lambda x: x[1].get("score", 0.0), reverse=True)
    top_strength = sorted_factors[0][0].replace("_", " ").title() if sorted_factors else "Savings"
    top_attention = sorted_factors[-1][0].replace("_", " ").title() if sorted_factors else "Emergency Fund"

    st.markdown(f"""
    <div class="fh-narrative-box">
        <div class="fh-narrative-title">
            <i class="fa-solid fa-circle-info" style="color: #4F8CFF;"></i>
            <span>What This Means</span>
        </div>
        <p class="fh-narrative-text">
            Your <strong>{top_strength}</strong> is currently one of the strongest positive drivers of your overall score. Conversely, <strong>{top_attention}</strong> represents your primary opportunity for enhancement. Systematically addressing lower-scoring categories could noticeably elevate your financial health and future resilience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Score Factors (Compact Fintech Cards) ──
    st.markdown("""
    <div style="margin-bottom: 0.85rem;">
        <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
            Scoring Factors
        </h3>
        <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">
            Diagnostic breakdown of all multi-dimensional health metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fh-factor-grid">', unsafe_allow_html=True)
    factor_cols = st.columns(3)
    
    for idx, (key, val) in enumerate(components.items()):
        f_name = key.replace("_", " ").title()
        f_score = val.get("score", 0.0)
        f_raw = val.get("raw_value", "0")
        f_unit = val.get("unit", "")
        f_bench = val.get("benchmark", "")
        f_weight = val.get("weight", 0)
        f_label, f_color, _ = get_classification(f_score)
        f_icon_class = _factor_icons.get(key, "fa-solid fa-circle-info")

        # Format display value
        if key == "savings_rate" or key == "emi_burden" or key == "investment_ratio":
            display_val = f"{f_raw}{f_unit}"
        elif key == "emergency_fund":
            display_val = f"{f_raw} months"
        elif key == "debt_level":
            display_val = f"{f_raw}x income"
        elif key == "insurance_adequacy":
            display_val = f"{f_score:.0f}% adequate"
        else:
            display_val = f"{f_raw}{f_unit}"

        with factor_cols[idx % 3]:
            st.markdown(f"""
            <div class="fh-factor-card">
                <div>
                    <div class="fh-factor-header">
                        <span class="fh-factor-name">{f_name}</span>
                        <i class="{f_icon_class}" style="color: #4F8CFF; font-size: 0.9rem;"></i>
                    </div>
                    <div class="fh-factor-value">{display_val}</div>
                </div>
                <div class="fh-factor-footer">
                    <span class="fh-factor-status" style="color: {f_color};">{f_label}</span>
                    <span class="fh-factor-score">{f_score:.0f} / 100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Details", expanded=False):
                st.markdown(f"""
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
                    <div><strong>Weight:</strong> {f_weight}%</div>
                    <div><strong>Actual:</strong> {f_raw}{f_unit}</div>
                    <div><strong>Benchmark:</strong> {f_bench}</div>
                    <div><strong>Score:</strong> {f_score:.1f} / 100</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(f_score / 100.0)

                # Actionable Recommendation Snippet
                rec_action = ""
                if key == "savings_rate":
                    rec_action = "Maintain regular monthly surplus above 25% to maximize wealth creation." if f_score >= 80 else f"Trim non-essential lifestyle outflows to target the {f_bench} benchmark."
                elif key == "emi_burden":
                    rec_action = "Debt obligations are low and manageable." if f_score >= 80 else "Avoid high-interest consumer debt and prioritize prepaying outstanding balances."
                elif key == "emergency_fund":
                    target = (twin.basic_expenses + twin.monthly_emi) * 6.0
                    rec_action = "Emergency reserves are well-buffered." if f_score >= 80 else f"Build liquid savings towards ₹{target:,.0f} (6 months of commitments)."
                elif key == "investment_ratio":
                    target_sip = twin.total_income * 0.15
                    rec_action = "Systematic investments are on target." if f_score >= 80 else f"Increase automated SIP allocations towards ₹{target_sip:,.0f}/month."
                elif key == "insurance_adequacy":
                    rec_action = "Family and health protections are active." if f_score >= 80 else "Ensure adequate term life (10x annual income) and health insurance coverage."
                elif key == "debt_level":
                    rec_action = "Overall leverage is safe." if f_score >= 80 else "Focus surplus cash flows on reducing total outstanding loan principal."

                st.markdown(f"""
                <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.80rem; color: #94A3B8;">
                    <strong style="color: #F8FAFC;">Recommended:</strong> {rec_action}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Financial Score Simulator
# ─────────────────────────────────────────────────────────────────────────────
with tab_simulator:
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
            Financial Score Simulator
        </h3>
        <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
            See how changes to your income, expenses, savings, and debt dynamically affect your health score in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_sim_inputs, col_sim_output = st.columns([1.3, 0.9])

    with col_sim_inputs:
        st.markdown("#### Income & Expenses")
        sim_income = st.slider("Monthly Salary (₹)", min_value=10000, max_value=500000, value=int(twin.monthly_income), step=5000, key="sim_inc")
        sim_rent = st.slider("Monthly Rent (₹)", min_value=0, max_value=150000, value=int(twin.rent), step=1000, key="sim_rnt")
        other_expenses = twin.basic_expenses - twin.rent
        sim_expenses = st.slider("Other Expenses (Food, Utilities, Transport, Shopping) (₹)", min_value=0, max_value=200000, value=int(other_expenses), step=1000, key="sim_exp")
        sim_emi = st.slider("Monthly EMI (₹)", min_value=0, max_value=200000, value=int(twin.monthly_emi), step=1000, key="sim_emi_val")

        st.markdown("---")
        st.markdown("#### Savings & Investments")
        sim_emergency = st.slider("Emergency Fund (₹)", min_value=0, max_value=2000000, value=int(twin.emergency_fund), step=10000, key="sim_ef_val")
        sim_sip = st.slider("Monthly SIP Investment (₹)", min_value=0, max_value=150000, value=int(twin.sip_amount), step=1000, key="sim_sip_val")

        st.markdown("---")
        st.markdown("#### Protection")
        sim_life_ins = st.slider("Term Life Insurance Cover (₹)", min_value=0, max_value=30000000, value=int(twin.life_insurance), step=500000, key="sim_life_val")
        sim_health_ins = st.slider("Health Insurance Cover (₹)", min_value=0, max_value=2000000, value=int(twin.health_insurance), step=50000, key="sim_health_val")

        st.markdown("---")
        st.markdown("#### Debt")
        sim_loans = st.slider("Total Outstanding Loans (₹)", min_value=0, max_value=15000000, value=int(twin.loan_amount), step=50000, key="sim_loan_val")

    # Perform Simulated Calculation
    sim_demographics = {
        "name": twin.name,
        "age": twin.age,
        "city": twin.city,
        "occupation": twin.occupation,
        "monthly_income": float(sim_income),
        "bonus": twin.bonus,
        "additional_income": twin.additional_income
    }

    sim_balance_sheet = {
        "net_worth": float(twin.net_worth),
        "bank_savings": float(twin.bank_savings),
        "fd_amount": float(twin.fd_amount),
        "emergency_fund": float(sim_emergency),
        "sip_amount": float(sim_sip),
        "mutual_funds": float(twin.mutual_funds),
        "stocks": float(twin.stocks),
        "ppf_investment": float(twin.ppf_investment),
        "nps_investment": float(twin.nps_investment),
        "loan_amount": float(sim_loans),
        "car_loan": twin.car_loan,
        "home_loan": twin.home_loan,
        "credit_card_debt": twin.credit_card_debt,
        "monthly_emi": float(sim_emi),
        "health_insurance": float(sim_health_ins),
        "life_insurance": float(sim_life_ins),
        "rent": float(sim_rent),
        "groceries": twin.groceries,
        "utilities": twin.utilities,
        "transport": twin.transport,
        "food_delivery": twin.food_delivery,
        "entertainment": twin.entertainment,
        "shopping": twin.shopping,
        "goal_type": twin.goal_type,
        "goal_amount": twin.goal_amount
    }

    sim_twin = FinancialDigitalTwin(twin.user_id, sim_demographics, sim_balance_sheet)
    sim_twin.basic_expenses = float(sim_rent + sim_expenses)

    sim_engine = HealthScoreEngine(sim_twin)
    sim_score_data = sim_engine.compute_overall_health_score()
    sim_overall = float(sim_score_data["overall_score"])
    sim_grade = sim_score_data["financial_grade"]
    score_delta = round(sim_overall - overall_score, 2)
    sim_label, sim_color, _ = get_classification(sim_overall)

    with col_sim_output:
        delta_color = "#10B981" if score_delta > 0 else ("#EF4444" if score_delta < 0 else "#94A3B8")
        delta_sign = "+" if score_delta > 0 else ""

        st.markdown(f"""<div class="fh-sim-result-card">
<div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">Current vs. Simulated Score</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.06);">
<div>
<div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Current Score</div>
<div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 700; color: #F8FAFC;">{overall_score:.1f} <span style="font-size: 0.8rem; color: #64748B;">/ 100</span></div>
</div>
<div style="text-align: right;">
<div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Simulated Score</div>
<div style="font-family: 'Outfit', sans-serif; font-size: 1.85rem; font-weight: 800; color: {sim_color};">{sim_overall:.1f} <span style="font-size: 0.85rem; color: #64748B;">/ 100</span></div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
<span style="font-size: 0.82rem; font-weight: 600; color: #CBD5E1;">Score Change:</span>
<span style="font-size: 1rem; font-weight: 700; color: {delta_color};">{delta_sign}{score_delta:.2f} points</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.35rem;">
<span style="font-size: 0.82rem; font-weight: 600; color: #CBD5E1;">Simulated Grade:</span>
<span style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC;">Grade {sim_grade} ({sim_label})</span>
</div>
</div>""", unsafe_allow_html=True)

        fig_sim_gauge = PlotlyVisualizer.plot_health_score_gauge(sim_overall)
        st.plotly_chart(fig_sim_gauge, use_container_width=True, key="fh_simulator_gauge")

        if st.button("Reset Simulation", icon=":material/refresh:", use_container_width=True):
            for key in ["sim_inc", "sim_rnt", "sim_exp", "sim_emi_val", "sim_ef_val", "sim_sip_val", "sim_life_val", "sim_health_val", "sim_loan_val"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Peer Benchmarking
# ─────────────────────────────────────────────────────────────────────────────
with tab_peers:
    st.markdown(f"""
    <div style="margin-bottom: 1.25rem;">
        <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
            Peer Cohort Comparison
        </h3>
        <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
            Compare your monthly spending and savings profile against other professionals working as a <strong>{_he(twin.occupation)}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cohort_avg = DBManager.get_cohort_averages(twin.occupation)
    
    if cohort_avg:
        user_spend = {
            "Rent": twin.rent,
            "Groceries": twin.groceries,
            "Utilities": twin.utilities,
            "Transport": twin.transport,
            "Food Delivery": twin.food_delivery,
            "Entertainment": twin.entertainment,
            "Shopping": twin.shopping,
            "EMI": twin.monthly_emi,
            "SIP (Invest)": twin.sip_amount
        }
        
        peer_spend = {
            "Rent": cohort_avg.get("rent", 0.0),
            "Groceries": cohort_avg.get("groceries", 0.0),
            "Utilities": cohort_avg.get("utilities", 0.0),
            "Transport": cohort_avg.get("transport", 0.0),
            "Food Delivery": cohort_avg.get("food_delivery", 0.0),
            "Entertainment": cohort_avg.get("entertainment", 0.0),
            "Shopping": cohort_avg.get("shopping", 0.0),
            "EMI": cohort_avg.get("monthly_emi", 0.0),
            "SIP (Invest)": cohort_avg.get("sip_amount", 0.0)
        }
        
        fig_peer = PlotlyVisualizer.plot_peer_comparison(user_spend, peer_spend)
        st.plotly_chart(fig_peer, use_container_width=True, key="fh_peer_comparison_chart")
        
        user_total_spend = sum(user_spend.values()) - user_spend["SIP (Invest)"]
        peer_total_spend = sum(peer_spend.values()) - peer_spend["SIP (Invest)"]
        diff = user_total_spend - peer_total_spend
        
        if diff < 0:
            st.success(f"**Cohort Insight:** You spend ₹{abs(diff):,.0f} LESS than the average **{twin.occupation}** each month, providing a strong financial buffer.")
        else:
            st.info(f"**Cohort Insight:** Your monthly discretionary spending is ₹{diff:,.0f} higher than the average **{twin.occupation}**. Check categories above for potential savings optimization.")
    else:
        st.info("Cohort benchmarks are loading. Ensure dataset is fully populated.")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT EXPORT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 2.25rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 1.5rem;'></div>", unsafe_allow_html=True)

try:
    _pdf_health = FinTwinPDFReport(
        report_title="Financial Health Intelligence Report",
        user_id=twin.user_id,
        report_id=f"FH-{twin.user_id}",
        subtitle="Comprehensive 0-100 score breakdown, factor benchmarks, and peer comparison",
    )
    _pdf_health.add_cover_page(
        user_name=getattr(twin, "name", "") or "",
        report_type="Financial Health Intelligence Report",
    )
    # KPI summary row
    _pdf_health.add_kpi_summary_row([
        {"label": "Overall Score", "value": f"{overall_score:.1f} / 100",
         "status": "positive" if overall_score >= 70 else ("negative" if overall_score < 50 else "neutral")},
        {"label": "Financial Grade", "value": f"Grade {financial_grade}", "status": "neutral"},
        {"label": "Classification", "value": str(overall_label), "status": "neutral"},
    ])

    # Health interpretation callout
    if overall_score >= 80:
        _pdf_health.add_callout(
            f"Score {overall_score:.1f}/100 — Excellent financial health. You are on a strong trajectory.",
            style="success", title="Health Assessment"
        )
    elif overall_score >= 65:
        _pdf_health.add_callout(
            f"Score {overall_score:.1f}/100 — Good financial health with targeted improvement opportunities.",
            style="info", title="Health Assessment"
        )
    elif overall_score >= 50:
        _pdf_health.add_callout(
            f"Score {overall_score:.1f}/100 — Moderate financial health. Review scoring factors below.",
            style="warning", title="Health Assessment"
        )
    else:
        _pdf_health.add_callout(
            f"Score {overall_score:.1f}/100 — Financial health needs immediate attention. Prioritise AI Coach actions.",
            style="danger", title="Health Assessment"
        )

    _pdf_health.add_section_divider("Scoring Factor Breakdown")
    rows = []
    for key, val in components.items():
        rows.append({
            "Factor": key.replace("_", " ").title(),
            "Score": f"{val.get('score', 0):.1f} / 100",
            "Actual": f"{val.get('raw_value', '')}{val.get('unit', '')}",
            "Benchmark": val.get("benchmark", ""),
            "Weight": f"{val.get('weight', '')}%",
        })
    _pdf_health.add_dataframe_table(pd.DataFrame(rows))

    pdf_health_bytes = _pdf_health.build()
    pdf_filename = _pdf_health.suggested_filename("financial_health_report")
except Exception as e:
    pdf_health_bytes = None
    pdf_filename = "financial_health_report.pdf"

c_rep1, c_rep2 = st.columns([1.7, 1.3], vertical_alignment="center")
with c_rep1:
    st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.25rem;"></i>
<h3 style="margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">Your Financial Health Report</h3>
</div>
<p style="color: #94A3B8; font-size: 0.86rem; margin: 0; line-height: 1.45;">
Download a detailed summary of your financial health score, contributing factors, benchmarks, and recommendations.
</p>""", unsafe_allow_html=True)

with c_rep2:
    if pdf_health_bytes:
        st.download_button(
            label="Download Financial Health Report",
            icon=":material/download:",
            data=pdf_health_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            key="btn_download_fh_report"
        )
    else:
        st.error("Report is currently unavailable.")


