"""
Streamlit Page: My Financial Digital Twin
Visualizes the complete Financial Digital Twin — income, expense, debt, investment,
risk level, and health score — with interactive charts, sub-profiles, and export.
Supports creating and editing custom profiles.
"""

import json
import pandas as pd
import streamlit as st
from html import escape as _he
from utils.session import render_sidebar_user_selector, invalidate_session_twin
from utils.security import validate_name
from database.db_manager import DBManager
from utils.visualizer import PlotlyVisualizer
from utils.pdf_report import FinTwinPDFReport

st.set_page_config(
    page_title="My Financial Digital Twin — FinTwin AI",
    page_icon="assets/logos/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar User Selector ──────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

if not twin:
    st.markdown("""
    <div style="padding: 0 0 1rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 1.25rem;">
        <h1 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin: 0 0 0.5rem 0; letter-spacing: -0.03em;">
            My Financial Digital Twin
        </h1>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
            A complete view of your current financial position, behavior, and financial health.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Please log in or create an account via the sidebar to view your Financial Digital Twin.")
    st.stop()

# ── Custom Page CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.twin-header-box {
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.twin-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
.twin-subtitle {
    color: #94A3B8;
    font-size: 0.95rem;
    margin: 0;
}

/* Profile Summary Card */
.twin-profile-card {
    background: linear-gradient(135deg, rgba(31, 41, 55, 0.7) 0%, rgba(17, 24, 39, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.twin-profile-left {
    display: flex;
    align-items: center;
    gap: 1rem;
    min-width: 0;
}
.twin-avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    border: 2px solid rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.1rem;
    color: #FFFFFF;
    flex-shrink: 0;
    overflow: hidden;
    letter-spacing: 0.03em;
}
.twin-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
    display: block;
}
.twin-profile-meta {
    display: flex;
    flex-direction: column;
}
.twin-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}
.twin-details {
    font-size: 0.84rem;
    color: #94A3B8;
    margin-top: 3px;
}

/* Snapshot Metric Cards */
.twin-snapshot-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}
@media (max-width: 1100px) {
    .twin-snapshot-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}
@media (max-width: 700px) {
    .twin-snapshot-grid {
        grid-template-columns: 1fr;
    }
}
.twin-snapshot-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 98px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}
.twin-snapshot-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
.twin-snapshot-label {
    font-size: 0.70rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.twin-snapshot-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.twin-snapshot-desc {
    font-size: 0.74rem;
    color: #64748B;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* Overview Narrative Box */
.twin-narrative-box {
    background: rgba(31, 41, 55, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid #4F8CFF;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.75rem;
}
.twin-narrative-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.twin-narrative-text {
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.55;
    margin: 0;
}

/* Breakdown & Chart Summary Cards */
.chart-summary-box {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    height: 100%;
}
.chart-summary-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.chart-summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.82rem;
}
.chart-summary-row:last-child {
    border-bottom: none;
}

/* Action Box */
.twin-action-card {
    background: linear-gradient(135deg, rgba(79, 140, 255, 0.08) 0%, rgba(17, 24, 39, 0.9) 100%);
    border: 1px solid rgba(79, 140, 255, 0.25);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="twin-header-box">
    <h1 class="twin-title">My Financial Digital Twin</h1>
    <p class="twin-subtitle">A complete view of your current financial position, behavior, and financial health.</p>
</div>
""", unsafe_allow_html=True)

# ── Build Twin Export Payload via export_twin() ───────────────────────────────
twin_payload = twin.export_twin()
income  = twin_payload["income_profile"]
expense = twin_payload["expense_profile"]
debt    = twin_payload["debt_profile"]
invest  = twin_payload["investment_profile"]
risk    = twin_payload["risk_level"]
health  = twin_payload["health_score"]

# Helper: Risk badge HTML
_risk_meta = {
    "Low":      ("#10B981", "fa-solid fa-circle-check", "Low Risk"),
    "Moderate": ("#F59E0B", "fa-solid fa-triangle-exclamation", "Moderate Risk"),
    "High":     ("#EF4444", "fa-solid fa-circle-exclamation", "High Risk"),
    "Critical": ("#7C3AED", "fa-solid fa-circle-xmark", "Critical Risk"),
}

def risk_badge(level: str) -> str:
    color, fa_class, label = _risk_meta.get(level, ("#6B7280", "fa-solid fa-circle-info", f"{level} Risk"))
    return (
        f'<span style="background:{color}22;border:1px solid {color}55;'
        f'color:{color};border-radius:20px;padding:4px 14px;font-weight:700;'
        f'font-size:0.82rem;display:inline-flex;align-items:center;gap:6px;">'
        f'<i class="{fa_class}" style="font-size:0.85rem;"></i> {label}</span>'
    )

# Savings rate calculation
monthly_surplus = income["total_monthly_income"] - expense["total_monthly_expenses"] - debt["monthly_emi"]
savings_rate_pct = round(
    (monthly_surplus / income["total_monthly_income"] * 100), 1
) if income["total_monthly_income"] > 0 else 0.0

# User Initials
name_parts = [p for p in twin.name.strip().split() if p]
if len(name_parts) >= 2:
    initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
elif len(name_parts) == 1 and name_parts[0]:
    initials = name_parts[0][:2].upper()
else:
    initials = "FT"

has_financial_data = bool(
    income["total_monthly_income"] > 0 or twin.net_worth != 0 or twin.bank_savings > 0 or expense["total_monthly_expenses"] > 0
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_subprofiles, tab_edit, tab_export = st.tabs([
    ":material/dashboard: Overview",
    ":material/folder_open: Sub-Profiles",
    ":material/edit: Edit Profile",
    ":material/download: Download My Data",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    if not has_financial_data:
        st.markdown("""
        <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 2.5rem 2rem; text-align: center; margin: 1.5rem 0;">
            <i class="fa-solid fa-user-pen" style="color: #4F8CFF; font-size: 2.5rem; margin-bottom: 1rem; display: inline-block;"></i>
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.5rem 0;">
                Complete Your Financial Digital Twin
            </h3>
            <p style="color: #94A3B8; font-size: 0.90rem; max-width: 520px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                Add your income, expense, and balance sheet details to unlock personalized financial analysis, forecasting, stress testing, and AI recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Switch to the **Edit Profile** tab above to configure your financial numbers.")
    else:
        # ── 1. Profile Summary Card ──
        from utils.avatar import render_avatar_html
        twin_avatar_img = render_avatar_html(getattr(twin, "avatar_id", "avatar_01"), size=46, class_name="twin-avatar-img")
        st.markdown(f"""
        <div class="twin-profile-card">
            <div class="twin-profile-left">
                <div class="twin-avatar">{twin_avatar_img}</div>
                <div class="twin-profile-meta">
                    <div class="twin-name">{_he(twin.name)}</div>
                    <div class="twin-details">{_he(twin.occupation or 'Salaried Professional')} &middot; Age {twin.age} &middot; {_he(twin.city or 'India')}</div>
                </div>
            </div>
            <div class="twin-profile-right">
                {risk_badge(risk)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 2. Financial Snapshot (5 Metrics) ──
        emi_pct = debt.get("emi_to_income_ratio", 0.0) * 100
        health_score_num = float(health.get("overall_score", 0.0))
        
        # Classification for savings and health
        savings_status = "Healthy" if savings_rate_pct >= 20 else ("Moderate" if savings_rate_pct >= 10 else "Needs Attention")
        health_status = "Excellent" if health_score_num >= 80 else ("Good" if health_score_num >= 65 else ("Moderate" if health_score_num >= 50 else "Needs Attention"))

        st.markdown(f"""
        <div class="twin-snapshot-grid">
            <div class="twin-snapshot-card">
                <div class="twin-snapshot-header">
                    <span class="twin-snapshot-label">Monthly Income</span>
                    <i class="fa-solid fa-wallet" style="color: #4F8CFF; font-size: 0.9rem;"></i>
                </div>
                <div class="twin-snapshot-value">₹{income['total_monthly_income']:,.0f}</div>
                <div class="twin-snapshot-desc">Regular monthly income</div>
            </div>
            <div class="twin-snapshot-card">
                <div class="twin-snapshot-header">
                    <span class="twin-snapshot-label">Net Worth</span>
                    <i class="fa-solid fa-landmark" style="color: #22C55E; font-size: 0.9rem;"></i>
                </div>
                <div class="twin-snapshot-value">₹{twin.net_worth:,.0f}</div>
                <div class="twin-snapshot-desc">Estimated current value</div>
            </div>
            <div class="twin-snapshot-card">
                <div class="twin-snapshot-header">
                    <span class="twin-snapshot-label">Monthly EMI</span>
                    <i class="fa-solid fa-credit-card" style="color: #00D4FF; font-size: 0.9rem;"></i>
                </div>
                <div class="twin-snapshot-value">₹{debt['monthly_emi']:,.0f}</div>
                <div class="twin-snapshot-desc">{emi_pct:.1f}% of monthly income</div>
            </div>
            <div class="twin-snapshot-card">
                <div class="twin-snapshot-header">
                    <span class="twin-snapshot-label">Savings Rate</span>
                    <i class="fa-solid fa-piggy-bank" style="color: #F59E0B; font-size: 0.9rem;"></i>
                </div>
                <div class="twin-snapshot-value">{savings_rate_pct}%</div>
                <div class="twin-snapshot-desc" style="color: {'#10B981' if savings_rate_pct >= 20 else ('#F59E0B' if savings_rate_pct >= 10 else '#EF4444')}; font-weight: 600;">{savings_status}</div>
            </div>
            <div class="twin-snapshot-card">
                <div class="twin-snapshot-header">
                    <span class="twin-snapshot-label">Financial Health</span>
                    <i class="fa-solid fa-gauge" style="color: #EC4899; font-size: 0.9rem;"></i>
                </div>
                <div class="twin-snapshot-value">{health_score_num:.1f} <span style="font-size: 0.78rem; font-weight: 500; color: #64748B;">/ 100</span></div>
                <div class="twin-snapshot-desc" style="color: {'#10B981' if health_score_num >= 80 else ('#3B82F6' if health_score_num >= 65 else ('#F59E0B' if health_score_num >= 50 else '#EF4444'))}; font-weight: 600;">{health_status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3. Financial Overview Narrative ──
        status_term = "strong" if health_score_num >= 75 else ("stable" if health_score_num >= 55 else "developing")
        if savings_rate_pct >= 20:
            sav_msg = f"Your current savings rate is healthy at {savings_rate_pct}%"
        elif savings_rate_pct >= 10:
            sav_msg = f"Your current savings rate is moderate at {savings_rate_pct}%"
        else:
            sav_msg = f"Your current savings rate indicates an opportunity for optimization at {savings_rate_pct}%"

        if emi_pct <= 25:
            debt_msg = f"debt obligations remain manageable at {emi_pct:.1f}% of income"
        elif emi_pct <= 45:
            debt_msg = f"debt obligations currently consume {emi_pct:.1f}% of income"
        else:
            debt_msg = f"debt obligations represent a notable commitment at {emi_pct:.1f}% of income"

        st.markdown(f"""
        <div class="twin-narrative-box">
            <div class="twin-narrative-title">
                <i class="fa-solid fa-circle-info" style="color: #4F8CFF;"></i>
                <span>Financial Overview</span>
            </div>
            <p class="twin-narrative-text">
                Your current financial position suggests a <strong>{status_term}</strong> foundation. {sav_msg}, while your {debt_msg}. Maintaining emergency reserves and continuing systematic investments could further support long-term stability.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── 4. Cash Flow Section ("Where Your Income Goes") ──
        st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
                Where Your Income Goes
            </h3>
            <p style="color: #94A3B8; font-size: 0.85rem; margin: 0 0 0.75rem 0;">
                Monthly income allocation and major outflows.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: rgba(79, 140, 255, 0.08); border: 1px solid rgba(79, 140, 255, 0.2); border-radius: 8px; padding: 0.45rem 0.85rem; display: inline-block; font-size: 0.82rem; color: #F8FAFC; font-weight: 600; margin-bottom: 1rem;">
            Monthly Income: <span style="color: #4F8CFF; font-weight: 700;">₹{income['total_monthly_income']:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

        outflows = {
            "Rent":          twin.rent,
            "Groceries":     twin.groceries,
            "Utilities":     twin.utilities,
            "Transport":     twin.transport,
            "Food Delivery": twin.food_delivery,
            "Entertainment": twin.entertainment,
            "Shopping":      twin.shopping,
            "EMI":           twin.monthly_emi,
        }
        active_outflows = {k: v for k, v in outflows.items() if v > 0}
        surplus = income["total_monthly_income"] - sum(active_outflows.values())

        c_flow1, c_flow2 = st.columns([1.5, 1])
        with c_flow1:
            fig_sankey = PlotlyVisualizer.plot_cash_flow_sankey(
                income["total_monthly_income"], active_outflows, max(surplus, 0)
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

        with c_flow2:
            st.markdown("""<div class="chart-summary-box">
                <div class="chart-summary-title">Allocation Summary</div>
            """, unsafe_allow_html=True)
            if surplus > 0:
                st.markdown(f"""<div class="chart-summary-row"><span style="color: #10B981; font-weight: 600;">Monthly Surplus / Savings</span><span style="color: #10B981; font-weight: 700;">₹{surplus:,.0f}</span></div>""", unsafe_allow_html=True)
            for cat, amt in active_outflows.items():
                st.markdown(f"""<div class="chart-summary-row"><span style="color: #CBD5E1;">{cat}</span><span style="color: #F8FAFC; font-weight: 600;">₹{amt:,.0f}</span></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="chart-summary-row" style="border-top: 1px solid rgba(255,255,255,0.08); margin-top: 4px; padding-top: 6px;"><span style="color: #94A3B8; font-weight: 600;">Total Outflows</span><span style="color: #F8FAFC; font-weight: 700;">₹{sum(active_outflows.values()):,.0f}</span></div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        # ── 5. Assets vs Liabilities ──
        assets_dict = {k: v for k, v in {
            "Bank Savings":   twin.bank_savings,
            "Fixed Deposits": twin.fd_amount,
            "Emergency Fund": twin.emergency_fund,
            "Mutual Funds":   twin.mutual_funds,
            "Stocks":         twin.stocks,
            "PPF":            twin.ppf_investment,
            "NPS":            twin.nps_investment,
        }.items() if v > 0}
        
        liab_dict = {k: v for k, v in {
            "Home Loan":   twin.home_loan,
            "Car Loan":    twin.car_loan,
            "Credit Card": twin.credit_card_debt,
        }.items() if v > 0}

        total_assets_val = sum(assets_dict.values())
        total_liab_val = sum(liab_dict.values())

        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
                Assets vs Liabilities
            </h3>
            <p style="color: #94A3B8; font-size: 0.85rem; margin: 0 0 0.75rem 0;">
                Portfolio balance between accumulated assets and current liabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # High-level summary row
        col_as1, col_as2 = st.columns(2)
        with col_as1:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 0.75rem 1rem; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.82rem; font-weight: 600; color: #A7F3D0; text-transform: uppercase;">Total Assets</span>
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #10B981;">₹{total_assets_val:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_as2:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 0.75rem 1rem; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.82rem; font-weight: 600; color: #FECACA; text-transform: uppercase;">Total Liabilities</span>
                <span style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #EF4444;">₹{total_liab_val:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)

        if assets_dict or liab_dict:
            fig_donut = PlotlyVisualizer.plot_asset_liability_donut(
                assets_dict if assets_dict else {"Savings": 1},
                liab_dict if liab_dict else {"No Debt": 1}
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No asset or liability records available.")

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        # ── 6. Financial Health Breakdown ──
        if isinstance(health, dict) and "components" in health and health["components"]:
            st.markdown("""
            <div style="margin-bottom: 0.5rem;">
                <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
                    Financial Health Breakdown
                </h3>
                <p style="color: #94A3B8; font-size: 0.85rem; margin: 0 0 0.85rem 0;">
                    Core factors contributing to your overall Financial Health score.
                </p>
            </div>
            """, unsafe_allow_html=True)

            comp_items = health["components"]
            cols_comp = st.columns(len(comp_items) if len(comp_items) <= 6 else 3)
            for idx, (comp_key, comp_val) in enumerate(comp_items.items()):
                c_score = comp_val.get("score", 0.0)
                c_weight = comp_val.get("weight", 0.0)
                c_label = comp_key.replace("_", " ").title()
                c_grade = "Strong" if c_score >= 80 else ("Good" if c_score >= 65 else ("Moderate" if c_score >= 45 else "Needs Attention"))
                c_color = "#10B981" if c_score >= 80 else ("#3B82F6" if c_score >= 65 else ("#F59E0B" if c_score >= 45 else "#EF4444"))

                with cols_comp[idx % len(cols_comp)]:
                    st.markdown(f"""
                    <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 0.75rem 0.85rem; margin-bottom: 0.5rem;">
                        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; margin-bottom: 3px;">{c_label} ({c_weight}%)</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; font-family: 'Outfit', sans-serif;">{c_score:.0f} <span style="font-size: 0.72rem; color: #64748B;">/ 100</span></div>
                        <div style="font-size: 0.74rem; font-weight: 600; color: {c_color}; margin-top: 2px;">{c_grade}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── 7. Recommended Action ──
        st.markdown("""
        <div style="margin-top: 2rem; margin-bottom: 0.75rem;">
            <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
                Recommended Action
            </h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(79, 140, 255, 0.08) 0%, rgba(17, 24, 39, 0.95) 100%);
                    border: 1px solid rgba(79, 140, 255, 0.25);
                    border-radius: 12px;
                    padding: 1.25rem 1.5rem;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    gap: 1.25rem;">
            <div style="flex: 1; min-width: 260px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <i class="fa-solid fa-lightbulb" style="color: #00D4FF; font-size: 1.1rem;"></i>
                    <span style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">Scenario Exploration</span>
                </div>
                <p style="color: #94A3B8; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                    Your current savings behavior contributes positively to your financial position. Explore how different income, expense, and investment scenarios could affect your projected future.
                </p>
            </div>
            <a href="/Scenario_Simulator" target="_self" style="
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: #2563EB;
                color: #FFFFFF;
                text-decoration: none;
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 0.88rem;
                padding: 0.6rem 1.25rem;
                border-radius: 8px;
                transition: all 0.2s ease;
                white-space: nowrap;
                box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
            ">
                <i class="fa-solid fa-sliders" style="font-size: 0.85rem;"></i> Run Scenario &rarr;
            </a>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Sub-Profiles
# ─────────────────────────────────────────────────────────────────────────────
with tab_subprofiles:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
            Detailed Financial Sub-Profiles
        </h3>
        <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">
            Breakdown across income, expense, debt, investment, risk, and health score models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. Income Profile ──────────────────────────────────────────────────
    with st.expander("Income Profile", expanded=True):
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Monthly Salary",       f"₹{income['monthly_salary']:,.0f}")
        ic2.metric("Annual Bonus",         f"₹{income['annual_bonus']:,.0f}")
        ic3.metric("Additional Monthly",   f"₹{income['additional_monthly_income']:,.0f}")
        ic4, ic5, _ = st.columns(3)
        ic4.metric("Total Monthly Income", f"₹{income['total_monthly_income']:,.0f}")
        ic5.metric("Annual Income",        f"₹{income['annual_income']:,.0f}")

    # ── 2. Expense Profile ─────────────────────────────────────────────────
    with st.expander("Expense Profile"):
        ex_items = {
            "Rent": expense["rent"], "Groceries": expense["groceries"],
            "Utilities": expense["utilities"], "Transport": expense["transport"],
            "Food Delivery": expense["food_delivery"],
            "Entertainment": expense["entertainment"], "Shopping": expense["shopping"]
        }
        ec = st.columns(4)
        for i, (label, val) in enumerate(ex_items.items()):
            ec[i % 4].metric(label, f"₹{val:,.0f}")
        st.markdown("---")
        et1, et2 = st.columns(2)
        et1.metric("Total Monthly Expenses",  f"₹{expense['total_monthly_expenses']:,.0f}")
        et2.metric("Expense-to-Income Ratio", f"{expense['expense_to_income_ratio']*100:.1f}%")

    # ── 3. Debt Profile ────────────────────────────────────────────────────
    with st.expander("Debt Profile"):
        db1, db2, db3 = st.columns(3)
        db1.metric("Total Outstanding Loans", f"₹{debt['total_outstanding_loans']:,.0f}")
        db2.metric("Home Loan",               f"₹{debt['home_loan']:,.0f}")
        db3.metric("Car Loan",                f"₹{debt['car_loan']:,.0f}")
        db4, db5, db6 = st.columns(3)
        db4.metric("Credit Card Debt",        f"₹{debt['credit_card_debt']:,.0f}")
        db5.metric("Monthly EMI",             f"₹{debt['monthly_emi']:,.0f}")
        db6.metric("EMI-to-Income Ratio",     f"{debt['emi_to_income_ratio']*100:.1f}%")
        st.metric("Debt-to-Annual Income Ratio", f"{debt['debt_to_annual_income_ratio']:.2f}x")

    # ── 4. Investment Profile ──────────────────────────────────────────────
    with st.expander("Investment Profile"):
        fig_inv = PlotlyVisualizer.plot_investment_breakdown_bar(invest)
        st.plotly_chart(fig_inv, use_container_width=True)
        inv1, inv2 = st.columns(2)
        inv1.metric("Total Invested Corpus",       f"₹{invest['total_invested_corpus']:,.0f}")
        inv2.metric("Monthly SIP",                 f"₹{invest['monthly_sip']:,.0f}")
        inv3, inv4 = st.columns(2)
        inv3.metric("Investment-to-Income Ratio",  f"{invest['investment_to_income_ratio']*100:.1f}%")
        inv4.metric("Emergency Fund",              f"₹{invest['emergency_fund']:,.0f}")

    # ── 5. Risk Level ──────────────────────────────────────────────────────
    with st.expander("Risk Level"):
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px 0;">
                <div style="font-size:1rem;color:#94A3B8;margin-bottom:12px">
                    Overall Financial Risk Assessment
                </div>
                {risk_badge(risk)}
                <div style="margin-top:16px;color:#CBD5E1;font-size:0.86rem;max-width:500px;
                            margin-left:auto;margin-right:auto;line-height:1.6">
                    Risk is derived from your EMI-to-income ratio and monthly savings rate.<br>
                    <strong>Low</strong>: EMI &le; 20% &amp; Savings &ge; 15% &nbsp;|&nbsp;
                    <strong>Moderate</strong>: EMI &le; 35% &amp; Savings &ge; 5% &nbsp;|&nbsp;
                    <strong>High</strong>: EMI &le; 50% &nbsp;|&nbsp;
                    <strong>Critical</strong>: EMI &gt; 50% or negative savings
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        rl1, rl2 = st.columns(2)
        rl1.metric("EMI-to-Income Ratio", f"{debt['emi_to_income_ratio']*100:.1f}%")
        rl2.metric("Monthly Savings Rate", f"{savings_rate_pct}%")

    # ── 6. Health Score Summary ────────────────────────────────────────────
    with st.expander("Health Score Summary"):
        fig_gauge = PlotlyVisualizer.plot_health_score_gauge(health["overall_score"])
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("**Component Scores:**")
        comp_cols = st.columns(3)
        for i, (key, val) in enumerate(health["components"].items()):
            label  = key.replace("_", " ").title()
            score  = val["score"]
            weight = val["weight"]
            comp_cols[i % 3].metric(f"{label} ({weight}%)", f"{score:.0f} / 100")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Edit Profile
# ─────────────────────────────────────────────────────────────────────────────
with tab_edit:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">
            Edit Financial Profile
        </h3>
        <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">
            Update your income parameters, balance sheet assets, liabilities, and recurring expenses.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle between Edit Active Profile and Create New Profile
    profile_action = st.radio(
        "Select Action:",
        ["Edit Current Values", "Start Fresh (blank template)"],
        horizontal=True
    )
    
    # Gather initial values based on selection
    if profile_action == "Edit Current Values":
        init_name = twin.name
        init_age = int(twin.age)
        init_city = twin.city
        init_occ = twin.occupation
        init_income = float(twin.monthly_income)
        init_bonus = float(twin.bonus)
        init_add_income = float(twin.additional_income)
        
        init_savings = float(twin.bank_savings)
        init_fd = float(twin.fd_amount)
        init_ef = float(twin.emergency_fund)
        init_sip = float(twin.sip_amount)
        init_mf = float(twin.mutual_funds)
        init_stocks = float(twin.stocks)
        init_ppf = float(twin.ppf_investment)
        init_nps = float(twin.nps_investment)
        
        init_loan = float(twin.loan_amount)
        init_car = float(twin.car_loan)
        init_home = float(twin.home_loan)
        init_cc = float(twin.credit_card_debt)
        init_emi = float(twin.monthly_emi)
        
        init_health_ins = float(twin.health_insurance)
        init_life_ins = float(twin.life_insurance)
        
        init_rent = float(twin.rent)
        init_groceries = float(twin.groceries)
        init_utilities = float(twin.utilities)
        init_transport = float(twin.transport)
        init_food = float(twin.food_delivery)
        init_ent = float(twin.entertainment)
        init_shop = float(twin.shopping)
        
        init_goal_type = twin.goal_type
        init_goal_amt = float(twin.goal_amount)
    else:
        init_name = ""
        init_age = 30
        init_city = "Mumbai"
        init_occ = "Software Engineer"
        init_income = 100000.0
        init_bonus = 100000.0
        init_add_income = 0.0
        
        init_savings = 100000.0
        init_fd = 200000.0
        init_ef = 150000.0
        init_sip = 15000.0
        init_mf = 300000.0
        init_stocks = 100000.0
        init_ppf = 50000.0
        init_nps = 50000.0
        
        init_loan = 0.0
        init_car = 0.0
        init_home = 0.0
        init_cc = 0.0
        init_emi = 0.0
        
        init_health_ins = 250000.0
        init_life_ins = 10000000.0
        
        init_rent = 25000.0
        init_groceries = 10000.0
        init_utilities = 5000.0
        init_transport = 5000.0
        init_food = 5000.0
        init_ent = 5000.0
        init_shop = 10000.0
        
        init_goal_type = "House"
        init_goal_amt = 5000000.0

    # Renders the Input Form
    with st.form("profile_form"):
        st.markdown("#### Demographics")
        f_name = st.text_input("Full Name", value=init_name, max_chars=80)
        
        c_age, c_city, c_occ = st.columns(3)
        f_age = c_age.number_input("Age", min_value=18, max_value=100, value=init_age)
        f_city = c_city.text_input("City", value=init_city, max_chars=100)
        f_occ = c_occ.selectbox(
            "Occupation",
            ["Software Engineer", "Consultant", "Doctor", "Product Manager", "Data Scientist", "Banker", "Teacher", "Entrepreneur", "Sales Executive", "Marketing Manager"],
            index=0 if init_occ not in ["Software Engineer", "Consultant", "Doctor", "Product Manager", "Data Scientist", "Banker", "Teacher", "Entrepreneur", "Sales Executive", "Marketing Manager"] else ["Software Engineer", "Consultant", "Doctor", "Product Manager", "Data Scientist", "Banker", "Teacher", "Entrepreneur", "Sales Executive", "Marketing Manager"].index(init_occ)
        )
        
        from utils.validators import (
            validate_financial_profile,
            MAX_FINANCIAL_AMOUNT,
            MAX_MONTHLY_INCOME,
            MAX_MONTHLY_EXPENSE,
            MAX_GOAL_AMOUNT,
        )

        c_inc, c_bon, c_add = st.columns(3)
        f_income = c_inc.number_input("Monthly Income (₹)", min_value=0.0, max_value=MAX_MONTHLY_INCOME, value=min(init_income, MAX_MONTHLY_INCOME), step=5000.0)
        f_bonus = c_bon.number_input("Annual Bonus (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_bonus, MAX_FINANCIAL_AMOUNT), step=10000.0)
        f_add_income = c_add.number_input("Additional Income/mo (₹)", min_value=0.0, max_value=MAX_MONTHLY_INCOME, value=min(init_add_income, MAX_MONTHLY_INCOME), step=2000.0)
        
        st.markdown("#### Balance Sheet Assets")
        ca1, ca2, ca3 = st.columns(3)
        f_savings = ca1.number_input("Bank Savings (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_savings, MAX_FINANCIAL_AMOUNT), step=10000.0)
        f_fd = ca2.number_input("Fixed Deposits (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_fd, MAX_FINANCIAL_AMOUNT), step=10000.0)
        f_ef = ca3.number_input("Emergency Fund (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_ef, MAX_FINANCIAL_AMOUNT), step=10000.0)
        
        ca4, ca5, ca6 = st.columns(3)
        f_sip = ca4.number_input("Monthly Mutual Fund SIP (₹)", min_value=0.0, max_value=MAX_MONTHLY_INCOME, value=min(init_sip, MAX_MONTHLY_INCOME), step=1000.0)
        f_mf = ca5.number_input("Mutual Funds Corpus (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_mf, MAX_FINANCIAL_AMOUNT), step=20000.0)
        f_stocks = ca6.number_input("Stocks Corpus (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_stocks, MAX_FINANCIAL_AMOUNT), step=20000.0)
        
        ca7, ca8 = st.columns(2)
        f_ppf = ca7.number_input("PPF Investment Corpus (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_ppf, MAX_FINANCIAL_AMOUNT), step=10000.0)
        f_nps = ca8.number_input("NPS Investment Corpus (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_nps, MAX_FINANCIAL_AMOUNT), step=10000.0)
        
        st.markdown("#### Liabilities & Debt")
        cl1, cl2, cl3 = st.columns(3)
        f_loan = cl1.number_input("Total Outstanding Loan Amount (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_loan, MAX_FINANCIAL_AMOUNT), step=50000.0)
        f_car = cl2.number_input("Car Loan Amount (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_car, MAX_FINANCIAL_AMOUNT), step=25000.0)
        f_home = cl3.number_input("Home Loan Amount (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_home, MAX_FINANCIAL_AMOUNT), step=100000.0)
        
        cl4, cl5 = st.columns(2)
        f_cc = cl4.number_input("Outstanding Credit Card Debt (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_cc, MAX_FINANCIAL_AMOUNT), step=5000.0)
        f_emi = cl5.number_input("Monthly EMIs (₹)", min_value=0.0, max_value=MAX_MONTHLY_INCOME, value=min(init_emi, MAX_MONTHLY_INCOME), step=2000.0)
        
        st.markdown("#### Insurance Cover")
        ci1, ci2 = st.columns(2)
        f_health = ci1.number_input("Health Insurance Premium / Cover (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_health_ins, MAX_FINANCIAL_AMOUNT), step=1000.0)
        f_life = ci2.number_input("Term Life Insurance Cover (₹)", min_value=0.0, max_value=MAX_FINANCIAL_AMOUNT, value=min(init_life_ins, MAX_FINANCIAL_AMOUNT), step=500000.0)
        
        st.markdown("#### Monthly Expenses")
        ce1, ce2, ce3, ce4 = st.columns(4)
        f_rent = ce1.number_input("Rent (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_rent, MAX_MONTHLY_EXPENSE), step=2000.0)
        f_groceries = ce2.number_input("Groceries (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_groceries, MAX_MONTHLY_EXPENSE), step=1000.0)
        f_utilities = ce3.number_input("Utilities (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_utilities, MAX_MONTHLY_EXPENSE), step=500.0)
        f_transport = ce4.number_input("Transport (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_transport, MAX_MONTHLY_EXPENSE), step=500.0)
        
        ce5, ce6, ce7 = st.columns(3)
        f_food = ce5.number_input("Food Delivery (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_food, MAX_MONTHLY_EXPENSE), step=500.0)
        f_ent = ce6.number_input("Entertainment (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_ent, MAX_MONTHLY_EXPENSE), step=500.0)
        f_shop = ce7.number_input("Shopping (₹)", min_value=0.0, max_value=MAX_MONTHLY_EXPENSE, value=min(init_shop, MAX_MONTHLY_EXPENSE), step=1000.0)
        
        st.markdown("#### Financial Goal")
        cg1, cg2 = st.columns(2)
        f_goal_type = cg1.selectbox("Goal Type", ["House", "Car", "Education", "Marriage", "Vacation"], index=["House", "Car", "Education", "Marriage", "Vacation"].index(init_goal_type) if init_goal_type in ["House", "Car", "Education", "Marriage", "Vacation"] else 0)
        f_goal_amt = cg2.number_input("Goal Target Amount (₹)", min_value=0.0, max_value=MAX_GOAL_AMOUNT, value=min(init_goal_amt, MAX_GOAL_AMOUNT), step=50000.0)
        
        submit_btn = st.form_submit_button("Save & Activate Digital Twin", icon=":material/save:", use_container_width=True, type="primary")
        
        if submit_btn:
            name_ok, name_err = validate_name(f_name)
            if not name_ok:
                st.error(name_err)
            else:
                combined_check = {
                    "monthly_income": f_income,
                    "rent": f_rent,
                    "groceries": f_groceries,
                    "utilities": f_utilities,
                    "transport": f_transport,
                    "food_delivery": f_food,
                    "entertainment": f_ent,
                    "shopping": f_shop,
                    "monthly_emi": f_emi,
                    "bank_savings": f_savings,
                    "fd_amount": f_fd,
                    "emergency_fund": f_ef,
                    "sip_amount": f_sip,
                    "mutual_funds": f_mf,
                    "stocks": f_stocks,
                    "ppf_investment": f_ppf,
                    "nps_investment": f_nps,
                    "loan_amount": f_loan,
                    "car_loan": f_car,
                    "home_loan": f_home,
                    "credit_card_debt": f_cc,
                    "health_insurance": f_health,
                    "life_insurance": f_life,
                }
                prof_ok, prof_err = validate_financial_profile(combined_check)
                if not prof_ok:
                    st.error(prof_err)
                else:
                    save_user_id = twin.user_id
                    computed_net_worth = (f_savings + f_fd + f_mf + f_stocks + f_ppf + f_nps) - f_loan
                    
                    user_payload = {
                        "user_id": save_user_id,
                        "name": f_name,
                        "age": int(f_age),
                        "city": f_city,
                        "occupation": f_occ,
                        "monthly_income": float(f_income),
                        "bonus": float(f_bonus),
                        "additional_income": float(f_add_income)
                    }
                    
                    twin_payload = {
                        "user_id": save_user_id,
                        "net_worth": float(computed_net_worth),
                        "bank_savings": float(f_savings),
                        "fd_amount": float(f_fd),
                        "emergency_fund": float(f_ef),
                        "sip_amount": float(f_sip),
                        "mutual_funds": float(f_mf),
                        "stocks": float(f_stocks),
                        "ppf_investment": float(f_ppf),
                        "nps_investment": float(f_nps),
                        "loan_amount": float(f_loan),
                        "car_loan": float(f_car),
                        "home_loan": float(f_home),
                        "credit_card_debt": float(f_cc),
                        "monthly_emi": float(f_emi),
                        "health_insurance": float(f_health),
                        "life_insurance": float(f_life),
                        "rent": float(f_rent),
                        "groceries": float(f_groceries),
                        "utilities": float(f_utilities),
                        "transport": float(f_transport),
                        "food_delivery": float(f_food),
                        "entertainment": float(f_ent),
                        "shopping": float(f_shop),
                        "goal_type": f_goal_type,
                        "goal_amount": float(f_goal_amt)
                    }
                    
                    try:
                        db_user_ok = DBManager.save_user_profile(user_payload)
                        db_twin_ok = DBManager.save_digital_twin(twin_payload)
                        
                        if db_user_ok and db_twin_ok:
                            invalidate_session_twin()
                            st.success("Your Digital Twin has been saved successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to write profile details to SQLite Database.")
                    except ValueError as ve:
                        st.error(str(ve))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Download My Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("""<div style="margin-bottom: 1.5rem;">
<h3 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 0 0 2px 0;">Download My Data</h3>
<p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">Export a complete copy of your Financial Digital Twin profile in structured PDF or JSON format.</p>
</div>""", unsafe_allow_html=True)

    # 1. PDF Report
    try:
        _pdf_report = FinTwinPDFReport(
            report_title="Digital Twin Profile Report",
            user_id=twin.user_id,
            report_id=f"DT-{twin.user_id}",
            subtitle="Complete Virtual Financial Self — Income, Expenses, Debt, Investments & Health Score",
        )
        # Full-page branded cover
        _pdf_report.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Digital Twin Profile Report",
        )
        # KPI summary row
        _pdf_report.add_kpi_summary_row([
            {"label": "Total Monthly Income", "value": f"Rs. {getattr(twin, 'total_income', 0):,.0f}", "status": "positive"},
            {"label": "Net Worth", "value": f"Rs. {getattr(twin, 'net_worth', 0):,.0f}",
             "status": "positive" if getattr(twin, 'net_worth', 0) >= 0 else "negative"},
            {"label": "Health Score", "value": f"{health.get('overall_score', 0):.1f} / 100", "status": "neutral"},
            {"label": "Savings Rate", "value": f"{savings_rate_pct}%",
             "status": "positive" if savings_rate_pct >= 20 else ("negative" if savings_rate_pct < 10 else "neutral")},
        ])

        _pdf_report.add_section_divider("User Profile")
        _pdf_report.add_key_value_grid({
            "Name": getattr(twin, "name", "—"),
            "Risk Level": risk,
            "Financial Health Score": f"{health.get('overall_score', 0):.1f} / 100",
            "Monthly Savings Rate": f"{savings_rate_pct}%",
        })
        _pdf_report.add_section_divider("Income Profile")
        _pdf_report.add_key_value_grid(income)
        _pdf_report.add_section_divider("Expense Profile")
        _pdf_report.add_key_value_grid(expense)
        _pdf_report.add_section_divider("Debt Profile")
        _pdf_report.add_key_value_grid(debt)
        _pdf_report.add_section_divider("Investment Profile")
        _pdf_report.add_key_value_grid(invest)

        if isinstance(health, dict) and health:
            _pdf_report.add_section_divider("Financial Health Score Breakdown")
            _pdf_report.add_key_value_grid(
                {k: v for k, v in health.items() if not isinstance(v, (dict, list))}
            )
            h_score = health.get('overall_score', 0)
            if h_score >= 70:
                _pdf_report.add_callout(
                    f"Your financial health score of {h_score:.1f}/100 reflects strong financial management.",
                    style="success", title="Health Assessment"
                )
            elif h_score >= 50:
                _pdf_report.add_callout(
                    f"Your financial health score of {h_score:.1f}/100 is moderate. Review the breakdown above.",
                    style="warning", title="Health Assessment"
                )
            else:
                _pdf_report.add_callout(
                    f"Your financial health score of {h_score:.1f}/100 needs attention. Check AI Coach recommendations.",
                    style="danger", title="Health Assessment"
                )

        pdf_bytes = _pdf_report.build()
        pdf_fn = _pdf_report.suggested_filename("digital_twin_report")
    except Exception as e:
        pdf_bytes = None
        pdf_fn = "digital_twin_report.pdf"

    c_pdf1, c_pdf2 = st.columns([1.7, 1.3], vertical_alignment="center")
    with c_pdf1:
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.25rem;"></i>
<h3 style="margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">Digital Twin PDF Report</h3>
</div>
<p style="color: #94A3B8; font-size: 0.86rem; margin: 0; line-height: 1.45;">
Complete document summarizing your income, balance sheet, risk assessment, and sub-profiles.
</p>""", unsafe_allow_html=True)
    with c_pdf2:
        if pdf_bytes:
            st.download_button(
                label="Download Digital Twin PDF Report",
                icon=":material/download:",
                data=pdf_bytes,
                file_name=pdf_fn,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="btn_download_twin_pdf"
            )

    st.markdown("<div style='margin-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. JSON Data
    json_str = json.dumps(twin_payload, indent=2, ensure_ascii=False)
    c_json1, c_json2 = st.columns([1.7, 1.3], vertical_alignment="center")
    with c_json1:
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<i class="fa-solid fa-file-code" style="color: #22C55E; font-size: 1.25rem;"></i>
<h3 style="margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">Structured JSON Export</h3>
</div>
<p style="color: #94A3B8; font-size: 0.86rem; margin: 0; line-height: 1.45;">
Raw machine-readable JSON dataset containing all demographic and financial parameters.
</p>""", unsafe_allow_html=True)
    with c_json2:
        st.download_button(
            label="Download JSON Data",
            icon=":material/download:",
            data=json_str,
            file_name=f"digital_twin_{twin.user_id}.json",
            mime="application/json",
            use_container_width=True,
            key="btn_download_twin_json"
        )

    st.markdown("---")
    with st.expander("Preview JSON Payload"):
        st.json(twin_payload)

