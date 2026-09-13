"""
Streamlit Page: AI Financial Coach  (Module 10)
================================================
Rule-based financial coaching engine with 5 analysis domains.

Tabs:
  1. Coach Summary  — overall score, risk level, top alerts, narrative
  2. Savings        — habits analysis, discretionary leaks, recommendations
  3. Investment     — SIP adequacy, diversification, NPS/PPF gaps
  4. Debt           — EMI burden, CC debt, leverage alerts
  5. Emergency Fund — months coverage, top-up plan
  6. Tax Efficiency — 80C/80D/NPS utilisation, regime tips
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.session import render_sidebar_user_selector
from utils.coach import (
    FinancialCoach, CoachingReport, Recommendation, Alert,
    Domain, Severity, SEVERITY_META,
)
from utils.pdf_report import FinTwinPDFReport

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Financial Coach — FinTwin AI",
    page_icon  = "assets/logos/favicon.svg",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Custom Scoped Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tab Bar Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    margin-bottom: 2rem;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 8px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
}

/* Page Header */
.coach-header-box {
    margin-bottom: 2.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.coach-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.03em;
}
.coach-subtitle {
    color: #94A3B8;
    font-size: 0.96rem;
    margin: 0;
    line-height: 1.55;
}

/* Section Header */
.coach-section-header {
    margin: 2.25rem 0 1.25rem 0;
}
.coach-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.coach-section-desc {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 0 0 1.25rem 0;
    line-height: 1.5;
}

/* Top Metric Area */
.coach-metric-box {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    min-height: 116px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.coach-metric-box:hover {
    border-color: rgba(79, 140, 255, 0.4);
    transform: translateY(-1px);
}
.coach-metric-box-featured {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(79, 140, 255, 0.45);
    border-top: 3px solid #4F8CFF;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    min-height: 116px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
}
.coach-metric-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    color: #94A3B8;
    margin-bottom: 0.45rem;
}
.coach-metric-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.25;
    margin-bottom: 0.35rem;
}
.coach-metric-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}

/* Callout / Narrative Card */
.coach-narrative-box {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 1.35rem 1.6rem;
    margin: 1.25rem 0 2rem 0;
    color: #E2E8F0;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Priority Action Card */
.coach-priority-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1.5px solid rgba(245, 158, 11, 0.35);
    border-left: 5px solid #F59E0B;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1.25rem 0 2.25rem 0;
}
.coach-priority-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #F59E0B;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 0.75rem;
}
.coach-priority-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: #F8FAFC;
    line-height: 1.6;
    margin: 0;
}

/* Unified Domain Scorecard Card (Tab 1) */
.domain-card {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.35rem 1.15rem;
    text-align: center;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 200px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.domain-card:hover {
    transform: translateY(-2px);
    border-color: rgba(79, 140, 255, 0.4);
}
.domain-card-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: rgba(79, 140, 255, 0.15);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #4F8CFF;
    font-size: 1.1rem;
    margin: 0 auto 0.65rem auto;
}
.domain-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.45rem;
}
.domain-card-score {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
}
.domain-grade-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}
.domain-bar-bg {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    height: 6px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.domain-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}

/* Domain Header Container (Tabs 2-6) */
.domain-header-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Key Metrics & Score Box inside domain tabs */
.domain-score-panel {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.35rem;
    box-sizing: border-box;
    margin-bottom: 1.25rem;
}
.domain-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 0.88rem;
}
.domain-metric-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.domain-metric-label {
    color: #94A3B8;
    font-weight: 500;
}
.domain-metric-val {
    color: #F8FAFC;
    font-weight: 700;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Insights Panel */
.domain-insights-panel {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.5rem;
}
.domain-suggestion-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
    color: #CBD5E1;
    font-size: 0.91rem;
    line-height: 1.55;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.domain-suggestion-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.domain-suggestion-icon {
    color: #10B981;
    font-size: 0.85rem;
    margin-top: 4px;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar & Profile Selection ──────────────────────────────────────────────
twin = render_sidebar_user_selector()

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="coach-header-box">
    <h1 class="coach-title">AI Financial Coach</h1>
    <p class="coach-subtitle">
        Personalized financial guidance based on your current financial profile.
    </p>
</div>
""", unsafe_allow_html=True)

if not twin:
    st.warning("Please select or configure a financial profile in the sidebar to load your coaching report.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Generate report (cached per user_id)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(
    show_spinner="Generating personalised coaching report…",
    hash_funcs={object: lambda x: str(getattr(x, "user_id", id(x)))},
)
def get_coaching_report_v2(_twin) -> CoachingReport:
    return FinancialCoach(_twin).generate_report()

report = get_coaching_report_v2(twin)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Renderers
# ══════════════════════════════════════════════════════════════════════════════

def render_alert(alert: Alert) -> None:
    """Render a risk alert with severity styling and comfortable spacing."""
    meta = SEVERITY_META[alert.severity]
    st.markdown(
        f"""
        <div style="background: {meta['bg']}; border: 1px solid {meta['color']}40; border-left: 4px solid {meta['color']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="color: {meta['color']}; font-size: 1rem;">{meta['icon']}</span>
                <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: {meta['color']};">
                    {alert.headline}
                </span>
            </div>
            <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.55; padding-left: 26px;">
                {alert.message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation(rec: Recommendation, expanded: bool = False) -> None:
    """Render an expandable recommendation card with generous breathing room."""
    meta = SEVERITY_META[rec.severity]
    with st.expander(f"[{rec.severity.value}] {rec.title}", expanded=expanded):
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 16px 18px; margin-bottom: 10px;">
                <div style="color: #F8FAFC; font-size: 0.93rem; line-height: 1.6; margin-bottom: 12px;">
                    {rec.detail}
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                    <div style="margin-bottom: 8px; font-size: 0.89rem;">
                        <strong style="color: {meta['color']};">Recommended Action:</strong>
                        <span style="color: #CBD5E1;"> {rec.action}</span>
                    </div>
                    <div style="font-size: 0.89rem;">
                        <strong style="color: #38BDF8;">Expected Impact:</strong>
                        <span style="color: #CBD5E1;"> {rec.impact}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if rec.tags:
            tag_html = " ".join(
                f'<span style="background: #1E293B; border: 1px solid rgba(255,255,255,0.08); color: #94A3B8; border-radius: 4px; padding: 3px 9px; font-size: 0.74rem; margin-right: 6px;">#{t}</span>'
                for t in rec.tags
            )
            st.markdown(f"<div style='margin-top: 6px; margin-bottom: 4px;'>{tag_html}</div>", unsafe_allow_html=True)


def domain_recs(domain: Domain) -> list:
    return [r for r in report.recommendations if r.domain == domain]


DOMAIN_ICONS = {
    "Savings": "fa-wallet",
    "Investment": "fa-chart-pie",
    "Debt": "fa-credit-card",
    "Emergency Fund": "fa-shield-halved",
    "Tax Efficiency": "fa-file-invoice-dollar",
}


# ══════════════════════════════════════════════════════════════════════════════
# TOP METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════

risk_colors = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444", "Critical": "#7C3AED"}
risk_classes = {
    "Low": "fa-solid fa-circle-check",
    "Moderate": "fa-solid fa-triangle-exclamation",
    "High": "fa-solid fa-circle-exclamation",
    "Critical": "fa-solid fa-circle-xmark",
}
risk_c = risk_colors.get(report.risk_level, "#6B7280")
fa_class = risk_classes.get(report.risk_level, "fa-solid fa-circle-info")

score_c = "#10B981" if report.overall_score >= 80 else ("#3B82F6" if report.overall_score >= 60 else ("#F59E0B" if report.overall_score >= 40 else "#EF4444"))
score_status = "Strong" if report.overall_score >= 80 else ("Good" if report.overall_score >= 60 else ("Needs Attention" if report.overall_score >= 40 else "High Risk"))

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="coach-metric-box-featured">
        <div>
            <div class="coach-metric-label">Overall Coach Score</div>
            <div class="coach-metric-val" style="color: {score_c};">{report.overall_score:.0f} <span style="font-size: 1rem; color: #94A3B8; font-weight: 500;">/ 100</span></div>
        </div>
        <div class="coach-metric-sub" style="color: {score_c}; font-weight: 600;">
            <i class="fa-solid fa-gauge-high" style="margin-right: 4px;"></i> {score_status} Health
        </div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="coach-metric-box">
        <div>
            <div class="coach-metric-label">Risk Profile</div>
            <div class="coach-metric-val" style="color: {risk_c}; font-size: 1.5rem;">
                <i class="{fa_class}" style="font-size: 1.15rem; margin-right: 4px;"></i> {report.risk_level}
            </div>
        </div>
        <div class="coach-metric-sub">
            Composite risk assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    crit_count = len(report.critical_alerts)
    crit_label = f"{crit_count} Critical" if crit_count > 0 else "None Critical"
    crit_color = "#EF4444" if crit_count > 0 else "#10B981"
    st.markdown(f"""
    <div class="coach-metric-box">
        <div>
            <div class="coach-metric-label">Active Alerts</div>
            <div class="coach-metric-val">{len(report.alerts)}</div>
        </div>
        <div class="coach-metric-sub" style="color: {crit_color}; font-weight: 600;">
            {crit_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    high_rec_count = len([r for r in report.recommendations if r.severity in (Severity.CRITICAL, Severity.HIGH)])
    st.markdown(f"""
    <div class="coach-metric-box">
        <div>
            <div class="coach-metric-label">Action Items</div>
            <div class="coach-metric-val">{len(report.recommendations)}</div>
        </div>
        <div class="coach-metric-sub" style="color: #38BDF8;">
            {high_rec_count} Priority recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_home, tab_sav, tab_inv, tab_debt, tab_ef, tab_tax = st.tabs([
    ":material/dashboard: Coach Summary",
    ":material/savings: Savings",
    ":material/query_stats: Investment",
    ":material/credit_card: Debt",
    ":material/shield: Emergency Fund",
    ":material/receipt_long: Tax Efficiency",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Coach Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    # ── Executive Summary ──
    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-compass" style="color: #4F8CFF; font-size: 1.1rem;"></i> Executive Summary
        </h3>
        <p class="coach-section-desc">Personalized high-level assessment synthesized across all financial dimensions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="coach-narrative-box">
        {report.narrative}
    </div>
    """, unsafe_allow_html=True)

    # ── #1 Priority Action ──
    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-bullseye" style="color: #F59E0B; font-size: 1.1rem;"></i> Top Priority Action
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="coach-priority-box">
        <div class="coach-priority-badge">
            <i class="fa-solid fa-bolt"></i> Immediate Action Required
        </div>
        <p class="coach-priority-text">{report.top_action}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Domain Scorecards ──
    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-layer-group" style="color: #38BDF8; font-size: 1.1rem;"></i> Domain Scorecards
        </h3>
        <p class="coach-section-desc">Evaluated performance scores across the 5 core financial pillars.</p>
    </div>
    """, unsafe_allow_html=True)

    g_cols = st.columns(5)
    for i, (domain_name, summary) in enumerate(report.domain_summaries.items()):
        with g_cols[i]:
            icon_cls = DOMAIN_ICONS.get(domain_name, "fa-circle-dot")
            grade_color = "#10B981" if summary.score >= 80 else ("#3B82F6" if summary.score >= 60 else ("#F59E0B" if summary.score >= 40 else "#EF4444"))
            grade_bg = "rgba(16, 185, 129, 0.15)" if summary.score >= 80 else ("rgba(59, 130, 246, 0.15)" if summary.score >= 60 else ("rgba(245, 158, 11, 0.15)" if summary.score >= 40 else "rgba(239, 68, 68, 0.15)"))

            st.markdown(f"""
            <div class="domain-card">
                <div>
                    <div class="domain-card-icon">
                        <i class="fa-solid {icon_cls}"></i>
                    </div>
                    <div class="domain-card-title">{domain_name}</div>
                    <div class="domain-card-score" style="color: {grade_color};">{summary.score:.0f}<span style="font-size: 0.85rem; color: #64748B; font-weight: 500;">/100</span></div>
                    <div class="domain-grade-badge" style="background: {grade_bg}; color: {grade_color}; border: 1px solid {grade_color}40;">
                        Grade {summary.grade}
                    </div>
                </div>
                <div>
                    <div class="domain-bar-bg">
                        <div class="domain-bar-fill" style="width: {min(max(summary.score, 0), 100)}%; background: {grade_color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Risk Alerts ──
    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-triangle-exclamation" style="color: #EF4444; font-size: 1.1rem;"></i> Risk Alerts & Notifications
        </h3>
        <p class="coach-section-desc">Identified vulnerabilities and budget imbalances requiring user intervention.</p>
    </div>
    """, unsafe_allow_html=True)

    if report.alerts:
        for alert in report.alerts:
            render_alert(alert)
    else:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.10); border: 1.5px solid rgba(16, 185, 129, 0.35); border-radius: 10px; padding: 14px 18px; color: #10B981; font-size: 0.92rem; display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
            <i class="fa-solid fa-circle-check" style="font-size: 1.2rem;"></i>
            <span>No active risk alerts. Your financial foundations are currently in sound standing.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Top Priority Recommendations ──
    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-list-check" style="color: #10B981; font-size: 1.1rem;"></i> Top Action Recommendations
        </h3>
        <p class="coach-section-desc">High-impact steps prioritized by financial urgency and quantitative outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    top3 = [r for r in report.recommendations if r.severity in (Severity.CRITICAL, Severity.HIGH)][:5]
    if not top3:
        top3 = report.recommendations[:5]
    for rec in top3:
        render_recommendation(rec, expanded=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Savings
# ══════════════════════════════════════════════════════════════════════════════
with tab_sav:
    sav_sum = report.domain_summaries.get("Savings")
    if sav_sum:
        grade_c = "#10B981" if sav_sum.score >= 80 else ("#3B82F6" if sav_sum.score >= 60 else ("#F59E0B" if sav_sum.score >= 40 else "#EF4444"))
        grade_bg = f"{grade_c}20"

        st.markdown(f"""
        <div class="domain-header-card">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #F8FAFC;">
                        Savings Habits & Cash Flow
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700;">
                        Grade {sav_sum.grade} ({sav_sum.score:.0f}/100)
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.92rem;">{sav_sum.headline}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.9], gap="medium")
        with c1:
            st.markdown(f"""
            <div class="domain-score-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #F8FAFC;">
                        Savings Score
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700;">
                        Grade {sav_sum.grade}
                    </span>
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: {grade_c}; margin-bottom: 10px;">
                    {sav_sum.score:.0f} <span style="font-size: 1rem; color: #64748B; font-weight: 500;">/ 100</span>
                </div>
                <div class="domain-bar-bg" style="margin-bottom: 18px;">
                    <div class="domain-bar-fill" style="width: {min(max(sav_sum.score, 0), 100)}%; background: {grade_c};"></div>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 12px; margin-top: 6px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.92rem; color: #F8FAFC; margin-bottom: 8px;">
                        Key Savings Metrics
                    </div>
            """, unsafe_allow_html=True)
            for k, v in sav_sum.metrics.items():
                st.markdown(f"""
                <div class="domain-metric-row">
                    <span class="domain-metric-label">{k}</span>
                    <span class="domain-metric-val">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""<div class="domain-insights-panel">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.98rem; color: #F8FAFC; margin-bottom: 10px;">
                    Quick Insights & Suggestions
                </div>""", unsafe_allow_html=True)
            for s in sav_sum.suggestions:
                st.markdown(f"""
                <div class="domain-suggestion-item">
                    <i class="fa-solid fa-circle-arrow-right domain-suggestion-icon"></i>
                    <span>{s}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Expense breakdown mini-chart
            disc_data = {
                "Food Delivery":   twin.food_delivery,
                "Entertainment":   twin.entertainment,
                "Shopping":        twin.shopping,
                "Rent":            twin.rent,
                "Groceries":       twin.groceries,
                "Utilities":       twin.utilities,
                "Transport":       twin.transport,
            }
            spend_items = [{"Category": k, "Amount (₹)": v} for k, v in disc_data.items() if v > 0]
            if spend_items:
                df_disc = pd.DataFrame(spend_items).sort_values("Amount (₹)", ascending=False)
                fig_pie = go.Figure(go.Pie(
                    labels       = df_disc["Category"],
                    values       = df_disc["Amount (₹)"],
                    hole         = 0.55,
                    textinfo     = "label+percent",
                    textposition = "outside",
                    marker       = {"colors": [
                        "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4"
                    ]},
                ))
                fig_pie.update_layout(
                    title         = {"text": "Discretionary & Essential Spend Split", "font": {"size": 13, "color": "#94A3B8"}},
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                    height        = 280,
                    showlegend    = False,
                    margin        = dict(l=10, r=10, t=35, b=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_savings_breakdown")
            else:
                st.info("No expense outflows recorded yet. Update your budget in Digital Twin to visualize breakdown.")

    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-lightbulb" style="color: #4F8CFF; font-size: 1.1rem;"></i> Savings Optimization Recommendations
        </h3>
    </div>
    """, unsafe_allow_html=True)
    for rec in domain_recs(Domain.SAVINGS):
        render_recommendation(rec, expanded=(rec.severity in (Severity.CRITICAL, Severity.HIGH)))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Investment
# ══════════════════════════════════════════════════════════════════════════════
with tab_inv:
    inv_sum = report.domain_summaries.get("Investment")
    if inv_sum:
        grade_c = "#10B981" if inv_sum.score >= 80 else ("#3B82F6" if inv_sum.score >= 60 else ("#F59E0B" if inv_sum.score >= 40 else "#EF4444"))
        grade_bg = f"{grade_c}20"

        st.markdown(f"""
        <div class="domain-header-card">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #F8FAFC;">
                        Investment Portfolio & SIP Adequacy
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700;">
                        Grade {inv_sum.grade} ({inv_sum.score:.0f}/100)
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.92rem;">{inv_sum.headline}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.9], gap="medium")
        with c1:
            st.markdown(f"""
            <div class="domain-score-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #F8FAFC;">
                        Investment Score
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700;">
                        Grade {inv_sum.grade}
                    </span>
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: {grade_c}; margin-bottom: 10px;">
                    {inv_sum.score:.0f} <span style="font-size: 1rem; color: #64748B; font-weight: 500;">/ 100</span>
                </div>
                <div class="domain-bar-bg" style="margin-bottom: 18px;">
                    <div class="domain-bar-fill" style="width: {min(max(inv_sum.score, 0), 100)}%; background: {grade_c};"></div>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 12px; margin-top: 6px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.92rem; color: #F8FAFC; margin-bottom: 8px;">
                        Key Investment Metrics
                    </div>
            """, unsafe_allow_html=True)
            for k, v in inv_sum.metrics.items():
                st.markdown(f"""
                <div class="domain-metric-row">
                    <span class="domain-metric-label">{k}</span>
                    <span class="domain-metric-val">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""<div class="domain-insights-panel">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.98rem; color: #F8FAFC; margin-bottom: 10px;">
                    Strategic Insights
                </div>""", unsafe_allow_html=True)
            for s in inv_sum.suggestions:
                st.markdown(f"""
                <div class="domain-suggestion-item">
                    <i class="fa-solid fa-circle-arrow-right domain-suggestion-icon"></i>
                    <span>{s}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Investment corpus donut
            corpus_data = {
                "Mutual Funds":   twin.mutual_funds,
                "Stocks":         twin.stocks,
                "PPF":            twin.ppf_investment,
                "NPS":            twin.nps_investment,
                "Fixed Deposits": twin.fd_amount,
                "Bank Savings":   twin.bank_savings,
            }
            corpus_data = {k: v for k, v in corpus_data.items() if v > 0}
            if corpus_data:
                fig_corp = go.Figure(go.Pie(
                    labels       = list(corpus_data.keys()),
                    values       = list(corpus_data.values()),
                    hole         = 0.55,
                    textinfo     = "label+percent",
                    textposition = "outside",
                    marker       = {"colors": [
                        "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"
                    ]},
                ))
                fig_corp.update_layout(
                    title         = {"text": "Investment Corpus Allocation", "font": {"size": 13, "color": "#94A3B8"}},
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                    height        = 280,
                    showlegend    = False,
                    margin        = dict(l=10, r=10, t=35, b=10),
                )
                st.plotly_chart(fig_corp, use_container_width=True, key="pie_investment_corpus")
            else:
                st.warning("No investment corpus detected. Start your first SIP today to build compounding wealth.")

    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-lightbulb" style="color: #4F8CFF; font-size: 1.1rem;"></i> Investment Recommendations
        </h3>
    </div>
    """, unsafe_allow_html=True)
    for rec in domain_recs(Domain.INVESTMENT):
        render_recommendation(rec, expanded=(rec.severity in (Severity.CRITICAL, Severity.HIGH)))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Debt
# ══════════════════════════════════════════════════════════════════════════════
with tab_debt:
    debt_sum = report.domain_summaries.get("Debt")
    if debt_sum:
        grade_c = "#10B981" if debt_sum.score >= 80 else ("#3B82F6" if debt_sum.score >= 60 else ("#F59E0B" if debt_sum.score >= 40 else "#EF4444"))
        grade_bg = f"{grade_c}20"

        st.markdown(f"""
        <div class="domain-header-card">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #F8FAFC;">
                        Debt Burden & Leverage Analysis
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700;">
                        Grade {debt_sum.grade} ({debt_sum.score:.0f}/100)
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.92rem;">{debt_sum.headline}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.9], gap="medium")
        with c1:
            st.markdown(f"""
            <div class="domain-score-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #F8FAFC;">
                        Debt Score
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700;">
                        Grade {debt_sum.grade}
                    </span>
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: {grade_c}; margin-bottom: 10px;">
                    {debt_sum.score:.0f} <span style="font-size: 1rem; color: #64748B; font-weight: 500;">/ 100</span>
                </div>
                <div class="domain-bar-bg" style="margin-bottom: 18px;">
                    <div class="domain-bar-fill" style="width: {min(max(debt_sum.score, 0), 100)}%; background: {grade_c};"></div>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 12px; margin-top: 6px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.92rem; color: #F8FAFC; margin-bottom: 8px;">
                        Key Debt Metrics
                    </div>
            """, unsafe_allow_html=True)
            for k, v in debt_sum.metrics.items():
                st.markdown(f"""
                <div class="domain-metric-row">
                    <span class="domain-metric-label">{k}</span>
                    <span class="domain-metric-val">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""<div class="domain-insights-panel">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.98rem; color: #F8FAFC; margin-bottom: 10px;">
                    Debt Management Nudges
                </div>""", unsafe_allow_html=True)
            for s in debt_sum.suggestions:
                st.markdown(f"""
                <div class="domain-suggestion-item">
                    <i class="fa-solid fa-circle-arrow-right domain-suggestion-icon"></i>
                    <span>{s}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Loan breakdown bar chart
            loan_data = {
                "Home Loan":         twin.home_loan,
                "Car Loan":          twin.car_loan,
                "Credit Card Debt":  twin.credit_card_debt,
                "Other Loans":       max(twin.loan_amount - twin.home_loan - twin.car_loan - twin.credit_card_debt, 0),
            }
            loan_data = {k: v for k, v in loan_data.items() if v > 0}
            if loan_data:
                bar_colors = {
                    "Home Loan":        "#3B82F6",
                    "Car Loan":         "#10B981",
                    "Credit Card Debt": "#EF4444",
                    "Other Loans":      "#F59E0B",
                }
                fig_loans = go.Figure(go.Bar(
                    x            = list(loan_data.keys()),
                    y            = list(loan_data.values()),
                    marker_color = [bar_colors.get(k, "#6B7280") for k in loan_data.keys()],
                    text         = [f"₹{v:,.0f}" for v in loan_data.values()],
                    textposition = "outside",
                ))
                fig_loans.update_layout(
                    title         = {"text": "Outstanding Liability Breakdown", "font": {"size": 13, "color": "#94A3B8"}},
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                    height        = 280,
                    showlegend    = False,
                    margin        = dict(l=10, r=10, t=35, b=10),
                    yaxis         = {"gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
                    xaxis         = {"gridcolor": "rgba(0,0,0,0)"},
                )
                st.plotly_chart(fig_loans, use_container_width=True, key="bar_debt_loans")
            else:
                st.success("No outstanding loans or debt detected! Excellent balance sheet management.")

    # Domain alerts
    debt_alerts = [a for a in report.alerts if a.domain == Domain.DEBT]
    if debt_alerts:
        st.markdown("""
        <div class="coach-section-header">
            <h3 class="coach-section-title">
                <i class="fa-solid fa-triangle-exclamation" style="color: #EF4444; font-size: 1.1rem;"></i> Debt Risk Alerts
            </h3>
        </div>
        """, unsafe_allow_html=True)
        for alert in debt_alerts:
            render_alert(alert)

    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-lightbulb" style="color: #4F8CFF; font-size: 1.1rem;"></i> Debt Reduction Strategies
        </h3>
    </div>
    """, unsafe_allow_html=True)
    for rec in domain_recs(Domain.DEBT):
        render_recommendation(rec, expanded=(rec.severity in (Severity.CRITICAL, Severity.HIGH)))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Emergency Fund
# ══════════════════════════════════════════════════════════════════════════════
with tab_ef:
    ef_sum = report.domain_summaries.get("Emergency Fund")
    if ef_sum:
        grade_c = "#10B981" if ef_sum.score >= 80 else ("#3B82F6" if ef_sum.score >= 60 else ("#F59E0B" if ef_sum.score >= 40 else "#EF4444"))
        grade_bg = f"{grade_c}20"

        st.markdown(f"""
        <div class="domain-header-card">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #F8FAFC;">
                        Emergency Fund & Liquidity Buffer
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700;">
                        Grade {ef_sum.grade} ({ef_sum.score:.0f}/100)
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.92rem;">{ef_sum.headline}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.9], gap="medium")
        with c1:
            st.markdown(f"""
            <div class="domain-score-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #F8FAFC;">
                        Emergency Fund Score
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700;">
                        Grade {ef_sum.grade}
                    </span>
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: {grade_c}; margin-bottom: 10px;">
                    {ef_sum.score:.0f} <span style="font-size: 1rem; color: #64748B; font-weight: 500;">/ 100</span>
                </div>
                <div class="domain-bar-bg" style="margin-bottom: 18px;">
                    <div class="domain-bar-fill" style="width: {min(max(ef_sum.score, 0), 100)}%; background: {grade_c};"></div>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 12px; margin-top: 6px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.92rem; color: #F8FAFC; margin-bottom: 8px;">
                        Key Liquidity Metrics
                    </div>
            """, unsafe_allow_html=True)
            for k, v in ef_sum.metrics.items():
                st.markdown(f"""
                <div class="domain-metric-row">
                    <span class="domain-metric-label">{k}</span>
                    <span class="domain-metric-val">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""<div class="domain-insights-panel">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.98rem; color: #F8FAFC; margin-bottom: 10px;">
                    Buffer Targets & Guidance
                </div>""", unsafe_allow_html=True)
            for s in ef_sum.suggestions:
                st.markdown(f"""
                <div class="domain-suggestion-item">
                    <i class="fa-solid fa-circle-arrow-right domain-suggestion-icon"></i>
                    <span>{s}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Progress bar toward 6-month target
            monthly_out  = twin.basic_expenses + twin.monthly_emi
            target_ef    = monthly_out * 6
            current_ef   = twin.emergency_fund
            pct_complete = min(current_ef / max(target_ef, 1), 1.0)

            fig_prog = go.Figure()
            fig_prog.add_trace(go.Bar(
                x    = ["Coverage"],
                y    = [current_ef],
                name = f"Current (₹{current_ef:,.0f})",
                marker_color = "#3B82F6",
                text = [f"₹{current_ef:,.0f} ({pct_complete:.0%})"],
                textposition = "inside",
            ))
            fig_prog.add_trace(go.Bar(
                x    = ["Coverage"],
                y    = [max(target_ef - current_ef, 0)],
                name = f"Gap (₹{max(target_ef - current_ef, 0):,.0f})",
                marker_color = "rgba(255, 255, 255, 0.1)",
                text = [f"₹{max(target_ef - current_ef, 0):,.0f} deficit" if target_ef > current_ef else ""],
                textposition = "inside",
            ))
            fig_prog.add_hline(
                y = target_ef,
                line_color = "#10B981", line_width = 2, line_dash = "dash",
                annotation_text = f"6-Month Safe Target: ₹{target_ef:,.0f}",
                annotation_font_color = "#10B981",
            )
            fig_prog.update_layout(
                barmode       = "stack",
                title         = {"text": "6-Month Target Coverage", "font": {"size": 13, "color": "#94A3B8"}},
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                height        = 260,
                showlegend    = True,
                legend        = {"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": -0.2},
                yaxis         = {"title": "Amount (₹)", "gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
                xaxis         = {"visible": False},
                margin        = dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_prog, use_container_width=True, key="bar_ef_progress")

    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-lightbulb" style="color: #4F8CFF; font-size: 1.1rem;"></i> Liquidity Building Recommendations
        </h3>
    </div>
    """, unsafe_allow_html=True)
    for rec in domain_recs(Domain.EMERGENCY):
        render_recommendation(rec, expanded=(rec.severity in (Severity.CRITICAL, Severity.HIGH)))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Tax Efficiency
# ══════════════════════════════════════════════════════════════════════════════
with tab_tax:
    tax_sum = report.domain_summaries.get("Tax Efficiency")
    if tax_sum:
        grade_c = "#10B981" if tax_sum.score >= 80 else ("#3B82F6" if tax_sum.score >= 60 else ("#F59E0B" if tax_sum.score >= 40 else "#EF4444"))
        grade_bg = f"{grade_c}20"

        st.markdown(f"""
        <div class="domain-header-card">
            <div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #F8FAFC;">
                        Tax Efficiency & Deduction Optimization
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700;">
                        Grade {tax_sum.grade} ({tax_sum.score:.0f}/100)
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.92rem;">{tax_sum.headline}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.9], gap="medium")
        with c1:
            st.markdown(f"""
            <div class="domain-score-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #F8FAFC;">
                        Tax Efficiency Score
                    </div>
                    <span style="background: {grade_bg}; color: {grade_c}; border: 1px solid {grade_c}50; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700;">
                        Grade {tax_sum.grade}
                    </span>
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: {grade_c}; margin-bottom: 10px;">
                    {tax_sum.score:.0f} <span style="font-size: 1rem; color: #64748B; font-weight: 500;">/ 100</span>
                </div>
                <div class="domain-bar-bg" style="margin-bottom: 18px;">
                    <div class="domain-bar-fill" style="width: {min(max(tax_sum.score, 0), 100)}%; background: {grade_c};"></div>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 12px; margin-top: 6px;">
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.92rem; color: #F8FAFC; margin-bottom: 8px;">
                        Key Tax Metrics
                    </div>
            """, unsafe_allow_html=True)
            for k, v in tax_sum.metrics.items():
                st.markdown(f"""
                <div class="domain-metric-row">
                    <span class="domain-metric-label">{k}</span>
                    <span class="domain-metric-val">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""<div class="domain-insights-panel">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.98rem; color: #F8FAFC; margin-bottom: 10px;">
                    Tax Planning Strategies
                </div>""", unsafe_allow_html=True)
            for s in tax_sum.suggestions:
                st.markdown(f"""
                <div class="domain-suggestion-item">
                    <i class="fa-solid fa-circle-arrow-right domain-suggestion-icon"></i>
                    <span>{s}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Deduction utilisation bar chart
            _80c_limit = 150_000
            _nps_limit = 50_000
            _80d_limit = 25_000
            ppf_pa    = twin.ppf_investment
            sip_elss  = min(twin.sip_amount * 12 * 0.30, _80c_limit)
            life_prem = min(twin.life_insurance * 0.01, 50_000)
            used_80c  = min(ppf_pa + sip_elss + life_prem, _80c_limit)
            used_nps  = min(twin.nps_investment, _nps_limit)
            used_80d  = min(twin.health_insurance * 0.02, _80d_limit)

            categories = ["Sec 80C", "Sec 80CCD(1B) NPS", "Sec 80D Health"]
            used_vals   = [used_80c,  used_nps,  used_80d]
            unused_vals = [max(_80c_limit - used_80c, 0),
                           max(_nps_limit - used_nps, 0),
                           max(_80d_limit - used_80d, 0)]

            fig_ded = go.Figure()
            fig_ded.add_trace(go.Bar(
                name         = "Utilised",
                x            = categories,
                y            = used_vals,
                marker_color = "#10B981",
                text         = [f"₹{v:,.0f}" for v in used_vals],
                textposition = "inside",
            ))
            fig_ded.add_trace(go.Bar(
                name         = "Unused Headroom",
                x            = categories,
                y            = unused_vals,
                marker_color = "rgba(255, 255, 255, 0.1)",
                text         = [f"₹{v:,.0f}" if v > 0 else "" for v in unused_vals],
                textposition = "inside",
            ))
            fig_ded.update_layout(
                barmode       = "stack",
                title         = {"text": "Statutory Deduction Headroom", "font": {"size": 13, "color": "#94A3B8"}},
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                height        = 280,
                legend        = {"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": -0.2},
                yaxis         = {"title": "Amount (₹)", "gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
                xaxis         = {"gridcolor": "rgba(0,0,0,0)"},
                margin        = dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_ded, use_container_width=True, key="bar_tax_deduction")

    st.markdown("""
    <div class="coach-section-header">
        <h3 class="coach-section-title">
            <i class="fa-solid fa-lightbulb" style="color: #4F8CFF; font-size: 1.1rem;"></i> Tax Optimization Suggestions
        </h3>
    </div>
    """, unsafe_allow_html=True)
    for rec in domain_recs(Domain.TAX):
        render_recommendation(rec, expanded=True)

    st.markdown("""
    <div style="background: rgba(59, 130, 246, 0.10); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 14px 18px; margin-top: 1.5rem; color: #93C5FD; font-size: 0.9rem; display: flex; align-items: center; gap: 12px;">
        <i class="fa-solid fa-calculator" style="font-size: 1.25rem; color: #60A5FA;"></i>
        <span>
            For exact tax computation under Old vs New regime with full deduction breakdown, visit the <strong>Tax Intelligence</strong> page.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT EXPORT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)

try:
    _pdf_coach = FinTwinPDFReport(
        report_title="AI Financial Coach Advisory Report",
        user_id=twin.user_id,
        report_id=f"AC-{twin.user_id}",
        subtitle=f"Overall Score: {report.overall_score:.0f}/100 | Risk Level: {report.risk_level}",
    )
    _pdf_coach.add_cover_page(
        user_name=getattr(twin, "name", "") or "",
        report_type="AI Financial Coach Advisory Report",
    )
    _pdf_coach.add_kpi_summary_row([
        {"label": "Overall Score", "value": f"{report.overall_score:.0f} / 100",
         "status": "positive" if report.overall_score >= 70 else ("negative" if report.overall_score < 50 else "neutral")},
        {"label": "Risk Level", "value": str(report.risk_level), "status": "neutral"},
        {"label": "Top Priority Action", "value": str(report.top_action)[:50] + ("..." if len(str(report.top_action)) > 50 else ""), "status": "neutral"},
    ])

    _pdf_coach.add_section_divider("Executive Summary")
    if report.narrative:
        _pdf_coach.add_executive_summary(report.narrative)

    if report.alerts:
        _pdf_coach.add_section_divider("Risk Alerts")
        for a in report.alerts[:5]:
            sev = str(getattr(a, "severity", "") or "").lower()
            cs = "danger" if "critical" in sev or "high" in sev else "warning"
            headline = getattr(a, "headline", "")
            message = getattr(a, "message", "")
            _pdf_coach.add_callout(f"{message}", style=cs, title=headline)
        if len(report.alerts) > 5:
            alerts_df = pd.DataFrame([
                {"Domain": a.domain.value if hasattr(a.domain, "value") else str(a.domain),
                 "Severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                 "Headline": a.headline}
                for a in report.alerts[5:]
            ])
            _pdf_coach.add_dataframe_table(alerts_df)

    if report.recommendations:
        _pdf_coach.add_section_divider("Priority Recommendations")
        for r in report.recommendations[:3]:
            sev = str(getattr(r, "severity", "") or "").lower()
            cs = "danger" if "critical" in sev or "high" in sev else ("warning" if "medium" in sev else "info")
            _pdf_coach.add_callout(
                f"{r.action} \u2014 Impact: {r.impact}",
                style=cs, title=r.title
            )
        if len(report.recommendations) > 3:
            _pdf_coach.add_section("All Recommendations", level=2)
            rec_df = pd.DataFrame([
                {"Domain": r.domain.value if hasattr(r.domain, "value") else str(r.domain),
                 "Severity": r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                 "Title": r.title, "Action": r.action, "Impact": r.impact}
                for r in report.recommendations[3:]
            ])
            _pdf_coach.add_dataframe_table(rec_df)

    if report.domain_summaries:
        _pdf_coach.add_section_divider("Domain Summaries")
        dom_df = pd.DataFrame([
            {"Domain": k, "Score": f"{v.score:.0f}", "Grade": v.grade, "Headline": v.headline}
            for k, v in report.domain_summaries.items()
        ])
        _pdf_coach.add_dataframe_table(dom_df)

    pdf_coach_bytes = _pdf_coach.build()
    pdf_coach_fn = _pdf_coach.suggested_filename("ai_coach_advisory_report")
except Exception as e:
    pdf_coach_bytes = None
    pdf_coach_fn = "ai_coach_advisory_report.pdf"

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
                    AI Coach Advisory Report (PDF)
                </div>
                <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                    Download your personalized financial coaching report, risk alerts, and action roadmap.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_rep2:
        if pdf_coach_bytes:
            st.download_button(
                label="Download AI Coach Report",
                icon=":material/download:",
                data=pdf_coach_bytes,
                file_name=pdf_coach_fn,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="btn_download_coach_report"
            )
        else:
            st.error("Report is currently unavailable.")

