"""
Streamlit Page: Goal Planning Engine  (Module 11)
================================================
Comprehensive goal planning system with single-goal calculator,
multi-goal portfolio optimizer, and cross-goal comparison analytics.
"""

import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.session import render_sidebar_user_selector
from utils.goal_engine import (
    GoalType, GoalInput, GoalResult, GoalEngine,
    MultiGoalPlanner, GOAL_CONFIG,
)
from utils.pdf_report import FinTwinPDFReport

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Goal Planner — FinTwin AI",
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
.goal-header-box {
    margin-bottom: 2.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.goal-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.03em;
}
.goal-subtitle {
    color: #94A3B8;
    font-size: 0.96rem;
    margin: 0;
    line-height: 1.55;
}

/* Section Header */
.goal-section-header {
    margin: 2.25rem 0 1.25rem 0;
}
.goal-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.goal-section-desc {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 0 0 1.15rem 0;
    line-height: 1.5;
}

/* Metric Display Boxes */
.goal-metric-box {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.goal-metric-box:hover {
    border-color: rgba(79, 140, 255, 0.4);
    transform: translateY(-1px);
}
.goal-metric-box-featured {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(79, 140, 255, 0.45);
    border-top: 3px solid #4F8CFF;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
}
.goal-metric-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    color: #94A3B8;
    margin-bottom: 0.45rem;
}
.goal-metric-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.25;
    margin-bottom: 0.3rem;
}
.goal-metric-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}

/* Callout / Narrative Card */
.goal-narrative-box {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #38BDF8;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.25rem 0 2rem 0;
    color: #E2E8F0;
    font-size: 0.94rem;
    line-height: 1.65;
}

/* Tip Accent Box */
.goal-tip-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #4F8CFF;
    border-radius: 10px;
    padding: 0.95rem 1.25rem;
    margin-bottom: 0.75rem;
    color: #E2E8F0;
    font-size: 0.92rem;
    line-height: 1.55;
}

/* Expanders Spacing */
div[data-testid="stExpander"] {
    margin-top: 1rem !important;
    margin-bottom: 1.75rem !important;
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

# ── Sidebar ────────────────────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="goal-header-box">
    <h1 class="goal-title">Goal Planner</h1>
    <p class="goal-subtitle">Plan and track your financial goals with personalized savings and investment projections.</p>
</div>
""", unsafe_allow_html=True)

if not twin:
    st.markdown("""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; text-align: center; padding: 2.5rem 1.5rem; max-width: 680px; margin: 2rem auto; box-shadow: 0 4px 14px rgba(0,0,0,0.2);">
        <i class="fa-solid fa-bullseye" style="color: #4F8CFF; font-size: 2.2rem; margin-bottom: 0.85rem; display: inline-block;"></i>
        <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.5rem 0;">
            Complete Your Financial Profile
        </h2>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5; margin: 0 0 1.5rem 0;">
            Log in or load your financial twin profile in the sidebar to simulate personalized goal trajectories and SIP allocations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1.4, 1])
    with c_btn2:
        st.page_link("pages/01_Digital_Twin.py", label="Complete Financial Profile →", icon=":material/arrow_forward:", use_container_width=True)
    st.stop()

# ── Derived constants from twin ────────────────────────────────────────────────
_monthly_surplus = max(twin.total_income - twin.basic_expenses - twin.monthly_emi, 0.0)
_today = datetime.date.today()
_min_date = _today + datetime.timedelta(days=30)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: render probability badge
# ══════════════════════════════════════════════════════════════════════════════
def prob_badge(prob: float, label: str) -> str:
    c = "#10B981" if prob >= 0.80 else ("#F59E0B" if prob >= 0.55 else ("#EF4444" if prob >= 0.30 else "#7C3AED"))
    icon_cls = "fa-circle-check" if prob >= 0.80 else ("fa-triangle-exclamation" if prob >= 0.55 else "fa-circle-xmark")
    return f"""<span style="background:{c}22;border:1.5px solid {c};color:{c};border-radius:8px;padding:4px 14px;font-weight:700;font-size:0.92rem;display:inline-flex;align-items:center;gap:6px;"><i class="fa-solid {icon_cls}"></i>{prob:.0%} — {label}</span>"""


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build trajectory figure
# ══════════════════════════════════════════════════════════════════════════════
def build_trajectory_chart(result: GoalResult, show_milestones: bool = True) -> go.Figure:
    df = result.trajectory
    target = result.inflation_adj_goal
    color  = result.color
    target_months = result.goal_input.months_remaining

    fig = go.Figure()

    fill_color_rgba = color
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_color_rgba = f"rgba({r}, {g}, {b}, 0.08)"

    # Corpus area fill
    fig.add_trace(go.Scatter(
        x    = df["Month"],
        y    = df["Corpus"],
        name = "Projected Corpus",
        mode = "lines",
        line = {"color": color, "width": 2.5},
        fill = "tozeroy",
        fillcolor = fill_color_rgba,
        hovertemplate = "Month %{x}<br>Corpus: ₹%{y:,.0f}<extra>Corpus</extra>",
    ))

    # SIP cumulative
    fig.add_trace(go.Scatter(
        x    = df["Month"],
        y    = df["SIP_Cumulative"],
        name = "SIP Contributions",
        mode = "lines",
        line = {"color": "#64748B", "width": 1.5, "dash": "dot"},
        hovertemplate = "Month %{x}<br>SIP: ₹%{y:,.0f}<extra>SIP Cumul.</extra>",
    ))

    # Target horizontal line
    fig.add_hline(
        y                = target,
        line_color       = "#10B981",
        line_width       = 2,
        line_dash        = "dash",
        annotation_text  = f"Target ₹{target:,.0f}",
        annotation_position = "top right",
        annotation_font_color = "#10B981",
    )

    # Target date vertical line
    fig.add_vline(
        x                = target_months,
        line_color       = "#F59E0B",
        line_width       = 1.5,
        line_dash        = "dot",
        annotation_text  = "Target Date",
        annotation_position = "top left",
        annotation_font_color = "#F59E0B",
    )

    # Milestone markers
    if show_milestones and result.milestone_months:
        milestone_labels = ["25%", "50%", "75%", "100%"]
        milestone_pcts   = [0.25, 0.50, 0.75, 1.00]
        for m_month, m_label, m_pct in zip(result.milestone_months, milestone_labels, milestone_pcts):
            if 0 < m_month <= len(df):
                corpus_at_m = df[df["Month"] == m_month]["Corpus"].values
                if len(corpus_at_m) > 0:
                    fig.add_trace(go.Scatter(
                        x    = [m_month],
                        y    = [corpus_at_m[0]],
                        mode = "markers+text",
                        name = f"{m_label} milestone",
                        marker = {"size": 9, "color": "#F59E0B", "symbol": "diamond"},
                        text = [m_label],
                        textposition = "top center",
                        showlegend = False,
                        hovertemplate = f"Milestone {m_label}<br>Month %{{x}}<br>₹%{{y:,.0f}}<extra></extra>",
                    ))

    fig.update_layout(
        title         = f"{result.goal_name} — Corpus Trajectory",
        xaxis         = {"title": "Timeline (Months)", "gridcolor": "rgba(255, 255, 255, 0.06)"},
        yaxis         = {"title": "Corpus (₹)", "gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
        legend        = {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(255, 255, 255, 0.08)", "borderwidth": 1},
        height        = 420,
        margin        = dict(l=10, r=10, t=55, b=20),
        hovermode     = "x unified",
    )
    return fig


def build_corpus_breakdown_chart(result: GoalResult) -> go.Figure:
    """Stacked area: SIP contributions vs investment returns over time."""
    df = result.trajectory

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x         = df["Month"],
        y         = df["SIP_Cumulative"],
        name      = "Your Contributions",
        mode      = "lines",
        stackgroup = "one",
        fillcolor = "rgba(79, 140, 255, 0.45)",
        line      = {"color": "rgba(79, 140, 255, 0.8)", "width": 1.5},
        hovertemplate = "Month %{x}<br>Contributions: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x         = df["Month"],
        y         = df["Returns_Earned"].clip(lower=0),
        name      = "Compounded Returns",
        mode      = "lines",
        stackgroup = "one",
        fillcolor = "rgba(16, 185, 129, 0.45)",
        line      = {"color": "rgba(16, 185, 129, 0.8)", "width": 1.5},
        hovertemplate = "Month %{x}<br>Returns: ₹%{y:,.0f}<extra></extra>",
    ))

    fig.add_hline(
        y=result.inflation_adj_goal, line_color="#F59E0B", line_width=1.5, line_dash="dash",
        annotation_text=f"Target ₹{result.inflation_adj_goal:,.0f}",
        annotation_font_color="#F59E0B", annotation_position="top right",
    )

    fig.update_layout(
        title         = "Corpus Build-Up: Contributions vs Returns",
        xaxis         = {"title": "Timeline (Months)", "gridcolor": "rgba(255, 255, 255, 0.06)"},
        yaxis         = {"title": "Amount (₹)", "gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
        legend        = {"bgcolor": "rgba(0,0,0,0)"},
        height        = 380,
        margin        = dict(l=10, r=10, t=55, b=20),
        hovermode     = "x unified",
    )
    return fig


def build_probability_gauge(prob: float, goal_name: str, color: str) -> go.Figure:
    c = color
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        number = {"suffix": "%", "font": {"size": 34, "color": "#F8FAFC", "family": "Plus Jakarta Sans"}},
        title  = {"text": f"Estimated Success Probability", "font": {"size": 13, "color": "#94A3B8"}},
        gauge  = {
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#374151"},
            "bar":   {"color": c, "thickness": 0.25},
            "bgcolor": "#1E293B",
            "steps": [
                {"range": [0,  30], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [30, 55], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [55, 80], "color": "rgba(79, 140, 255, 0.15)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.15)"},
            ],
            "threshold": {"line": {"color": c, "width": 3}, "value": prob * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        height        = 240,
        margin        = dict(l=10, r=10, t=20, b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TWO-TAB STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    ":material/track_changes: Goal Calculator & Multi-Goal Planner",
    ":material/compare: Goal Comparison",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: GOAL CALCULATOR & MULTI-GOAL PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;">
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Financial Surplus Available</div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #10B981; margin-top: 2px;">
                ₹{0:,.0f}<span style="font-size: 0.85rem; color: #94A3B8; font-weight: 500;"> /month</span>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Current Liquid Net Worth</div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">
                ₹{1:,.0f}
            </div>
        </div>
    </div>
    """.format(_monthly_surplus, twin.net_worth), unsafe_allow_html=True)

    planner_mode = st.radio(
        "Select Planning Mode",
        ["Single Goal Deep-Dive", "Multi-Goal Portfolio Plan"],
        horizontal=True,
        label_visibility="collapsed",
        key="goal_planner_mode_selector",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-MODE A: SINGLE GOAL CALCULATOR
    # ─────────────────────────────────────────────────────────────────────────
    if planner_mode == "Single Goal Deep-Dive":
        st.markdown("""
        <div class="goal-section-header" style="margin-top: 1rem;">
            <h2 class="goal-section-title"><i class="fa-solid fa-bullseye" style="color: #4F8CFF; font-size: 1.15rem;"></i> Select Life Goal</h2>
            <p class="goal-section-desc">Choose a target milestone to calculate inflation-adjusted capital requirements and SIP allocations.</p>
        </div>
        """, unsafe_allow_html=True)

        goal_labels = [gt.value for gt in GoalType]
        selected_label = st.radio(
            "Select Goal Type:",
            goal_labels,
            horizontal=True,
            label_visibility="collapsed",
            key="single_goal_type_selector",
        )
        selected_goal_type = list(GoalType)[goal_labels.index(selected_label)]
        cfg = GOAL_CONFIG[selected_goal_type]

        st.markdown(f"""
        <div style="background:{cfg['bg']}; border: 1px solid {cfg['color']}44; border-left: 4px solid {cfg['color']}; border-radius: 10px; padding: 0.95rem 1.35rem; margin: 0.85rem 0 1.5rem 0; display: flex; align-items: center; gap: 12px;">
            <i class="fa-solid {cfg['icon_class']}" style="font-size: 1.35rem; color: {cfg['color']}; flex-shrink: 0;"></i>
            <div>
                <strong style="color: {cfg['color']}; font-size: 1.05rem;">{selected_goal_type.value}</strong>
                <span style="color: #94A3B8; font-size: 0.9rem; margin-left: 8px;">— {cfg['description']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Input form container ──
        st.markdown("""
        <div class="goal-section-header">
            <h3 class="goal-section-title"><i class="fa-solid fa-sliders" style="color: #4F8CFF; font-size: 1.1rem;"></i> Goal & Financial Assumptions</h3>
            <p class="goal-section-desc">Configure target valuation, horizon, inflation rate, and expected asset returns.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            form_col1, form_col2, form_col3 = st.columns(3)

            with form_col1:
                goal_amount = st.number_input(
                    "Target Goal Amount (₹)",
                    min_value        = 10_000,
                    max_value        = 100_000_000,
                    value            = cfg["default_amount"],
                    step             = 10_000,
                    help             = "Estimated cost in today's money.",
                    key              = "sg_goal_amount",
                )
                current_savings = st.number_input(
                    "Already Saved for this Goal (₹)",
                    min_value = 0,
                    max_value = 100_000_000,
                    value     = int(twin.bank_savings * 0.20),
                    step      = 5_000,
                    help      = "Savings already allocated specifically to this objective.",
                    key       = "sg_current_savings",
                )

            with form_col2:
                target_date = st.date_input(
                    "Target Completion Date",
                    value   = _today + datetime.timedelta(days=cfg["default_years"] * 365),
                    min_value = _min_date,
                    max_value = _today + datetime.timedelta(days=365 * 35),
                    help    = "Milestone date by which this capital is required.",
                    key     = "sg_target_date",
                )
                return_rate = st.slider(
                    "Expected Annual Return (%)",
                    min_value = 4.0,
                    max_value = 18.0,
                    value     = round(cfg["return_rate"] * 100, 1),
                    step      = 0.5,
                    help      = "Conservative: 6-7% (debt) · Moderate: 9-11% (hybrid) · Aggressive: 12-15% (equity)",
                    key       = "sg_return_rate",
                ) / 100

            with form_col3:
                inflation_rate = st.slider(
                    "Inflation Rate (%)",
                    min_value = 3.0,
                    max_value = 10.0,
                    value     = 6.0,
                    step      = 0.5,
                    help      = "Annual inflation rate to adjust target cost to future value.",
                    key       = "sg_inflation_rate",
                ) / 100
                monthly_commit = st.number_input(
                    "Monthly Contribution Commitment (₹)",
                    min_value = 0,
                    max_value = int(max(_monthly_surplus, 1)),
                    value     = min(int(_monthly_surplus * 0.4), cfg["default_amount"] // 60),
                    step      = 500,
                    help      = "How much monthly surplus you commit. (0 = auto-compute required SIP)",
                    key       = "sg_monthly_commit",
                )

        # ── Compute single goal ──
        gi = GoalInput(
            goal_type            = selected_goal_type,
            goal_amount          = float(goal_amount),
            current_savings      = float(current_savings),
            target_date          = target_date,
            monthly_contribution = float(monthly_commit),
            return_rate          = return_rate,
            inflation_rate       = inflation_rate,
        )
        result = GoalEngine(gi, monthly_surplus=_monthly_surplus).compute()

        # ── Primary Planning Results ──
        st.markdown(f"""
        <div class="goal-section-header">
            <h3 class="goal-section-title"><i class="fa-solid fa-chart-line" style="color: #4F8CFF; font-size: 1.1rem;"></i> Planning Result — {result.goal_name}</h3>
            <p class="goal-section-desc">Key requirements and projected horizon for your milestone.</p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="goal-metric-box">
                <div class="goal-metric-label">Inflation-Adjusted Target</div>
                <div class="goal-metric-val" style="font-size: 1.35rem;">₹{result.inflation_adj_goal:,.0f}</div>
                <div class="goal-metric-sub">+₹{result.inflation_adj_goal - goal_amount:,.0f} inflation buffer</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="goal-metric-box">
                <div class="goal-metric-label">FV of Current Savings</div>
                <div class="goal-metric-val" style="font-size: 1.35rem;">₹{result.fv_current_savings:,.0f}</div>
                <div class="goal-metric-sub">Compounded to target date</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="goal-metric-box">
                <div class="goal-metric-label">Remaining Gap</div>
                <div class="goal-metric-val" style="font-size: 1.35rem; color: #F59E0B;">₹{result.gap:,.0f}</div>
                <div class="goal-metric-sub">Net corpus to fund</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            sip_label_str = f"₹{result.sip_required:,.0f}/mo" if not result.is_already_funded else "Funded!"
            st.markdown(f"""
            <div class="goal-metric-box-featured">
                <div class="goal-metric-label" style="color: #60A5FA;">Required Monthly SIP</div>
                <div class="goal-metric-val" style="font-size: 1.4rem; color: #38BDF8;">{sip_label_str}</div>
                <div class="goal-metric-sub">At {return_rate:.1%} expected return</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            completion_date_str = result.projected_completion.strftime("%b %Y") if result.projected_completion else "Beyond 50Y"
            delay_text = f"{result.shortfall_months} months late" if result.shortfall_months > 0 else "On schedule"
            delay_color = "#EF4444" if result.shortfall_months > 0 else "#10B981"
            st.markdown(f"""
            <div class="goal-metric-box">
                <div class="goal-metric-label">Projected Completion</div>
                <div class="goal-metric-val" style="font-size: 1.35rem;">{completion_date_str}</div>
                <div class="goal-metric-sub" style="color: {delay_color}; font-weight: 600;">{delay_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Status / Probability ──
        if result.is_already_funded:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1.5px solid #10B981; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1.25rem 0; color: #F8FAFC;">
                <strong style="color: #10B981; font-size: 1.05rem;"><i class="fa-solid fa-circle-check"></i> Goal Fully Funded!</strong><br/>
                Your existing savings of ₹{current_savings:,.0f} will grow to approximately ₹{result.fv_current_savings:,.0f} by {target_date.strftime('%b %Y')}, fully satisfying the inflation-adjusted goal of ₹{result.inflation_adj_goal:,.0f} without requiring additional monthly SIP.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            ga1, ga2 = st.columns([1.2, 2.0])
            with ga1:
                st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>Achievement Feasibility</div>", unsafe_allow_html=True)
                fig_gauge = build_probability_gauge(result.probability, result.goal_name, result.color)
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown(f"<div style='text-align:center; margin-bottom: 1rem;'>{prob_badge(result.probability, result.probability_label)}</div>", unsafe_allow_html=True)

                if result.sip_required > 0:
                    coverage_pct = min(monthly_commit / max(result.sip_required, 1), 1.0)
                    st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #94A3B8; margin-bottom: 6px;'>Monthly SIP Commitment Coverage:</div>", unsafe_allow_html=True)
                    st.progress(coverage_pct)
                    st.markdown(f"<div style='font-size: 0.82rem; color: #94A3B8; margin-top: 4px;'>Committed: ₹{monthly_commit:,.0f} / Required: ₹{result.sip_required:,.0f} ({coverage_pct:.0%})</div>", unsafe_allow_html=True)
                    if result.sip_required > _monthly_surplus:
                        st.markdown(f"""
                        <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #EF4444; border-radius: 8px; padding: 8px 12px; margin-top: 10px; font-size: 0.84rem; color: #FCA5A5;">
                            Required SIP exceeds your total monthly surplus (₹{_monthly_surplus:,.0f}). Consider extending your target date.
                        </div>
                        """, unsafe_allow_html=True)

            with ga2:
                st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>Corpus Trajectory Curve</div>", unsafe_allow_html=True)
                fig_traj = build_trajectory_chart(result)
                st.plotly_chart(fig_traj, use_container_width=True)

        # ── Build-Up Chart ──
        st.markdown("""
        <div class="goal-section-header">
            <h3 class="goal-section-title">Corpus Build-Up: Contributions vs Compound Returns</h3>
            <p class="goal-section-desc">Visual breakdown of your invested principal versus accrued investment returns.</p>
        </div>
        """, unsafe_allow_html=True)
        fig_breakdown = build_corpus_breakdown_chart(result)
        st.plotly_chart(fig_breakdown, use_container_width=True)

        # ── Milestone Tracker & Tips ──
        m_col1, m_col2 = st.columns([1.6, 1.4])
        with m_col1:
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.75rem;'>Milestone Tracker</div>", unsafe_allow_html=True)
            m_labels = ["25% of Goal", "50% of Goal", "75% of Goal", "100% — Goal Achieved"]
            m_amounts = [result.inflation_adj_goal * p for p in [0.25, 0.50, 0.75, 1.00]]
            m_rows = []
            for m_month, label, amount in zip(result.milestone_months, m_labels, m_amounts):
                if m_month > 0:
                    est_date = _today + datetime.timedelta(days=m_month * 30.44)
                    m_rows.append({
                        "Milestone": label,
                        "Corpus Target": f"₹{amount:,.0f}",
                        "Month": m_month,
                        "Estimated Date": est_date.strftime("%b %Y"),
                        "Status": "Achieved" if m_month <= result.goal_input.months_remaining else "Projected",
                    })
                else:
                    m_rows.append({
                        "Milestone": label, "Corpus Target": f"₹{amount:,.0f}",
                        "Month": "—", "Estimated Date": "Beyond horizon", "Status": "Pending",
                    })
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

        with m_col2:
            st.markdown(f"<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.75rem;'>{result.goal_name} Insights & Tips</div>", unsafe_allow_html=True)
            for tip in result.tips:
                st.markdown(f"""
                <div class="goal-tip-box" style="border-left-color: {cfg['color']};">
                    <i class="fa-solid fa-lightbulb" style="color: {cfg['color']}; margin-right: 6px;"></i> {tip}
                </div>
                """, unsafe_allow_html=True)

        # ── Collapsible detailed projection ──
        with st.expander("View Month-by-Month Projection Schedule", expanded=False):
            display_df = result.trajectory.copy()
            display_df["Date"]           = pd.to_datetime(display_df["Date"]).dt.strftime("%b %Y")
            display_df["Corpus"]         = display_df["Corpus"].apply(lambda v: f"₹{v:,.0f}")
            display_df["SIP_Cumulative"] = display_df["SIP_Cumulative"].apply(lambda v: f"₹{v:,.0f}")
            display_df["Returns_Earned"] = display_df["Returns_Earned"].apply(lambda v: f"₹{v:,.0f}")
            display_df["Target"]         = display_df["Target"].apply(lambda v: f"₹{v:,.0f}")
            display_df["On_Track"]       = display_df["On_Track"].apply(lambda v: "Yes" if v else "No")
            display_df.columns = ["Month", "Date", "Corpus", "SIP Contributed", "Returns Earned", "Target", "On Track"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── PDF Download for Single Goal ──
        st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
        try:
            _pdf_sg = FinTwinPDFReport(
                report_title=f"{result.goal_name} Goal Planning Report",
                user_id=twin.user_id,
                report_id=f"GP-{twin.user_id}",
                subtitle=f"Personalised Goal Analysis — Target: Rs. {result.inflation_adj_goal:,.0f}",
            )
            _pdf_sg.add_cover_page(
                user_name=getattr(twin, "name", "") or "",
                report_type="Financial Goal Planning Report",
            )
            prob = result.probability
            _pdf_sg.add_kpi_summary_row([
                {"label": "Goal Type", "value": str(result.goal_name), "status": "neutral"},
                {"label": "Inflation-Adjusted Target", "value": f"Rs. {result.inflation_adj_goal:,.0f}", "status": "neutral"},
                {"label": "Required Monthly SIP", "value": f"Rs. {result.sip_required if not result.is_already_funded else 0:,.0f}", "status": "neutral"},
                {"label": "Success Probability", "value": f"{prob:.0%}",
                 "status": "positive" if prob >= 0.75 else ("negative" if prob < 0.5 else "neutral")},
            ])
            if result.is_already_funded:
                _pdf_sg.add_callout(
                    "Your current savings are already sufficient to fund this goal. No additional SIP is required.",
                    style="success", title="Goal Status"
                )
            elif prob >= 0.75:
                _pdf_sg.add_callout(
                    f"High success probability ({prob:.0%}). Continue your current SIP plan to achieve this goal.",
                    style="success", title="Goal Status"
                )
            elif prob >= 0.5:
                _pdf_sg.add_callout(
                    f"Moderate probability ({prob:.0%}). Consider increasing your monthly SIP contribution.",
                    style="warning", title="Goal Status"
                )
            else:
                _pdf_sg.add_callout(
                    f"Low success probability ({prob:.0%}). Review your goal timeline or increase contributions significantly.",
                    style="danger", title="Goal Status"
                )

            _pdf_sg.add_section_divider("Goal Parameters")
            _pdf_sg.add_key_value_grid({
                "Goal Type": result.goal_name,
                "Goal Amount (Today)": goal_amount,
                "Inflation-Adjusted Target": result.inflation_adj_goal,
                "Current Savings FV": result.fv_current_savings,
                "Required Monthly SIP": result.sip_required if not result.is_already_funded else 0,
                "Success Probability": f"{result.probability:.0%}",
                "Target Date": target_date.strftime("%b %Y"),
                "Projected Completion": result.projected_completion.strftime("%b %Y") if result.projected_completion else "N/A",
            })
            _pdf_sg.add_plotly_figure(fig_traj, caption=f"{result.goal_name} Corpus Trajectory")
            _pdf_sg.add_plotly_figure(fig_breakdown, caption="Contributions vs Investment Returns")
            _pdf_sg.add_section_divider("Milestone Tracker")
            _pdf_sg.add_dataframe_table(pd.DataFrame(m_rows))
            pdf_sg_bytes = _pdf_sg.build()
            pdf_sg_fn = _pdf_sg.suggested_filename("single_goal_plan_report")
        except Exception:
            pdf_sg_bytes = None
            pdf_sg_fn = "goal_planning_report.pdf"

        with st.container(border=True):
            c_rep1, c_rep2 = st.columns([2.2, 1.1], vertical_alignment="center")
            with c_rep1:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 14px; padding: 4px 0;">
                    <div style="background: rgba(79, 140, 255, 0.15); border: 1px solid rgba(79, 140, 255, 0.3); border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                        <i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.3rem;"></i>
                    </div>
                    <div>
                        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;">
                            Download {result.goal_name} Plan Report (PDF)
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                            Comprehensive export including trajectory schedules, SIP requirement, and milestone dates.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                if pdf_sg_bytes:
                    st.download_button(
                        label="Download PDF Report",
                        icon=":material/download:",
                        data=pdf_sg_bytes,
                        file_name=pdf_sg_fn,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="btn_download_sg_report"
                    )
                else:
                    st.error("Report is currently unavailable.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-MODE B: MULTI-GOAL PORTFOLIO PLANNER
    # ─────────────────────────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="goal-section-header" style="margin-top: 1rem;">
            <h2 class="goal-section-title"><i class="fa-solid fa-layer-group" style="color: #4F8CFF; font-size: 1.15rem;"></i> Configure Multiple Goals</h2>
            <p class="goal-section-desc">Select and configure up to 5 life milestones. Surplus is distributed by priority (Education → Home → Marriage → Car → Vacation).</p>
        </div>
        """, unsafe_allow_html=True)

        active_goals: list[GoalInput] = []
        goal_types_list = list(GoalType)

        for gt in goal_types_list:
            cfg_g = GOAL_CONFIG[gt]
            with st.expander(f"{gt.value} Settings", expanded=False):
                enabled = st.checkbox(f"Include {gt.value} in portfolio plan", value=True, key=f"mg_en_{gt.value}")
                if enabled:
                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        g_amt = st.number_input(
                            "Target Amount (₹)", min_value=10_000, max_value=100_000_000,
                            value=cfg_g["default_amount"], step=10_000, key=f"mg_amt_{gt.value}"
                        )
                        g_cur = st.number_input(
                            "Already Saved (₹)", min_value=0, max_value=100_000_000,
                            value=0, step=5_000, key=f"mg_cur_{gt.value}"
                        )
                    with gc2:
                        g_date = st.date_input(
                            "Target Date", key=f"mg_date_{gt.value}",
                            value=_today + datetime.timedelta(days=cfg_g["default_years"] * 365),
                            min_value=_min_date,
                            max_value=_today + datetime.timedelta(days=365 * 35),
                        )
                        g_ret = st.slider(
                            "Annual Return (%)", 4.0, 18.0,
                            round(cfg_g["return_rate"] * 100, 1), 0.5,
                            key=f"mg_ret_{gt.value}"
                        )
                    with gc3:
                        g_infl = st.slider(
                            "Inflation (%)", 3.0, 10.0, 6.0, 0.5,
                            key=f"mg_infl_{gt.value}"
                        )
                        g_contrib = st.number_input(
                            "Fixed Monthly Contribution (₹, 0 = auto)",
                            min_value=0, max_value=int(max(_monthly_surplus, 1)),
                            value=0, step=500, key=f"mg_contrib_{gt.value}",
                            help="Set 0 for automated priority-based allocation from surplus."
                        )

                    active_goals.append(GoalInput(
                        goal_type            = gt,
                        goal_amount          = float(g_amt),
                        current_savings      = float(g_cur),
                        target_date          = g_date,
                        monthly_contribution = float(g_contrib),
                        return_rate          = g_ret / 100,
                        inflation_rate       = g_infl / 100,
                    ))

        if not active_goals:
            st.warning("Please toggle at least one goal on to calculate a multi-goal portfolio plan.")
            st.stop()

        # ── Run Multi-Goal Plan ──
        planner = MultiGoalPlanner(twin, active_goals)
        results = planner.plan()

        st.markdown(f"""
        <div class="goal-section-header">
            <h3 class="goal-section-title"><i class="fa-solid fa-table-list" style="color: #4F8CFF; font-size: 1.1rem;"></i> Portfolio Plan Summary ({len(results)} Active Goals)</h3>
            <p class="goal-section-desc">Overview of required SIP contributions, achievement probabilities, and horizon schedules.</p>
        </div>
        """, unsafe_allow_html=True)

        sum_df = planner.summary_table(results)
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        # ── SIP Donut & Probability Bars ──
        total_sip_required = sum(r.sip_required for r in results)
        sip_labels = [r.goal_name for r in results]
        sip_vals   = [r.sip_required for r in results]
        sip_colors = [GOAL_CONFIG[r.goal_input.goal_type]["color"] for r in results]

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        donut_col, bar_col = st.columns(2)

        with donut_col:
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>Monthly SIP Allocation by Goal</div>", unsafe_allow_html=True)
            if total_sip_required > 0:
                fig_donut = go.Figure(go.Pie(
                    labels       = sip_labels,
                    values       = sip_vals,
                    hole         = 0.55,
                    marker_colors = sip_colors,
                    textinfo     = "label+percent",
                    hovertemplate = "%{label}<br>₹%{value:,.0f}/mo<extra></extra>",
                ))
                fig_donut.add_annotation(
                    text     = f"Total SIP<br>₹{total_sip_required:,.0f}",
                    x        = 0.5, y = 0.5,
                    font     = {"size": 13, "color": "#F8FAFC", "family": "Plus Jakarta Sans"},
                    showarrow = False,
                )
                fig_donut.update_layout(
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                    height        = 360,
                    showlegend    = False,
                    margin        = dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

                remaining = _monthly_surplus - total_sip_required
                if remaining >= 0:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; border-radius: 8px; padding: 10px 14px; font-size: 0.88rem; color: #A7F3D0;">
                        <strong>Surplus Sufficient:</strong> Total SIP required is <strong>₹{total_sip_required:,.0f}/mo</strong>. You retain an unallocated monthly buffer of <strong>₹{remaining:,.0f}/mo</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #EF4444; border-radius: 8px; padding: 10px 14px; font-size: 0.88rem; color: #FCA5A5;">
                        <strong>Surplus Deficit:</strong> Total SIP required (₹{total_sip_required:,.0f}) exceeds available monthly surplus (₹{_monthly_surplus:,.0f}) by <strong>₹{abs(remaining):,.0f}</strong>. Consider extending lower-priority dates.
                    </div>
                    """, unsafe_allow_html=True)

        with bar_col:
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;'>Probability of Achievement per Goal</div>", unsafe_allow_html=True)
            sorted_results = sorted(results, key=lambda r: r.probability, reverse=True)
            bar_colors_prob = [r.probability_color for r in sorted_results]
            fig_prob = go.Figure(go.Bar(
                x            = [r.goal_name for r in sorted_results],
                y            = [r.probability * 100 for r in sorted_results],
                marker_color = bar_colors_prob,
                text         = [f"{r.probability:.0%}" for r in sorted_results],
                textposition = "outside",
                hovertemplate = "%{x}<br>Probability: %{y:.1f}%<extra></extra>",
            ))
            fig_prob.add_hline(y=80, line_color="#10B981", line_dash="dash",
                               annotation_text="80% Target Benchmark", annotation_font_color="#10B981")
            fig_prob.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                yaxis         = {"title": "Probability (%)", "range": [0, 115], "gridcolor": "rgba(255, 255, 255, 0.06)"},
                xaxis         = {"gridcolor": "rgba(0,0,0,0)"},
                height        = 360,
                showlegend    = False,
                margin        = dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # ── Individual Goal Trajectory Cards ──
        st.markdown("""
        <div class="goal-section-header">
            <h3 class="goal-section-title">Individual Goal Trajectories</h3>
            <p class="goal-section-desc">Parallel projection paths for each active milestone in your plan.</p>
        </div>
        """, unsafe_allow_html=True)

        num_cols = min(len(results), 3)
        traj_cols = st.columns(num_cols)
        for i, res in enumerate(results):
            with traj_cols[i % num_cols]:
                fig_mini = go.Figure()
                df_m = res.trajectory.head(res.goal_input.months_remaining + 6)
                fill_color_rgba = res.color
                if res.color.startswith("#") and len(res.color) == 7:
                    r, g, b = int(res.color[1:3], 16), int(res.color[3:5], 16), int(res.color[5:7], 16)
                    fill_color_rgba = f"rgba({r}, {g}, {b}, 0.12)"

                fig_mini.add_trace(go.Scatter(
                    x    = df_m["Month"],
                    y    = df_m["Corpus"],
                    mode = "lines",
                    line = {"color": res.color, "width": 2},
                    fill = "tozeroy",
                    fillcolor = fill_color_rgba,
                    name = res.goal_name,
                ))
                fig_mini.add_hline(
                    y=res.inflation_adj_goal,
                    line_color="#10B981", line_width=1.5, line_dash="dash",
                )
                fig_mini.update_layout(
                    title         = f"<b>{res.goal_name}</b>",
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
                    height        = 230,
                    showlegend    = False,
                    margin        = dict(l=10, r=10, t=45, b=10),
                    xaxis         = {"gridcolor": "rgba(255, 255, 255, 0.06)"},
                    yaxis         = {"gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
                )
                st.plotly_chart(fig_mini, use_container_width=True)
                pct_done = min(res.goal_input.current_savings / max(res.inflation_adj_goal, 1), 1.0)
                st.progress(pct_done)
                delay_str = "On time" if res.shortfall_months == 0 else f"{res.shortfall_months}M late"
                st.caption(f"Required SIP: ₹{res.sip_required:,.0f}/mo · P: {res.probability:.0%} · {delay_str}")

        # ── Export Multi-Goal Report ──
        st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
        try:
            _pdf_goal = FinTwinPDFReport(
                report_title="Multi-Goal Portfolio Report",
                user_id=twin.user_id,
                report_id=f"GP-PORTFOLIO-{twin.user_id}",
                subtitle=f"Portfolio Goal Plan — {len(results)} Active Goals & Surplus Allocation",
            )
            _pdf_goal.add_cover_page(
                user_name=getattr(twin, "name", "") or "",
                report_type="Multi-Goal Portfolio Report",
            )
            _rem_status = "positive" if remaining >= 0 else "negative"
            _pdf_goal.add_kpi_summary_row([
                {"label": "Active Goals", "value": str(len(results)), "status": "neutral"},
                {"label": "Total Required SIP", "value": f"Rs. {total_sip_required:,.0f}/mo", "status": "neutral"},
                {"label": "Monthly Surplus", "value": f"Rs. {_monthly_surplus:,.0f}/mo", "status": "neutral"},
                {"label": "Remaining Buffer", "value": f"Rs. {remaining:+,.0f}/mo", "status": _rem_status},
            ])
            if remaining >= 0:
                _pdf_goal.add_callout(
                    f"Surplus Sufficient: Total required monthly SIP is Rs. {total_sip_required:,.0f}/mo across all {len(results)} goals, "
                    f"leaving an unallocated safety buffer of Rs. {remaining:,.0f}/mo.",
                    style="success",
                    title="Portfolio Feasibility Status"
                )
            else:
                _pdf_goal.add_callout(
                    f"Surplus Deficit: Total required monthly SIP (Rs. {total_sip_required:,.0f}/mo) exceeds current monthly surplus "
                    f"(Rs. {_monthly_surplus:,.0f}/mo) by Rs. {abs(remaining):,.0f}/mo. Consider extending lower-priority goal timelines.",
                    style="danger",
                    title="Portfolio Feasibility Alert"
                )
            _pdf_goal.add_section_divider("Goal Planning Summary Table")
            _pdf_goal.add_dataframe_table(sum_df)
            try:
                _pdf_goal.add_plotly_figure(fig_donut, caption="Monthly SIP Allocation by Goal")
            except NameError:
                pass
            try:
                _pdf_goal.add_plotly_figure(fig_prob, caption="Probability of Achievement per Goal")
            except NameError:
                pass
            pdf_goal_bytes = _pdf_goal.build()
            pdf_goal_fn = _pdf_goal.suggested_filename("multi_goal_planning_report")
        except Exception:
            pdf_goal_bytes = None
            pdf_goal_fn = "multi_goal_planning_report.pdf"

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
                            Download Multi-Goal Portfolio Report (PDF)
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                            Export summary of all planned milestones, surplus allocation, and probability metrics.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                if pdf_goal_bytes:
                    st.download_button(
                        label="Download PDF Report",
                        icon=":material/download:",
                        data=pdf_goal_bytes,
                        file_name=pdf_goal_fn,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="btn_download_mg_report"
                    )
                else:
                    st.error("Report is currently unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: GOAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="goal-section-header" style="margin-top: 0.5rem;">
        <h2 class="goal-section-title"><i class="fa-solid fa-scale-balanced" style="color: #4F8CFF; font-size: 1.15rem;"></i> Goal Feasibility & Horizon Analysis</h2>
        <p class="goal-section-desc">Evaluate the relative difficulty, time horizons, and monthly capital demands across standard life milestones.</p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(
        show_spinner="Computing comparison for all goals…",
        hash_funcs={object: lambda x: str(getattr(x, "user_id", id(x)))},
    )
    def compute_all_defaults(_twin) -> list:
        results_all = []
        for gt in GoalType:
            cfg_d = GOAL_CONFIG[gt]
            gi_d = GoalInput(
                goal_type       = gt,
                goal_amount     = cfg_d["default_amount"],
                current_savings = 0.0,
                target_date     = _today + datetime.timedelta(days=cfg_d["default_years"] * 365),
                return_rate     = cfg_d["return_rate"],
                inflation_rate  = 0.06,
                monthly_contribution = 0.0,
            )
            surplus = max(_twin.total_income - _twin.basic_expenses - _twin.monthly_emi, 0.0)
            r = GoalEngine(gi_d, monthly_surplus=surplus).compute()
            results_all.append(r)
        return results_all

    all_results = compute_all_defaults(twin)

    # ── Bubble Chart ──
    st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.35rem;'>Goal Feasibility Map</div><div style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.75rem;'>Bubble Size = Inflation-Adjusted Target · X-Axis = Time Horizon (Months) · Y-Axis = Probability of Success</div>", unsafe_allow_html=True)

    bubble_data = []
    for r in all_results:
        bubble_data.append({
            "Goal":         r.goal_name,
            "Months":       r.goal_input.months_remaining,
            "Probability":  r.probability * 100,
            "SIP_Required": r.sip_required,
            "Amount":       r.inflation_adj_goal,
            "Color":        r.color,
        })
    df_bubble = pd.DataFrame(bubble_data)

    fig_bubble = go.Figure()
    for _, row in df_bubble.iterrows():
        fig_bubble.add_trace(go.Scatter(
            x    = [row["Months"]],
            y    = [row["Probability"]],
            mode = "markers+text",
            name = row["Goal"],
            marker = {
                "size":  max(int(row["Amount"] / 100_000) + 18, 24),
                "color": row["Color"],
                "opacity": 0.8,
                "line":  {"color": "#1E293B", "width": 2},
            },
            text         = [row["Goal"]],
            textposition = "top center",
            hovertemplate = (
                f"<b>{row['Goal']}</b><br>"
                f"Horizon: {row['Months']} months<br>"
                f"Probability: {row['Probability']:.0f}%<br>"
                f"SIP Required: ₹{row['SIP_Required']:,.0f}/mo<br>"
                f"Target: ₹{row['Amount']:,.0f}<extra></extra>"
            ),
        ))

    fig_bubble.add_hline(y=80, line_color="#10B981", line_dash="dash",
                         annotation_text="80% Feasibility Benchmark",
                         annotation_font_color="#10B981")
    fig_bubble.update_layout(
        xaxis         = {"title": "Time Horizon to Goal (Months)", "gridcolor": "rgba(255, 255, 255, 0.06)"},
        yaxis         = {"title": "Probability of Success (%)", "gridcolor": "rgba(255, 255, 255, 0.06)", "range": [0, 115]},
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
        height        = 420,
        showlegend    = False,
        margin        = dict(l=10, r=10, t=25, b=20),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Required Monthly SIP Bar Chart ──
    st.markdown("""
    <div class="goal-section-header">
        <h3 class="goal-section-title">Required Monthly SIP per Goal</h3>
        <p class="goal-section-desc">Comparison of monthly capital commitments against your available surplus buffer.</p>
    </div>
    """, unsafe_allow_html=True)

    fig_sip_comp = go.Figure(go.Bar(
        x            = [r.goal_name for r in all_results],
        y            = [r.sip_required for r in all_results],
        marker_color = [r.color for r in all_results],
        text         = [f"₹{r.sip_required:,.0f}" for r in all_results],
        textposition = "outside",
        hovertemplate = "%{x}<br>Required SIP: ₹%{y:,.0f}/mo<extra></extra>",
    ))
    fig_sip_comp.add_hline(
        y=_monthly_surplus, line_color="#F59E0B", line_width=1.5, line_dash="dash",
        annotation_text=f"Your Available Surplus: ₹{_monthly_surplus:,.0f}/mo",
        annotation_font_color="#F59E0B",
    )
    fig_sip_comp.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"color": "#CBD5E1", "family": "Plus Jakarta Sans, sans-serif", "size": 11},
        yaxis         = {"title": "Monthly SIP Required (₹)", "gridcolor": "rgba(255, 255, 255, 0.06)", "tickformat": "₹,.0f"},
        xaxis         = {"gridcolor": "rgba(0,0,0,0)"},
        height        = 360,
        showlegend    = False,
        margin        = dict(l=10, r=10, t=25, b=20),
    )
    st.plotly_chart(fig_sip_comp, use_container_width=True)

    # ── Comparison Data Matrix ──
    st.markdown("""
    <div class="goal-section-header">
        <h3 class="goal-section-title">Cross-Goal Comparison Matrix</h3>
        <p class="goal-section-desc">Side-by-side metric comparison across all 5 standard goal benchmarks.</p>
    </div>
    """, unsafe_allow_html=True)

    cmp_rows = []
    for r in all_results:
        cmp_rows.append({
            "Goal":               r.goal_name,
            "Target (Today)":     f"₹{r.goal_input.goal_amount:,.0f}",
            "Infl-Adj Target":    f"₹{r.inflation_adj_goal:,.0f}",
            "Horizon":            f"{r.goal_input.months_remaining} months",
            "Return Rate":        f"{r.goal_input.effective_return_rate:.1%}",
            "SIP Required":       f"₹{r.sip_required:,.0f}/mo",
            "Probability":        f"{r.probability:.0%}",
            "Feasibility":        r.probability_label,
            "Est. Completion":    r.projected_completion.strftime("%b %Y") if r.projected_completion else "N/A",
        })
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # ── Status Categorization Cards ──
    st.markdown("""
    <div class="goal-section-header">
        <h3 class="goal-section-title">Affordability Status Categorization</h3>
    </div>
    """, unsafe_allow_html=True)

    feasible = [r for r in all_results if r.sip_required <= _monthly_surplus * 0.40]
    stretch  = [r for r in all_results if _monthly_surplus * 0.40 < r.sip_required <= _monthly_surplus]
    hard     = [r for r in all_results if r.sip_required > _monthly_surplus]

    ia1, ia2, ia3 = st.columns(3)
    with ia1:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.10); border: 1.5px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="margin-bottom: 6px;"><i class="fa-solid fa-circle-check" style="font-size: 1.75rem; color: #10B981;"></i></div>
            <div style="color: #10B981; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">Comfortably Achievable</div>
            <div style="color: #CBD5E1; font-size: 0.9rem;">
                {', '.join([r.goal_name for r in feasible]) or '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ia2:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.10); border: 1.5px solid rgba(245, 158, 11, 0.35); border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="margin-bottom: 6px;"><i class="fa-solid fa-triangle-exclamation" style="font-size: 1.75rem; color: #F59E0B;"></i></div>
            <div style="color: #F59E0B; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">Stretch Goals</div>
            <div style="color: #CBD5E1; font-size: 0.9rem;">
                {', '.join([r.goal_name for r in stretch]) or '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ia3:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.10); border: 1.5px solid rgba(239, 68, 68, 0.35); border-radius: 12px; padding: 1.25rem; text-align: center;">
            <div style="margin-bottom: 6px;"><i class="fa-solid fa-clock-rotate-left" style="font-size: 1.75rem; color: #EF4444;"></i></div>
            <div style="color: #EF4444; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">Need Longer Horizon</div>
            <div style="color: #CBD5E1; font-size: 0.9rem;">
                {', '.join([r.goal_name for r in hard]) or '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Export Comparison Report ──
    st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
    try:
        _pdf_cmp = FinTwinPDFReport(
            report_title="Goal Comparison Analysis Report",
            user_id=twin.user_id,
            report_id=f"GP-CMP-{twin.user_id}",
            subtitle="Cross-Goal Feasibility & Required Monthly SIP Matrix",
        )
        _pdf_cmp.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Goal Comparison Analysis Report",
        )
        _pdf_cmp.add_kpi_summary_row([
            {"label": "Total Analyzed Goals", "value": str(len(cmp_rows)), "status": "neutral"},
            {"label": "Comfortably Achievable", "value": str(len(feasible)), "status": "positive"},
            {"label": "Stretch Goals", "value": str(len(stretch)), "status": "neutral"},
            {"label": "Needs Longer Horizon", "value": str(len(hard)), "status": "negative" if hard else "neutral"},
        ])
        if hard:
            _pdf_cmp.add_callout(
                f"{len(hard)} goal(s) ({', '.join([r.goal_name for r in hard])}) have ambitious targets "
                f"or short horizons relative to current savings. Extending timelines or prioritizing capital can raise achievement odds.",
                style="warning",
                title="Feasibility Observations"
            )
        else:
            _pdf_cmp.add_callout(
                f"All {len(cmp_rows)} evaluated goals have favorable feasibility profiles under current projections.",
                style="success",
                title="Feasibility Observations"
            )
        _pdf_cmp.add_section_divider("Cross-Goal Comparison Matrix")
        _pdf_cmp.add_dataframe_table(pd.DataFrame(cmp_rows))
        if fig_bubble:
            _pdf_cmp.add_plotly_figure(fig_bubble, caption="Goal Feasibility Map")
        if fig_sip_comp:
            _pdf_cmp.add_plotly_figure(fig_sip_comp, caption="Required Monthly SIP Comparison")
        pdf_cmp_bytes = _pdf_cmp.build()
        pdf_cmp_fn = _pdf_cmp.suggested_filename("goal_comparison_report")
    except Exception:
        pdf_cmp_bytes = None
        pdf_cmp_fn = "goal_comparison_report.pdf"

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
                        Download Goal Comparison Report (PDF)
                    </div>
                    <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                        Export side-by-side analysis of feasibility scores, time horizons, and monthly capital demands.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_rep2:
            if pdf_cmp_bytes:
                st.download_button(
                    label="Download PDF Report",
                    icon=":material/download:",
                    data=pdf_cmp_bytes,
                    file_name=pdf_cmp_fn,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="btn_download_cmp_report"
                )
            else:
                st.error("Report is currently unavailable.")
