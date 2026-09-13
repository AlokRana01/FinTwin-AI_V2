"""
Streamlit Page: Financial Forecasting
====================================
Features XGBoost-powered savings and net worth forecasts, 12-month trajectory projection,
and Monte Carlo probabilistic wealth path modeling in a polished fintech UI.
"""

import streamlit as st
import pandas as pd
from utils.session import render_sidebar_user_selector
from utils.visualizer import PlotlyVisualizer
from utils.simulator import ScenarioSimulator
from utils.pdf_report import FinTwinPDFReport
from models.predictor import FinancialPredictor
from utils.ui_components import PremiumLoader

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Forecasting — FinTwin AI",
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
.forecast-header-box {
    margin-bottom: 2.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.forecast-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.03em;
}
.forecast-subtitle {
    color: #94A3B8;
    font-size: 0.96rem;
    margin: 0;
    line-height: 1.55;
}

/* Card Containers */
.fc-card {
    background-color: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.6rem 1.85rem;
    margin: 1.5rem 0 2rem 0;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}
.fc-card-highlight {
    background: linear-gradient(135deg, rgba(79, 140, 255, 0.08) 0%, #111827 100%);
    border: 1px solid rgba(79, 140, 255, 0.3);
    border-left: 4px solid #4F8CFF;
    border-radius: 14px;
    padding: 1.6rem 1.85rem;
    margin: 1.5rem 0 2rem 0;
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
}

/* Section Header */
.fc-section-header {
    margin: 2.25rem 0 1.25rem 0;
}
.fc-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.fc-section-desc {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 0 0 1.15rem 0;
    line-height: 1.5;
}

/* Metric Display Boxes */
.fc-metric-box {
    background: #1F2937;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    box-sizing: border-box;
}
.fc-metric-box:hover {
    border-color: rgba(79, 140, 255, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
}
.fc-metric-box-featured {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(79, 140, 255, 0.45);
    border-top: 3px solid #4F8CFF;
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    box-sizing: border-box;
}
.fc-metric-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    color: #94A3B8;
    margin-bottom: 0.45rem;
}
.fc-metric-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.25;
    margin-bottom: 0.3rem;
}
.fc-metric-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}
.fc-delta-pos {
    color: #10B981;
    font-weight: 600;
    font-size: 0.84rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.fc-delta-neg {
    color: #EF4444;
    font-weight: 600;
    font-size: 0.84rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* Narrative Card */
.fc-narrative-box {
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

/* Callout Note */
.fc-info-note {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.15rem 1.45rem;
    color: #94A3B8;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 1.25rem 0 2.25rem 0;
}

/* Goal Probability Card */
.fc-goal-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1.5rem 0 2.25rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.75rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}
.fc-goal-badge {
    border-radius: 12px;
    padding: 0.75rem 1.35rem;
    font-weight: 800;
    font-size: 1.75rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 115px;
    text-align: center;
}
.fc-goal-high {
    background: rgba(16, 185, 129, 0.14);
    border: 1.5px solid #10B981;
    color: #10B981;
}
.fc-goal-mid {
    background: rgba(245, 158, 11, 0.14);
    border: 1.5px solid #F59E0B;
    color: #F59E0B;
}
.fc-goal-low {
    background: rgba(239, 68, 68, 0.14);
    border: 1.5px solid #EF4444;
    color: #EF4444;
}

/* Report Download Card */
.fc-report-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-top: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

/* Expanders Spacing */
div[data-testid="stExpander"] {
    margin-top: 1.25rem !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    background: #111827 !important;
}

/* Bordered Action Banners */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 0.6rem 0.75rem !important;
}

/* Action Link Button Styling */
div[data-testid="stPageLink-NavLink"] {
    background: #4F8CFF !important;
    border: 1px solid #4F8CFF !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-decoration: none !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(79, 140, 255, 0.3) !important;
}
div[data-testid="stPageLink-NavLink"]:hover {
    background: #3B7BF6 !important;
    border-color: #3B7BF6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79, 140, 255, 0.45) !important;
}
div[data-testid="stPageLink-NavLink"] span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}

/* Download Button Polish */
div[data-testid="stDownloadButton"] > button {
    background: #4F8CFF !important;
    color: #FFFFFF !important;
    border: 1px solid #4F8CFF !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.2rem !important;
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

# ── Sidebar & Authentication Check ───────────────────────────────────────────
twin = render_sidebar_user_selector()

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="forecast-header-box">
    <h1 class="forecast-title">Financial Forecasting</h1>
    <p class="forecast-subtitle">See where your finances could be headed and explore different future scenarios.</p>
</div>
""", unsafe_allow_html=True)

# ── Empty / Incomplete Profile State ─────────────────────────────────────────
if not twin:
    st.markdown("""
    <div class="fc-card" style="text-align: center; padding: 2.5rem 1.5rem; max-width: 680px; margin: 2rem auto;">
        <i class="fa-solid fa-user-gear" style="color: #4F8CFF; font-size: 2.2rem; margin-bottom: 0.85rem; display: inline-block;"></i>
        <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.5rem 0;">
            Complete Your Financial Profile
        </h2>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5; margin: 0 0 1.5rem 0;">
            Add your financial information to generate personalized financial forecasts and probabilistic wealth simulations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1.4, 1])
    with c_btn2:
        st.page_link("pages/01_Digital_Twin.py", label="Complete Financial Profile →", icon=":material/arrow_forward:", use_container_width=True)
    st.stop()

# ── Pretrained Model Loader ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor(horizon: int):
    predictor = FinancialPredictor()
    loaded = predictor.load_only(horizon=horizon)
    return predictor, loaded

# ══════════════════════════════════════════════════════════════════════════════
# TWO-TAB STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    ":material/trending_up: XGBoost Forecast",
    ":material/stacked_line_chart: Monte Carlo Simulation"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: XGBOOST FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="fc-section-header" style="margin-top: 0.5rem;">
        <h2 class="fc-section-title">Financial Forecast</h2>
        <p class="fc-section-desc">See your projected savings and net worth based on your current financial profile.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How this forecast works", expanded=False):
        st.markdown("""
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
            This forecast uses gradient boosted regression trees (XGBoost) trained on demographic and financial trajectories.
            It evaluates your current cash flows, savings rate, existing assets, and monthly commitments to project expected future balances at 6-month and 12-month horizons.
        </div>
        """, unsafe_allow_html=True)

    # ── Horizon Selector ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    col_hz1, col_hz2 = st.columns([1.5, 2.5], vertical_alignment="center")
    with col_hz1:
        st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>Forecast Horizon</div>", unsafe_allow_html=True)
        horizon = st.radio(
            "Forecast Horizon",
            [6, 12],
            horizontal=True,
            format_func=lambda x: f"{x} Months",
            label_visibility="collapsed",
            key="xgb_forecast_horizon_radio"
        )

    # ── Prediction Calculations ──────────────────────────────────────────────
    predictor, model_loaded = get_predictor(horizon)
    if not model_loaded:
        st.info("Using baseline linear projection while offline XGBoost weights initialize.")

    try:
        pred_6 = predictor.predict(twin, horizon_months=6)
        pred_12 = predictor.predict(twin, horizon_months=12)
    except Exception:
        pred_6 = {
            "predicted_savings": twin.monthly_savings * 6,
            "predicted_net_worth": twin.net_worth + (twin.monthly_savings * 6)
        }
        pred_12 = {
            "predicted_savings": twin.monthly_savings * 12,
            "predicted_net_worth": twin.net_worth + (twin.monthly_savings * 12)
        }

    # Active projection figures
    active_pred = pred_6 if horizon == 6 else pred_12
    active_savings = active_pred["predicted_savings"]
    active_net_worth = active_pred["predicted_net_worth"]
    delta_net_worth = active_net_worth - twin.net_worth

    # ── Forecast Summary Section ─────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header">
        <h3 class="fc-section-title">Your Financial Forecast</h3>
        <p class="fc-section-desc">Key projected milestones for your selected {0}-month horizon.</p>
    </div>
    """.format(horizon), unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="fc-metric-box">
            <div class="fc-metric-label">Savings</div>
            <div class="fc-metric-val">₹{active_savings:,.0f}</div>
            <div class="fc-metric-sub">{horizon}-month projection</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="fc-metric-box-featured">
            <div class="fc-metric-label">Net Worth</div>
            <div class="fc-metric-val">₹{active_net_worth:,.0f}</div>
            <div class="fc-metric-sub">{horizon}-month projection</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        if delta_net_worth >= 0:
            change_html = f'<div class="fc-metric-val" style="color: #10B981;">↑ ₹{delta_net_worth:,.0f}</div>'
            sub_html = f'<div class="fc-delta-pos">Growth compared with today</div>'
        else:
            change_html = f'<div class="fc-metric-val" style="color: #EF4444;">↓ ₹{abs(delta_net_worth):,.0f}</div>'
            sub_html = f'<div class="fc-delta-neg">Deficit compared with today</div>'

        st.markdown(f"""
        <div class="fc-metric-box">
            <div class="fc-metric-label">Projected Change</div>
            {change_html}
            {sub_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Forecast Trajectory Chart ────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header">
        <h3 class="fc-section-title">Projected Financial Trajectory</h3>
        <p class="fc-section-desc">Your projected savings and net worth over the next 12 months.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df_traj = predictor.generate_trajectory(twin, months=12)
        fig_traj = PlotlyVisualizer.plot_forecast_trajectory(df_traj)
        st.plotly_chart(fig_traj, use_container_width=True)
    except Exception as e:
        df_traj = pd.DataFrame()
        fig_traj = None
        st.error("We couldn't generate the trajectory chart right now. Please check your inputs and try again.")

    # ── Human Interpretation ─────────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header" style="margin-top: 1rem;">
        <h3 class="fc-section-title">What This Forecast Means</h3>
    </div>
    """, unsafe_allow_html=True)

    if delta_net_worth > 0:
        interpretation_text = (
            f"Based on your current financial profile, your projected net worth is expected to increase by "
            f"<strong>₹{delta_net_worth:,.0f}</strong> over the next <strong>{horizon} months</strong>. "
            f"Your current monthly surplus and investment allocations indicate steady wealth accumulation under typical financial trajectories."
        )
    elif delta_net_worth == 0:
        interpretation_text = (
            f"Based on your current financial profile, your projected net worth remains neutral over the next "
            f"<strong>{horizon} months</strong>, indicating cash inflow and expenses are closely balanced."
        )
    else:
        interpretation_text = (
            f"Based on your current financial profile, your projected net worth indicates a potential reduction of "
            f"<strong>₹{abs(delta_net_worth):,.0f}</strong> over the next <strong>{horizon} months</strong>. "
            f"Reviewing high monthly debt EMIs or discretionary expenses can help reverse this trend."
        )

    st.markdown(f"""
    <div class="fc-narrative-box">
        {interpretation_text}
    </div>
    """, unsafe_allow_html=True)

    # ── Financial Snapshot ───────────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header">
        <h3 class="fc-section-title">Your Financial Snapshot</h3>
        <p class="fc-section-desc">Key baseline cash flow metrics used in generating this projection.</p>
    </div>
    """, unsafe_allow_html=True)

    monthly_surplus = max(twin.total_income - twin.basic_expenses - twin.monthly_emi, 0)
    
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 14px; margin-bottom: 2rem;">
        <div class="fc-metric-box" style="border-top: 2px solid #38BDF8;">
            <div class="fc-metric-label">Monthly Income</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{twin.total_income:,.0f}</div>
        </div>
        <div class="fc-metric-box" style="border-top: 2px solid #F87171;">
            <div class="fc-metric-label">Monthly Expenses</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{twin.basic_expenses:,.0f}</div>
        </div>
        <div class="fc-metric-box" style="border-top: 2px solid #34D399;">
            <div class="fc-metric-label">Monthly Savings</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{monthly_surplus:,.0f}</div>
        </div>
        <div class="fc-metric-box" style="border-top: 2px solid #818CF8;">
            <div class="fc-metric-label">Monthly Investments</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{twin.sip_amount:,.0f}</div>
        </div>
        <div class="fc-metric-box" style="border-top: 2px solid #FBBF24;">
            <div class="fc-metric-label">Monthly EMI</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{twin.monthly_emi:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Progressive Disclosure: Model Details & Performance ──────────────────
    with st.expander("Model Details — How reliable is this forecast?", expanded=False):
        metrics = predictor.get_model_metrics()
        if metrics:
            st.markdown("""
            <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 1rem;">
                Statistical validation and out-of-sample cross-validation metrics for gradient boosted regression models:
            </p>
            """, unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**Savings Model Performance**")
                sav_m = metrics.get("savings", {})
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("MAE", f"₹{sav_m.get('MAE', 0):,.0f}")
                sm2.metric("RMSE", f"₹{sav_m.get('RMSE', 0):,.0f}")
                sm3.metric("R²", f"{sav_m.get('R2', 0):.4f}")
                st.caption(f"5-Fold CV R²: {sav_m.get('CV_R2_mean', 0):.4f} ± {sav_m.get('CV_R2_std', 0):.4f}")
                st.caption(f"Training Samples: {sav_m.get('train_size', 0):,} | Test Samples: {sav_m.get('test_size', 0):,}")
            with mc2:
                st.markdown("**Net Worth Model Performance**")
                nw_m = metrics.get("net_worth", {})
                nm1, nm2, nm3 = st.columns(3)
                nm1.metric("MAE", f"₹{nw_m.get('MAE', 0):,.0f}")
                nm2.metric("RMSE", f"₹{nw_m.get('RMSE', 0):,.0f}")
                nm3.metric("R²", f"{nw_m.get('R2', 0):.4f}")
                st.caption(f"5-Fold CV R²: {nw_m.get('CV_R2_mean', 0):.4f} ± {nw_m.get('CV_R2_std', 0):.4f}")
                st.caption(f"Anchor Horizon: {nw_m.get('horizon_months', horizon)} months")
        else:
            st.info("Evaluation metrics will appear once offline model weights are loaded.")

    # ── Forecast Report Row ──────────────────────────────────────────────────
    try:
        _pdf_traj = FinTwinPDFReport(
            report_title="Financial Forecast & Projection Report",
            user_id=twin.user_id,
            report_id=f"FC-{twin.user_id}",
            subtitle="AI-powered savings and net worth trajectory over 6 and 12 months",
        )
        _pdf_traj.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Financial Forecast & Projection Report",
        )
        _pdf_traj.add_kpi_summary_row([
            {"label": "6-Month Projected Savings", "value": f"Rs. {pred_6['predicted_savings']:,.0f}", "status": "positive"},
            {"label": "6-Month Projected Net Worth", "value": f"Rs. {pred_6['predicted_net_worth']:,.0f}", "status": "positive"},
            {"label": "12-Month Projected Savings", "value": f"Rs. {pred_12['predicted_savings']:,.0f}", "status": "positive"},
            {"label": "12-Month Projected Net Worth", "value": f"Rs. {pred_12['predicted_net_worth']:,.0f}", "status": "positive"},
        ])
        _pdf_traj.add_callout(
            "These projections are generated by an AI model trained on financial trajectories. "
            "Actual outcomes may vary based on market conditions and personal decisions.",
            style="info", title="Forecast Disclaimer"
        )
        _pdf_traj.add_section_divider("Forecasting Results")
        _pdf_traj.add_key_value_grid({
            "6-Month Projected Savings": pred_6["predicted_savings"],
            "6-Month Projected Net Worth": pred_6["predicted_net_worth"],
            "12-Month Projected Savings": pred_12["predicted_savings"],
            "12-Month Projected Net Worth": pred_12["predicted_net_worth"],
        })
        if fig_traj:
            _pdf_traj.add_plotly_figure(fig_traj, caption="12-Month Projected Financial Trajectory")
        if not df_traj.empty:
            _pdf_traj.add_section_divider("Trajectory Schedule")
            _pdf_traj.add_dataframe_table(df_traj)
        pdf_traj_bytes = _pdf_traj.build()
        pdf_traj_fn = _pdf_traj.suggested_filename("forecast_report")
    except Exception:
        pdf_traj_bytes = None
        pdf_traj_fn = "forecast_report.pdf"

    with st.container(border=True):
        c_rep1, c_rep2 = st.columns([2.0, 1.2], vertical_alignment="center")
        with c_rep1:
            st.markdown("""
            <div style="display: flex; align-items: flex-start; gap: 12px; padding: 2px 0;">
                <div style="background: rgba(79, 140, 255, 0.15); border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px;">
                    <i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.15rem;"></i>
                </div>
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;">
                        Your Forecast Report
                    </div>
                    <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                        Download a comprehensive PDF summary of your projected savings and net worth trajectory.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_rep2:
            if pdf_traj_bytes:
                st.download_button(
                    label="Download Forecast Report",
                    icon=":material/download:",
                    data=pdf_traj_bytes,
                    file_name=pdf_traj_fn,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="btn_download_forecast_report_tab1"
                )
            else:
                st.error("Report is currently unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="fc-section-header" style="margin-top: 0.5rem;">
        <h2 class="fc-section-title">Explore Future Scenarios</h2>
        <p class="fc-section-desc">See a range of possible financial outcomes based on different market and financial conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Simulation Controls Card ─────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.85rem; display: flex; align-items: center; gap: 8px;">
            <i class="fa-solid fa-sliders" style="color: #4F8CFF;"></i> Simulation Settings
        </div>
        """, unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        mc_years = mc1.slider("Forecast Horizon (Years)", 1, 10, 5, key="mc_slider_years")
        mc_sims  = mc2.slider("Simulations", 100, 2000, 1000, step=100, key="mc_slider_sims")
        mc_ret   = mc3.slider("Expected Annual Return (%)", 6.0, 20.0, 12.0, step=0.5, key="mc_slider_return") / 100

    # ── Run Simulation with Loading State ────────────────────────────────────
    simulator_mc = ScenarioSimulator(twin)
    with PremiumLoader(title="Running Financial Scenarios", initial_subtitle=f"Simulating {mc_sims:,} market pathways over {mc_years} years...", start_progress=20) as loader:
        loader.update(50, "Evaluating compounding interest & return distributions...")
        df_mc = simulator_mc.run_monte_carlo(
            years=mc_years,
            num_simulations=mc_sims,
            annual_return_mean=mc_ret
        )
        loader.update(100, "Compiling percentile ranges...")

    # ── Monte Carlo Chart ────────────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header">
        <h3 class="fc-section-title">Possible Future Net Worth</h3>
        <p class="fc-section-desc">The simulation shows a range of possible outcomes rather than a single prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    fig_mc = PlotlyVisualizer.plot_monte_carlo_fan(df_mc, years=mc_years)
    st.plotly_chart(fig_mc, use_container_width=True)

    # Explanatory Educational Note
    st.markdown("""
    <div class="fc-info-note">
        <strong>Understanding Percentiles:</strong> The middle line represents the median projected outcome. The surrounding shaded ranges show how results could vary across different simulated market scenarios over time.
    </div>
    """, unsafe_allow_html=True)

    # ── Outcome Summary ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header">
        <h3 class="fc-section-title">Possible Outcomes at the End of the Forecast</h3>
        <p class="fc-section-desc">Projected net worth distribution at Year {0}:</p>
    </div>
    """.format(mc_years), unsafe_allow_html=True)

    final_month = df_mc[df_mc["Month"] == df_mc["Month"].max()].iloc[0]
    gain_p50 = final_month["p50"] - twin.net_worth
    delta_str = f"↑ ₹{gain_p50:,.0f}" if gain_p50 >= 0 else f"↓ ₹{abs(gain_p50):,.0f}"

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 14px; margin-bottom: 2rem;">
        <div class="fc-metric-box">
            <div class="fc-metric-label" title="10th Percentile">Worst Case Range</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{final_month['p10']:,.0f}</div>
            <div class="fc-metric-sub">10th Percentile</div>
        </div>
        <div class="fc-metric-box">
            <div class="fc-metric-label" title="25th Percentile">Lower Range</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{final_month['p25']:,.0f}</div>
            <div class="fc-metric-sub">25th Percentile</div>
        </div>
        <div class="fc-metric-box-featured">
            <div class="fc-metric-label" style="color: #60A5FA;">Median Projection</div>
            <div class="fc-metric-val" style="font-size: 1.45rem;">₹{final_month['p50']:,.0f}</div>
            <div class="fc-delta-pos" style="font-size: 0.78rem;">{delta_str} vs today</div>
        </div>
        <div class="fc-metric-box">
            <div class="fc-metric-label" title="75th Percentile">Upper Range</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{final_month['p75']:,.0f}</div>
            <div class="fc-metric-sub">75th Percentile</div>
        </div>
        <div class="fc-metric-box">
            <div class="fc-metric-label" title="90th Percentile">Best Case Range</div>
            <div class="fc-metric-val" style="font-size: 1.3rem;">₹{final_month['p90']:,.0f}</div>
            <div class="fc-metric-sub">90th Percentile</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── What This Means Narrative ────────────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header" style="margin-top: 1rem;">
        <h3 class="fc-section-title">What This Forecast Means</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fc-narrative-box">
        Your median projected net worth is <strong>₹{final_month['p50']:,.0f}</strong> after <strong>{mc_years} years</strong>, 
        representing an estimated gain of <strong>₹{gain_p50:,.0f}</strong> compared to your current net worth. 
        Under 80% of simulated market conditions (between the 10th and 90th percentiles), your net worth is estimated to land between 
        <strong>₹{final_month['p10']:,.0f}</strong> and <strong>₹{final_month['p90']:,.0f}</strong>.
    </div>
    """, unsafe_allow_html=True)

    # ── Financial Goal / Target Section ──────────────────────────────────────
    st.markdown("""
    <div class="fc-section-header">
        <h3 class="fc-section-title">Set Your Financial Goal</h3>
        <p class="fc-section-desc">How much would you like your net worth to reach?</p>
    </div>
    """, unsafe_allow_html=True)

    target_nw = st.number_input(
        "Target Net Worth (₹)",
        min_value=100000,
        max_value=100000000,
        value=max(int(twin.net_worth * 2), 100000),
        step=100000,
        key="target_nw_monte_carlo"
    )

    # Probability estimation
    p50_final = final_month["p50"]
    if target_nw <= final_month["p10"]:
        prob_pct = 90
        badge_cls = "fc-goal-high"
        prob_text = "Your target appears highly achievable under the vast majority of simulated scenarios."
    elif target_nw <= final_month["p25"]:
        prob_pct = 75
        badge_cls = "fc-goal-high"
        prob_text = "Your target appears achievable under most simulated market scenarios."
    elif target_nw <= p50_final:
        prob_pct = 50
        badge_cls = "fc-goal-mid"
        prob_text = "Your target aligns with the central median projection over this timeframe."
    elif target_nw <= final_month["p75"]:
        prob_pct = 25
        badge_cls = "fc-goal-mid"
        prob_text = "Your target is ambitious and requires favorable market returns or increased monthly savings."
    elif target_nw <= final_month["p90"]:
        prob_pct = 10
        badge_cls = "fc-goal-low"
        prob_text = "Your target is challenging within this horizon; consider extending your timeline or raising savings."
    else:
        prob_pct = 5
        badge_cls = "fc-goal-low"
        prob_text = "Target exceeds 90% of simulated scenarios. Adjusting horizon or monthly investments is recommended."

    st.markdown(f"""
    <div class="fc-goal-card">
        <div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Estimated Probability of Reaching Your Goal
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
                Target: ₹{target_nw:,.0f} in {mc_years} Years
            </div>
            <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
                {prob_text}
            </div>
        </div>
        <div class="fc-goal-badge {badge_cls}">
            ~{prob_pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Connect to Scenario Simulator ────────────────────────────────────────
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        col_act1, col_act2 = st.columns([2.0, 1.2], vertical_alignment="center")
        with col_act1:
            st.markdown("""
            <div style="display: flex; align-items: flex-start; gap: 12px; padding: 2px 0;">
                <div style="background: rgba(79, 140, 255, 0.15); border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px;">
                    <i class="fa-solid fa-wand-magic-sparkles" style="color: #4F8CFF; font-size: 1.15rem;"></i>
                </div>
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;">
                        Want to test specific life changes or asset shocks?
                    </div>
                    <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                        Simulate job transitions, home purchases, inflation spikes, or career adjustments.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_act2:
            st.page_link(
                "pages/06_Scenario_Simulator.py",
                label="Explore Another Scenario →",
                icon=":material/tune:",
                use_container_width=True
            )

    # ── Progressive Disclosure: Monte Carlo Methodology ──────────────────────
    with st.expander("Technical Methodology — How the Monte Carlo simulation works", expanded=False):
        st.markdown("""
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
            <p><strong>Probabilistic Wealth Engine:</strong></p>
            <ul>
                <li>Simulates portfolio paths using lognormal return distributions calibrated to your expected mean return and historical asset-class volatility.</li>
                <li>Accounts for monthly recurring savings, investments (SIP), and debt outflows (EMI).</li>
                <li>Calculates non-parametric percentile bounds (10th, 25th, 50th, 75th, 90th) at every monthly step to establish robust confidence intervals.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Monte Carlo Report Row ───────────────────────────────────────────────
    try:
        _pdf_mc = FinTwinPDFReport(
            report_title="Monte Carlo Wealth Projection Report",
            user_id=twin.user_id,
            report_id=f"MC-{twin.user_id}",
            subtitle=f"Probabilistic {mc_years}-Year Net Worth Fan Path & Goal Feasibility",
        )
        _pdf_mc.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Monte Carlo Wealth Projection Report",
        )
        _prob_status = "positive" if prob_pct >= 60 else ("neutral" if prob_pct >= 35 else "negative")
        _pdf_mc.add_kpi_summary_row([
            {"label": "Target Net Worth", "value": f"Rs. {target_nw:,.0f}", "status": "neutral"},
            {"label": "Time Horizon", "value": f"{mc_years} Years", "status": "neutral"},
            {"label": "Target Probability", "value": f"~{prob_pct}%", "status": _prob_status},
            {"label": "Median Net Worth (p50)", "value": f"Rs. {final_month['p50']:,.0f}", "status": "positive"},
        ])
        _callout_style = "success" if prob_pct >= 60 else ("warning" if prob_pct >= 35 else "danger")
        _pdf_mc.add_callout(
            f"Based on {mc_years}-year stochastic modeling with calibrated return distributions: {prob_text}",
            style=_callout_style,
            title=f"Goal Feasibility Analysis (~{prob_pct}% Probability)"
        )
        _pdf_mc.add_section_divider("Simulation Key Metrics")
        _pdf_mc.add_key_value_grid({
            "Target Net Worth": f"Rs. {target_nw:,.0f}",
            "Horizon (Years)": f"{mc_years} Years",
            "Probability of Reaching Target": f"~{prob_pct}%",
            "Median Projected Net Worth (p50)": f"Rs. {final_month['p50']:,.0f}",
            "10th Percentile (Worst Case)": f"Rs. {final_month['p10']:,.0f}",
            "90th Percentile (Best Case)": f"Rs. {final_month['p90']:,.0f}",
        })
        if fig_mc:
            _pdf_mc.add_plotly_figure(fig_mc, caption="Monte Carlo Percentile Fan Path")
        if df_mc is not None and not df_mc.empty:
            _pdf_mc.add_section_divider("Monte Carlo Percentile Schedule")
            _pdf_mc.add_dataframe_table(df_mc)
        pdf_mc_bytes = _pdf_mc.build()
        pdf_mc_fn = _pdf_mc.suggested_filename("monte_carlo_report")
    except Exception:
        pdf_mc_bytes = None
        pdf_mc_fn = "monte_carlo_report.pdf"

    with st.container(border=True):
        c_mc1, c_mc2 = st.columns([2.0, 1.2], vertical_alignment="center")
        with c_mc1:
            st.markdown("""
            <div style="display: flex; align-items: flex-start; gap: 12px; padding: 2px 0;">
                <div style="background: rgba(79, 140, 255, 0.15); border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px;">
                    <i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.15rem;"></i>
                </div>
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;">
                        Monte Carlo Simulation Report
                    </div>
                    <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                        Download a detailed analysis of possible wealth outcomes and probability ranges.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_mc2:
            if pdf_mc_bytes:
                st.download_button(
                    label="Download Monte Carlo Report",
                    icon=":material/download:",
                    data=pdf_mc_bytes,
                    file_name=pdf_mc_fn,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="btn_download_mc_report_tab2"
                )
            else:
                st.error("Report is currently unavailable.")

