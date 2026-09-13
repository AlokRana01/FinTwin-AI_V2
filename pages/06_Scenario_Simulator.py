"""
Streamlit Page: Scenario Simulator
=================================
Allows simulating critical life scenarios (job loss, home purchase, market shocks, wedding, salary hike) 
and displays the before/after impact side-by-side with comparison charts and PDF report generation.
"""

import streamlit as st
import pandas as pd
from utils.session import render_sidebar_user_selector
from utils.visualizer import PlotlyVisualizer
from utils.simulator import ScenarioSimulator, SCENARIO_REGISTRY
from utils.pdf_report import FinTwinPDFReport

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scenario Simulator — FinTwin AI",
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
.sim-header-box {
    margin-bottom: 2.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.sim-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.03em;
}
.sim-subtitle {
    color: #94A3B8;
    font-size: 0.96rem;
    margin: 0;
    line-height: 1.55;
}

/* Section Header */
.sim-section-header {
    margin: 2.25rem 0 1.25rem 0;
}
.sim-section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sim-section-desc {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 0 0 1.15rem 0;
    line-height: 1.5;
}

/* Metric Display Boxes */
.sim-metric-box {
    background: #1F2937;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    min-height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    box-sizing: border-box;
}
.sim-metric-box:hover {
    border-color: rgba(79, 140, 255, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
}
.sim-impact-metric-box {
    background: #1E293B;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.95rem 1.1rem;
    min-height: 98px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.sim-impact-metric-box:hover {
    border-color: rgba(79, 140, 255, 0.4);
    transform: translateY(-1px);
}
.sim-metric-box-featured {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(79, 140, 255, 0.45);
    border-top: 3px solid #4F8CFF;
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    min-height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    box-sizing: border-box;
}
.sim-metric-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 700;
    color: #94A3B8;
    margin-bottom: 0.45rem;
}
.sim-metric-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.25;
    margin-bottom: 0.3rem;
}
.sim-metric-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}
.sim-delta-pos {
    color: #10B981;
    font-weight: 600;
    font-size: 0.84rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.sim-delta-neg {
    color: #EF4444;
    font-weight: 600;
    font-size: 0.84rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* Narrative Card */
.sim-narrative-box {
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

/* Assumptions Pill Strip */
.sim-pill-strip {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.75rem 0 1.75rem 0;
    color: #CBD5E1;
    font-size: 0.88rem;
    font-weight: 600;
}
.sim-pill-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #F8FAFC;
}
.sim-pill-sep {
    color: #64748B;
    font-weight: 400;
}

/* Impact Column Box */
.sim-impact-col {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem 1.65rem;
    box-sizing: border-box;
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
    padding: 0.6rem 0.75rem !important;
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
<div class="sim-header-box">
    <h1 class="sim-title">Scenario Simulator</h1>
    <p class="sim-subtitle">Explore how financial conditions and major life events could affect your future financial position.</p>
</div>
""", unsafe_allow_html=True)

# ── Empty / Incomplete Profile State ─────────────────────────────────────────
if not twin:
    st.markdown("""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; text-align: center; padding: 2.5rem 1.5rem; max-width: 680px; margin: 2rem auto; box-shadow: 0 4px 14px rgba(0,0,0,0.2);">
        <i class="fa-solid fa-user-gear" style="color: #4F8CFF; font-size: 2.2rem; margin-bottom: 0.85rem; display: inline-block;"></i>
        <h2 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.5rem 0;">
            Complete Your Financial Profile
        </h2>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5; margin: 0 0 1.5rem 0;">
            Add your financial information to run personalized scenario simulations and stress-test life events.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1.4, 1])
    with c_btn2:
        st.page_link("pages/01_Digital_Twin.py", label="Complete Financial Profile →", icon=":material/arrow_forward:", use_container_width=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TWO-TAB STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    ":material/trending_up: Asset Projections & Shocks",
    ":material/event: Specialized Life Events"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: ASSET PROJECTIONS & SHOCKS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    simulator = ScenarioSimulator(twin)

    st.markdown("""
    <div class="sim-section-header" style="margin-top: 0.5rem;">
        <h2 class="sim-section-title">Choose a Financial Scenario</h2>
        <p class="sim-section-desc">Explore how different financial conditions could affect your projected net worth.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        scenario = st.selectbox(
            "Select Scenario",
            ["Normal Trajectory (Inflation Model)",
             "Job Loss Shock",
             "Home Purchase",
             "Inflation & Market Returns"],
            label_visibility="collapsed",
            key="tab1_scenario_selector"
        )

    # ── Scenario: Normal Trajectory (Inflation Model) ────────────────────────
    if scenario == "Normal Trajectory (Inflation Model)":
        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">Scenario Settings</h3>
            <p class="sim-section-desc">Adjust timeline, inflation assumptions, and investment growth rates.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2 = st.columns(2)
            years = c1.slider("Projection Years", 1, 10, 5, key="norm_years")
            inflation_rate = c2.slider("Inflation Rate (%)", 3.0, 10.0, 6.0, step=0.5, key="norm_infl") / 100
            inv_growth = c1.slider("Investment Return (%)", 6.0, 20.0, 12.0, step=0.5, key="norm_ret") / 100

        # Current Scenario Summary Pill
        st.markdown(f"""
        <div class="sim-pill-strip">
            <span class="sim-pill-item"><i class="fa-solid fa-calendar-days" style="color:#4F8CFF;"></i> {years} Years</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-percent" style="color:#F59E0B;"></i> {inflation_rate:.1%} Inflation</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-chart-line" style="color:#10B981;"></i> {inv_growth:.1%} Return</span>
        </div>
        """, unsafe_allow_html=True)

        df_proj = simulator.simulate_market_returns_and_inflation(years, inflation_rate, inv_growth)
        fig_proj = PlotlyVisualizer.plot_forecast_trajectories(df_proj)

        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">How Your Net Worth Could Change</h3>
            <p class="sim-section-desc">Compare your projected wealth under different financial scenarios.</p>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_proj, use_container_width=True)

        final_row = df_proj.iloc[-1]
        base_val = final_row.get("Base Case", 0)
        opt_val = final_row.get("Optimistic Scenario", 0)
        pess_val = final_row.get("Pessimistic Scenario", 0)

        # Final-Year Outlook
        st.markdown(f"""
        <div class="sim-section-header">
            <h3 class="sim-section-title">{years}-Year Outlook</h3>
            <p class="sim-section-desc">Projected net worth distribution at Year {years}:</p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Pessimistic Scenario</div>
                <div class="sim-metric-val" style="font-size: 1.45rem;">₹{pess_val:,.0f}</div>
                <div class="sim-metric-sub">Conservative returns</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="sim-metric-box-featured">
                <div class="sim-metric-label" style="color: #60A5FA;">Base Case</div>
                <div class="sim-metric-val" style="font-size: 1.55rem;">₹{base_val:,.0f}</div>
                <div class="sim-metric-sub">Expected market baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Optimistic Scenario</div>
                <div class="sim-metric-val" style="font-size: 1.45rem; color: #10B981;">₹{opt_val:,.0f}</div>
                <div class="sim-metric-sub">Favorable equity growth</div>
            </div>
            """, unsafe_allow_html=True)

        # What This Scenario Means
        st.markdown("""
        <div class="sim-section-header" style="margin-top: 1rem;">
            <h3 class="sim-section-title">What This Scenario Means</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sim-narrative-box">
            Under the selected assumptions, your projected net worth reaches approximately <strong>₹{base_val:,.0f}</strong> in the base scenario over <strong>{years} years</strong>. 
            Under favorable market conditions, the optimistic scenario is estimated to reach approximately <strong>₹{opt_val:,.0f}</strong>, while under more conservative market conditions, the pessimistic scenario is estimated at <strong>₹{pess_val:,.0f}</strong>.
        </div>
        """, unsafe_allow_html=True)

        # Detailed Projection Table
        with st.expander("View Detailed Projection", expanded=False):
            st.dataframe(
                df_proj.style.format({"Base Case": "₹{:,.0f}", "Optimistic Scenario": "₹{:,.0f}", "Pessimistic Scenario": "₹{:,.0f}"}),
                use_container_width=True, hide_index=True
            )

        # Scenario Report
        try:
            _pdf_proj = FinTwinPDFReport(
                report_title="Scenario Impact Simulation Report",
                user_id=twin.user_id,
                report_id=f"SC-{twin.user_id}",
                subtitle="Inflation & investment growth model — base, optimistic, and pessimistic trajectories",
            )
            _pdf_proj.add_cover_page(
                user_name=getattr(twin, "name", "") or "",
                report_type="Scenario Impact Simulation Report",
            )
            _pdf_proj.add_kpi_summary_row([
                {"label": f"Base Case (Year {years})", "value": f"Rs. {base_val:,.0f}", "status": "neutral"},
                {"label": f"Optimistic (Year {years})", "value": f"Rs. {opt_val:,.0f}", "status": "positive"},
                {"label": f"Pessimistic (Year {years})", "value": f"Rs. {pess_val:,.0f}", "status": "negative"},
            ])
            _pdf_proj.add_callout(
                f"Over {years} years, the base case projects a net worth of Rs. {base_val:,.0f}. "
                f"In a favorable scenario (higher returns), this could reach Rs. {opt_val:,.0f}. "
                f"Under adverse conditions, the pessimistic projection is Rs. {pess_val:,.0f}.",
                style="info", title="Scenario Summary"
            )
            _pdf_proj.add_section_divider("Scenario Parameters")
            _pdf_proj.add_key_value_grid({
                "Projection Years": years,
                "Inflation Rate": f"{inflation_rate:.1%}",
                "Investment Return": f"{inv_growth:.1%}",
                f"Base Case (Year {years})": base_val,
                f"Optimistic (Year {years})": opt_val,
                f"Pessimistic (Year {years})": pess_val,
            })
            _pdf_proj.add_plotly_figure(fig_proj, caption="Base / Optimistic / Pessimistic Trajectories")
            _pdf_proj.add_section_divider("Year-by-Year Projections")
            _pdf_proj.add_dataframe_table(df_proj)
            pdf_proj_bytes = _pdf_proj.build()
            pdf_proj_fn = _pdf_proj.suggested_filename("inflation_model_projections")
        except Exception:
            pdf_proj_bytes = None
            pdf_proj_fn = "inflation_model_projections.pdf"

        st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
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
                            Scenario Report
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                            Download a detailed summary of how this scenario could affect your projected savings and net worth.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                if pdf_proj_bytes:
                    st.download_button(
                        label="Download Scenario Report",
                        icon=":material/download:",
                        data=pdf_proj_bytes,
                        file_name=pdf_proj_fn,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="btn_download_norm_report"
                    )
                else:
                    st.error("Report is currently unavailable.")

    # ── Scenario: Job Loss Shock ─────────────────────────────────────────────
    elif scenario == "Job Loss Shock":
        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">Scenario Settings</h3>
            <p class="sim-section-desc">Select how many months without primary income to simulate.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            duration = st.slider("Job Loss Duration (months)", 1, 24, 6, key="jl_dur_tab1")

        result = simulator.simulate_job_loss(duration)
        monthly_burn = max(twin.basic_expenses + twin.monthly_emi, 1)
        ef_coverage = twin.emergency_fund / monthly_burn

        # Current Scenario Summary Pill
        st.markdown(f"""
        <div class="sim-pill-strip">
            <span class="sim-pill-item"><i class="fa-solid fa-briefcase" style="color:#EF4444;"></i> {duration} Months Job Loss</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-shield-halved" style="color:#38BDF8;"></i> ₹{twin.emergency_fund:,.0f} Emergency Fund</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-fire" style="color:#F59E0B;"></i> ₹{monthly_burn:,.0f}/mo Fixed Burn</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">Financial Survival Metrics</h3>
            <p class="sim-section-desc">Impact on emergency reserves and liquidity during the simulated gap.</p>
        </div>
        """, unsafe_allow_html=True)

        jc1, jc2, jc3 = st.columns(3)
        with jc1:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Emergency Fund Coverage</div>
                <div class="sim-metric-val">{ef_coverage:.1f} mo</div>
                <div class="sim-metric-sub">Total reserve duration</div>
            </div>
            """, unsafe_allow_html=True)
        with jc2:
            exhausted_label = f"Month {result['exhaustion_month_ef']}" if result["exhaustion_month_ef"] != -1 else "Fully Intact"
            ex_color = "#EF4444" if result["exhaustion_month_ef"] != -1 else "#10B981"
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">EF Depletion Point</div>
                <div class="sim-metric-val" style="color: {ex_color};">{exhausted_label}</div>
                <div class="sim-metric-sub">When emergency fund ends</div>
            </div>
            """, unsafe_allow_html=True)
        with jc3:
            deficit_val = result["total_deficit"]
            def_str = f"₹{deficit_val:,.0f}" if deficit_val > 0 else "₹0 — Survived"
            def_color = "#EF4444" if deficit_val > 0 else "#10B981"
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Total Cash Deficit</div>
                <div class="sim-metric-val" style="color: {def_color};">{def_str}</div>
                <div class="sim-metric-sub">Net shortfall after reserves</div>
            </div>
            """, unsafe_allow_html=True)

        # What This Scenario Means
        st.markdown("""
        <div class="sim-section-header" style="margin-top: 1rem;">
            <h3 class="sim-section-title">What This Scenario Means</h3>
        </div>
        """, unsafe_allow_html=True)

        if result["total_deficit"] > 0:
            jl_explanation = (
                f"Under a <strong>{duration}-month</strong> income disruption, your existing emergency reserves are estimated to cover "
                f"approximately <strong>{ef_coverage:.1f} months</strong> of fixed obligations. "
                f"A cumulative cash shortfall of approximately <strong>₹{result['total_deficit']:,.0f}</strong> could occur before re-employment."
            )
        else:
            jl_explanation = (
                f"Under a <strong>{duration}-month</strong> income disruption, your emergency reserves and liquid savings are estimated to "
                f"fully sustain your living expenses and debt commitments without creating a cash deficit."
            )

        st.markdown(f"""
        <div class="sim-narrative-box">
            {jl_explanation}
        </div>
        """, unsafe_allow_html=True)

        # Detailed Projection Table
        with st.expander("View Monthly Survival Timeline", expanded=False):
            st.dataframe(
                result["df"].style.format({"Emergency Fund": "₹{:,.0f}", "Bank Savings": "₹{:,.0f}", "Monthly Burn": "₹{:,.0f}"}),
                use_container_width=True, hide_index=True
            )

        # Scenario Report
        try:
            _pdf_jl = FinTwinPDFReport(
                report_title="Job Loss Shock Scenario Report",
                user_id=twin.user_id,
                report_id=f"SC-{twin.user_id}",
                subtitle=f"Simulated {duration}-month income disruption survival analysis",
            )
            _pdf_jl.add_cover_page(
                user_name=getattr(twin, "name", "") or "",
                report_type="Job Loss Shock Scenario Report",
            )
            _pdf_jl.add_kpi_summary_row([
                {"label": "Shock Duration", "value": f"{duration} Months", "status": "neutral"},
                {"label": "EF Coverage", "value": f"{ef_coverage:.1f} Months",
                 "status": "positive" if ef_coverage >= duration else "negative"},
                {"label": "Total Cash Deficit", "value": f"Rs. {result['total_deficit']:,.0f}",
                 "status": "positive" if result['total_deficit'] <= 0 else "negative"},
            ])
            if result["total_deficit"] <= 0:
                _pdf_jl.add_callout(
                    f"Your emergency reserves are sufficient to sustain a {duration}-month income disruption with no cash deficit.",
                    style="success", title="Resilience Assessment"
                )
            else:
                _pdf_jl.add_callout(
                    f"A cumulative cash deficit of Rs. {result['total_deficit']:,.0f} is projected during a {duration}-month income disruption. "
                    "Build your emergency fund to at least 6 months of fixed obligations.",
                    style="danger", title="Resilience Alert"
                )
            _pdf_jl.add_section_divider("Financial Risk Assessment")
            _pdf_jl.add_key_value_grid({
                "Job Loss Duration (months)": duration,
                "Emergency Fund Coverage (months)": f"{ef_coverage:.1f}",
                "EF Exhausted at Month": result["exhaustion_month_ef"] if result["exhaustion_month_ef"] != -1 else "Still intact",
                "Total Cash Deficit": result["total_deficit"],
            })
            _pdf_jl.add_section_divider("Monthly Survival Timeline")
            _pdf_jl.add_dataframe_table(result["df"])
            pdf_jl_bytes = _pdf_jl.build()
            pdf_jl_fn = _pdf_jl.suggested_filename("job_loss_shock_report")
        except Exception:
            pdf_jl_bytes = None
            pdf_jl_fn = "job_loss_shock_report.pdf"

        st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
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
                            Scenario Report
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                            Download a detailed survival timeline and cash deficit assessment.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                if pdf_jl_bytes:
                    st.download_button(
                        label="Download Scenario Report",
                        icon=":material/download:",
                        data=pdf_jl_bytes,
                        file_name=pdf_jl_fn,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="btn_download_jl_report"
                    )
                else:
                    st.error("Report is currently unavailable.")

    # ── Scenario: Home Purchase ──────────────────────────────────────────────
    elif scenario == "Home Purchase":
        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">Scenario Settings</h3>
            <p class="sim-section-desc">Configure property pricing, downpayment ratio, and mortgage terms.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            hc1, hc2 = st.columns(2)
            price = hc1.number_input("Property Price (₹)", min_value=500000, max_value=50000000, value=5000000, step=500000, key="hp_price_tab1")
            dp_pct = hc2.slider("Downpayment (%)", 10, 40, 20, key="hp_dp_tab1") / 100
            loan_rate = hc1.slider("Home Loan Rate (%)", 6.5, 12.0, 8.5, step=0.1, key="hp_rate_tab1") / 100
            tenure = hc2.slider("Loan Tenure (Years)", 5, 30, 20, key="hp_tenure_tab1")

        result = simulator.simulate_home_purchase(price, dp_pct, loan_rate, tenure)

        # Current Scenario Summary Pill
        st.markdown(f"""
        <div class="sim-pill-strip">
            <span class="sim-pill-item"><i class="fa-solid fa-house" style="color:#38BDF8;"></i> ₹{price:,.0f} Property</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-money-bill-wave" style="color:#10B981;"></i> {dp_pct:.0%} Downpayment</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-landmark" style="color:#F59E0B;"></i> {tenure} Yrs @ {loan_rate:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">Post-Purchase Financial Impact</h3>
            <p class="sim-section-desc">Upfront capital requirements, new monthly EMI burden, and liquidity reserve.</p>
        </div>
        """, unsafe_allow_html=True)

        hm1, hm2, hm3, hm4 = st.columns(4)
        with hm1:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Total Upfront Cash</div>
                <div class="sim-metric-val" style="font-size: 1.35rem;">₹{result['total_upfront_cash']:,.0f}</div>
                <div class="sim-metric-sub">Downpayment required</div>
            </div>
            """, unsafe_allow_html=True)
        with hm2:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">New Monthly EMI</div>
                <div class="sim-metric-val" style="font-size: 1.35rem;">₹{result['new_emi']:,.0f}</div>
                <div class="sim-metric-sub">{tenure}-year loan amortization</div>
            </div>
            """, unsafe_allow_html=True)
        with hm3:
            surplus_delta_val = result['surplus_change']
            surplus_delta_html = f'<div class="sim-delta-pos">↑ ₹{surplus_delta_val:,.0f}</div>' if surplus_delta_val >= 0 else f'<div class="sim-delta-neg">↓ ₹{abs(surplus_delta_val):,.0f}</div>'
            st.markdown(f"""
            <div class="sim-metric-box-featured">
                <div class="sim-metric-label" style="color:#60A5FA;">New Monthly Surplus</div>
                <div class="sim-metric-val" style="font-size: 1.35rem;">₹{result['new_monthly_surplus']:,.0f}</div>
                {surplus_delta_html}
            </div>
            """, unsafe_allow_html=True)
        with hm4:
            rem_sav = result['remaining_savings']
            rem_color = "#10B981" if rem_sav >= 0 else "#EF4444"
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Remaining Savings</div>
                <div class="sim-metric-val" style="font-size: 1.35rem; color: {rem_color};">₹{rem_sav:,.0f}</div>
                <div class="sim-metric-sub">After upfront outflow</div>
            </div>
            """, unsafe_allow_html=True)

        # What This Scenario Means
        st.markdown("""
        <div class="sim-section-header" style="margin-top: 1rem;">
            <h3 class="sim-section-title">What This Scenario Means</h3>
        </div>
        """, unsafe_allow_html=True)

        if result["can_afford"]:
            hp_explanation = (
                f"Your current liquid savings are sufficient to cover the estimated <strong>₹{result['total_upfront_cash']:,.0f}</strong> upfront downpayment. "
                f"Following the purchase, your ongoing monthly commitment increases with a new EMI of <strong>₹{result['new_emi']:,.0f}</strong>, "
                f"leaving an estimated monthly surplus of <strong>₹{result['new_monthly_surplus']:,.0f}</strong>."
            )
        else:
            hp_explanation = (
                f"The estimated upfront cash requirement of <strong>₹{result['total_upfront_cash']:,.0f}</strong> exceeds your current liquid savings. "
                f"Consider adjusting the downpayment percentage, choosing a different price point, or accumulating additional savings before purchase."
            )

        st.markdown(f"""
        <div class="sim-narrative-box">
            {hp_explanation}
        </div>
        """, unsafe_allow_html=True)

        # Detailed Projection Table
        with st.expander("View Post-Purchase Savings Trajectory", expanded=False):
            st.dataframe(
                result["df"].style.format({"Monthly Surplus": "₹{:,.0f}", "Bank Savings": "₹{:,.0f}"}),
                use_container_width=True, hide_index=True
            )

        # Scenario Report
        try:
            _pdf_hp = FinTwinPDFReport(
                report_title="Home Purchase Scenario Report",
                user_id=twin.user_id,
                report_id=f"SC-{twin.user_id}",
                subtitle="Property affordability, mortgage impact & post-purchase savings trajectory",
            )
            _pdf_hp.add_cover_page(
                user_name=getattr(twin, "name", "") or "",
                report_type="Home Purchase Scenario Report",
            )
            can_afford = result["can_afford"]
            _pdf_hp.add_kpi_summary_row([
                {"label": "Affordability", "value": "Yes" if can_afford else "No",
                 "status": "positive" if can_afford else "negative"},
                {"label": "Total Upfront Cash", "value": f"Rs. {result['total_upfront_cash']:,.0f}", "status": "neutral"},
                {"label": "New Home Loan EMI", "value": f"Rs. {result['new_emi']:,.0f}", "status": "neutral"},
                {"label": "New Monthly Surplus", "value": f"Rs. {result['new_monthly_surplus']:,.0f}",
                 "status": "positive" if result['new_monthly_surplus'] > 0 else "negative"},
            ])
            if can_afford:
                _pdf_hp.add_callout(
                    f"You can afford the downpayment of Rs. {result['total_upfront_cash']:,.0f}. "
                    f"After purchase, your monthly surplus will be Rs. {result['new_monthly_surplus']:,.0f}.",
                    style="success", title="Affordability"
                )
            else:
                _pdf_hp.add_callout(
                    f"The upfront cash requirement of Rs. {result['total_upfront_cash']:,.0f} exceeds your current liquid savings. "
                    "Consider adjusting the downpayment percentage or accumulating additional savings first.",
                    style="danger", title="Affordability Alert"
                )
            _pdf_hp.add_section_divider("Purchase Analysis")
            _pdf_hp.add_key_value_grid({
                "Property Price": price,
                "Downpayment %": f"{dp_pct:.0%}",
                "Home Loan Rate": f"{loan_rate:.1%}",
                "Loan Tenure (Years)": tenure,
                "Total Upfront Cash": result["total_upfront_cash"],
                "New Home Loan EMI": result["new_emi"],
                "New Monthly Surplus": result["new_monthly_surplus"],
                "Remaining Savings": result["remaining_savings"],
                "Can Afford Downpayment": "Yes" if can_afford else "No",
            })
            _pdf_hp.add_section_divider("12-Month Post-Purchase Savings Trajectory")
            _pdf_hp.add_dataframe_table(result["df"])
            pdf_hp_bytes = _pdf_hp.build()
            pdf_hp_fn = _pdf_hp.suggested_filename("home_purchase_report")
        except Exception:
            pdf_hp_bytes = None
            pdf_hp_fn = "home_purchase_report.pdf"

        st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
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
                            Scenario Report
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                            Download a detailed analysis of upfront downpayment and recurring mortgage impact.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                if pdf_hp_bytes:
                    st.download_button(
                        label="Download Scenario Report",
                        icon=":material/download:",
                        data=pdf_hp_bytes,
                        file_name=pdf_hp_fn,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="btn_download_hp_report"
                    )
                else:
                    st.error("Report is currently unavailable.")

    # ── Scenario: Inflation & Market Returns ─────────────────────────────────
    elif scenario == "Inflation & Market Returns":
        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">Scenario Settings</h3>
            <p class="sim-section-desc">Test different macroeconomic inflation environments and equity returns.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            ic1, ic2 = st.columns(2)
            years_i = ic1.slider("Projection Years", 1, 10, 5, key="im_years_tab1")
            inflation = ic2.slider("Base Inflation (%)", 3.0, 12.0, 6.0, step=0.5, key="im_infl_tab1") / 100
            equity_ret = ic1.slider("Base Equity Return (%)", 6.0, 20.0, 12.0, step=0.5, key="im_ret_tab1") / 100

        # Current Scenario Summary Pill
        st.markdown(f"""
        <div class="sim-pill-strip">
            <span class="sim-pill-item"><i class="fa-solid fa-calendar-days" style="color:#4F8CFF;"></i> {years_i} Years</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-percent" style="color:#F59E0B;"></i> {inflation:.1%} Inflation</span>
            <span class="sim-pill-sep">•</span>
            <span class="sim-pill-item"><i class="fa-solid fa-chart-line" style="color:#10B981;"></i> {equity_ret:.1%} Equity Return</span>
        </div>
        """, unsafe_allow_html=True)

        df_infl = simulator.simulate_market_returns_and_inflation(years_i, inflation, equity_ret)
        fig_infl = PlotlyVisualizer.plot_forecast_trajectories(df_infl)

        st.markdown("""
        <div class="sim-section-header">
            <h3 class="sim-section-title">How Your Net Worth Could Change</h3>
            <p class="sim-section-desc">Compare your projected wealth under different macroeconomic conditions.</p>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_infl, use_container_width=True)

        final_row_i = df_infl.iloc[-1]
        base_val_i = final_row_i.get("Base Case", 0)
        opt_val_i = final_row_i.get("Optimistic Scenario", 0)
        pess_val_i = final_row_i.get("Pessimistic Scenario", 0)

        # Final-Year Outlook
        st.markdown(f"""
        <div class="sim-section-header">
            <h3 class="sim-section-title">{years_i}-Year Outlook</h3>
            <p class="sim-section-desc">Projected net worth distribution at Year {years_i}:</p>
        </div>
        """, unsafe_allow_html=True)

        fi1, fi2, fi3 = st.columns(3)
        with fi1:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Pessimistic Scenario</div>
                <div class="sim-metric-val" style="font-size: 1.45rem;">₹{pess_val_i:,.0f}</div>
                <div class="sim-metric-sub">Lower equity growth</div>
            </div>
            """, unsafe_allow_html=True)
        with fi2:
            st.markdown(f"""
            <div class="sim-metric-box-featured">
                <div class="sim-metric-label" style="color: #60A5FA;">Base Case</div>
                <div class="sim-metric-val" style="font-size: 1.55rem;">₹{base_val_i:,.0f}</div>
                <div class="sim-metric-sub">Expected market baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with fi3:
            st.markdown(f"""
            <div class="sim-metric-box">
                <div class="sim-metric-label">Optimistic Scenario</div>
                <div class="sim-metric-val" style="font-size: 1.45rem; color: #10B981;">₹{opt_val_i:,.0f}</div>
                <div class="sim-metric-sub">Higher market returns</div>
            </div>
            """, unsafe_allow_html=True)

        # What This Scenario Means
        st.markdown("""
        <div class="sim-section-header" style="margin-top: 1rem;">
            <h3 class="sim-section-title">What This Scenario Means</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sim-narrative-box">
            Under a <strong>{inflation:.1%}</strong> inflation baseline and <strong>{equity_ret:.1%}</strong> expected equity return, 
            your projected net worth reaches approximately <strong>₹{base_val_i:,.0f}</strong> over <strong>{years_i} years</strong>. 
            In an optimistic market cycle, it could reach approximately <strong>₹{opt_val_i:,.0f}</strong>, while under conservative returns it is estimated at <strong>₹{pess_val_i:,.0f}</strong>.
        </div>
        """, unsafe_allow_html=True)

        # Detailed Projection Table
        with st.expander("View Detailed Projection", expanded=False):
            st.dataframe(
                df_infl.style.format({"Base Case": "₹{:,.0f}", "Optimistic Scenario": "₹{:,.0f}", "Pessimistic Scenario": "₹{:,.0f}"}),
                use_container_width=True, hide_index=True
            )

        # Scenario Report
        try:
            _pdf_infl = FinTwinPDFReport(
                report_title="Inflation & Market Returns Scenario Report",
                user_id=twin.user_id,
                report_id=f"SC-{twin.user_id}",
                subtitle=f"Base / optimistic / pessimistic wealth projections over {years_i} years",
            )
            _pdf_infl.add_cover_page(
                user_name=getattr(twin, "name", "") or "",
                report_type="Inflation & Market Returns Scenario Report",
            )
            _pdf_infl.add_kpi_summary_row([
                {"label": f"Base Case (Year {years_i})", "value": f"Rs. {base_val_i:,.0f}", "status": "neutral"},
                {"label": f"Optimistic (Year {years_i})", "value": f"Rs. {opt_val_i:,.0f}", "status": "positive"},
                {"label": f"Pessimistic (Year {years_i})", "value": f"Rs. {pess_val_i:,.0f}", "status": "negative"},
            ])
            _pdf_infl.add_callout(
                f"Over {years_i} years at {inflation:.1%} inflation and {equity_ret:.1%} base equity return, "
                f"your projected net worth ranges from Rs. {pess_val_i:,.0f} (pessimistic) to Rs. {opt_val_i:,.0f} (optimistic).",
                style="info", title="Scenario Summary"
            )
            _pdf_infl.add_section_divider("Scenario Parameters")
            _pdf_infl.add_key_value_grid({
                "Projection Years": years_i,
                "Base Inflation": f"{inflation:.1%}",
                "Base Equity Return": f"{equity_ret:.1%}",
                f"Base Case (Year {years_i})": base_val_i,
                f"Optimistic (Year {years_i})": opt_val_i,
                f"Pessimistic (Year {years_i})": pess_val_i,
            })
            _pdf_infl.add_plotly_figure(fig_infl, caption="Inflation & Market Return Scenarios")
            _pdf_infl.add_section_divider("Scenario Data")
            _pdf_infl.add_dataframe_table(df_infl)
            pdf_infl_bytes = _pdf_infl.build()
            pdf_infl_fn = _pdf_infl.suggested_filename("inflation_market_returns_report")
        except Exception:
            pdf_infl_bytes = None
            pdf_infl_fn = "inflation_market_returns_report.pdf"

        st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
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
                            Scenario Report
                        </div>
                        <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                            Download a detailed analysis of compounding wealth under variable macroeconomic regimes.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_rep2:
                if pdf_infl_bytes:
                    st.download_button(
                        label="Download Scenario Report",
                        icon=":material/download:",
                        data=pdf_infl_bytes,
                        file_name=pdf_infl_fn,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                        key="btn_download_infl_report"
                    )
                else:
                    st.error("Report is currently unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SPECIALIZED LIFE EVENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="sim-section-header" style="margin-top: 0.5rem;">
        <h2 class="sim-section-title">Explore Life Events</h2>
        <p class="sim-section-desc">See how major life changes could affect your income, savings, net worth and financial health.</p>
    </div>
    """, unsafe_allow_html=True)

    sim = ScenarioSimulator(twin)

    # ── Choose a Life Event ──────────────────────────────────────────────────
    all_scenarios = list(SCENARIO_REGISTRY.keys())
    with st.container(border=True):
        scenario_name = st.selectbox(
            "Choose a Life Event",
            all_scenarios,
            index=0,
            label_visibility="collapsed",
            key="tab2_life_event_selector"
        )

    # ── Scenario Settings ────────────────────────────────────────────────────
    st.markdown("""
    <div class="sim-section-header">
        <h3 class="sim-section-title">Scenario Settings</h3>
        <p class="sim-section-desc">Configure parameters for the selected life milestone.</p>
    </div>
    """, unsafe_allow_html=True)

    params = {}
    with st.container(border=True):
        if scenario_name == "Salary Hike":
            pc1, pc2 = st.columns(2)
            params["hike_percent"] = pc1.slider(
                "Salary Hike (%)", min_value=5, max_value=100, value=20, step=5, key="le_hike_pct"
            )

        elif scenario_name == "Job Loss":
            pc1, pc2 = st.columns(2)
            params["duration_months"] = pc1.slider("Job Loss Duration (months)", 1, 24, 6, key="le_jl_dur")
            params["recovery_salary_pct"] = pc2.slider(
                "Recovery Salary (% of old salary)", 50, 130, 100, step=5, key="le_rec_sal"
            )

        elif scenario_name == "Car Purchase":
            pc1, pc2 = st.columns(2)
            params["car_price"] = pc1.number_input(
                "Car On-Road Price (₹)", min_value=200000, max_value=20000000, value=800000, step=50000, key="le_car_price"
            )
            params["downpayment_pct"] = pc2.slider("Downpayment (%)", 10, 50, 20, key="le_car_dp") / 100
            params["loan_rate"] = pc1.slider("Car Loan Rate (%)", 6.0, 15.0, 8.5, step=0.25, key="le_car_rate") / 100
            params["tenure_years"] = pc2.slider("Tenure (Years)", 1, 7, 5, key="le_car_tenure")

        elif scenario_name == "Home Loan":
            pc1, pc2 = st.columns(2)
            params["purchase_price"] = pc1.number_input(
                "Property Price (₹)", min_value=500000, max_value=100000000, value=5000000, step=500000, key="le_home_price"
            )
            params["downpayment_pct"] = pc2.slider("Downpayment (%)", 10, 40, 20, key="le_home_dp") / 100
            params["home_loan_rate"] = pc1.slider("Home Loan Rate (%)", 6.5, 12.0, 8.5, step=0.1, key="le_home_rate") / 100
            params["tenure_years"] = pc2.slider("Loan Tenure (Years)", 5, 30, 20, key="le_home_tenure")

        elif scenario_name == "Marriage Expense":
            pc1, pc2 = st.columns(2)
            params["one_time_cost"] = pc1.number_input(
                "Total Wedding Cost (₹)", min_value=50000, max_value=10000000, value=500000, step=50000, key="le_wed_cost"
            )
            params["monthly_expense_hike"] = pc2.number_input(
                "Monthly Expense Increase (₹)", min_value=0, max_value=100000, value=10000, step=1000, key="le_wed_exp_hike"
            )

        elif scenario_name == "Increase SIP":
            pc1, pc2 = st.columns(2)
            sip_mode = pc1.radio(
                "Specify by", ["Percentage Increase", "Fixed Amount"],
                horizontal=True, key="le_sip_mode"
            )
            if sip_mode == "Percentage Increase":
                params["sip_increase_pct"] = pc2.slider(
                    "SIP Increase (%)", 10, 200, 50, step=10, key="le_sip_pct"
                )
            else:
                params["new_sip_amount"] = pc2.number_input(
                    "New Monthly SIP (₹)",
                    min_value=int(twin.sip_amount) if twin else 0,
                    max_value=500000,
                    value=int(twin.sip_amount * 2) if twin else 10000,
                    step=1000,
                    key="le_sip_amt"
                )

    # Run Scenario Calculation
    result = sim.run_scenario(scenario_name, **params)
    before = result.before
    after = result.after

    # Risk badge helper
    _risk_colors = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444", "Critical": "#7C3AED"}
    _risk_classes = {"Low": "fa-solid fa-circle-check", "Moderate": "fa-solid fa-triangle-exclamation", "High": "fa-solid fa-circle-exclamation", "Critical": "fa-solid fa-circle-xmark"}
    def risk_pill(level):
        c = _risk_colors.get(level, "#6B7280")
        fa_class = _risk_classes.get(level, "fa-solid fa-circle-info")
        return f'<span style="background:{c}22;border:1px solid {c};color:{c};border-radius:8px;padding:3px 12px;font-weight:700;display:inline-flex;align-items:center;gap:6px;"><i class="{fa_class}" style="font-size:0.95rem;line-height:1;"></i>{level}</span>'

    # ── Financial Impact (Before vs After) ───────────────────────────────────
    st.markdown(f"""
    <div class="sim-section-header">
        <h3 class="sim-section-title">Financial Impact</h3>
        <p class="sim-section-desc">Side-by-side comparison before and after simulating {result.scenario_name}.</p>
    </div>
    """, unsafe_allow_html=True)

    income_delta  = after.monthly_income  - before.monthly_income
    surplus_delta = after.monthly_surplus - before.monthly_surplus
    nw_delta      = after.net_worth       - before.net_worth
    hs_delta      = after.health_score    - before.health_score
    sav_delta     = after.future_savings_12m  - before.future_savings_12m
    fnw_delta     = after.future_net_worth_5y - before.future_net_worth_5y

    def format_delta(d, is_currency=True, prefix="₹"):
        if d > 0:
            val_str = f"↑ {prefix}{d:,.0f}" if is_currency else f"↑ {d:.1f}"
            return f'<div class="sim-delta-pos" style="font-size: 0.8rem; margin-top: 4px;">{val_str}</div>'
        elif d < 0:
            val_str = f"↓ {prefix}{abs(d):,.0f}" if is_currency else f"↓ {abs(d):.1f}"
            return f'<div class="sim-delta-neg" style="font-size: 0.8rem; margin-top: 4px;">{val_str}</div>'
        return '<div class="sim-metric-sub" style="font-size: 0.8rem; margin-top: 4px; color: #64748B;">No Change</div>'

    before_card_html = f"""<div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.5rem 1.6rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);"><div><div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.12rem; font-weight: 700; color: #94A3B8; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 8px;"><i class="fa-solid fa-clock-rotate-left" style="color: #64748B;"></i> Before Event</div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;"><div class="sim-impact-metric-box"><div class="sim-metric-label">Monthly Income</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{before.monthly_income:,.0f}</div><div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline</div></div><div class="sim-impact-metric-box"><div class="sim-metric-label">Monthly Surplus</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{before.monthly_surplus:,.0f}</div><div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline</div></div><div class="sim-impact-metric-box"><div class="sim-metric-label">Net Worth</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{before.net_worth:,.0f}</div><div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline</div></div><div class="sim-impact-metric-box"><div class="sim-metric-label">Health Score</div><div class="sim-metric-val" style="font-size: 1.35rem;">{before.health_score:.1f} <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">/100</span></div><div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline</div></div><div class="sim-impact-metric-box"><div class="sim-metric-label">Savings (12M)</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{before.future_savings_12m:,.0f}</div><div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline</div></div><div class="sim-impact-metric-box"><div class="sim-metric-label">Net Worth (5Y)</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{before.future_net_worth_5y:,.0f}</div><div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline</div></div></div></div><div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08);"><span style="font-size: 0.88rem; font-weight: 600; color: #94A3B8;">Risk Level</span>{risk_pill(before.risk_level)}</div></div>"""

    after_card_html = f"""<div style="background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, #111827 100%); border: 1.5px solid rgba(79, 140, 255, 0.45); border-radius: 14px; padding: 1.5rem 1.6rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);"><div><div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.12rem; font-weight: 700; color: #F8FAFC; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 8px;"><i class="fa-solid fa-wand-magic-sparkles" style="color: #4F8CFF;"></i> After {result.scenario_name}</div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;"><div class="sim-impact-metric-box" style="border-color: rgba(79, 140, 255, 0.25);"><div class="sim-metric-label">Monthly Income</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{after.monthly_income:,.0f}</div>{format_delta(income_delta)}</div><div class="sim-impact-metric-box" style="border-color: rgba(79, 140, 255, 0.25);"><div class="sim-metric-label">Monthly Surplus</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{after.monthly_surplus:,.0f}</div>{format_delta(surplus_delta)}</div><div class="sim-impact-metric-box" style="border-color: rgba(79, 140, 255, 0.25);"><div class="sim-metric-label">Net Worth</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{after.net_worth:,.0f}</div>{format_delta(nw_delta)}</div><div class="sim-impact-metric-box" style="border-color: rgba(79, 140, 255, 0.25);"><div class="sim-metric-label">Health Score</div><div class="sim-metric-val" style="font-size: 1.35rem;">{after.health_score:.1f} <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">/100</span></div>{format_delta(hs_delta, is_currency=False)}</div><div class="sim-impact-metric-box" style="border-color: rgba(79, 140, 255, 0.25);"><div class="sim-metric-label">Savings (12M)</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{after.future_savings_12m:,.0f}</div>{format_delta(sav_delta)}</div><div class="sim-impact-metric-box" style="border-color: rgba(79, 140, 255, 0.25);"><div class="sim-metric-label">Net Worth (5Y)</div><div class="sim-metric-val" style="font-size: 1.35rem;">₹{after.future_net_worth_5y:,.0f}</div>{format_delta(fnw_delta)}</div></div></div><div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08);"><span style="font-size: 0.88rem; font-weight: 600; color: #94A3B8;">Risk Level</span>{risk_pill(after.risk_level)}</div></div>"""

    impact_grid_html = f"""<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin: 1.25rem 0 2rem 0;">{before_card_html}{after_card_html}</div>"""
    st.markdown(impact_grid_html, unsafe_allow_html=True)

    # ── Scenario Impact Summary ──────────────────────────────────────────────
    st.markdown("""
    <div class="sim-section-header" style="margin-top: 1.5rem;">
        <h3 class="sim-section-title">Scenario Impact</h3>
    </div>
    """, unsafe_allow_html=True)

    if surplus_delta >= 0 and hs_delta >= 0:
        impact_narrative = (
            f"This scenario increases your monthly surplus by <strong>₹{surplus_delta:,.0f}</strong> and improves your financial health score by <strong>{hs_delta:.1f} points</strong>. "
            f"Your projected 5-year net worth is estimated to expand by <strong>₹{fnw_delta:,.0f}</strong>."
        )
    elif surplus_delta < 0 and hs_delta <= 0:
        impact_narrative = (
            f"This scenario reduces your monthly surplus by <strong>₹{abs(surplus_delta):,.0f}</strong> and adjusts your health score by <strong>{hs_delta:.1f} points</strong>. "
            f"Reviewing discretionary outflows or adjusting recurring commitments can help buffer this transition."
        )
    else:
        impact_narrative = (
            f"This scenario results in a monthly surplus change of <strong>₹{surplus_delta:,.0f}</strong> with a financial health score adjustment of <strong>{hs_delta:.1f} points</strong>. "
            f"Your long-term 5-year wealth position shifts by an estimated <strong>₹{fnw_delta:,.0f}</strong>."
        )

    st.markdown(f"""
    <div class="sim-narrative-box">
        {impact_narrative}
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison Chart ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="sim-section-header">
        <h3 class="sim-section-title">Financial Impact Comparison</h3>
        <p class="sim-section-desc">Key metrics evaluated before versus after {0}.</p>
    </div>
    """.format(result.scenario_name), unsafe_allow_html=True)

    chart_metrics = {
        "Monthly Surplus":  (before.monthly_surplus,  after.monthly_surplus),
        "Health Score":     (before.health_score,      after.health_score),
        "Savings 12M":      (before.future_savings_12m, after.future_savings_12m),
        "Net Worth 5Y":     (before.future_net_worth_5y, after.future_net_worth_5y),
    }
    before_chart = {k: v[0] for k, v in chart_metrics.items()}
    after_chart  = {k: v[1] for k, v in chart_metrics.items()}

    fig_cmp = PlotlyVisualizer.plot_scenario_comparison_bar(
        before_chart, after_chart,
        title=f"Key Metrics: Before vs After {result.scenario_name}"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── 12-Month Savings Trajectory ──────────────────────────────────────────
    st.markdown("""
    <div class="sim-section-header">
        <h3 class="sim-section-title">12-Month Savings Outlook</h3>
        <p class="sim-section-desc">Projected savings trajectory curve compared before and after the event.</p>
    </div>
    """, unsafe_allow_html=True)

    TRAJ_MONTHS = 12
    before_traj = [before._project_savings(m) for m in range(1, TRAJ_MONTHS + 1)]
    after_traj = [after._project_savings(m) for m in range(1, TRAJ_MONTHS + 1)]

    fig_traj = PlotlyVisualizer.plot_scenario_trajectory_comparison(
        before_traj, after_traj, months=TRAJ_MONTHS, scenario_name=result.scenario_name
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Detailed Comparison Table ────────────────────────────────────────────
    df_metrics = result.comparison_df()
    with st.expander("View Detailed Metrics", expanded=False):
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ── Scenario-specific extra details ──────────────────────────────────────
    extras = result.extra_details
    if extras:
        with st.expander("View Scenario Details", expanded=False):
            if "timeline_df" in extras:
                st.markdown("**Monthly Survival Timeline:**")
                st.dataframe(
                    extras["timeline_df"].style.format(
                        {"Emergency Fund": "₹{:,.0f}", "Bank Savings": "₹{:,.0f}", "Monthly Burn": "₹{:,.0f}"}
                    ),
                    use_container_width=True, hide_index=True,
                )
                if extras.get("total_deficit", 0) > 0:
                    st.error(f"Total Cash Deficit: ₹{extras['total_deficit']:,.0f}")
                else:
                    st.success("Reserves are sufficient to cover the entire simulated period.")
            else:
                safe_extras = {k: v for k, v in extras.items() if not hasattr(v, "to_dict")}
                detail_rows = []
                for k, v in safe_extras.items():
                    if isinstance(v, float):
                        label = k.replace("_", " ").title()
                        detail_rows.append({"Detail": label, "Value": f"₹{v:,.0f}"})
                    elif isinstance(v, bool):
                        label = k.replace("_", " ").title()
                        detail_rows.append({"Detail": label, "Value": "Yes" if v else "No"})
                    elif isinstance(v, (int, str)):
                        label = k.replace("_", " ").title()
                        detail_rows.append({"Detail": label, "Value": str(v)})
                if detail_rows:
                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True, hide_index=True
                    )

    # ── Scenario Report ──────────────────────────────────────────────────────
    try:
        _pdf_cmp = FinTwinPDFReport(
            report_title="Life Event Scenario Impact Report",
            user_id=twin.user_id,
            report_id=f"SC-{twin.user_id}",
            subtitle=f"Before vs After — {result.scenario_name}",
        )
        _pdf_cmp.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Life Event Scenario Impact Report",
        )
        _pdf_cmp.add_kpi_summary_row([
            {"label": "Monthly Surplus (Before)", "value": f"Rs. {before.monthly_surplus:,.0f}", "status": "neutral"},
            {"label": "Monthly Surplus (After)", "value": f"Rs. {after.monthly_surplus:,.0f}",
             "status": "positive" if surplus_delta >= 0 else "negative"},
            {"label": "Health Score Change", "value": f"{hs_delta:+.1f} pts",
             "status": "positive" if hs_delta >= 0 else "negative"},
            {"label": "5-Year Net Worth Change", "value": f"Rs. {fnw_delta:+,.0f}",
             "status": "positive" if fnw_delta >= 0 else "negative"},
        ])
        if surplus_delta >= 0 and hs_delta >= 0:
            _pdf_cmp.add_callout(
                f"The {result.scenario_name} scenario improves your monthly surplus"
                f" and enhances your financial health score by {hs_delta:.1f} points.",
                style="success", title="Scenario Impact"
            )
        elif surplus_delta < 0 and hs_delta < 0:
            _pdf_cmp.add_callout(
                f"The {result.scenario_name} scenario reduces your monthly surplus"
                f" and lowers your financial health score by {abs(hs_delta):.1f} points.",
                style="danger", title="Scenario Impact"
            )
        else:
            _pdf_cmp.add_callout(
                f"The {result.scenario_name} scenario results in a mixed impact with a health score adjustment of {hs_delta:.1f} points.",
                style="warning", title="Scenario Impact"
            )
        _pdf_cmp.add_section_divider("Impact Comparison")
        _pdf_cmp.add_plotly_figure(fig_cmp, caption=f"Key Metrics: Before vs After {result.scenario_name}")
        _pdf_cmp.add_plotly_figure(fig_traj, caption="12-Month Savings Trajectory Comparison")
        _pdf_cmp.add_section_divider("Full Metrics Table")
        _pdf_cmp.add_dataframe_table(df_metrics)
        pdf_cmp_bytes = _pdf_cmp.build()
        pdf_cmp_fn = _pdf_cmp.suggested_filename("scenario_comparison_report")
    except Exception:
        pdf_cmp_bytes = None
        pdf_cmp_fn = "scenario_comparison_report.pdf"

    st.markdown("<div style='margin-top: 2.25rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        c_rep1, c_rep2 = st.columns([2.0, 1.2], vertical_alignment="center")
        with c_rep1:
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; gap: 12px; padding: 2px 0;">
                <div style="background: rgba(79, 140, 255, 0.15); border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px;">
                    <i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.15rem;"></i>
                </div>
                <div>
                    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 3px;">
                        Scenario Comparison Report
                    </div>
                    <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">
                        Download a comprehensive PDF summary of {result.scenario_name} impact on your savings and net worth.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_rep2:
            if pdf_cmp_bytes:
                st.download_button(
                    label="Download Scenario Report",
                    icon=":material/download:",
                    data=pdf_cmp_bytes,
                    file_name=pdf_cmp_fn,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="btn_download_scenario_report"
                )
            else:
                st.error("Report is currently unavailable.")
