"""
Streamlit Page: Explainable AI & Forecast Insights
Human-centered AI transparency showing how financial profile factors shape predictions.

Layout:
  Tab 1: Overview                   — 6M/12M forecast cards, plain-English synthesis, meaning & potential improvements
  Tab 2: What affects my forecast?  — Ranked factor impact bars, supporting vs limiting drivers
  Tab 3: Technical Details          — Full XGBoost + SHAP TreeExplainer waterfalls, summary plots, & feature tables
"""

import html
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.session import render_sidebar_user_selector
from utils.visualizer import PlotlyVisualizer
from models.explainability import ExplainableAI
from models.predictor import FinancialPredictor
from models.twin_engine import HealthScoreEngine
from utils.pdf_report import FinTwinPDFReport

st.set_page_config(
    page_title="Explainable AI",
    page_icon="assets/logos/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar selector ───────────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

st.markdown("""<div style="margin-bottom: 1.5rem;">
<h1 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.35rem; font-weight: 800; color: #F8FAFC; margin: 0 0 0.35rem 0; letter-spacing: -0.03em;">
    Explainable AI
</h1>
<p style="color: #94A3B8; font-size: 0.95rem; margin: 0; line-height: 1.5;">
    Transparent AI decomposition explaining the exact factors, weights, and SHAP values driving your forecasts.
</p>
</div>""", unsafe_allow_html=True)

if not twin:
    st.warning("Log in (or register) in the sidebar to load your personalized forecast explanation.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Cached Model & Explainability Computations (Preserving existing logic)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading pretrained XGBoost models for explanation…")
def get_predictor():
    predictor = FinancialPredictor()
    predictor.load_only(horizon=6)
    return predictor


@st.cache_data(show_spinner="Analyzing forecast drivers…", hash_funcs={object: lambda x: str(getattr(x, 'user_id', id(x)))})
def compute_forecast_explanation(_twin, horizon: int):
    predictor = get_predictor()
    return ExplainableAI().explain_forecast(_twin, predictor, horizon=horizon)


@st.cache_data(show_spinner="Computing health score explanation…", hash_funcs={object: lambda x: str(getattr(x, 'user_id', id(x)))})
def compute_health_explanation(_twin):
    return ExplainableAI().explain_health_score(_twin)


# ── Horizon Selector ───────────────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3, 1])
with h_col2:
    horizon = st.radio(
        "Forecast Horizon",
        [6, 12],
        horizontal=True,
        format_func=lambda x: f"{x} Months",
        label_visibility="collapsed",
    )

with h_col1:
    st.title(f"Why is my {horizon}-month forecast this way?")
    st.caption("Your forecast is based on your financial profile. Here are the main factors influencing the prediction.")

# Compute explanation results
fc_result = compute_forecast_explanation(twin, horizon)
hs_result = compute_health_explanation(twin)

if fc_result is None:
    st.warning("Forecast model is not yet trained. Navigate to the Forecasting page first to train the model, then return here.")
    st.stop()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Primary Navigation Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_factors, tab_tech = st.tabs([
    ":material/visibility: Overview",
    ":material/tune: What affects my forecast?",
    ":material/analytics: Technical Details",
])

# ── Helper: Feature name mapping & Impact Classification ───────────────────────
FEATURE_LABELS = {
    "monthly_income":      "Monthly Income",
    "monthly_expenses":    "Monthly Expenses",
    "monthly_savings":     "Monthly Savings",
    "monthly_investments": "Monthly SIP",
    "monthly_emi":         "Monthly EMI",
    "income":              "Monthly Income",
    "expenses":            "Monthly Expenses",
    "savings":             "Monthly Savings",
    "investments":         "Monthly SIP",
    "debt":                "Monthly EMI",
}

def get_feature_name(name: str) -> str:
    key = str(name).lower().strip()
    return FEATURE_LABELS.get(key, name.replace("_", " ").title())

def get_impact_tier(rel_pct: float) -> tuple[str, str, str]:
    """Returns (Tier Label, Badge BG, Badge Text Color)."""
    if rel_pct >= 55:
        return "High Impact", "rgba(245, 158, 11, 0.15)", "#F59E0B"
    elif rel_pct >= 25:
        return "Moderate Impact", "rgba(59, 130, 246, 0.15)", "#60A5FA"
    else:
        return "Low Impact", "rgba(107, 114, 128, 0.15)", "#9CA3AF"


# Extract savings & net worth contributions
sav_contribs = fc_result.savings_contributions
nw_contribs = fc_result.net_worth_contributions

# Combine contributions by feature to gauge overall impact
combined_dict = {}
for c in sav_contribs:
    fname = get_feature_name(c.display_name)
    combined_dict[fname] = {
        "feature": fname,
        "raw_value": c.value,
        "sav_shap": c.shap_value,
        "nw_shap": 0.0,
        "mean_abs_shap": abs(c.shap_value),
        "direction": c.direction,
    }

for c in nw_contribs:
    fname = get_feature_name(c.display_name)
    if fname in combined_dict:
        combined_dict[fname]["nw_shap"] = c.shap_value
        combined_dict[fname]["mean_abs_shap"] = (abs(combined_dict[fname]["sav_shap"]) + abs(c.shap_value)) / 2.0
    else:
        combined_dict[fname] = {
            "feature": fname,
            "raw_value": c.value,
            "sav_shap": 0.0,
            "nw_shap": c.shap_value,
            "mean_abs_shap": abs(c.shap_value),
            "direction": c.direction,
        }

ranked_factors = sorted(combined_dict.values(), key=lambda x: x["mean_abs_shap"], reverse=True)
max_shap_magnitude = max([f["mean_abs_shap"] for f in ranked_factors], default=1.0)
if max_shap_magnitude <= 0:
    max_shap_magnitude = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### Your forecast at a glance")

    # ── 2 Prediction Cards ─────────────────────────────────────────────────────
    card_col1, card_col2 = st.columns(2)
    with card_col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(15, 23, 42, 0.6));
                        border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px;
                        padding: 22px 24px; text-align: left;">
                <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    {horizon}-Month Projected Savings
                </div>
                <div style="font-size: 2.3rem; font-weight: 800; color: #10B981; margin: 6px 0 2px 0;">
                    ₹{fc_result.savings_prediction:,.0f}
                </div>
                <div style="color: #6B7280; font-size: 0.85rem;">
                    Estimated cumulative cash accumulated over {horizon} months
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(15, 23, 42, 0.6));
                        border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 12px;
                        padding: 22px 24px; text-align: left;">
                <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    {horizon}-Month Projected Net Worth
                </div>
                <div style="font-size: 2.3rem; font-weight: 800; color: #60A5FA; margin: 6px 0 2px 0;">
                    ₹{fc_result.net_worth_prediction:,.0f}
                </div>
                <div style="color: #6B7280; font-size: 0.85rem;">
                    Projected total assets minus liabilities after {horizon} months
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── Plain-English Explanation ──────────────────────────────────────────────
    top_3_names = [f["feature"] for f in ranked_factors[:3]]
    top_drivers_text = ", ".join(top_3_names[:-1]) + f" and {top_3_names[-1]}" if len(top_3_names) > 1 else top_3_names[0]

    st.info(
        f"Your {horizon}-month financial forecast is mainly influenced by your **{top_drivers_text}**. "
        "The model analyzes your current income, living costs, and debt obligations to project how your wealth will evolve."
    )

    st.markdown("---")

    # ── What this means for you & Potential improvements ───────────────────────
    info_col1, info_col2 = st.columns(2)

    def _dominant_sign(f):
        return f["sav_shap"] if abs(f["sav_shap"]) > abs(f["nw_shap"]) else f["nw_shap"]

    with info_col1:
        st.markdown("#### What this means for you")
        
        # Check supporting vs limiting drivers based on dominant SHAP impact
        pos_drivers = [f for f in ranked_factors if _dominant_sign(f) >= 0]
        neg_drivers = [f for f in ranked_factors if _dominant_sign(f) < 0]

        meaning_points = []
        if pos_drivers:
            top_pos_name = pos_drivers[0]["feature"]
            meaning_points.append(
                f"Your <strong>{top_pos_name}</strong> provides the strongest positive momentum, supporting your forward wealth trajectory."
            )
        if neg_drivers:
            top_neg_name = neg_drivers[0]["feature"]
            meaning_points.append(
                f"Your current <strong>{top_neg_name}</strong> represents the primary constraint limiting faster savings growth."
            )
        else:
            meaning_points.append(
                "Your financial ratios are well-calibrated, with steady cash flow supporting continuous compounding."
            )

        meaning_points.append(
            f"Over the next {horizon} months, maintaining regular surplus allocation will be essential to achieving or exceeding this baseline."
        )

        for pt in meaning_points:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.4); border-left: 3px solid #3B82F6;
                            border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.92rem; color: #E2E8F0;">
                    {pt}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with info_col2:
        st.markdown("#### Potential improvements")

        improvements = []
        
        if twin.sip_amount < (twin.total_income * 0.15):
            improvements.append(
                "<strong>Increase Monthly SIP:</strong> Directing an additional 5%–10% of monthly surplus into disciplined index or flexi-cap funds could significantly compound your projected net worth."
            )
        if twin.monthly_emi > (twin.total_income * 0.30):
            improvements.append(
                "<strong>Prepay High-Interest Debt:</strong> Prioritizing loan prepayment reduces long-term interest drag and directly improves monthly cash retention."
            )
        if twin.basic_expenses > (twin.total_income * 0.55):
            improvements.append(
                "<strong>Optimize Living Expenses:</strong> Trimming discretionary outflows could yield immediate incremental surplus to boost your monthly savings buffer."
            )
        if not improvements:
            improvements.append(
                "<strong>Step-up Strategy:</strong> Consider an annual 10% SIP step-up to accelerate your goal timelines alongside expected income increments."
            )

        for imp in improvements:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.05); border-left: 3px solid #10B981;
                            border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.92rem; color: #E2E8F0;">
                    {imp}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — What affects my forecast?
# ══════════════════════════════════════════════════════════════════════════════
with tab_factors:
    st.markdown("### What's affecting your forecast?")
    st.caption("Factors are ranked by their overall relative influence on your prediction.")

    for factor in ranked_factors:
        fname = factor["feature"]
        raw_val = factor["raw_value"]
        rel_pct = (factor["mean_abs_shap"] / max_shap_magnitude) * 100
        tier_label, badge_bg, badge_color = get_impact_tier(rel_pct)
        
        is_positive = (_dominant_sign(factor) >= 0)
        
        if is_positive:
            dir_label = "Supporting your forecast"
            dir_icon = '<i class="fa-solid fa-arrow-trend-up" style="color: #10B981; margin-right: 5px;"></i>'
            dir_color = "#10B981"
            dir_bg = "rgba(16, 185, 129, 0.12)"
            bar_color = "#10B981"
            explanation = f"Your current {fname.lower()} of ₹{raw_val:,.0f}/mo is providing positive upward momentum to your projection."
        else:
            dir_label = "Limiting your forecast"
            dir_icon = '<i class="fa-solid fa-arrow-trend-down" style="color: #EF4444; margin-right: 5px;"></i>'
            dir_color = "#EF4444"
            dir_bg = "rgba(239, 68, 68, 0.12)"
            bar_color = "#EF4444"
            explanation = f"Your current {fname.lower()} of ₹{raw_val:,.0f}/mo is creating a drag on your projected accumulation."

        bar_width = max(int(rel_pct), 8)

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid #334155;
                        border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">{fname}</span>
                        <span style="color: #94A3B8; font-size: 0.85rem; margin-left: 8px;">(₹{raw_val:,.0f}/mo)</span>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span style="background: {badge_bg}; color: {badge_color}; border-radius: 6px;
                                     padding: 3px 10px; font-size: 0.78rem; font-weight: 600;">
                            {tier_label}
                        </span>
                        <span style="background: {dir_bg}; color: {dir_color}; border-radius: 6px;
                                     padding: 3px 10px; font-size: 0.78rem; font-weight: 600; display: inline-flex; align-items: center;">
                            {dir_icon} {dir_label}
                        </span>
                    </div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 6px; height: 8px; width: 100%; overflow: hidden; margin-bottom: 8px;">
                    <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 6px;"></div>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem;">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Factor Category Breakdown (Supporting vs Limiting) ──────────────────────
    col_supp, col_limit = st.columns(2)

    with col_supp:
        st.markdown("#### Supporting your forecast")
        pos_list = [f for f in ranked_factors if _dominant_sign(f) >= 0]
        if pos_list:
            for pf in pos_list:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981;
                                border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #F3F4F6;">{pf['feature']}</span>
                        <div style="color: #94A3B8; font-size: 0.84rem; margin-top: 2px;">
                            Contributing positively to your forecast.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strongly positive factors detected.")

    with col_limit:
        st.markdown("#### Limiting your forecast")
        neg_list = [f for f in ranked_factors if _dominant_sign(f) < 0]
        if neg_list:
            for nf in neg_list:
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444;
                                border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #F3F4F6;">{nf['feature']}</span>
                        <div style="color: #94A3B8; font-size: 0.84rem; margin-top: 2px;">
                            Creating a drag on your forecast.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("All analyzed factors are currently supporting your forecast!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Technical Details (Preserving full academic & ML explainability)
# ══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    st.markdown("### Technical Explainability Details")
    st.caption("Comprehensive XGBoost regression decomposition and SHAP TreeExplainer metrics for academic evaluation.")

    # Model specifications
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Model Architecture", "XGBoost Regressor")
    mcol2.metric("XAI Algorithm", "SHAP TreeExplainer")
    mcol3.metric("Savings Base Value", f"₹{fc_result.savings_base_value:,.0f}")
    mcol4.metric("Net Worth Base Value", f"₹{fc_result.net_worth_base_value:,.0f}")

    st.markdown("---")

    # ── Section 1: Forecast Force Plots ────────────────────────────────────────
    st.markdown("#### SHAP Force Plots (Waterfall Decomposition)")
    st.caption("Visualizes the step-by-step contribution of each feature from the neutral base value to the final prediction.")

    sav_force = {
        "baseline":   fc_result.savings_base_value,
        "prediction": fc_result.savings_prediction,
        "all_sorted": [
            type("C", (), {
                "component":   c.display_name,
                "contribution": c.shap_value,
            })()
            for c in fc_result.savings_contributions
        ],
    }

    nw_force = {
        "baseline":   fc_result.net_worth_base_value,
        "prediction": fc_result.net_worth_prediction,
        "all_sorted": [
            type("C", (), {
                "component":   c.display_name,
                "contribution": c.shap_value,
            })()
            for c in fc_result.net_worth_contributions
        ],
    }

    fp_col1, fp_col2 = st.columns(2)
    with fp_col1:
        st.markdown(f"**Savings Model ({horizon}M)**")
        fig_sav_force = PlotlyVisualizer.plot_shap_force_plot(
            sav_force,
            title   = f"Savings SHAP Waterfall ({horizon}M)",
            unit    = "₹",
            x_label = "Monthly Savings (₹)",
        )
        st.plotly_chart(fig_sav_force, use_container_width=True)

    with fp_col2:
        st.markdown(f"**Net Worth Model ({horizon}M)**")
        fig_nw_force = PlotlyVisualizer.plot_shap_force_plot(
            nw_force,
            title   = f"Net Worth SHAP Waterfall ({horizon}M)",
            unit    = "₹",
            x_label = "Net Worth (₹)",
        )
        st.plotly_chart(fig_nw_force, use_container_width=True)

    st.markdown("---")

    # ── Section 2: Summary Plots ───────────────────────────────────────────────
    st.markdown("#### SHAP Summary & Feature Importance Plots")
    
    sp_col1, sp_col2 = st.columns(2)
    with sp_col1:
        st.markdown("**Savings Feature Contributions**")
        sav_df = pd.DataFrame([{
            "Feature":     c.display_name,
            "SHAP Value":  c.shap_value,
        } for c in fc_result.savings_contributions])

        fig_sav_bar = PlotlyVisualizer.plot_shap_summary_bar(
            summary_df = sav_df,
            value_col  = "SHAP Value",
            label_col  = "Feature",
            title      = f"Savings SHAP Values ({horizon}M)",
            unit       = "₹",
        )
        st.plotly_chart(fig_sav_bar, use_container_width=True)

    with sp_col2:
        st.markdown("**Net Worth Feature Contributions**")
        nw_df = pd.DataFrame([{
            "Feature":     c.display_name,
            "SHAP Value":  c.shap_value,
        } for c in fc_result.net_worth_contributions])

        fig_nw_bar = PlotlyVisualizer.plot_shap_summary_bar(
            summary_df = nw_df,
            value_col  = "SHAP Value",
            label_col  = "Feature",
            title      = f"Net Worth SHAP Values ({horizon}M)",
            unit       = "₹",
        )
        st.plotly_chart(fig_nw_bar, use_container_width=True)

    st.markdown("**Combined Feature Importance (Mean |SHAP| across models)**")
    fig_combined = PlotlyVisualizer.plot_shap_summary_bar(
        summary_df = fc_result.combined_importance.assign(
            **{"SHAP Value": fc_result.combined_importance["Mean |SHAP|"]}
        ),
        value_col  = "SHAP Value",
        label_col  = "Feature",
        title      = "Combined Model Feature Importance",
        unit       = "₹",
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    st.markdown("---")

    # ── Section 3: Detailed Tables ─────────────────────────────────────────────
    st.markdown("#### Full SHAP Value Data Tables")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(f"**Savings Model Data ({horizon}M)**")
        sav_table = pd.DataFrame([{
            "Feature":        c.display_name,
            "Your Value (₹)": f"₹{c.value:,.0f}",
            "SHAP Value":     f"{'+' if c.shap_value >= 0 else ''}{c.shap_value:,.0f} ₹",
            "Direction":      "Positive" if c.direction == "positive" else "Negative",
        } for c in fc_result.savings_contributions])
        st.dataframe(sav_table, use_container_width=True, hide_index=True)

    with t_col2:
        st.markdown(f"**Net Worth Model Data ({horizon}M)**")
        nw_table = pd.DataFrame([{
            "Feature":        c.display_name,
            "Your Value (₹)": f"₹{c.value:,.0f}",
            "SHAP Value":     f"{'+' if c.shap_value >= 0 else ''}{c.shap_value:,.0f} ₹",
            "Direction":      "Positive" if c.direction == "positive" else "Negative",
        } for c in fc_result.net_worth_contributions])
        st.dataframe(nw_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 4: Health Score Explainability ─────────────────────────────────
    with st.expander("Financial Health Score Explainability (Decomposition from 50.0 Baseline)"):
        h_top_pos = hs_result.top_positive
        h_top_neg = hs_result.top_negative

        hkc1, hkc2, hkc3, hkc4 = st.columns(4)
        hkc1.metric("Final Score", f"{hs_result.overall_score:.1f} / 100", delta=f"{hs_result.overall_score - hs_result.baseline_score:.1f} vs baseline")
        hkc2.metric("Baseline Score", f"{hs_result.baseline_score:.1f} / 100")
        hkc3.metric("Biggest Booster", h_top_pos[0].component if h_top_pos else "None", delta=f"{h_top_pos[0].contribution:.1f} pts" if h_top_pos else None)
        hkc4.metric("Biggest Drag", h_top_neg[0].component if h_top_neg else "None", delta=f"{h_top_neg[0].contribution:.1f} pts" if h_top_neg else None)

        fig_hs_force = PlotlyVisualizer.plot_shap_force_plot(
            hs_result.force_plot_data,
            title   = "Health Score Waterfall — Baseline (50 pts) → Final Score",
            unit    = "pts",
            x_label = "Health Score (pts)",
        )
        st.plotly_chart(fig_hs_force, use_container_width=True)

        hs_display_df = hs_result.summary_data[[
            "Feature", "Sub-Score", "SHAP Value", "Weight (%)", "Benchmark",
        ]].copy()
        hs_display_df["SHAP Value"] = hs_display_df["SHAP Value"].apply(
            lambda v: f"+{v:.2f} pts" if v >= 0 else f"{v:.2f} pts"
        )
        hs_display_df["Sub-Score"] = hs_display_df["Sub-Score"].apply(lambda v: f"{v:.1f} / 100")
        st.dataframe(hs_display_df, use_container_width=True, hide_index=True)

    # ── Section 5: Export Technical PDF ────────────────────────────────────────
    st.markdown("<div style='margin-top: 2.25rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 1.5rem;'></div>", unsafe_allow_html=True)

    try:
        _pdf_xai = FinTwinPDFReport(
            report_title="Explainable AI Technical Report",
            user_id=twin.user_id,
            report_id=f"XAI-{twin.user_id}",
            subtitle=f"{horizon}-Month Forecast & Health Score Decomposition",
        )
        _pdf_xai.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Explainable AI Technical Report",
        )
        _pdf_xai.add_kpi_summary_row([
            {"label": "Health Score (Final)", "value": f"{fc_result.savings_prediction if hasattr(fc_result, 'savings_prediction') else '---'}", "status": "neutral"},
            {"label": "Predicted Savings", "value": f"Rs. {fc_result.savings_prediction:,.0f}" if hasattr(fc_result, 'savings_prediction') else "---", "status": "positive"},
            {"label": "Predicted Net Worth", "value": f"Rs. {fc_result.net_worth_prediction:,.0f}" if hasattr(fc_result, 'net_worth_prediction') else "---", "status": "positive"},
            {"label": "Health Score (Final)", "value": f"{hs_result.overall_score:.1f}" if hasattr(hs_result, 'overall_score') else "---", "status": "neutral"},
        ])
        _pdf_xai.add_callout(
            "This report explains how the AI model calculates your financial health score and "
            "financial projections using SHAP-based feature attribution. Feature contributions "
            "show which factors had the most positive or negative influence on predictions.",
            style="info", title="About This Report"
        )

        _pdf_xai.add_section_divider("Forecast & Explainability Summary")
        _pdf_xai.add_key_value_grid({
            "Forecast Horizon": f"{horizon} Months",
            "Predicted Savings": fc_result.savings_prediction,
            "Predicted Net Worth": fc_result.net_worth_prediction,
            "Savings Base Value": fc_result.savings_base_value,
            "Net Worth Base Value": fc_result.net_worth_base_value,
            "Health Score (Final)": hs_result.overall_score,
            "Health Score (Baseline)": hs_result.baseline_score,
        })

        _pdf_xai.add_section_divider("Forecast Model Contributions")
        for line in (fc_result.savings_narrative + fc_result.net_worth_narrative):
            _pdf_xai.add_paragraph(line)
        _pdf_xai.add_dataframe_table(fc_result.combined_importance)

        _pdf_xai.add_section_divider("Health Score Component Breakdown")
        for line in hs_result.narrative:
            _pdf_xai.add_paragraph(line)
        contrib_df = pd.DataFrame([
            {"Component": c.component, "Raw Value": c.raw_value, "Score": c.score,
             "Weight": c.weight, "Contribution (pts)": c.contribution}
            for c in hs_result.contributions
        ])
        _pdf_xai.add_dataframe_table(contrib_df)

        pdf_xai_bytes = _pdf_xai.build()
        pdf_xai_fn = _pdf_xai.suggested_filename(f"xai_{horizon}m_report")
    except Exception as e:
        pdf_xai_bytes = None
        pdf_xai_fn = f"xai_{horizon}m_report.pdf"

    c_rep1, c_rep2 = st.columns([1.7, 1.3], vertical_alignment="center")
    with c_rep1:
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.25rem;"></i>
<h3 style="margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">Your Explainable AI Report</h3>
</div>
<p style="color: #94A3B8; font-size: 0.86rem; margin: 0; line-height: 1.45;">
Download a comprehensive PDF summary of SHAP feature attributions, model narratives, and sub-score decompositions.
</p>""", unsafe_allow_html=True)

    with c_rep2:
        if pdf_xai_bytes:
            st.download_button(
                label="Download Explainability Report",
                icon=":material/download:",
                data=pdf_xai_bytes,
                file_name=pdf_xai_fn,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="btn_download_xai_report"
            )
        else:
            st.error("Report is currently unavailable.")
