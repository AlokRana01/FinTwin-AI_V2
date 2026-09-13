"""
Streamlit Page: AI Behavioral Coach  (Module 5)
Financial Personality Detection using K-Means clustering.

Detects one of 5 archetypes:
  Saver | Investor | Spender | Debt Heavy | Balanced Planner

Features used:
  savings_ratio, expense_ratio, investment_ratio, debt_to_income_ratio

Tabs:
  1. Personality Profile — radar chart + nudges
  2. Cluster Visualization — population scatter + stats table
  3. Financial Twin Coach — personality-aware chat
"""

import streamlit as st
from utils.session import render_sidebar_user_selector
from utils.visualizer import PlotlyVisualizer
from utils.pdf_report import FinTwinPDFReport
from models.clustering import FinancialPersonalityClusterer, PERSONALITY_META

st.set_page_config(page_title="Financial Personality", page_icon="assets/logos/favicon.svg", layout="wide", initial_sidebar_state="collapsed")

# ── Sidebar ────────────────────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

st.markdown("""<div style="margin-bottom: 1.5rem;">
<h1 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.35rem; font-weight: 800; color: #F8FAFC; margin: 0 0 0.35rem 0; letter-spacing: -0.03em;">
    Financial Personality
</h1>
<p style="color: #94A3B8; font-size: 0.95rem; margin: 0; line-height: 1.5;">
    Understand your financial mind, behavioral archetype, and AI-tailored decision nudges.
</p>
</div>""", unsafe_allow_html=True)

if not twin:
    st.warning("Log in (or register) in the sidebar to begin personality analysis.")
    st.stop()

# ── Color map for all 5 archetypes ────────────────────────────────────────────
COLOR_MAP = {meta_key: meta["color"] for meta_key, meta in PERSONALITY_META.items()}

# ── Load pretrained clusterer (trained OFFLINE by training/train_offline.py) ──
# The live app never re-fits KMeans on database rows — it only loads the
# saved centroids/scaler artifact and predicts the current user's cluster.
@st.cache_resource(show_spinner="Loading personality model...")
def load_clusterer():
    return FinancialPersonalityClusterer.load()

clusterer = load_clusterer()

if not clusterer.is_fitted():
    st.error(
        "The personality model hasn't been trained yet. Run "
        "`python training/train_offline.py` once to generate "
        "`data/models/personality_cluster_model.json`, then reload this page."
    )
    st.stop()

# ── Compute current user's personality ────────────────────────────────────────
user_features   = clusterer.get_feature_vector(twin)
personality     = clusterer.predict_personality(twin)
meta            = PERSONALITY_META.get(personality, PERSONALITY_META["Balanced Planner"])

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "Personality Profile",
    "Cluster Visualization"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Personality Profile
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    p1_col1, p1_col2 = st.columns([1, 1.1])

    with p1_col1:
        # Personality card
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{meta['color']}22,#0f172a);
                        border:1.5px solid {meta['color']}55;border-radius:16px;
                        padding:28px 24px;margin-bottom:12px;text-align:center;">
                <div style="font-size:3.2rem;margin-bottom:8px">{meta['icon']}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{meta['color']}">
                    {personality}
                </div>
                <div style="font-size:0.9rem;color:#94A3B8;margin-top:4px;font-style:italic">
                    {meta['tagline']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Raw feature metrics
        st.markdown("**Your Financial Ratios:**")
        m1, m2 = st.columns(2)
        m1.metric("Savings Ratio",    f"{user_features['savings_ratio']:.1%}")
        m2.metric("Expense Ratio",    f"{user_features['expense_ratio']:.1%}")
        m3, m4 = st.columns(2)
        m3.metric("Investment Ratio", f"{user_features['investment_ratio']:.1%}")
        m4.metric("Debt Ratio",       f"{user_features['debt_to_income_ratio']:.1%}")

        # Personality description
        st.markdown("---")
        st.markdown(f"**About this Personality:**")
        st.info(meta["description"])

    with p1_col2:
        # Radar chart
        fig_radar = PlotlyVisualizer.plot_personality_radar(
            user_features, personality, color=meta["color"]
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Behavioural nudges
        st.markdown("**Personalised Coach Nudges:**")
        for nudge in meta["nudges"]:
            st.success(nudge)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Cluster Visualization
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Personality Cluster Benchmarks")
    st.caption(
        f"Cluster centroids trained once, offline, on anonymized aggregate data "
        f"(see `training/train_offline.py`). Your position = **{personality}**."
    )

    stats_df = clusterer.load_saved_cluster_benchmarks()

    if stats_df.empty:
        st.warning(
            "Cluster benchmark statistics aren't available yet. Run "
            "`python training/train_offline.py` once to generate them."
        )
    else:
        def _parse_val(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.replace("%", "").strip()) / 100.0
                except (ValueError, TypeError):
                    return 0.0
            return 0.0

        def _get_cluster_val(row, *col_names):
            for c in col_names:
                if c in row.columns:
                    return _parse_val(row[c].values[0])
            return 0.0

        your_row = stats_df[stats_df["Personality"] == personality]
        if not your_row.empty:
            cluster_means = {
                "Savings Ratio":    _get_cluster_val(your_row, "Savings Ratio", "savings_ratio"),
                "Expense Ratio":    _get_cluster_val(your_row, "Expense Ratio", "expense_ratio"),
                "Investment Ratio": _get_cluster_val(your_row, "Investment Ratio", "investment_ratio"),
                "Debt Ratio":       _get_cluster_val(your_row, "Debt Ratio", "debt_to_income_ratio"),
            }
            user_vals = {
                "Savings Ratio":    user_features["savings_ratio"],
                "Expense Ratio":    user_features["expense_ratio"],
                "Investment Ratio": user_features["investment_ratio"],
                "Debt Ratio":       user_features["debt_to_income_ratio"],
            }
            fig_bar = PlotlyVisualizer.plot_cluster_comparison_bar(
                user_vals, cluster_means, personality, color=meta["color"]
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Full benchmark table (static cluster means)
        st.markdown("**Archetype Reference Table:**")
        display_df = stats_df.copy()
        # If columns are lowercased/raw floats, format them
        if "savings_ratio" in display_df.columns:
            display_df["savings_ratio"] = display_df["savings_ratio"].apply(lambda v: f"{_parse_val(v):.1%}")
            display_df["expense_ratio"] = display_df["expense_ratio"].apply(lambda v: f"{_parse_val(v):.1%}")
            display_df["investment_ratio"] = display_df["investment_ratio"].apply(lambda v: f"{_parse_val(v):.1%}")
            display_df["debt_to_income_ratio"] = display_df["debt_to_income_ratio"].apply(lambda v: f"{_parse_val(v):.1%}")
            display_df = display_df.rename(columns={
                "personality": "Personality Archetype",
                "Personality": "Personality Archetype",
                "savings_ratio": "Avg Savings Ratio",
                "expense_ratio": "Avg Expense Ratio",
                "investment_ratio": "Avg Investment Ratio",
                "debt_to_income_ratio": "Avg Debt Ratio",
            })
        else:
            display_df = display_df.rename(columns={
                "Personality": "Personality Archetype",
                "Count": "Sample Size",
                "Savings Ratio": "Avg Savings Ratio",
                "Expense Ratio": "Avg Expense Ratio",
                "Investment Ratio": "Avg Investment Ratio",
                "Debt Ratio": "Avg Debt Ratio",
            })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Personality archetype cards row
        st.markdown("**All 5 Personality Archetypes:**")
        card_cols = st.columns(5)
        for i, (label, pmeta) in enumerate(PERSONALITY_META.items()):
            with card_cols[i]:
                st.markdown(
                    f"""
                    <div style="border:1px solid {pmeta['color']}44;border-radius:10px;
                                padding:10px;text-align:center;background:{pmeta['color']}11;">
                        <div style="font-size:1.5rem">{pmeta['icon']}</div>
                        <div style="font-size:0.78rem;font-weight:700;color:{pmeta['color']};
                                    margin-top:4px">{label}</div>
                        <div style="font-size:0.7rem;color:#94A3B8;margin-top:3px">
                            {pmeta['tagline']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<div style='margin-top: 2.25rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 1.5rem;'></div>", unsafe_allow_html=True)

    try:
        _pdf_pers = FinTwinPDFReport(
            report_title="Financial Personality Profile Report",
            user_id=twin.user_id,
            report_id=f"FP-{twin.user_id}",
            subtitle=f"Detected Archetype: {personality} — Behavioural ratios and coaching insights",
        )
        _pdf_pers.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Financial Personality Profile Report",
        )
        _pdf_pers.add_kpi_summary_row([
            {"label": "Personality Archetype", "value": str(personality), "status": "neutral"},
            {"label": "Savings Ratio", "value": f"{user_features['savings_ratio']:.1%}",
             "status": "positive" if user_features['savings_ratio'] >= 0.2 else "negative"},
            {"label": "Investment Ratio", "value": f"{user_features['investment_ratio']:.1%}", "status": "neutral"},
            {"label": "Debt Ratio", "value": f"{user_features['debt_to_income_ratio']:.1%}",
             "status": "positive" if user_features['debt_to_income_ratio'] < 0.3 else "negative"},
        ])
        _pdf_pers.add_callout(
            f"Your financial personality archetype is: {personality}. " + (meta.get("description", "") or ""),
            style="success", title="Personality Archetype"
        )

        _pdf_pers.add_section_divider("Financial Ratios")
        _pdf_pers.add_key_value_grid({
            "Savings Ratio": f"{user_features['savings_ratio']:.1%}",
            "Expense Ratio": f"{user_features['expense_ratio']:.1%}",
            "Investment Ratio": f"{user_features['investment_ratio']:.1%}",
            "Debt Ratio": f"{user_features['debt_to_income_ratio']:.1%}",
        })

        try:
            _pdf_pers.add_plotly_figure(fig_radar, caption="Personality Ratio Radar")
        except NameError:
            pass

        if meta.get("nudges"):
            _pdf_pers.add_section_divider("AI Coaching Recommendations")
            for nudge in meta["nudges"][:3]:
                _pdf_pers.add_callout(nudge, style="info")

        try:
            _pdf_pers.add_section_divider("Cluster Statistics")
            _pdf_pers.add_dataframe_table(stats_df)
        except NameError:
            pass

        pdf_pers_bytes = _pdf_pers.build()
        pdf_fn = _pdf_pers.suggested_filename("financial_personality_report")
    except Exception as e:
        pdf_pers_bytes = None
        pdf_fn = "financial_personality_report.pdf"

    c_rep1, c_rep2 = st.columns([1.7, 1.3], vertical_alignment="center")
    with c_rep1:
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.25rem;"></i>
<h3 style="margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">Your Financial Personality Report</h3>
</div>
<p style="color: #94A3B8; font-size: 0.86rem; margin: 0; line-height: 1.45;">
Download a detailed summary of your detected personality archetype, behavioral ratios, and coaching recommendations.
</p>""", unsafe_allow_html=True)

    with c_rep2:
        if pdf_pers_bytes:
            st.download_button(
                label="Download Personality Report",
                icon=":material/download:",
                data=pdf_pers_bytes,
                file_name=pdf_fn,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="btn_download_personality_report"
            )
        else:
            st.error("Report is currently unavailable.")
