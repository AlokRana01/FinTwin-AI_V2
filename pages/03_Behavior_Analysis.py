"""
Streamlit Page: Behavior Analysis & Cash Outflows
Profiles spending habits, identifies lifestyle leaks, benchmarks allocations against peer cohorts, 
and provides interactive simulation for expense tuning.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.session import render_sidebar_user_selector
from utils.visualizer import PlotlyVisualizer
from utils.pdf_report import FinTwinPDFReport
from database.db_manager import DBManager

st.set_page_config(page_title="Behavior Analysis", page_icon="assets/logos/favicon.svg", layout="wide", initial_sidebar_state="collapsed")

# ── Sidebar Selector ──────────────────────────────────────────────────────────
twin = render_sidebar_user_selector()

st.markdown("""<div style="margin-bottom: 1.5rem;">
<h1 style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.35rem; font-weight: 800; color: #F8FAFC; margin: 0 0 0.35rem 0; letter-spacing: -0.03em;">
    Behavior Analysis
</h1>
<p style="color: #94A3B8; font-size: 0.95rem; margin: 0; line-height: 1.5;">
    Profile cash outflows, identify lifestyle leaks, and benchmark your spending against peer cohorts.
</p>
</div>""", unsafe_allow_html=True)

if not twin:
    st.warning("Log in (or register) in the sidebar to begin behavior analysis.")
    st.stop()

# ── Calculations ──────────────────────────────────────────────────────────────
needs = twin.rent + twin.groceries + twin.utilities + twin.transport
wants = twin.food_delivery + twin.entertainment + twin.shopping
debt_service = twin.monthly_emi
investments = twin.sip_amount
surplus = max(twin.total_income - (needs + wants + debt_service), 0.0)

needs_pct = needs / twin.total_income if twin.total_income > 0 else 0.0
wants_pct = wants / twin.total_income if twin.total_income > 0 else 0.0
debt_pct = debt_service / twin.total_income if twin.total_income > 0 else 0.0
invest_pct = investments / twin.total_income if twin.total_income > 0 else 0.0
surplus_pct = surplus / twin.total_income if twin.total_income > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    ":material/query_stats: Outflow Allocation",
    ":material/group: Peer Comparison",
    ":material/psychology: Behavioral Insights & Report"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Outflow Allocation
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1f2937,#111827);
                    border:1px solid #374151;border-radius:12px;padding:18px 24px;
                    margin-bottom:20px;">
            <span style="color:#9CA3AF;font-size:0.85rem;text-transform:uppercase;font-weight:700">Active Profile</span>
            <h3 style="margin:0 0 4px 0;color:#F9FAFB">{twin.name}</h3>
            <span style="color:#6B7280;font-size:0.88rem">{twin.occupation} &middot; Age {twin.age} &middot; {twin.city}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Columns of Outflows
    col_n, col_w, col_d, col_i = st.columns(4)
    col_n.metric("Essential Needs", f"₹{needs:,.0f}", f"{needs_pct:.1%}")
    col_w.metric("Lifestyle Wants", f"₹{wants:,.0f}", f"{wants_pct:.1%}")
    col_d.metric("Debt EMI", f"₹{debt_service:,.0f}", f"{debt_pct:.1%}")
    col_i.metric("Investments & Surplus", f"₹{(investments + surplus):,.0f}", f"{(invest_pct + surplus_pct):.1%}")

    st.markdown("---")

    ch_col1, ch_col2 = st.columns([1.1, 1])

    with ch_col1:
        st.markdown("#### Outflow Distribution Donut")
        
        # Donut Chart values
        labels = ['Needs', 'Wants', 'Debt Service', 'SIP Investments', 'Cash Surplus']
        values = [needs, wants, debt_service, investments, surplus]
        colors = ['#3B82F6', '#EF4444', '#F59E0B', '#10B981', '#6EE7B7']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker_colors=colors,
            textinfo='percent+label',
            textfont=dict(color='#F3F4F6', size=11),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F3F4F6", 'family': "Outfit, Inter, sans-serif"},
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch_col2:
        st.markdown("#### Discretionary Expense Simulator")
        st.caption("Trim lifestyle outflows to see savings increase in real-time.")
        
        sim_food = st.slider("Trim Food Delivery (₹)", 0, max(int(twin.food_delivery), 500), int(twin.food_delivery), step=500)
        sim_ent = st.slider("Trim Entertainment (₹)", 0, max(int(twin.entertainment), 500), int(twin.entertainment), step=500)
        sim_shop = st.slider("Trim Shopping (₹)", 0, max(int(twin.shopping), 500), int(twin.shopping), step=500)
        
        saved_food = twin.food_delivery - sim_food
        saved_ent = twin.entertainment - sim_ent
        saved_shop = twin.shopping - sim_shop
        total_monthly_saving = saved_food + saved_ent + saved_shop
        total_annual_saving = total_monthly_saving * 12
        
        st.markdown(
            f"""
            <div style="background:#1e293b;border-radius:10px;padding:16px;border:1px solid #3b82f6;">
                <h4 style="margin:0 0 8px 0;color:#60A5FA;">Simulation Results</h4>
                <p style="margin:4px 0;font-size:0.95rem;color:#E2E8F0;">
                    Monthly Cash Freed: <b>₹{total_monthly_saving:,.0f}</b>
                </p>
                <p style="margin:4px 0;font-size:0.95rem;color:#E2E8F0;">
                    Annual Extra Savings: <b>₹{total_annual_saving:,.0f}</b>
                </p>
                <p style="margin:8px 0 0 0;font-size:0.85rem;color:#94A3B8;font-style:italic;">
                    That extra amount compounded at 12% annually for 10 years would grow to <b>₹{total_monthly_saving * 232.33:,.0f}</b>!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Peer Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Peer Group Cohort Comparison")
    st.write(f"Compare your outflows with peers working as **{twin.occupation}**.")
    
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
        st.plotly_chart(fig_peer, use_container_width=True)
        
        st.markdown("#### Detailed Variance Table")
        comparison_rows = []
        for cat in user_spend.keys():
            u_val = user_spend[cat]
            p_val = peer_spend.get(cat, 0.0)
            diff = u_val - p_val
            pct_diff = (diff / p_val * 100) if p_val > 0 else 0.0
            
            status = "Overspending" if diff > 500 else ("Under Cohort" if diff < -500 else "Similar")
            comparison_rows.append({
                "Category": cat,
                "Your Spend": u_val,
                "Peer Average": p_val,
                "Difference": diff,
                "Variance (%)": f"{pct_diff:+.1f}%",
                "Status": status
            })
            
        df_comp = pd.DataFrame(comparison_rows)
        st.dataframe(
            df_comp.style.format({
                "Your Spend": "₹{:,.0f}",
                "Peer Average": "₹{:,.0f}",
                "Difference": "₹{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Cohort data is loading. Ensure database is fully populated.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Behavioral Insights & Report
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Behavioral Outflow Insights")
    
    # Simple rule-based alerts
    alerts_triggered = False
    
    if wants_pct > 0.30:
        st.warning(f"**High Lifestyle Expenses:** Lifestyle spending (wants) consumes **{wants_pct:.1%}** of your income (Limit: 30%). Try trimming discretionary shopping or subscriptions.")
        alerts_triggered = True
        
    if debt_pct > 0.35:
        st.error(f"**High Debt Burden:** EMIs eat up **{debt_pct:.1%}** of your monthly cash flow. Refrain from taking new loans and consider prepayment.")
        alerts_triggered = True
        
    if twin.total_income > 0 and (investments + surplus) / twin.total_income < 0.20:
        st.warning(f"**Savings Cushion Deficit:** Your savings & investment rate is **{((investments + surplus) / twin.total_income):.1%}** (Target: 20%+). Pay yourself first by scheduling SIPs immediately after salary credits.")
        alerts_triggered = True
        
    if twin.total_income > 0 and twin.food_delivery > (twin.total_income * 0.08):
        st.warning(f"**Food Ordering Leak:** Monthly food delivery expenditures (₹{twin.food_delivery:,.0f}) exceed 8% of your income. Preparing home-cooked meals could yield quick surpluses.")
        alerts_triggered = True
        
    if not alerts_triggered:
        st.success("**Balanced Allocations:** Your outflow patterns look healthy and within target bounds! Keep maintaining this structure.")

    st.markdown("---")
    st.markdown("#### Export Outflows Report")
    st.caption("Download your category expenditures and demographic summaries as a professional PDF report.")
    
    # Safe income percent formatter
    def _pct(val: float) -> str:
        return f"{(val / twin.total_income):.1%}" if twin.total_income > 0 else "0.0%"

    # Prepare export dataframe (unchanged data, reused as-is)
    export_rows = [
        {"Category": "Total Income", "Amount (₹)": twin.total_income, "Type": "Inflow", "Pct of Income": "100.0%"},
        {"Category": "Rent", "Amount (₹)": twin.rent, "Type": "Need", "Pct of Income": _pct(twin.rent)},
        {"Category": "Groceries", "Amount (₹)": twin.groceries, "Type": "Need", "Pct of Income": _pct(twin.groceries)},
        {"Category": "Utilities", "Amount (₹)": twin.utilities, "Type": "Need", "Pct of Income": _pct(twin.utilities)},
        {"Category": "Transport", "Amount (₹)": twin.transport, "Type": "Need", "Pct of Income": _pct(twin.transport)},
        {"Category": "Food Delivery", "Amount (₹)": twin.food_delivery, "Type": "Want", "Pct of Income": _pct(twin.food_delivery)},
        {"Category": "Entertainment", "Amount (₹)": twin.entertainment, "Type": "Want", "Pct of Income": _pct(twin.entertainment)},
        {"Category": "Shopping", "Amount (₹)": twin.shopping, "Type": "Want", "Pct of Income": _pct(twin.shopping)},
        {"Category": "EMIs", "Amount (₹)": twin.monthly_emi, "Type": "Debt Service", "Pct of Income": _pct(twin.monthly_emi)},
        {"Category": "SIP Investments", "Amount (₹)": twin.sip_amount, "Type": "Investment", "Pct of Income": _pct(twin.sip_amount)},
        {"Category": "Cash Surplus", "Amount (₹)": surplus, "Type": "Savings Buffer", "Pct of Income": _pct(surplus)},
    ]
    df_export = pd.DataFrame(export_rows)

    st.markdown("<div style='margin-top: 2.25rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 1.5rem;'></div>", unsafe_allow_html=True)

    try:
        _pdf_report = FinTwinPDFReport(
            report_title="Spending Behaviour Analysis Report",
            user_id=twin.user_id,
            report_id=f"BA-{twin.user_id}",
            subtitle="Category outflows, needs vs. wants analysis, and peer cohort benchmarking",
        )
        _pdf_report.add_cover_page(
            user_name=getattr(twin, "name", "") or "",
            report_type="Spending Behaviour Analysis Report",
        )
        total_spend = twin.rent + twin.groceries + twin.utilities + twin.transport + twin.food_delivery + twin.entertainment + twin.shopping
        needs_total = twin.rent + twin.groceries + twin.utilities + twin.transport
        wants_total = twin.food_delivery + twin.entertainment + twin.shopping
        income = getattr(twin, "total_income", 1) or 1
        _pdf_report.add_kpi_summary_row([
            {"label": "Total Monthly Spend", "value": f"Rs. {total_spend:,.0f}", "status": "neutral"},
            {"label": "Needs", "value": f"Rs. {needs_total:,.0f} ({needs_total/income:.0%})",
             "status": "positive" if needs_total / income < 0.5 else "warning"},
            {"label": "Wants", "value": f"Rs. {wants_total:,.0f} ({wants_total/income:.0%})",
             "status": "positive" if wants_total / income < 0.3 else "negative"},
            {"label": "Surplus", "value": f"Rs. {surplus:,.0f}",
             "status": "positive" if surplus > 0 else "negative"},
        ])

        _pdf_report.add_section_divider("Expense Breakdown")
        _pdf_report.add_dataframe_table(df_export)

        try:
            _pdf_report.add_section_divider("Needs vs. Wants Allocation")
            _pdf_report.add_plotly_figure(fig, caption="Needs vs. Wants Allocation")
        except NameError:
            pass

        try:
            _pdf_report.add_section_divider("Peer Cohort Comparison")
            _pdf_report.add_plotly_figure(fig_peer, caption="Your Spending vs. Peer Average")
        except NameError:
            pass

        pdf_bytes = _pdf_report.build()
        pdf_fn = _pdf_report.suggested_filename("spending_behavior_report")
    except Exception as e:
        pdf_bytes = None
        pdf_fn = "spending_behavior_report.pdf"

    c_rep1, c_rep2 = st.columns([1.7, 1.3], vertical_alignment="center")
    with c_rep1:
        st.markdown("""<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<i class="fa-solid fa-file-pdf" style="color: #4F8CFF; font-size: 1.25rem;"></i>
<h3 style="margin: 0; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">Your Spending Behavior Report</h3>
</div>
<p style="color: #94A3B8; font-size: 0.86rem; margin: 0; line-height: 1.45;">
Download a detailed summary of your spending behavior, needs vs. wants breakdown, and peer comparison.
</p>""", unsafe_allow_html=True)

    with c_rep2:
        if pdf_bytes:
            st.download_button(
                label="Download Spending Behavior Report",
                icon=":material/download:",
                data=pdf_bytes,
                file_name=pdf_fn,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="btn_download_behavior_report"
            )
        else:
            st.error("Report is currently unavailable.")
