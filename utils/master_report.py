"""
utils/master_report.py
========================
Builds ONE consolidated, professional PDF report spanning every module of
FinTwin AI (User Profile, Financial Health Score, Income/Expense/Savings/
Investment Analysis, Net Worth, Financial Risk Assessment, Tax Intelligence,
AI Coach Recommendations, Financial Personality, Behaviour Analysis,
Forecasting, and Explainable AI).

Design principles (matching the per-page PDF exports):
  * Every number is produced by calling the SAME engines the Streamlit pages
    already use (HealthScoreEngine, IndianTaxCalculator, FinancialCoach,
    FinancialPersonalityClusterer, FinancialPredictor, ExplainableAI). No
    financial calculation is duplicated or re-derived here.
  * Each section is wrapped in its own try/except so that a missing model
    artifact (e.g. clusterer not yet trained) or an edge-case input simply
    skips that ONE section rather than failing the whole report -- matching
    the "only include sections that contain data" requirement.
"""

from __future__ import annotations

import pandas as pd
import uuid

from utils.pdf_report import FinTwinPDFReport
from utils.visualizer import PlotlyVisualizer
from models.twin_engine import HealthScoreEngine


def build_master_report(twin) -> bytes:
    random_id = uuid.uuid4().hex[:8].upper()
    user_name = getattr(twin, "name", "") or "FinTwin AI User"

    report = FinTwinPDFReport(
        report_title="FinTwin AI — Complete Financial Intelligence Report",
        user_id=twin.user_id,
        report_id=f"MASTER-{random_id}",
        subtitle=(
            "Consolidated Digital Twin, Health Score, Forecasting, "
            "Tax, Coaching & AI Explainability"
        ),
    )

    # Full-page branded cover
    report.add_cover_page(
        user_name=user_name,
        report_type="Complete Financial Intelligence Report",
    )

    # Table of Contents
    report.add_table_of_contents([
        "User Profile",
        "Financial Health Score",
        "Financial Summary",
        "Income Analysis",
        "Expense Analysis",
        "Savings Analysis",
        "Investment Analysis",
        "Net Worth",
        "Financial Risk Assessment",
        "Tax Intelligence",
        "AI Financial Coach Recommendations",
        "Financial Personality",
        "Behaviour Analysis",
        "Forecasting Results",
        "Explainable AI Summary",
    ])

    # ---------------------------------------------------------------- #
    # 1. User Profile
    # ---------------------------------------------------------------- #
    try:
        payload = twin.export_twin()
        report.add_section_divider("1. User Profile")
        report.add_kpi_summary_row([
            {"label": "Name", "value": str(getattr(twin, "name", "—")), "status": "neutral"},
            {"label": "Occupation", "value": str(getattr(twin, "occupation", "—")), "status": "neutral"},
            {"label": "Risk Level", "value": str(payload.get("risk_level", "—")), "status": "neutral"},
        ])
        report.add_key_value_grid({
            "User Profile ID": twin.user_id,
            "Name": getattr(twin, "name", "—"),
            "Occupation": getattr(twin, "occupation", "—"),
            "Risk Level": payload.get("risk_level"),
        })
    except Exception:
        payload = {}

    # ---------------------------------------------------------------- #
    # 2. Financial Health Score
    # ---------------------------------------------------------------- #
    try:
        engine = HealthScoreEngine(twin)
        score_data = engine.compute_overall_health_score()
        overall_score = score_data.get("overall_score", 0)
        grade = score_data.get("financial_grade", "—")
        label = score_data.get("label", "")

        report.add_section_divider("2. Financial Health Score")
        report.add_kpi_summary_row([
            {"label": "Overall Score", "value": f"{overall_score:.1f} / 100",
             "status": "positive" if overall_score >= 70 else ("negative" if overall_score < 50 else "neutral")},
            {"label": "Financial Grade", "value": f"Grade {grade}", "status": "neutral"},
            {"label": "Classification", "value": label or "—", "status": "neutral"},
        ])

        # Interpretation callout
        if overall_score >= 80:
            callout_style, callout_msg = "success", (
                f"Your financial health score of {overall_score:.1f}/100 is excellent. "
                "You demonstrate strong financial discipline and resilience."
            )
        elif overall_score >= 65:
            callout_style, callout_msg = "info", (
                f"Your financial health score of {overall_score:.1f}/100 is good. "
                "There are targeted areas where further improvement can unlock greater stability."
            )
        elif overall_score >= 50:
            callout_style, callout_msg = "warning", (
                f"Your financial health score of {overall_score:.1f}/100 is moderate. "
                "Review the scoring factors to identify areas requiring attention."
            )
        else:
            callout_style, callout_msg = "danger", (
                f"Your financial health score of {overall_score:.1f}/100 indicates "
                "significant risk areas. Prioritise the AI Coach recommendations."
            )
        report.add_callout(callout_msg, style=callout_style, title="Health Assessment")

        try:
            fig = PlotlyVisualizer.plot_health_score_gauge(overall_score)
            report.add_plotly_figure(fig, caption="Financial Health Score Gauge")
        except Exception:
            pass
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 3-8. Financial Summary / Income / Expense / Savings / Investment / Net Worth
    # ---------------------------------------------------------------- #
    try:
        income = payload.get("income_profile", {})
        expense = payload.get("expense_profile", {})
        debt = payload.get("debt_profile", {})
        invest = payload.get("investment_profile", {})

        monthly_income = getattr(twin, "total_income", 0) or 0
        net_worth = getattr(twin, "net_worth", 0) or 0
        surplus = getattr(twin, "monthly_surplus", 0) or 0

        report.add_section_divider("3. Financial Summary")
        report.add_kpi_summary_row([
            {"label": "Total Monthly Income", "value": f"Rs. {monthly_income:,.0f}",
             "status": "positive" if monthly_income > 0 else "neutral"},
            {"label": "Net Worth", "value": f"Rs. {net_worth:,.0f}",
             "status": "positive" if net_worth >= 0 else "negative"},
            {"label": "Monthly Surplus", "value": f"Rs. {surplus:,.0f}",
             "status": "positive" if surplus > 0 else "negative"},
        ])

        if income:
            report.add_section_divider("4. Income Analysis")
            report.add_key_value_grid(income)

        if expense:
            report.add_section_divider("5. Expense Analysis")
            report.add_key_value_grid(expense)
            try:
                needs = twin.rent + twin.groceries + twin.utilities + twin.transport
                wants = twin.food_delivery + twin.entertainment + twin.shopping
                fig_nw = PlotlyVisualizer.plot_cash_flow_sankey(
                    twin.total_income, expense, max(twin.total_income - needs - wants, 0)
                )
                report.add_plotly_figure(fig_nw, caption="Cash Flow Breakdown")
            except Exception:
                pass

        report.add_section_divider("6. Savings Analysis")
        report.add_key_value_grid({
            "SIP Amount": getattr(twin, "sip_amount", None),
            "Emergency Fund": getattr(twin, "emergency_fund", None),
        })

        if invest:
            report.add_section_divider("7. Investment Analysis")
            report.add_key_value_grid(invest)
            try:
                fig_inv = PlotlyVisualizer.plot_investment_breakdown_bar(invest)
                report.add_plotly_figure(fig_inv, caption="Investment Breakdown")
            except Exception:
                pass

        report.add_section_divider("8. Net Worth")
        nw_data = {"Net Worth": twin.net_worth}
        if isinstance(debt, (int, float)):
            nw_data["Total Debt"] = debt
        report.add_key_value_grid(nw_data)
        if net_worth >= 0:
            report.add_callout(
                f"Your net worth is Rs. {net_worth:,.0f}. "
                "Continue building assets to accelerate wealth creation.",
                style="success",
                title="Net Worth",
            )
        else:
            report.add_callout(
                f"Your net worth is negative (Rs. {net_worth:,.0f}). "
                "Prioritise debt reduction to restore a positive net-worth position.",
                style="danger",
                title="Net Worth Alert",
            )
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 9. Financial Risk Assessment
    # ---------------------------------------------------------------- #
    try:
        risk_level = str(getattr(twin, "risk_level", "—"))
        report.add_section_divider("9. Financial Risk Assessment")
        risk_style = "success" if "low" in risk_level.lower() else (
            "danger" if "high" in risk_level.lower() else "warning"
        )
        report.add_kpi_summary_row([
            {"label": "Risk Level", "value": risk_level, "status": risk_style.replace("success", "positive").replace("danger", "negative").replace("warning", "neutral")},
        ])
        report.add_callout(
            f"Your assessed financial risk level is: {risk_level}. "
            "Review your emergency fund coverage and debt obligations to manage exposure.",
            style=risk_style,
            title="Risk Profile",
        )
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 10. Tax Intelligence
    # ---------------------------------------------------------------- #
    try:
        from utils.tax_calculator import IndianTaxCalculator, DeductionProfile
        annual_income = float(getattr(twin, "monthly_income", 0)) * 12
        deductions = DeductionProfile(
            investment_80c=min(getattr(twin, "sip_amount", 0) * 12, 150000),
            health_insurance_self=getattr(twin, "health_insurance", 0),
            health_insurance_parents=0,
            nps_80ccd1b=0,
            home_loan_interest=0,
            other_deductions=0,
        )
        calc = IndianTaxCalculator()
        result = calc.compare_and_optimize(annual_income, deductions)
        report.add_section_divider("10. Tax Intelligence")
        report.add_kpi_summary_row([
            {"label": "Recommended Regime", "value": result["recommended_regime"], "status": "positive"},
            {"label": "Annual Tax Saving", "value": f"Rs. {result['annual_savings']:,.0f}", "status": "positive"},
            {"label": "Old Regime Tax", "value": f"Rs. {result['old_regime'].total_tax:,.0f}", "status": "neutral"},
            {"label": "New Regime Tax", "value": f"Rs. {result['new_regime'].total_tax:,.0f}", "status": "neutral"},
        ])
        report.add_key_value_grid({
            "Recommended Regime": result["recommended_regime"],
            "Annual Savings": result["annual_savings"],
            "Old Regime Tax": result["old_regime"].total_tax,
            "New Regime Tax": result["new_regime"].total_tax,
        })
        report.add_callout(
            f"Switch to the {result['recommended_regime']} to save Rs. {result['annual_savings']:,.0f} annually.",
            style="success",
            title="Tax Optimisation",
        )
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 11. AI Financial Coach Recommendations
    # ---------------------------------------------------------------- #
    try:
        from utils.coach import FinancialCoach
        coach_report = FinancialCoach(twin).generate_report()
        report.add_section_divider("11. AI Financial Coach Recommendations")
        if coach_report.narrative:
            report.add_executive_summary(coach_report.narrative)
        if coach_report.recommendations:
            # Top 3 as callouts
            for r in coach_report.recommendations[:3]:
                action_text = f"{r.title}. {r.action} — Impact: {r.impact}"
                sev = str(getattr(r, "severity", "") or "").lower()
                cs = "danger" if "critical" in sev or "high" in sev else (
                    "warning" if "medium" in sev else "info"
                )
                report.add_callout(action_text, style=cs)
            # Remaining as table
            if len(coach_report.recommendations) > 3:
                rec_df = pd.DataFrame([
                    {
                        "Domain": r.domain.value if hasattr(r.domain, "value") else r.domain,
                        "Title": r.title,
                        "Action": r.action,
                        "Impact": r.impact,
                    }
                    for r in coach_report.recommendations[3:15]
                ])
                report.add_dataframe_table(rec_df, title="Additional Recommendations")
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 12. Financial Personality
    # ---------------------------------------------------------------- #
    try:
        from models.clustering import FinancialPersonalityClusterer
        clusterer = FinancialPersonalityClusterer.load()
        if clusterer.is_fitted():
            personality = clusterer.predict_personality(twin)
            report.add_section_divider("12. Financial Personality")
            report.add_kpi_summary_row([
                {"label": "Detected Archetype", "value": str(personality), "status": "neutral"},
            ])
            report.add_callout(
                f"Your financial personality archetype is: {personality}. "
                "This profile shapes your spending, saving, and investment decisions.",
                style="info",
                title="Personality Archetype",
            )
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 13. Behaviour Analysis
    # ---------------------------------------------------------------- #
    try:
        from database.db_manager import DBManager
        cohort_avg = DBManager.get_cohort_averages(twin.occupation)
        if cohort_avg:
            user_spend = {
                "Rent": twin.rent, "Groceries": twin.groceries, "Utilities": twin.utilities,
                "Transport": twin.transport, "Food Delivery": twin.food_delivery,
                "Entertainment": twin.entertainment, "Shopping": twin.shopping,
            }
            peer_spend = {k: cohort_avg.get(k.lower().replace(" ", "_"), 0.0) for k in user_spend}
            report.add_section_divider("13. Behaviour Analysis")
            fig_peer = PlotlyVisualizer.plot_peer_comparison(user_spend, peer_spend)
            report.add_plotly_figure(fig_peer, caption="Your Spending vs. Peer Average")
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 14. Forecasting Results
    # ---------------------------------------------------------------- #
    try:
        from models.predictor import FinancialPredictor
        predictor = FinancialPredictor()
        if predictor.load_only(horizon=6):
            pred_6 = predictor.predict(twin, horizon_months=6)
            pred_12 = predictor.predict(twin, horizon_months=12) if predictor.load_only(horizon=12) else {}
            report.add_section_divider("14. Forecasting Results")
            kpis = [
                {"label": "Predicted Savings (6M)", "value": f"Rs. {pred_6.get('predicted_savings', 0):,.0f}", "status": "positive"},
                {"label": "Predicted Net Worth (6M)", "value": f"Rs. {pred_6.get('predicted_net_worth', 0):,.0f}", "status": "positive"},
            ]
            if pred_12:
                kpis += [
                    {"label": "Predicted Savings (12M)", "value": f"Rs. {pred_12.get('predicted_savings', 0):,.0f}", "status": "positive"},
                    {"label": "Predicted Net Worth (12M)", "value": f"Rs. {pred_12.get('predicted_net_worth', 0):,.0f}", "status": "positive"},
                ]
            report.add_kpi_summary_row(kpis[:4])
            try:
                df_traj = predictor.generate_trajectory(twin, months=12)
                fig_traj = PlotlyVisualizer.plot_forecast_trajectory(df_traj)
                report.add_plotly_figure(fig_traj, caption="12-Month Forecast Trajectory")
            except Exception:
                pass
    except Exception:
        pass

    # ---------------------------------------------------------------- #
    # 15. Explainable AI Summary
    # ---------------------------------------------------------------- #
    try:
        from models.explainability import ExplainableAI
        hs_result = ExplainableAI().explain_health_score(twin)
        report.add_section_divider("15. Explainable AI Summary")
        report.add_callout(
            "The following analysis breaks down what factors drive your financial health score "
            "and how the AI model arrives at its predictions.",
            style="info",
            title="About This Section",
        )
        for line in hs_result.narrative[:6]:
            report.add_paragraph(line)
    except Exception:
        pass

    return report.build()
