"""
Unit tests for PlotlyVisualizer utility.
Validates that all charts render figures without errors and return valid Plotly go.Figure objects.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
from utils.visualizer import PlotlyVisualizer


def test_plot_health_score_gauge():
    for score in [25, 45, 65, 85]:
        fig = PlotlyVisualizer.plot_health_score_gauge(score)
        assert isinstance(fig, go.Figure)


def test_plot_cash_flow_sankey():
    income = 75000
    expenses = {"Rent": 20000, "Food": 15000, "Utilities": 5000}
    savings = 35000
    fig = PlotlyVisualizer.plot_cash_flow_sankey(income, expenses, savings)
    assert isinstance(fig, go.Figure)


def test_plot_peer_comparison():
    user_spend = {"Housing": 25000, "Food": 12000}
    peer_spend = {"Housing": 20000, "Food": 10000}
    fig = PlotlyVisualizer.plot_peer_comparison(user_spend, peer_spend)
    assert isinstance(fig, go.Figure)


def test_plot_asset_liability_donut():
    assets = {"Cash": 100000, "Mutual Funds": 250000}
    liabilities = {"Car Loan": 150000}
    fig = PlotlyVisualizer.plot_asset_liability_donut(assets, liabilities)
    assert isinstance(fig, go.Figure)


def test_plot_investment_breakdown_bar():
    investments = {"Equities": 50000, "Fixed Deposit": 100000}
    fig = PlotlyVisualizer.plot_investment_breakdown_bar(investments)
    assert isinstance(fig, go.Figure)


def test_plot_personality_radar():
    features = {
        "savings_ratio": 0.35,
        "expense_ratio": 0.45,
        "investment_ratio": 0.20,
        "debt_to_income_ratio": 0.10,
    }
    fig = PlotlyVisualizer.plot_personality_radar(features, "Balanced Wealth Builder")
    assert isinstance(fig, go.Figure)


def test_plot_cluster_comparison_bar():
    user_vals = {"Savings Ratio": 0.35, "Expense Ratio": 0.45}
    cluster_means = {"Savings Ratio": 0.30, "Expense Ratio": 0.50}
    fig = PlotlyVisualizer.plot_cluster_comparison_bar(
        user_vals, cluster_means, "Balanced Wealth Builder", color="#6366F1"
    )
    assert isinstance(fig, go.Figure)


def test_plot_cluster_scatter():
    df_pop = pd.DataFrame({
        "savings_ratio": [0.2, 0.4, 0.1],
        "expense_ratio": [0.6, 0.4, 0.8],
        "personality": ["Saver", "Investor", "Spender"],
    })
    current = {"savings_ratio": 0.35, "expense_ratio": 0.45}
    colors = {"Saver": "#10B981", "Investor": "#3B82F6", "Spender": "#EF4444"}
    fig = PlotlyVisualizer.plot_cluster_scatter(df_pop, current, "Saver", colors)
    assert isinstance(fig, go.Figure)


def test_plot_monte_carlo_fan():
    df_mc = pd.DataFrame({
        "Month": list(range(1, 13)),
        "p10": [1000 * i for i in range(1, 13)],
        "p25": [1200 * i for i in range(1, 13)],
        "p50": [1500 * i for i in range(1, 13)],
        "p75": [1800 * i for i in range(1, 13)],
        "p90": [2000 * i for i in range(1, 13)],
    })
    fig = PlotlyVisualizer.plot_monte_carlo_fan(df_mc, years=1)
    assert isinstance(fig, go.Figure)


def test_plot_forecast_trajectory():
    df = pd.DataFrame({
        "Month": list(range(1, 13)),
        "Projected_Savings": [10000] * 12,
        "Projected_Net_Worth": [50000 + 10000 * i for i in range(1, 13)],
    })
    fig = PlotlyVisualizer.plot_forecast_trajectory(df)
    assert isinstance(fig, go.Figure)


def test_plot_forecast_trajectories():
    df = pd.DataFrame({
        "Year": [1, 2, 3, 4, 5],
        "Base Case": [100, 200, 300, 400, 500],
        "Optimistic Scenario": [120, 250, 400, 600, 800],
        "Pessimistic Scenario": [90, 160, 220, 280, 320],
    })
    fig = PlotlyVisualizer.plot_forecast_trajectories(df)
    assert isinstance(fig, go.Figure)


def test_plot_scenario_comparison_bar():
    before = {"Net Worth": 500000, "Savings": 30000}
    after = {"Net Worth": 450000, "Savings": 25000}
    fig = PlotlyVisualizer.plot_scenario_comparison_bar(before, after)
    assert isinstance(fig, go.Figure)


def test_plot_scenario_trajectory_comparison():
    df_before = pd.DataFrame({"Month": [1, 2, 3], "Projected_Net_Worth": [100, 200, 300]})
    df_after = pd.DataFrame({"Month": [1, 2, 3], "Projected_Net_Worth": [90, 180, 270]})
    fig = PlotlyVisualizer.plot_scenario_trajectory_comparison(df_before, df_after)
    assert isinstance(fig, go.Figure)


def test_plot_shap_force_plot_with_objects():
    class Item:
        def __init__(self, comp, contrib):
            self.component = comp
            self.contribution = contrib

    force_data = {
        "baseline": 50.0,
        "prediction": 72.0,
        "all_sorted": [Item("Savings Rate", 15.0), Item("Debt Load", -3.0)],
    }
    fig = PlotlyVisualizer.plot_shap_force_plot(force_data)
    assert isinstance(fig, go.Figure)


def test_plot_shap_force_plot_with_dicts():
    force_data = {
        "baseline": 50.0,
        "prediction": 65.0,
        "all_sorted": [
            {"component": "Emergency Fund", "contribution": 10.0},
            {"component": "Expense Ratio", "contribution": -5.0},
        ],
    }
    fig = PlotlyVisualizer.plot_shap_force_plot(force_data)
    assert isinstance(fig, go.Figure)


def test_plot_shap_summary_bar():
    df = pd.DataFrame({
        "Feature": ["Savings Rate", "Expense Ratio", "Investment Ratio"],
        "SHAP Value": [12.5, -6.2, 4.8],
    })
    fig = PlotlyVisualizer.plot_shap_summary_bar(df)
    assert isinstance(fig, go.Figure)


def test_plot_shap_contribution_heatmap():
    df = pd.DataFrame({
        "Feature": ["Savings", "Debt", "Investments"],
        "SHAP Value": [10.0, -4.0, 6.0],
        "Sub-Score": [80.0, 45.0, 70.0],
    })
    fig = PlotlyVisualizer.plot_shap_contribution_heatmap(df)
    assert isinstance(fig, go.Figure)
