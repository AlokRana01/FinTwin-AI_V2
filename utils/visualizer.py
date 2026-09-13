"""
Visualizer Wrapper Utility
Wraps Plotly to generate consistent, premium, interactive UI charts for the Streamlit dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

class PlotlyVisualizer:
    """
    Standard visualization module producing clean, dark-mode-compatible charts.
    """
    
    @staticmethod
    def plot_health_score_gauge(score: float) -> go.Figure:
        """
        Creates a clean, premium Gauge chart showing the user's Financial Health Score (0-100).
        """
        # Premium Color Palette
        if score >= 80:
            color = "#10B981"  # Emerald Green
            grade = "A"
        elif score >= 60:
            color = "#3B82F6"  # Royal Blue
            grade = "B"
        elif score >= 40:
            color = "#F59E0B"  # Amber Orange
            grade = "C"
        else:
            color = "#EF4444"  # Rose Red
            grade = "D"
            
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Financial Grade: {grade}", 'font': {'size': 22, 'color': '#FFFFFF', 'weight': 'bold'}},
            number={'font': {'color': color, 'size': 54}, 'suffix': "/100"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9CA3AF"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "#1F2937",
                'borderwidth': 1,
                'bordercolor': "#374151",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [60, 80], 'color': 'rgba(59, 130, 246, 0.15)'},
                    {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.8,
                    'value': score
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F3F4F6", 'family': "Outfit, Inter, sans-serif"},
            height=350,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        return fig

    @staticmethod
    def plot_cash_flow_sankey(income: float, expenses: Dict[str, float], savings: float) -> go.Figure:
        """
        Creates a Cash Flow Sankey diagram displaying inflows vs outflows.
        """
        # Node Labels: Total Income, Savings, Expenses, and then categories
        nodes = ["Monthly Income", "Surplus / Savings"]
        for cat in expenses.keys():
            nodes.append(cat)
            
        # Source, Target, Value lists
        sources = []
        targets = []
        values = []
        
        # Inflow to Savings
        sources.append(0)
        targets.append(1)
        values.append(savings if savings > 0 else 0)
        
        # Inflow to Expenses
        for idx, (cat, amount) in enumerate(expenses.items()):
            sources.append(0)
            targets.append(idx + 2)
            values.append(amount if amount > 0 else 0)

        outflow_palette = [
            "#8B5CF6", "#EC4899", "#F59E0B", "#06B6D4",
            "#6366F1", "#3B82F6", "#14B8A6", "#F97316"
        ]
            
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=18,
                thickness=22,
                line=dict(color="#1F2937", width=1),
                label=nodes,
                color=["#4F8CFF", "#10B981"] + [outflow_palette[i % len(outflow_palette)] for i in range(len(expenses))],
                hovertemplate="<b>%{label}</b><br>Total: ₹%{value:,.0f}<extra></extra>"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(79, 140, 255, 0.22)",
                hovertemplate="%{source.label} → %{target.label}<br><b>₹%{value:,.0f}</b><extra></extra>"
            )
        )])
        
        fig.update_layout(
            font_size=13,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F8FAFC", 'family': "Outfit, Inter, sans-serif"},
            height=370,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig


    @staticmethod
    def plot_peer_comparison(user_spend: Dict[str, float], peer_avg_spend: Dict[str, float]) -> go.Figure:
        """
        Plots a grouped bar chart comparing the user's monthly spending categories to peer average spending.
        """
        categories = list(user_spend.keys())
        user_vals = [user_spend[cat] for cat in categories]
        peer_vals = [peer_avg_spend.get(cat, 0.0) for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(name='Your Outflow', x=categories, y=user_vals, marker_color='#3B82F6'),
            go.Bar(name='Peer Outflow Average', x=categories, y=peer_vals, marker_color='#9CA3AF')
        ])
        
        fig.update_layout(
            barmode='group',
            title="Category Outflow: You vs. Peer Cohort",
            xaxis_title="Expense Category",
            yaxis_title="Monthly Outflow (₹)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F3F4F6", 'family': "Outfit, Inter, sans-serif"},
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def plot_asset_liability_donut(assets: Dict[str, float], liabilities: Dict[str, float]) -> go.Figure:
        """
        Creates a dual-donut chart: inner ring = asset breakdown, outer ring = liability breakdown.
        Uses a green-teal palette for assets and a red-orange palette for liabilities.
        """
        asset_labels = list(assets.keys())
        asset_values = list(assets.values())
        liability_labels = list(liabilities.keys())
        liability_values = list(liabilities.values())

        asset_colors = [
            "#10B981", "#34D399", "#6EE7B7", "#A7F3D0",
            "#059669", "#047857", "#065F46", "#D1FAE5"
        ]
        liability_colors = [
            "#EF4444", "#F87171", "#FCA5A5", "#FECACA",
            "#DC2626", "#B91C1C"
        ]

        fig = go.Figure()

        # Inner ring: Assets
        fig.add_trace(go.Pie(
            labels=asset_labels,
            values=asset_values,
            name="Assets",
            hole=0.55,
            domain={"x": [0.15, 0.85], "y": [0.15, 0.85]},
            marker={"colors": asset_colors[:len(asset_labels)], "line": {"color": "#111827", "width": 2}},
            textinfo="label+percent",
            textfont={"size": 11, "color": "#F3F4F6"},
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra>Assets</extra>",
            showlegend=True,
        ))

        # Outer ring: Liabilities
        fig.add_trace(go.Pie(
            labels=liability_labels,
            values=liability_values,
            name="Liabilities",
            hole=0.75,
            domain={"x": [0.05, 0.95], "y": [0.05, 0.95]},
            marker={"colors": liability_colors[:len(liability_labels)], "line": {"color": "#111827", "width": 2}},
            textinfo="label+percent",
            textfont={"size": 10, "color": "#F3F4F6"},
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra>Liabilities</extra>",
            showlegend=True,
        ))

        fig.update_layout(
            title={"text": "Asset Breakdown (Inner) vs Liability Breakdown (Outer)", "font": {"size": 14}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5, "font": {"size": 10}}
        )
        return fig

    @staticmethod
    def plot_investment_breakdown_bar(investment_profile: Dict[str, float]) -> go.Figure:
        """
        Creates a horizontal bar chart showing each investment bucket.
        Sorted by value descending. Color-coded from teal to blue gradient.
        """
        # Filter out non-numeric or ratio keys
        skip_keys = {"monthly_sip", "investment_to_income_ratio"}
        labels = []
        values = []
        for k, v in investment_profile.items():
            if k not in skip_keys and isinstance(v, (int, float)) and v >= 0:
                labels.append(k.replace("_", " ").title())
                values.append(v)

        # Sort descending by value
        paired = sorted(zip(values, labels), reverse=True)
        values_sorted = [p[0] for p in paired]
        labels_sorted = [p[1] for p in paired]

        # Gradient palette
        bar_colors = [
            "#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE",
            "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"
        ]

        fig = go.Figure(go.Bar(
            x=values_sorted,
            y=labels_sorted,
            orientation="h",
            marker={
                "color": bar_colors[:len(labels_sorted)],
                "line": {"color": "#1F2937", "width": 1}
            },
            text=[f"₹{v:,.0f}" for v in values_sorted],
            textposition="outside",
            textfont={"color": "#F3F4F6", "size": 11},
            hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"
        ))

        fig.update_layout(
            title="Investment Portfolio Breakdown",
            xaxis_title="Amount (₹)",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height=380,
            margin=dict(l=20, r=80, t=50, b=20),
            xaxis={"gridcolor": "#374151"},
            yaxis={"gridcolor": "rgba(0,0,0,0)"}
        )
        return fig

    @staticmethod
    def plot_personality_radar(
        feature_dict: Dict[str, float],
        personality_label: str,
        color: str = "#6366F1"
    ) -> go.Figure:
        """
        Radar / spider chart showing the user's 4 financial ratio features.
        Axes: Savings Ratio, Expense Ratio, Investment Ratio, Debt-to-Income Ratio.

        Args:
            feature_dict      : Dict with keys savings_ratio, expense_ratio,
                                investment_ratio, debt_to_income_ratio (values 0–1).
            personality_label : Label shown in the chart title.
            color             : Hex color for the radar fill & line.
        """
        labels = ["Savings", "Expenses", "Investing", "Debt Load"]
        values = [
            feature_dict.get("savings_ratio",        0.0),
            feature_dict.get("expense_ratio",         0.0),
            feature_dict.get("investment_ratio",      0.0),
            feature_dict.get("debt_to_income_ratio",  0.0),
        ]
        # Close the polygon
        labels_closed = labels + [labels[0]]
        values_closed = values + [values[0]]

        fill_color = color
        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_color = f"rgba({r}, {g}, {b}, 0.2)"

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=fill_color,
            line={"color": color, "width": 2.5},
            marker={"size": 7, "color": color},
            name=personality_label,
            hovertemplate="%{theta}: %{r:.1%}<extra></extra>"
        ))

        fig.update_layout(
            title={
                "text": f"Financial Ratio Profile — {personality_label}",
                "font": {"size": 15, "color": "#F3F4F6"}
            },
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1],
                    "tickformat": ".0%",
                    "gridcolor": "#374151",
                    "linecolor": "#374151",
                    "tickfont": {"color": "#9CA3AF", "size": 10}
                },
                "angularaxis": {
                    "tickfont": {"color": "#E5E7EB", "size": 12},
                    "gridcolor": "#374151",
                    "linecolor": "#374151"
                },
                "bgcolor": "rgba(0,0,0,0)"
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height=380,
            margin=dict(l=60, r=60, t=60, b=40),
            showlegend=False
        )
        return fig

    @staticmethod
    def plot_cluster_comparison_bar(
        user_vals: Dict[str, float],
        cluster_means: Dict[str, float],
        personality_label: str,
        color: str = "#6366F1"
    ) -> go.Figure:
        """
        Grouped bar chart comparing the user's ratio values against the cluster's benchmark average.

        Args:
            user_vals: Dict with feature names mapping to user's ratio values (0-1).
            cluster_means: Dict with feature names mapping to cluster average ratio values (0-1).
            personality_label: The name of the personality archetype.
            color: Accent color for the user's bars.
        """
        categories = list(user_vals.keys())
        user_data = [user_vals.get(c, 0.0) * 100 for c in categories]
        cluster_data = [cluster_means.get(c, 0.0) * 100 for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Your Metrics",
            x=categories,
            y=user_data,
            marker_color=color,
            text=[f"{v:.1f}%" for v in user_data],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Your Metric: %{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name=f"{personality_label} Average",
            x=categories,
            y=cluster_data,
            marker_color="#64748B",
            text=[f"{v:.1f}%" for v in cluster_data],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{personality_label} Avg: %{{y:.1f}}%<extra></extra>",
        ))

        fig.update_layout(
            title={
                "text": f"Your Ratios vs. {personality_label} Benchmark Average",
                "font": {"size": 15, "color": "#F3F4F6"}
            },
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            yaxis={"title": "Percentage (%)", "gridcolor": "#1F2937", "range": [0, max(max(user_data + cluster_data, default=50) * 1.25, 30)]},
            xaxis={"gridcolor": "#1F2937"},
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        return fig

    @staticmethod
    def plot_cluster_scatter(
        df_population: "pd.DataFrame",
        current_features: Dict[str, float],
        current_label: str,
        color_map: Dict[str, str]
    ) -> go.Figure:
        """
        2D scatter: all users plotted by savings_ratio vs expense_ratio,
        colored by personality cluster. Current user highlighted with a star.

        Args:
            df_population   : DataFrame with columns savings_ratio, expense_ratio, personality.
            current_features: Feature dict for the current user.
            current_label   : Personality label of the current user.
            color_map       : Dict mapping personality label -> hex color.
        """
        fig = go.Figure()

        if not df_population.empty and "personality" in df_population.columns:
            for label, grp in df_population.groupby("personality"):
                color = color_map.get(label, "#6B7280")
                fig.add_trace(go.Scatter(
                    x=grp["savings_ratio"],
                    y=grp["expense_ratio"],
                    mode="markers",
                    name=label,
                    marker={
                        "color": color,
                        "size": 6,
                        "opacity": 0.55,
                        "line": {"color": "#111827", "width": 0.5}
                    },
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Savings: %{x:.1%}<br>"
                        "Expense: %{y:.1%}<extra></extra>"
                    )
                ))

        # Current user — highlighted star
        user_color = color_map.get(current_label, "#FBBF24")
        fig.add_trace(go.Scatter(
            x=[current_features.get("savings_ratio", 0)],
            y=[current_features.get("expense_ratio",  0)],
            mode="markers+text",
            name=f"You ({current_label})",
            marker={
                "symbol": "star",
                "size": 20,
                "color": user_color,
                "line": {"color": "#FFFFFF", "width": 1.5}
            },
            text=["  ← You"],
            textfont={"color": "#FFFFFF", "size": 12},
            textposition="middle right",
            hovertemplate=(
                f"<b>You — {current_label}</b><br>"
                "Savings: %{x:.1%}<br>"
                "Expense: %{y:.1%}<extra></extra>"
            )
        ))

        fig.update_layout(
            title="Cluster Visualization — Savings vs Expense Ratio",
            xaxis={
                "title": "Savings Ratio",
                "tickformat": ".0%",
                "gridcolor": "#374151",
                "zerolinecolor": "#4B5563"
            },
            yaxis={
                "title": "Expense Ratio",
                "tickformat": ".0%",
                "gridcolor": "#374151",
                "zerolinecolor": "#4B5563"
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": "#374151", "borderwidth": 1},
            height=430,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def plot_monte_carlo_fan(df: "pd.DataFrame", years: int = 5) -> go.Figure:
        """
        Shaded fan chart showing Monte Carlo percentile bands of net worth.

        Args:
            df   : DataFrame with columns Month, p10, p25, p50, p75, p90.
            years: Used only for x-axis label formatting.
        """
        fig = go.Figure()

        # Band fills (outer to inner)
        bands = [
            ("p10", "p90", "rgba(99,102,241,0.10)", "10th–90th percentile"),
            ("p25", "p75", "rgba(99,102,241,0.20)", "25th–75th percentile"),
        ]
        for lo, hi, fill_color, name in bands:
            if lo in df.columns and hi in df.columns:
                fig.add_trace(go.Scatter(
                    x=list(df["Month"]) + list(df["Month"])[::-1],
                    y=list(df[hi]) + list(df[lo])[::-1],
                    fill="toself",
                    fillcolor=fill_color,
                    line={"color": "rgba(0,0,0,0)"},
                    name=name,
                    showlegend=True,
                    hoverinfo="skip"
                ))

        # Median line
        if "p50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Month"],
                y=df["p50"],
                mode="lines",
                name="Median (p50)",
                line={"color": "#818CF8", "width": 2.5},
                hovertemplate="Month %{x}<br>Net Worth: ₹%{y:,.0f}<extra>Median</extra>"
            ))

        # p10 / p90 dashed bounds
        for col, label, dash in [("p10", "Pessimistic (p10)", "dot"), ("p90", "Optimistic (p90)", "dot")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["Month"], y=df[col],
                    mode="lines", name=label,
                    line={"color": "#6366F1", "width": 1.2, "dash": dash},
                    hovertemplate=f"Month %{{x}}<br>₹%{{y:,.0f}}<extra>{label}</extra>"
                ))

        fig.update_layout(
            title=f"Monte Carlo Net Worth Projection — {years}-Year Horizon",
            xaxis={"title": "Month", "gridcolor": "#374151"},
            yaxis={"title": "Net Worth (₹)", "gridcolor": "#374151", "tickformat": ",.0f"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": "#374151", "borderwidth": 1},
            height=430,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def plot_forecast_trajectory(df: "pd.DataFrame") -> go.Figure:
        """
        Dual-axis line chart: monthly projected savings AND net worth over time.

        Args:
            df: DataFrame with columns Month, Projected_Savings, Projected_Net_Worth.
        """
        fig = go.Figure()

        if "Projected_Savings" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Month"], y=df["Projected_Savings"],
                mode="lines+markers", name="Monthly Savings",
                line={"color": "#10B981", "width": 2.5},
                marker={"size": 5},
                hovertemplate="Month %{x}<br>Savings: ₹%{y:,.0f}<extra></extra>"
            ))

        if "Projected_Net_Worth" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Month"], y=df["Projected_Net_Worth"],
                mode="lines+markers", name="Net Worth",
                line={"color": "#6366F1", "width": 2.5},
                marker={"size": 5},
                hovertemplate="Month %{x}<br>Net Worth: ₹%{y:,.0f}<extra></extra>"
            ))

        # Markers at month 6 and 12
        for m in [6, 12]:
            subset = df[df["Month"] == m]
            if not subset.empty and "Projected_Net_Worth" in subset.columns:
                fig.add_vline(
                    x=m, line_dash="dash",
                    line_color="#F59E0B", line_width=1.2,
                    annotation_text=f"Month {m}",
                    annotation_font_color="#F59E0B"
                )

        fig.update_layout(
            title="12-Month XGBoost Forecast — Savings & Net Worth Trajectory",
            xaxis={"title": "Month", "gridcolor": "#374151", "dtick": 1},
            yaxis={"title": "Amount (₹)", "gridcolor": "#374151", "tickformat": ",.0f"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            legend={"bgcolor": "rgba(0,0,0,0)"},
            height=390,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def plot_forecast_trajectories(df: "pd.DataFrame") -> go.Figure:
        """
        Multi-line chart for scenario projections (Base / Optimistic / Pessimistic).

        Args:
            df: DataFrame with columns Year + one column per scenario name.
        """
        SCENARIO_COLORS = {
            "Base Case":            "#6366F1",
            "Optimistic Scenario":  "#10B981",
            "Pessimistic Scenario": "#EF4444",
        }
        SCENARIO_DASH = {
            "Base Case":            "solid",
            "Optimistic Scenario":  "solid",
            "Pessimistic Scenario": "dash",
        }

        fig = go.Figure()
        x_col = "Year" if "Year" in df.columns else df.columns[0]

        for col in df.columns:
            if col == x_col:
                continue
            color = SCENARIO_COLORS.get(col, "#9CA3AF")
            dash  = SCENARIO_DASH.get(col, "solid")
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[col],
                mode="lines+markers", name=col,
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 6},
                hovertemplate=f"Year %{{x}}<br>₹%{{y:,.0f}}<extra>{col}</extra>"
            ))

        fig.update_layout(
            title="Net Worth Projection — Scenario Comparison",
            xaxis={"title": x_col, "gridcolor": "#374151", "dtick": 1},
            yaxis={"title": "Net Worth (₹)", "gridcolor": "#374151", "tickformat": ",.0f"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": "#374151", "borderwidth": 1},
            height=420,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def plot_scenario_comparison_bar(before_vals: dict, after_vals: dict, title: str = "Before vs After") -> go.Figure:
        """
        Creates a grouped bar chart comparing key financial metrics before and after a scenario.

        Args:
            before_vals: Dict of {metric_label: numeric_value} for the before state.
            after_vals:  Dict of {metric_label: numeric_value} for the after state.
            title:       Chart title.
        Returns:
            Plotly Figure.
        """
        labels = list(before_vals.keys())
        before = [before_vals[k] for k in labels]
        after  = [after_vals[k]  for k in labels]

        fig = go.Figure(data=[
            go.Bar(
                name="Before",
                x=labels,
                y=before,
                marker_color="rgba(107,114,128,0.8)",
                text=[f"₹{v:,.0f}" if isinstance(v, float) else str(v) for v in before],
                textposition="outside",
            ),
            go.Bar(
                name="After",
                x=labels,
                y=after,
                marker_color="rgba(59,130,246,0.85)",
                text=[f"₹{v:,.0f}" if isinstance(v, float) else str(v) for v in after],
                textposition="outside",
            ),
        ])
        fig.update_layout(
            title         = title,
            barmode       = "group",
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height        = 400,
            legend        = {"bgcolor": "rgba(0,0,0,0)"},
            margin        = dict(l=20, r=20, t=60, b=20),
            yaxis         = {"gridcolor": "#1F2937"},
        )
        return fig

    @staticmethod
    def plot_scenario_trajectory_comparison(
        before_savings: list,
        after_savings:  list,
        months:         int = 12,
        scenario_name:  str = "Scenario",
    ) -> go.Figure:
        """
        Plots 12-month savings trajectory before and after a scenario event.

        Args:
            before_savings: List of monthly savings projections (before).
            after_savings:  List of monthly savings projections (after).
            months:         Number of months on the x-axis.
            scenario_name:  Label for the scenario.
        Returns:
            Plotly Figure.
        """
        x = list(range(1, months + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=before_savings,
            name="Before",
            mode="lines+markers",
            line={"color": "#6B7280", "width": 2.5, "dash": "dash"},
            marker={"size": 6},
            hovertemplate="Month %{x}<br>₹%{y:,.0f}<extra>Before</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=after_savings,
            name=f"After ({scenario_name})",
            mode="lines+markers",
            line={"color": "#3B82F6", "width": 2.5},
            marker={"size": 6},
            hovertemplate="Month %{x}<br>₹%{y:,.0f}<extra>After</extra>",
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.07)",
        ))
        fig.update_layout(
            title         = f"12-Month Savings Trajectory — {scenario_name}",
            xaxis         = {"title": "Month", "gridcolor": "#374151", "dtick": 1},
            yaxis         = {"title": "Cumulative Savings (₹)", "gridcolor": "#374151", "tickformat": ",.0f"},
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            legend        = {"bgcolor": "rgba(0,0,0,0)"},
            height        = 380,
            margin        = dict(l=20, r=20, t=60, b=20),
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════
    # Module 9 — SHAP / Explainable AI Charts
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def plot_shap_force_plot(
        force_data: Dict[str, Any],
        title: str = "Force Plot — Score Decomposition",
        unit: str  = "pts",
        x_label: str = "Health Score",
    ) -> go.Figure:
        """
        Renders a horizontal waterfall / force plot showing how each factor
        pushes the prediction from the baseline to the final value.

        Args:
            force_data: Dict from HealthScoreExplainer._build_force_plot_data()
                        Keys: baseline, prediction, positives, negatives, all_sorted
            title:      Chart title.
            unit:       Unit label for tooltip (e.g. "pts" or "₹").
            x_label:    X-axis label.

        Returns:
            Plotly Figure.
        """
        baseline   = force_data.get("baseline",   50.0)
        prediction = force_data.get("prediction", 50.0)
        all_sorted = force_data.get("all_sorted", [])

        def _get_comp(c):
            return getattr(c, "component", c.get("component", "") if isinstance(c, dict) else str(c))

        def _get_contrib(c):
            return float(getattr(c, "contribution", c.get("contribution", 0.0) if isinstance(c, dict) else 0.0))

        # Build waterfall: baseline → each contribution → prediction
        labels = ["Baseline"] + [_get_comp(c) for c in all_sorted] + ["Final Score"]
        values = [baseline]   + [_get_contrib(c) for c in all_sorted] + [0]
        measure = ["absolute"] + ["relative"] * len(all_sorted) + ["total"]

        colors = []
        for i, m in enumerate(measure):
            if m == "absolute":
                colors.append("#6B7280")   # grey for baseline
            elif m == "total":
                colors.append("#F59E0B")   # amber for final
            else:
                contrib = _get_contrib(all_sorted[i - 1])
                colors.append("#10B981" if contrib >= 0 else "#EF4444")

        text_labels = [f"{baseline:.1f}"]
        for c in all_sorted:
            contrib = _get_contrib(c)
            sign = "+" if contrib >= 0 else ""
            text_labels.append(f"{sign}{contrib:.1f} {unit}")
        text_labels.append(f"{prediction:.1f}")

        fig = go.Figure(go.Waterfall(
            orientation  = "v",
            measure      = measure,
            x            = labels,
            y            = values,
            text         = text_labels,
            textposition = "outside",
            connector    = {"line": {"color": "#374151", "width": 1}},
            increasing   = {"marker": {"color": "#10B981"}},
            decreasing   = {"marker": {"color": "#EF4444"}},
            totals       = {"marker": {"color": "#F59E0B"}},
        ))

        fig.update_layout(
            title         = title,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height        = 420,
            showlegend    = False,
            margin        = dict(l=10, r=10, t=60, b=10),
            yaxis         = {"title": x_label, "gridcolor": "#1F2937"},
            xaxis         = {"tickangle": -20},
        )
        return fig

    @staticmethod
    def plot_shap_summary_bar(
        summary_df: "pd.DataFrame",
        value_col:  str = "SHAP Value",
        label_col:  str = "Feature",
        title: str      = "Feature Importance — SHAP Values",
        unit: str       = "pts",
    ) -> go.Figure:
        """
        Renders a ranked horizontal bar chart of signed SHAP contributions.
        Green bars for positive contributions, red for negative.
        Sorted by absolute value (most impactful at the top).

        Args:
            summary_df: DataFrame with at least `label_col` and `value_col`.
            value_col:  Column name for the SHAP values (signed).
            label_col:  Column name for feature/component labels.
            title:      Chart title.
            unit:       Unit suffix for hover tooltip.

        Returns:
            Plotly Figure.
        """
        df = summary_df.copy()
        df = df.sort_values(value_col, key=abs, ascending=True)  # ascending for horizontal

        colors = ["#10B981" if v >= 0 else "#EF4444" for v in df[value_col]]
        text   = [
            f"+{v:.1f} {unit}" if v >= 0 else f"{v:.1f} {unit}"
            for v in df[value_col]
        ]

        fig = go.Figure(go.Bar(
            y            = df[label_col],
            x            = df[value_col],
            orientation  = "h",
            marker_color = colors,
            text         = text,
            textposition = "outside",
            hovertemplate = "%{y}<br>SHAP: %{x:.2f} " + unit + "<extra></extra>",
        ))

        # Zero reference line
        fig.add_vline(x=0, line_color="#6B7280", line_width=1.5, line_dash="dash")

        fig.update_layout(
            title         = title,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height        = max(300, len(df) * 55 + 80),
            showlegend    = False,
            margin        = dict(l=10, r=60, t=60, b=10),
            xaxis         = {"title": f"SHAP Value ({unit})", "gridcolor": "#1F2937", "zeroline": False},
            yaxis         = {"gridcolor": "#1F2937"},
        )
        return fig

    @staticmethod
    def plot_shap_contribution_heatmap(
        summary_df:   "pd.DataFrame",
        label_col:    str = "Feature",
        value_col:    str = "SHAP Value",
        subscore_col: str = "Sub-Score",
        title:        str = "Component Deep-Dive",
    ) -> go.Figure:
        """
        Renders a colour-coded horizontal bar heatmap showing sub-scores
        with SHAP contribution annotations.

        Useful for showing the full component breakdown in one glance.

        Args:
            summary_df:   DataFrame from HealthScoreResult.summary_data.
            label_col:    Column name for component labels.
            value_col:    Column name for signed SHAP values.
            subscore_col: Column name for sub-scores (0-100).
            title:        Chart title.

        Returns:
            Plotly Figure.
        """
        df     = summary_df.copy().sort_values(value_col, key=abs, ascending=False)
        labels = df[label_col].tolist()
        scores = df[subscore_col].tolist()
        shapvs = df[value_col].tolist()

        # Color: green gradient for high score, red for low
        bar_colors = []
        for s in scores:
            if s >= 80:
                bar_colors.append("#10B981")
            elif s >= 60:
                bar_colors.append("#3B82F6")
            elif s >= 40:
                bar_colors.append("#F59E0B")
            else:
                bar_colors.append("#EF4444")

        text_labels = [
            f"{s:.0f}/100  (SHAP: {'+' if v >= 0 else ''}{v:.1f} pts)"
            for s, v in zip(scores, shapvs)
        ]

        fig = go.Figure(go.Bar(
            y            = labels,
            x            = scores,
            orientation  = "h",
            marker_color = bar_colors,
            text         = text_labels,
            textposition = "inside",
            insidetextanchor = "middle",
            hovertemplate = (
                "<b>%{y}</b><br>"
                "Sub-Score: %{x:.1f}/100<br>"
                "<extra></extra>"
            ),
        ))

        # Add benchmark line at 80
        fig.add_vline(x=80, line_color="#10B981", line_width=1.5,
                      line_dash="dot", annotation_text="Target 80",
                      annotation_position="top right",
                      annotation_font_color="#10B981")
        fig.add_vline(x=50, line_color="#6B7280", line_width=1,
                      line_dash="dash", annotation_text="Baseline 50",
                      annotation_position="bottom right",
                      annotation_font_color="#6B7280")

        fig.update_layout(
            title         = title,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#F3F4F6", "family": "Outfit, Inter, sans-serif"},
            height        = max(320, len(df) * 60 + 80),
            showlegend    = False,
            margin        = dict(l=10, r=20, t=60, b=10),
            xaxis         = {"title": "Sub-Score (0–100)", "range": [0, 110],
                             "gridcolor": "#1F2937"},
            yaxis         = {"gridcolor": "#1F2937"},
        )
        return fig
