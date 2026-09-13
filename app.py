"""
Main Streamlit Application Entrypoint.
Renders the homepage/dashboard summary and configures the multi-page navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from html import escape as _he
from utils.session import render_sidebar_user_selector
from utils.splash_screen import render_startup_splash
from models.twin_engine import HealthScoreEngine
from utils.master_report import build_master_report

# Page Configuration
st.set_page_config(
    page_title="FinTwin AI — Financial Digital Twin",
    page_icon="assets/logos/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Animated startup splash screen — shown once per browser session, on first
# load of the Home page only. It never appears again when navigating between
# pages. Purely additive: does not affect any logic below.
_current_dir_for_splash = os.path.dirname(os.path.abspath(__file__))
render_startup_splash(_current_dir_for_splash)

# Render User Selector Sidebar
twin = render_sidebar_user_selector()

# Load official logo for Hero section
current_dir = os.path.dirname(os.path.abspath(__file__))

logo_path = os.path.join(current_dir, "assets", "logos", "3_icon_only.svg")
logo_src = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode('utf-8')
    logo_src = f"data:image/svg+xml;base64,{logo_base64}"

# Compute data for layout
if twin:
    # Compute active health score summary
    engine = HealthScoreEngine(twin)
    score_data = engine.compute_overall_health_score()
    score = score_data["overall_score"]
    grade = score_data["financial_grade"]
    
    kpis = [
        {"icon": '<i class="fa-solid fa-shield-halved" style="color:#4F8CFF; font-size:1.35rem;"></i>', "val": f"{score} / 100", "label": "Financial Health Score"},
        {"icon": '<i class="fa-solid fa-award" style="color:#22C55E; font-size:1.35rem;"></i>', "val": grade, "label": "Financial Grade"},
        {"icon": '<i class="fa-solid fa-money-bill-wave" style="color:#00D4FF; font-size:1.35rem;"></i>', "val": f"₹{twin.total_income:,.0f}", "label": "Monthly Net Income"},
        {"icon": '<i class="fa-solid fa-gem" style="color:#EC4899; font-size:1.35rem;"></i>', "val": f"₹{twin.net_worth:,.0f}", "label": "Net Worth"}
    ]
    from utils.avatar import render_avatar_html
    greeting_avatar = render_avatar_html(getattr(twin, "avatar_id", "avatar_01"), size=26)
    info_html = f'<div class="info-card success"><span class="info-icon">{greeting_avatar}</span><span class="info-text">Welcome back, <strong>{_he(twin.name)}</strong>! Your Financial Digital Twin is active.</span></div>'
else:
    kpis = [
        {"icon": '<i class="fa-solid fa-folder-open" style="color:#4F8CFF; font-size:1.35rem;"></i>', "val": "10", "label": "Financial Modules"},
        {"icon": '<i class="fa-solid fa-brain" style="color:#22C55E; font-size:1.35rem;"></i>', "val": "5", "label": "AI Models"},
        {"icon": '<i class="fa-solid fa-chart-line" style="color:#00D4FF; font-size:1.35rem;"></i>', "val": "6M", "label": "Financial Forecasting"},
        {"icon": '<i class="fa-solid fa-lightbulb" style="color:#EC4899; font-size:1.35rem;"></i>', "val": '<span style="font-size: 0.88rem; font-weight: 600; color: #F8FAFC;">Available after sign-in</span>', "label": "Personalized Insights"}
    ]
    info_html = '<div class="info-card"><span class="info-icon"><i class="fa-solid fa-lock" style="vertical-align:middle; color:#94A3B8; font-size:1.25rem;"></i></span><span class="info-text">Sign in or register via the top navigation to unlock personalized financial insights and AI predictions.</span></div>'

# Inject custom stylesheet and render the original clean dashboard layout
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
/* Custom app background styling */
.stApp {{
    background-color: #0B1220;
    background-image: radial-gradient(circle at 50% 0%, rgba(79, 140, 255, 0.08) 0%, transparent 60%),
                      radial-gradient(circle at 100% 100%, rgba(0, 212, 255, 0.04) 0%, transparent 40%);
    background-attachment: fixed;
}}

/* Stable background layout - prevent elements from flashing to opacity:0 on reruns */
.hero-container, .info-card, .kpi-grid, .module-section-title, .module-grid, .features-section-title, .features-grid {{
    opacity: 1;
}}

/* Hero Section */
.hero-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, rgba(31, 41, 55, 0.5) 0%, rgba(17, 24, 39, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    gap: 2rem;
    backdrop-filter: blur(10px);
}}
.hero-text {{
    flex: 1.3;
}}
.badge {{
    background: rgba(79, 140, 255, 0.12);
    color: #4F8CFF;
    border: 1px solid rgba(79, 140, 255, 0.25);
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    display: inline-block;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}}
.hero-title {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.03em;
    line-height: 1.2;
}}
.hero-subtitle {{
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
    font-size: 1.0rem;
    font-weight: 400;
    line-height: 1.6;
    margin: 0;
    max-width: 650px;
}}
.hero-graphics {{
    flex: 0.7;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.hero-logo-img {{
    animation: premiumLogoAnim 4s ease-in-out infinite;
    filter: drop-shadow(0 0 15px rgba(79, 140, 255, 0.25));
}}
@keyframes premiumLogoAnim {{
    0%, 100% {{
        transform: translateY(0) scale(1);
        filter: drop-shadow(0 0 15px rgba(79, 140, 255, 0.25));
    }}
    50% {{
        transform: translateY(-6px) scale(1.03);
        filter: drop-shadow(0 0 25px rgba(0, 212, 255, 0.45));
    }}
}}

/* Info Notification Cards */
.info-card {{
    background: rgba(79, 140, 255, 0.06);
    border: 1px solid rgba(79, 140, 255, 0.15);
    border-radius: 12px;
    padding: 0.9rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2rem;
    color: #F8FAFC;
    font-size: 0.9rem;
}}
.info-card.success {{
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.15);
    color: #F8FAFC;
}}
.info-icon {{
    font-size: 1.2rem;
}}

/* Quick Statistics Grid */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2.25rem;
}}
@media (max-width: 1080px) {{
    .kpi-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}
@media (max-width: 600px) {{
    .kpi-grid {{
        grid-template-columns: 1fr;
    }}
}}
.kpi-card {{
    background: #1F2937;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    min-height: 82px;
    box-sizing: border-box;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(79, 140, 255, 0.3);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}}
.kpi-icon-wrap {{
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 10px;
    background: rgba(79, 140, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}
.kpi-info {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 0;
    flex: 1;
}}
.kpi-value {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.015em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.kpi-label {{
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

/* Platform Modules Grid - Parallel 2 rows x 5 columns */
.module-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.9rem;
    margin-top: 1rem;
    margin-bottom: 2rem;
}}
@media (max-width: 1024px) {{
    .module-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}
@media (max-width: 600px) {{
    .module-grid {{
        grid-template-columns: 1fr;
    }}
}}
.module-card {{
    background: #1F2937;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.45rem;
    height: 100%;
    box-sizing: border-box;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}}
.module-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 20px -8px rgba(79, 140, 255, 0.22), 0 0 12px -3px rgba(79, 140, 255, 0.12);
    border-color: rgba(79, 140, 255, 0.35);
}}
.module-icon {{
    font-size: 1.5rem;
    margin-bottom: 0.15rem;
}}
.module-title {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.02rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 0;
    line-height: 1.25;
}}
.module-desc {{
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.4;
    margin: 0;
}}

/* Why Choose FinTwin AI Grid */
.features-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.9rem;
    margin-top: 1rem;
    margin-bottom: 2.5rem;
}}
.feature-card {{
    background: rgba(31, 41, 55, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.4rem;
    box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    transition: all 0.25s ease;
}}
.feature-card:hover {{
    background: rgba(31, 41, 55, 0.6);
    border-color: rgba(79, 140, 255, 0.15);
}}
.feature-icon {{
    font-size: 1.45rem;
    margin-bottom: 0.15rem;
}}
.feature-title {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 0;
}}
.feature-desc {{
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.4;
    margin: 0;
}}
</style>

<div class="hero-container">
    <div class="hero-text">
        <div class="badge">AI POWERED FINANCIAL INTELLIGENCE</div>
        <h1 class="hero-title">Welcome to FinTwin AI</h1>
        <p class="hero-subtitle">Build your AI-powered Financial Digital Twin to analyze your financial health, predict future outcomes, and receive personalized recommendations.</p>
    </div>
    <div class="hero-graphics">
        <img src="{logo_src}" class="hero-logo-img" width="130" height="130" style="border: none; background: transparent;" />
    </div>
</div>

{info_html}

<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon-wrap">{kpis[0]['icon']}</div>
        <div class="kpi-info">
            <div class="kpi-value">{kpis[0]['val']}</div>
            <div class="kpi-label">{kpis[0]['label']}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-wrap">{kpis[1]['icon']}</div>
        <div class="kpi-info">
            <div class="kpi-value">{kpis[1]['val']}</div>
            <div class="kpi-label">{kpis[1]['label']}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-wrap">{kpis[2]['icon']}</div>
        <div class="kpi-info">
            <div class="kpi-value">{kpis[2]['val']}</div>
            <div class="kpi-label">{kpis[2]['label']}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-wrap">{kpis[3]['icon']}</div>
        <div class="kpi-info">
            <div class="kpi-value">{kpis[3]['val']}</div>
            <div class="kpi-label">{kpis[3]['label']}</div>
        </div>
    </div>
</div>

<div class="module-section-title" style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <h3 style="color: #F8FAFC; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em;">Platform Modules</h3>
</div>

<div class="module-grid">
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-circle-user" style="color:#4F8CFF; font-size: 2.2rem;"></i></div>
        <div class="module-title">Digital Twin</div>
        <div class="module-desc">View your complete AI-powered financial profile.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-heart-pulse" style="color:#22C55E; font-size: 2.2rem;"></i></div>
        <div class="module-title">Financial Health</div>
        <div class="module-desc">Monitor overall health score and metrics.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-chart-pie" style="color:#00D4FF; font-size: 2.2rem;"></i></div>
        <div class="module-title">Behavior Analysis</div>
        <div class="module-desc">Analyze spending habits and income patterns.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-fingerprint" style="color:#A855F7; font-size: 2.2rem;"></i></div>
        <div class="module-title">Financial Personality</div>
        <div class="module-desc">Discover your AI-driven financial archetype.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-chart-line" style="color:#EC4899; font-size: 2.2rem;"></i></div>
        <div class="module-title">Forecasting</div>
        <div class="module-desc">Predict future savings, expenses, and net worth.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-sliders" style="color:#EF4444; font-size: 2.2rem;"></i></div>
        <div class="module-title">Scenario Simulator</div>
        <div class="module-desc">Test financial scenarios and compare outcomes.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-landmark" style="color:#10B981; font-size: 2.2rem;"></i></div>
        <div class="module-title">Tax Intelligence</div>
        <div class="module-desc">Get tax-saving insights and compare regimes.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-comments-dollar" style="color:#F59E0B; font-size: 2.2rem;"></i></div>
        <div class="module-title">AI Coach</div>
        <div class="module-desc">Personalized financial guidance powered by AI.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-bullseye" style="color:#6366F1; font-size: 2.2rem;"></i></div>
        <div class="module-title">Goal Planner</div>
        <div class="module-desc">Set and track milestones for your life ambitions.</div>
    </div>
    <div class="module-card">
        <div class="module-icon"><i class="fa-solid fa-magnifying-glass-chart" style="color:#38BDF8; font-size: 2.2rem;"></i></div>
        <div class="module-title">Explainable AI</div>
        <div class="module-desc">Understand why AI generated each prediction.</div>
    </div>
</div>

<div class="features-section-title" style="margin-top: 3rem; margin-bottom: 0.5rem;">
    <h3 style="color: #F8FAFC; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em;">Why Choose FinTwin AI</h3>
</div>

<div class="features-grid">
    <div class="feature-card">
        <div class="feature-icon"><i class="fa-solid fa-robot" style="color:#4F8CFF; font-size: 1.8rem;"></i></div>
        <div class="feature-title">AI Powered Analysis</div>
        <div class="feature-desc">Deep neural networks compute accurate financial grades and stress test reserves.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon"><i class="fa-solid fa-chart-line" style="color:#22C55E; font-size: 1.8rem;"></i></div>
        <div class="feature-title">Financial Forecasting</div>
        <div class="feature-desc">Pretrained ML models predict your long-term expense and asset trends.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon"><i class="fa-solid fa-chart-pie" style="color:#00D4FF; font-size: 1.8rem;"></i></div>
        <div class="feature-title">Behavior Intelligence</div>
        <div class="feature-desc">Identify anomalies and detect optimization vectors inside spending tracks.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon"><i class="fa-solid fa-brain" style="color:#EC4899; font-size: 1.8rem;"></i></div>
        <div class="feature-title">Explainable AI</div>
        <div class="feature-desc">SHAP explanations provide full transparency behind forecast logic.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Consolidated Master PDF Report ─────────────────────────────────────────────
if twin:
    st.markdown("---")
    st.markdown("### Download Your Complete Financial Report")
    st.caption(
        "One consolidated, professional PDF covering your Digital Twin profile, "
        "Financial Health Score, Income/Expense/Investment analysis, Net Worth, "
        "Tax Intelligence, AI Coach recommendations, Personality, Behaviour, "
        "Forecasting, and Explainable AI \u2014 all in a single document."
    )
    rep_col, _ = st.columns([1, 2])
    with rep_col:
        if st.button("Generate Full PDF Report", icon=":material/picture_as_pdf:", use_container_width=True, type="primary"):
            with st.spinner("Compiling your complete financial report..."):
                try:
                    st.session_state["_master_pdf_bytes"] = build_master_report(twin)
                    st.session_state["_master_pdf_uid"] = twin.user_id
                except Exception as e:
                    st.error(f"PDF report could not be generated: {e}")

        if st.session_state.get("_master_pdf_bytes") and st.session_state.get("_master_pdf_uid") == twin.user_id:
            from utils.pdf_report import FinTwinPDFReport as _FTR
            _fname = _FTR(report_title="x", user_id=twin.user_id).suggested_filename(
                "complete_report_active_user"
            )
            st.download_button(
                label="Download PDF",
                icon=":material/download:",
                data=st.session_state["_master_pdf_bytes"],
                file_name=_fname,
                mime="application/pdf",
                use_container_width=True,
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "MSc Data Science Project &copy; 2026. Built with Python & Streamlit."
    "</div>",
    unsafe_allow_html=True
)
