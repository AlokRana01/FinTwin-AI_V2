"""
Session & Authentication UI

Replaces the old sidebar logic with a premium fintech navigation system
featuring custom CSS, grouped links, active indicators, financial status, and modal authentication.
"""
import time
import streamlit as st
from database.connection import init_db, get_db_cursor
from database.db_manager import DBManager
from models.twin_engine import FinancialDigitalTwin, HealthScoreEngine
from utils import auth
from utils.chatbot import render_chatbot
from utils.security import (
    session_is_expired,
    session_touch,
    validate_name,
    validate_email,
    MAX_NAME_LEN,
    MAX_EMAIL_LEN,
)
from typing import Optional
from html import escape as _he
import logging

logger = logging.getLogger(__name__)

ACTIVE_TWIN_KEY = "ftw_active_twin"


def invalidate_session_twin() -> None:
    """Invalidates the session-scoped FinancialDigitalTwin cache."""
    st.session_state.pop(ACTIVE_TWIN_KEY, None)


def _init_db_once():
    if not st.session_state.get("_db_initialized"):
        try:
            init_db()
            st.session_state["_db_initialized"] = True
        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            st.error("A database error occurred. Please try again later.")


@st.cache_data
def _get_font_css() -> str:
    """Encodes local WOFF2 font files as base64 @font-face rules for Space Grotesk and Inter."""
    import base64
    from pathlib import Path
    fonts_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"

    font_defs = [
        ("Space Grotesk", 400, "normal", fonts_dir / "SpaceGrotesk" / "SpaceGrotesk-Regular.woff2"),
        ("Space Grotesk", 500, "normal", fonts_dir / "SpaceGrotesk" / "SpaceGrotesk-Medium.woff2"),
        ("Space Grotesk", 600, "normal", fonts_dir / "SpaceGrotesk" / "SpaceGrotesk-SemiBold.woff2"),
        ("Space Grotesk", 700, "normal", fonts_dir / "SpaceGrotesk" / "SpaceGrotesk-Bold.woff2"),
        ("Inter", 400, "normal", fonts_dir / "Inter" / "Inter-Regular.woff2"),
        ("Inter", 500, "normal", fonts_dir / "Inter" / "Inter-Medium.woff2"),
        ("Inter", 600, "normal", fonts_dir / "Inter" / "Inter-SemiBold.woff2"),
        ("Inter", 700, "normal", fonts_dir / "Inter" / "Inter-Bold.woff2"),
    ]

    css_parts = []
    for family, weight, style, path in font_defs:
        if path.exists():
            try:
                b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
                css_parts.append(
                    f"@font-face {{\n"
                    f"  font-family: '{family}';\n"
                    f"  font-style: {style};\n"
                    f"  font-weight: {weight};\n"
                    f"  font-display: swap;\n"
                    f"  src: url('data:font/woff2;charset=utf-8;base64,{b64}') format('woff2');\n"
                    f"}}"
                )
            except Exception as e:
                logger.warning(f"Could not load font {path}: {e}")

    return "\n".join(css_parts)


def load_css():
    """Injects premium CSS for the fintech top navigation redesign and global spacing system."""
    font_face_css = _get_font_css()
    if font_face_css:
        st.markdown(f"<style>\n{font_face_css}\n</style>", unsafe_allow_html=True)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');

        :root {
            --space-2xs: 4px;
            --space-xs: 8px;
            --space-sm: 12px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;
            --space-2xl: 48px;
            --space-3xl: 64px;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --radius-xl: 18px;
            --card-bg: #1F2937;
            --card-border: rgba(255, 255, 255, 0.06);
        }

        /* Completely eliminate Streamlit native sidebar & expand/collapse controls */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        button[data-testid="stSidebarCollapseButton"],
        div[data-testid="stSidebarNav"],
        div[data-testid="stSidebarUserContent"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"],
        div[data-testid="stSidebarHeader"],
        [data-testid="stSidebarContent"],
        header[data-testid="stHeader"] button[data-testid="collapsedControl"] {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            opacity: 0 !important;
            overflow: hidden !important;
        }

        /* Completely eliminate Streamlit native header and its min-height */
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
            position: absolute !important;
            top: -9999px !important;
        }

        /* Float the topbar container out of normal document flow so it adds 0px gap */
        div[data-testid="stElementContainer"]:has(.fintwin-topbar) {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            overflow: visible !important;
            z-index: 99999 !important;
        }

        /* Completely collapse all ghost/empty/style-only Streamlit containers */
        div[data-testid="stElementContainer"]:has(style:only-child),
        div[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > style:only-child),
        div[data-testid="stElementContainer"]:has(> [data-testid="stMarkdownContainer"] > style:only-child),
        div[data-testid="stElementContainer"]:has(style):not(:has(.hero-container)):not(:has(h1)):not(:has(h2)):not(:has(h3)):not(:has(p)):not(:has(button)):not(:has(div[data-testid="stBaseButton-secondary"])):not(:has(.info-card)):not(:has(.kpi-grid)):not(:has(.module-grid)):not(:has(.features-grid)):not(:has([class*="-header-box"])):not(:has(.fintwin-topbar)),
        div[data-testid="stElementContainer"]:empty,
        div.stMarkdown:empty {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
        }

        /* Global Page Container Standardization (Crisp 14px spacing directly under 68px top header) */
        .main .block-container,
        [data-testid="stMainBlockContainer"],
        div[data-testid="stAppViewBlockContainer"] {
            max-width: 1680px !important;
            width: 100% !important;
            padding-top: 84px !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin: 0 auto !important;
            box-sizing: border-box !important;
        }
        @media (max-width: 768px) {
            .main .block-container,
            [data-testid="stMainBlockContainer"],
            div[data-testid="stAppViewBlockContainer"] {
                padding-left: 1.15rem !important;
                padding-right: 1.15rem !important;
                padding-top: 78px !important;
                padding-bottom: 3rem !important;
            }
        }

        /* Global Font Override */
        html, body, [data-testid="stAppViewContainer"], .stApp, p, li, label, input, select, textarea {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* App Background Styling (Matching Dashboard across all pages) */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: #0B1220 !important;
            background-image: radial-gradient(circle at 50% 0%, rgba(79, 140, 255, 0.08) 0%, transparent 60%),
                              radial-gradient(circle at 100% 100%, rgba(0, 212, 255, 0.04) 0%, transparent 40%) !important;
            background-attachment: fixed !important;
        }

        /* Restore Material Icons Font Family for Streamlit native icons */
        .material-icons,
        [data-testid="stIcon"],
        [class*="Icon"],
        [class*="icon"] {
            font-family: 'Material Icons', 'Material Symbols Outlined', 'Material Symbols Rounded', sans-serif !important;
        }

        /* Titles and Headers */
        h1, [data-testid="stMarkdownContainer"] h1, .hero-title, .twin-title {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 2.35rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            line-height: 1.2 !important;
            color: #F8FAFC !important;
            margin-bottom: 0.35rem !important;
        }
        
        h2, [data-testid="stMarkdownContainer"] h2 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.25 !important;
            color: #F8FAFC !important;
            margin-top: 1.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h3, [data-testid="stMarkdownContainer"] h3, .module-title, .feature-title, .card-title {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
            line-height: 1.3 !important;
            color: #F8FAFC !important;
            margin-bottom: 0.35rem !important;
        }
        
        h4, h5, h6, [data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] h5, [data-testid="stMarkdownContainer"] h6 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            color: #E2E8F0 !important;
        }

        /* Financial Figures & Metrics */
        [data-testid="stMetricValue"], .kpi-value, .financial-number, .metric-value, .score-value {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em !important;
            color: #F8FAFC !important;
        }

        /* Metric Labels */
        [data-testid="stMetricLabel"], .kpi-label, .fintwin-health-label {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            color: #94A3B8 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        /* Widget inputs, sliders, select boxes */
        [data-testid="stWidgetLabel"] p, label, .stWidgetLabel {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: #CBD5E1 !important;
            margin-bottom: 0.3rem !important;
        }
        
        div[data-baseweb="select"] * {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.92rem !important;
        }

        /* Tables & DataFrames */
        div[data-testid="stDataFrame"], [data-testid="stTable"] {
            font-family: 'Inter', sans-serif !important;
        }
        div[data-testid="stDataFrame"] th, [data-testid="stTable"] th {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
        }
        div[data-testid="stDataFrame"] td, [data-testid="stTable"] td {
            font-family: 'Inter', sans-serif !important;
            font-weight: 400 !important;
        }

        /* Native Streamlit Buttons */
        button[data-testid="stBaseButton-secondary"], 
        button[data-testid="stBaseButton-primary"],
        .stButton > button {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            border-radius: 8px !important;
            padding: 0.55rem 1.25rem !important;
            min-height: 42px !important;
        }

        /* Standardize Tabs */
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 8px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
            margin-bottom: 1.5rem !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            border-radius: 6px 6px 0 0 !important;
        }

        /* Standardize Expanders */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 10px !important;
            background: #1F2937 !important;
            margin-bottom: 0.75rem !important;
        }

        /* ======================================================= */
        /* TOP NAVIGATION HEADER & FINTECH DROPDOWN STYLING        */
        /* ======================================================= */

        .fintwin-topbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            width: 100%;
            height: 68px;
            z-index: 99999;
            background: rgba(11, 18, 32, 0.94);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        }
        .fintwin-topbar-inner {
            max-width: 1680px;
            height: 100%;
            margin: 0 auto;
            padding: 0 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            box-sizing: border-box;
        }
        @media (max-width: 768px) {
            .fintwin-topbar-inner {
                padding: 0 1.15rem;
            }
        }

        /* Brand Area */
        .fintwin-brand-link {
            display: inline-flex;
            align-items: center;
            text-decoration: none !important;
            flex-shrink: 0;
            outline: none;
            transition: opacity 0.2s ease;
        }
        .fintwin-brand-link:hover {
            opacity: 0.92;
        }
        .fintwin-brand-link svg {
            display: block;
            height: 44px;
            width: auto;
            max-width: 180px;
            transition: transform 0.2s ease;
        }
        .fintwin-brand-link:hover svg {
            transform: scale(1.02);
        }
        @media (max-width: 480px) {
            .fintwin-brand-link svg {
                height: 38px;
                max-width: 155px;
            }
        }

        /* Center: Horizontal Navigation Menu */
        .fintwin-nav-center {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            flex: 1;
            justify-content: center;
        }

        /* Nav Trigger Item */
        .fintwin-nav-item {
            position: relative;
            display: inline-flex;
            align-items: center;
        }
        .fintwin-nav-link {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 0.48rem 0.85rem;
            color: #94A3B8;
            text-decoration: none !important;
            font-family: 'Manrope', sans-serif;
            font-size: 0.88rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            border-radius: 8px;
            background: transparent;
            border: 1px solid transparent;
            transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
            cursor: pointer;
            white-space: nowrap;
            user-select: none;
            line-height: 1.2;
        }
        .fintwin-nav-link:hover {
            color: #F8FAFC;
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.08);
        }
        .fintwin-nav-link.active {
            color: #4F8CFF;
            background: rgba(79, 140, 255, 0.12);
            border-color: rgba(79, 140, 255, 0.28);
            box-shadow: 0 0 12px -2px rgba(79, 140, 255, 0.2);
        }
        .fintwin-nav-link .nav-chevron {
            font-size: 0.68rem;
            transition: transform 0.2s ease;
            opacity: 0.7;
            margin-left: 2px;
        }
        .fintwin-nav-item:hover .fintwin-nav-link .nav-chevron,
        .fintwin-nav-item:focus-within .fintwin-nav-link .nav-chevron {
            transform: rotate(180deg);
            opacity: 1;
        }

        /* Fintech Dropdown Menu Overlay */
        .fintwin-dropdown {
            position: absolute;
            top: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%) translateY(-6px);
            width: 320px;
            background: #111827;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 12px;
            padding: 0.6rem;
            box-shadow: 0 20px 40px -8px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.04);
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            transition: opacity 0.18s cubic-bezier(0.16, 1, 0.3, 1), transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), visibility 0.18s;
            z-index: 100000;
        }
        .fintwin-dropdown::before {
            content: '';
            position: absolute;
            top: -12px;
            left: 0;
            right: 0;
            height: 12px;
        }
        .fintwin-nav-item:hover .fintwin-dropdown,
        .fintwin-nav-item:focus-within .fintwin-dropdown {
            opacity: 1;
            visibility: visible;
            pointer-events: auto;
            transform: translateX(-50%) translateY(0);
        }

        /* Dropdown Header */
        .fintwin-dropdown-header {
            font-family: 'Manrope', sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B;
            padding: 0.4rem 0.65rem 0.45rem 0.65rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 0.4rem;
        }

        /* Dropdown Item Link */
        .fintwin-dropdown-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 0.55rem 0.7rem;
            border-radius: 9px;
            text-decoration: none !important;
            transition: all 0.16s ease;
            border: 1px solid transparent;
            margin-bottom: 2px;
        }
        .fintwin-dropdown-item:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.07);
        }
        .fintwin-dropdown-item.active-item {
            background: rgba(79, 140, 255, 0.12);
            border-color: rgba(79, 140, 255, 0.28);
        }
        .fintwin-item-icon {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            color: #94A3B8;
            transition: all 0.16s ease;
            margin-top: 1px;
            flex-shrink: 0;
        }
        .fintwin-dropdown-item:hover .fintwin-item-icon {
            background: rgba(79, 140, 255, 0.16);
            color: #4F8CFF;
        }
        .fintwin-dropdown-item.active-item .fintwin-item-icon {
            background: rgba(79, 140, 255, 0.24);
            color: #4F8CFF;
        }
        .fintwin-item-text {
            flex: 1;
            min-width: 0;
        }
        .fintwin-item-title {
            font-family: 'Manrope', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.25;
            margin-bottom: 2px;
        }
        .fintwin-dropdown-item.active-item .fintwin-item-title {
            color: #4F8CFF;
        }
        .fintwin-item-desc {
            font-family: 'Inter', sans-serif;
            font-size: 0.74rem;
            color: #94A3B8;
            line-height: 1.35;
        }

        /* Right Side: Auth / User Area */
        .fintwin-nav-right {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            flex-shrink: 0;
        }

        /* Auth Buttons (Unauthenticated) */
        .fintwin-btn-auth {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.45rem 0.95rem;
            font-family: 'Manrope', sans-serif;
            font-size: 0.84rem;
            font-weight: 700;
            border-radius: 8px;
            text-decoration: none !important;
            transition: all 0.18s ease;
            cursor: pointer;
            white-space: nowrap;
            height: 36px;
            box-sizing: border-box;
        }
        .fintwin-btn-auth.primary {
            background: #4F8CFF;
            color: #FFFFFF !important;
            border: 1px solid #4F8CFF;
        }
        .fintwin-btn-auth.primary:hover {
            background: #3B82F6;
            border-color: #3B82F6;
            box-shadow: 0 4px 12px rgba(79, 140, 255, 0.35);
        }
        .fintwin-btn-auth.outline {
            background: rgba(255, 255, 255, 0.04);
            color: #F8FAFC !important;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .fintwin-btn-auth.outline:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.22);
        }

        /* Fixed Streamlit topbar auth button (Zero page reload) */
        .st-key-fintwin_top_auth_btn {
            position: fixed !important;
            top: 15px !important;
            right: max(2rem, calc((100vw - 1680px) / 2 + 2rem)) !important;
            z-index: 100001 !important;
            margin: 0 !important;
            padding: 0 !important;
            width: auto !important;
        }
        .st-key-fintwin_top_auth_btn button {
            height: 38px !important;
            min-height: 38px !important;
            padding: 0 1.25rem !important;
            border-radius: 8px !important;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            color: #FFFFFF !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            cursor: pointer !important;
            white-space: nowrap !important;
        }
        .st-key-fintwin_top_auth_btn button:hover {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        .st-key-fintwin_top_auth_btn button:active {
            transform: translateY(0) !important;
        }
        @media (max-width: 1024px) {
            .st-key-fintwin_top_auth_btn {
                right: 4.6rem !important;
                top: 15px !important;
            }
            .st-key-fintwin_top_auth_btn button {
                height: 36px !important;
                min-height: 36px !important;
                padding: 0 0.85rem !important;
                font-size: 0.80rem !important;
            }
        }
        @media (max-width: 768px) {
            .st-key-fintwin_top_auth_btn {
                right: 4.1rem !important;
                top: 15px !important;
            }
        }
        button.fintwin-mobile-link {
            background: transparent;
            border: 1px solid transparent;
            width: 100%;
            text-align: left;
            cursor: pointer;
        }

        /* Authenticated User Pill Trigger */
        .fintwin-user-trigger {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            padding: 0.35rem 0.75rem 0.35rem 0.45rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 99px;
            text-decoration: none !important;
            cursor: pointer;
            transition: all 0.2s ease;
            height: 38px;
            box-sizing: border-box;
        }
        .fintwin-user-trigger:hover,
        .fintwin-nav-item.user-item:hover .fintwin-user-trigger {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(79, 140, 255, 0.3);
        }
        .fintwin-user-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            border: 1.5px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            color: #FFFFFF;
            flex-shrink: 0;
            overflow: hidden;
        }
        .fintwin-user-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%;
            display: block;
        }
        .fintwin-user-name {
            font-family: 'Manrope', sans-serif;
            font-size: 0.84rem;
            font-weight: 700;
            color: #F8FAFC;
            max-width: 120px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Authenticated User Dropdown */
        .fintwin-dropdown-user {
            right: 0 !important;
            left: auto !important;
            transform: translateY(-6px) !important;
            width: 300px !important;
        }
        .fintwin-nav-item.user-item:hover .fintwin-dropdown-user,
        .fintwin-nav-item.user-item:focus-within .fintwin-dropdown-user {
            transform: translateY(0) !important;
        }

        .fintwin-user-card {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.55rem 0.65rem 0.65rem 0.65rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 0.5rem;
        }
        .fintwin-user-card-avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            border: 1.5px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 700;
            color: #FFFFFF;
            flex-shrink: 0;
            overflow: hidden;
        }
        .fintwin-user-card-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%;
            display: block;
        }
        .fintwin-user-card-meta {
            flex: 1;
            min-width: 0;
        }
        .fintwin-user-card-name {
            font-family: 'Manrope', sans-serif;
            font-size: 0.90rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .fintwin-user-card-tag {
            font-size: 0.72rem;
            color: #10B981;
            font-weight: 600;
            margin-top: 2px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        /* User Health Status Widget in Dropdown */
        .fintwin-user-health-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 9px;
            padding: 0.65rem 0.75rem;
            margin-bottom: 0.5rem;
        }
        .fintwin-health-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.4rem;
        }
        .fintwin-health-label {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
        }
        .fintwin-health-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
        }
        .fintwin-health-score-val {
            font-family: 'Manrope', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .fintwin-health-bar {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 99px;
            overflow: hidden;
        }
        .fintwin-health-fill {
            height: 100%;
            border-radius: 99px;
        }

        /* Dropdown Divider */
        .fintwin-dropdown-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.06);
            margin: 0.4rem 0;
        }

        /* Mobile Responsive Navigation Drawer */
        .fintwin-mobile-nav {
            display: none;
        }
        @media (max-width: 1024px) {
            .fintwin-nav-center {
                display: none !important;
            }
            .fintwin-mobile-nav {
                display: block !important;
            }
        }

        details.fintwin-mobile-menu {
            position: relative;
        }
        summary.fintwin-mobile-toggle {
            list-style: none;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #F8FAFC;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            outline: none;
        }
        summary.fintwin-mobile-toggle::-webkit-details-marker {
            display: none;
        }
        summary.fintwin-mobile-toggle:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(79, 140, 255, 0.3);
        }
        .fintwin-mobile-drawer {
            position: fixed;
            top: 68px;
            left: 0;
            right: 0;
            width: 100vw;
            max-height: calc(100vh - 68px);
            overflow-y: auto;
            background: #0B1220;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.25rem 1.5rem 2.5rem 1.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8);
            z-index: 99998;
            box-sizing: border-box;
        }
        .fintwin-mobile-section {
            margin-bottom: 1.25rem;
        }
        .fintwin-mobile-section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #64748B;
            margin-bottom: 0.5rem;
            padding-left: 0.25rem;
        }
        .fintwin-mobile-link {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.55rem 0.75rem;
            border-radius: 9px;
            text-decoration: none !important;
            color: #E2E8F0;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.92rem;
            font-weight: 500;
            border: 1px solid transparent;
            margin-bottom: 4px;
            transition: all 0.15s ease;
        }
        .fintwin-mobile-link:hover {
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.08);
        }
        .fintwin-mobile-link.active-item {
            color: #FFFFFF;
            background: rgba(79, 140, 255, 0.14);
            border-color: rgba(79, 140, 255, 0.32);
        }
        .fintwin-mobile-link i {
            width: 32px;
            height: 32px;
            min-width: 32px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.88rem;
            color: #FFFFFF !important;
            transition: all 0.15s ease;
            box-sizing: border-box;
            flex-shrink: 0;
        }
        .fintwin-mobile-link:hover i {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.20);
            color: #FFFFFF !important;
        }
        .fintwin-mobile-link.active-item i {
            background: rgba(79, 140, 255, 0.28);
            border-color: rgba(79, 140, 255, 0.45);
            color: #FFFFFF !important;
        }
        .fintwin-mobile-link.logout-item i {
            background: rgba(239, 68, 68, 0.14) !important;
            border-color: rgba(239, 68, 68, 0.28) !important;
            color: #F87171 !important;
        }

        /* Streamlit Toast Notification Popup */
        div[data-testid="stToast"] {
            background-color: #111827 !important;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, #111827 100%) !important;
            border: 1px solid rgba(16, 185, 129, 0.4) !important;
            border-left: 4px solid #10B981 !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(16, 185, 129, 0.2) !important;
        }

        div[data-testid="stToast"] [data-testid="stIcon"],
        div[data-testid="stToast"] .material-symbols-rounded,
        div[data-testid="stToast"] [class*="Icon"] {
            color: #10B981 !important;
            font-size: 1.35rem !important;
        }

        div[data-testid="stToast"] [data-testid="stMarkdownContainer"] p {
            color: #F8FAFC !important;
            font-weight: 500 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Try to use st.dialog if available, fallback to experimental
try:
    dialog_decorator = st.dialog
except AttributeError:
    try:
        dialog_decorator = st.experimental_dialog
    except AttributeError:
        # Dummy decorator if neither exists (fallback mechanism)
        def dialog_decorator(title):
            def decorator(func):
                return func
            return decorator


@dialog_decorator("Reset Your Password")
def reset_password_modal(raw_token: str):
    """Modal dialog for resetting password via secure email link."""
    st.write("Enter a strong new password for your FinTwin AI account.")
    is_valid, msg, email = auth.validate_reset_token(raw_token)
    if not is_valid:
        st.error(msg)
        if st.button("Close", use_container_width=True):
            if "reset_password_token" in st.query_params:
                del st.query_params["reset_password_token"]
            st.rerun()
        return

    if email:
        st.caption(f"Account: **{email}**")

    with st.form("reset_password_form"):
        new_pw = st.text_input("New Password", type="password", max_chars=64)
        confirm_pw = st.text_input("Confirm New Password", type="password", max_chars=64)
        st.caption("Must be at least 8 characters, with uppercase, lowercase, and a number or symbol.")
        submitted = st.form_submit_button("Reset Password", use_container_width=True, type="primary")
        if submitted:
            success, reset_msg = auth.reset_password_with_token(raw_token, new_pw, confirm_pw)
            if success:
                st.success(reset_msg)
                if "reset_password_token" in st.query_params:
                    del st.query_params["reset_password_token"]
                st.info("You can now close this window and log in with your new password.")
            else:
                st.error(reset_msg)


@dialog_decorator("Welcome to FinTwin AI")
def login_modal():
    st.write("Log in to your own Financial Twin, or create a free account.")

    # Global feedback from previous actions (e.g. email verified)
    if "_auth_success_msg" in st.session_state:
        st.success(st.session_state.pop("_auth_success_msg"))

    tab_login, tab_register, tab_forgot = st.tabs([
        ":material/login: Log In",
        ":material/person_add: Register",
        ":material/lock_reset: Forgot Password",
    ])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email Address", key="login_email", max_chars=MAX_EMAIL_LEN)
            password = st.text_input("Password", type="password", key="login_password", max_chars=64)
            submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")
            if submitted:
                ok, status_code, user = auth.login_user(email, password)
                if ok:
                    st.session_state["user_id"] = user["user_id"]
                    st.query_params["uid"] = user["user_id"]
                    session_touch(st.session_state)
                    st.session_state["_auth_success_msg"] = f"Welcome back, {user.get('name', 'User')}!"
                    st.rerun()
                else:
                    if status_code == "UNVERIFIED_EMAIL":
                        st.session_state["_unverified_email"] = email
                        st.error(user)
                    else:
                        st.error(user)

        if st.session_state.get("_unverified_email"):
            unverified = st.session_state["_unverified_email"]
            st.markdown("---")
            st.markdown(
                f"<div style='font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem;'>"
                f"Need a new activation link for <strong>{unverified}</strong>?</div>",
                unsafe_allow_html=True,
            )
            if st.button("Resend Verification Email", use_container_width=True, type="secondary"):
                resend_ok, resend_msg = auth.resend_verification_email(unverified)
                if resend_ok:
                    st.success(resend_msg)
                else:
                    st.error(resend_msg)

    with tab_register:
        with st.form("register_form"):
            r_name = st.text_input("Full Name", key="register_name", max_chars=MAX_NAME_LEN)
            r_email = st.text_input("Email Address", key="register_email", max_chars=MAX_EMAIL_LEN)
            r_age = st.number_input("Age", min_value=18, max_value=100, value=25, key="register_age")
            r_password = st.text_input(
                "Password",
                type="password",
                help="At least 8 characters, with uppercase, lowercase, and a number or symbol.",
                key="register_password",
                max_chars=64,
            )
            r_confirm = st.text_input(
                "Confirm Password",
                type="password", key="register_confirm_password", max_chars=64
            )

            from utils.avatar import render_avatar_selector
            r_avatar = render_avatar_selector(
                key="register_avatar_selector",
                label="Choose your profile avatar",
            )

            submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if submitted:
                if not r_avatar:
                    st.error("Please select a profile avatar to continue.")
                else:
                    ok, status_code, result = auth.register_user(
                        email=r_email,
                        password=r_password,
                        name=r_name,
                        age=r_age,
                        confirm_password=r_confirm,
                        avatar_id=r_avatar,
                    )
                if ok:
                    if status_code == "SUCCESS":
                        st.success(
                            f"Account created successfully for {r_name}.\n\n"
                            f"We have sent a verification email to **{r_email}** with an activation link.\n\n"
                            "Please click the link in your email to verify your account, then switch to the **Log In** tab to access your Digital Twin."
                        )
                    else:
                        st.warning(
                            "Your account was created successfully, but we couldn't send the verification email. "
                            "Please try logging in and requesting a new verification link."
                        )
                else:
                    st.error(result)

    with tab_forgot:
        st.write("Enter your registered email address and we'll send you a secure password reset link.")
        with st.form("forgot_password_form"):
            f_email = st.text_input("Registered Email", key="forgot_email", max_chars=MAX_EMAIL_LEN)
            f_submitted = st.form_submit_button("Send Password Reset Link", use_container_width=True, type="primary")
            if f_submitted:
                ok, msg = auth.request_password_reset(f_email)
                if ok:
                    st.success(
                        "If an account exists for this email address, a password reset link has been sent to your inbox. "
                        "The link will expire in 30 minutes."
                    )
                else:
                    st.error(msg)


def get_current_page_name() -> str:
    """Inspects call frames to determine the currently executing Streamlit page filename."""
    try:
        import inspect
        import os
        for frame_info in inspect.stack():
            fname = os.path.basename(frame_info.filename)
            if fname == "app.py":
                return "app.py"
            if fname.endswith(".py") and any(fname.startswith(f"{i:02d}_") for i in range(1, 15)):
                return fname
    except Exception:
        pass
    return "app.py"


def _clean_html(html_str: str) -> str:
    """Removes leading indentation and blank lines so CommonMark never treats HTML as code blocks."""
    return "".join(line.strip() for line in html_str.splitlines() if line.strip())


def render_top_navigation() -> Optional[FinancialDigitalTwin]:
    """
    Renders the modern fintech horizontal top navigation header,
    dropdown menus, user profile controls, authentication modal,
    and returns the active account's FinancialDigitalTwin.
    """
    _init_db_once()
    load_css()

    # ── Handle Global Auth Success Feedback ──────────────────────────────
    if "_auth_success_msg" in st.session_state:
        st.toast(st.session_state.pop("_auth_success_msg"), icon=":material/check_circle:")

    # ── Handle Query Parameter Actions (Email Verification & Password Reset) ──
    params = st.query_params
    if "verify_email_token" in params:
        raw_token = params.get("verify_email_token")
        ok, msg = auth.verify_email_token(raw_token)
        if ok:
            st.toast("Email verified! Please log in to your account.", icon=":material/check_circle:")
            st.session_state["_auth_success_msg"] = msg
        else:
            st.toast(f"Verification failed: {msg}", icon=":material/error:")
        del st.query_params["verify_email_token"]
        st.rerun()

    if "reset_password_token" in params:
        raw_token = params.get("reset_password_token")
        reset_password_modal(raw_token)

    # ── Handle Top Nav Action Triggers (Login / Register / Logout) ────────
    action = st.query_params.get("action")
    if action == "logout":
        st.session_state.pop("user_id", None)
        st.session_state.pop(ACTIVE_TWIN_KEY, None)
        st.session_state.pop("_session_last_active", None)
        st.session_state.pop("_master_pdf_bytes", None)
        st.session_state.pop("_master_pdf_uid", None)
        st.session_state.pop("_reg_success_data", None)
        st.session_state.pop("_unverified_email", None)
        st.session_state.pop("ftw_cached_usage", None)
        st.session_state.pop("ftw_chat_history", None)
        st.session_state.pop("ftw_chat_open", None)
        if "uid" in st.query_params:
            del st.query_params["uid"]
        del st.query_params["action"]
        st.rerun()

    if action in ("login", "register"):
        del st.query_params["action"]
        login_modal()

    if "nav" in st.query_params:
        del st.query_params["nav"]

    # ── Session restoration & timeout enforcement ────────────────────────
    if "user_id" not in st.session_state:
        candidate_uid = st.query_params.get("uid")
        if candidate_uid and DBManager.get_user_profile(candidate_uid):
            st.session_state["user_id"] = candidate_uid
            session_touch(st.session_state)

    twin: Optional[FinancialDigitalTwin] = None
    if "user_id" in st.session_state:
        if session_is_expired(st.session_state):
            del st.session_state["user_id"]
            st.session_state.pop(ACTIVE_TWIN_KEY, None)
            st.session_state.pop("_session_last_active", None)
            if "uid" in st.query_params:
                del st.query_params["uid"]
            st.warning("Your session expired due to inactivity. Please log in again.")
            st.rerun()
        else:
            session_touch(st.session_state)
            user_id = st.session_state["user_id"]
            cached_twin = st.session_state.get(ACTIVE_TWIN_KEY)
            if cached_twin is not None and getattr(cached_twin, "user_id", None) == user_id:
                twin = cached_twin
            else:
                demographics = DBManager.get_user_profile(user_id)
                if demographics:
                    balance_sheet = DBManager.get_digital_twin(user_id)
                    if balance_sheet is None:
                        try:
                            with get_db_cursor() as cursor:
                                cursor.execute("INSERT OR IGNORE INTO digital_twins (user_id) VALUES (?)", (user_id,))
                            balance_sheet = DBManager.get_digital_twin(user_id) or {"user_id": user_id}
                        except Exception:
                            balance_sheet = {"user_id": user_id}
                    twin = FinancialDigitalTwin(user_id, demographics, balance_sheet)
                    st.session_state[ACTIVE_TWIN_KEY] = twin
                else:
                    st.session_state.pop("user_id", None)
                    st.session_state.pop(ACTIVE_TWIN_KEY, None)
    else:
        st.session_state.pop(ACTIVE_TWIN_KEY, None)

    # ── Active Page Determination ─────────────────────────────────────────
    active_page = get_current_page_name()
    uid_param = f"?uid={twin.user_id}" if twin else ""

    # ── SVG Logo ──────────────────────────────────────────────────────────
    brand_svg = """<svg width="176" height="44" viewBox="0 0 320 80" xmlns="http://www.w3.org/2000/svg" style="display: block; overflow: visible;">
<defs>
    <linearGradient id="navGradBlue" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#4F8CFF" />
        <stop offset="100%" stop-color="#00D4FF" />
    </linearGradient>
    <linearGradient id="navGradCyan" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#00D4FF" />
        <stop offset="100%" stop-color="#10B981" />
    </linearGradient>
</defs>
<g transform="translate(0, -2) scale(0.95)">
    <rect x="20" y="20" width="36" height="36" rx="10" fill="url(#navGradBlue)" opacity="0.9" transform="rotate(45, 38, 38)"/>
    <rect x="42" y="20" width="36" height="36" rx="10" fill="url(#navGradCyan)" opacity="0.9" transform="rotate(45, 60, 38)"/>
    <path d="M 28 50 L 46 32 L 56 42 L 78 20" fill="none" stroke="#F8FAFC" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="78" cy="20" r="3.5" fill="#F8FAFC"/>
    <circle cx="28" cy="50" r="3.5" fill="#F8FAFC"/>
</g>
<text x="88" y="52" font-family="'Space Grotesk', sans-serif" font-weight="700" font-size="38" fill="#F8FAFC">Fin<tspan font-weight="700" fill="#4F8CFF">Twin</tspan> <tspan font-weight="400" fill="#00D4FF">AI</tspan></text>
</svg>"""

    # ── Navigation Categories & Pages ─────────────────────────────────────
    nav_categories = [
        {
            "id": "financial_profile",
            "title": "Financial Profile",
            "pages": ["01_Digital_Twin.py", "02_Financial_Health.py"],
            "items": [
                {
                    "title": "Digital Twin",
                    "url": f"/Digital_Twin{uid_param}",
                    "icon": "fa-solid fa-circle-user",
                    "desc": "View your complete AI-powered financial profile",
                    "page": "01_Digital_Twin.py"
                },
                {
                    "title": "Financial Health",
                    "url": f"/Financial_Health{uid_param}",
                    "icon": "fa-solid fa-heart-pulse",
                    "desc": "Monitor overall health score and metrics",
                    "page": "02_Financial_Health.py"
                }
            ]
        },
        {
            "id": "analysis",
            "title": "Analysis",
            "pages": ["03_Behavior_Analysis.py", "04_Financial_Personality.py", "05_Forecasting.py"],
            "items": [
                {
                    "title": "Behavior Analysis",
                    "url": f"/Behavior_Analysis{uid_param}",
                    "icon": "fa-solid fa-chart-pie",
                    "desc": "Understand your financial behavior",
                    "page": "03_Behavior_Analysis.py"
                },
                {
                    "title": "Financial Personality",
                    "url": f"/Financial_Personality{uid_param}",
                    "icon": "fa-solid fa-fingerprint",
                    "desc": "Discover your financial profile",
                    "page": "04_Financial_Personality.py"
                },
                {
                    "title": "Forecasting",
                    "url": f"/Forecasting{uid_param}",
                    "icon": "fa-solid fa-chart-line",
                    "desc": "Project savings and net worth",
                    "page": "05_Forecasting.py"
                }
            ]
        },
        {
            "id": "planning",
            "title": "Planning",
            "pages": ["06_Scenario_Simulator.py", "07_Tax_Intelligence.py", "09_Goal_Planner.py"],
            "items": [
                {
                    "title": "Scenario Simulator",
                    "url": f"/Scenario_Simulator{uid_param}",
                    "icon": "fa-solid fa-sliders",
                    "desc": "Test financial scenarios & life events",
                    "page": "06_Scenario_Simulator.py"
                },
                {
                    "title": "Tax Intelligence",
                    "url": f"/Tax_Intelligence{uid_param}",
                    "icon": "fa-solid fa-receipt",
                    "desc": "Tax-saving insights & regime comparison",
                    "page": "07_Tax_Intelligence.py"
                },
                {
                    "title": "Goal Planner",
                    "url": f"/Goal_Planner{uid_param}",
                    "icon": "fa-solid fa-bullseye",
                    "desc": "Plan savings goals & track progress",
                    "page": "09_Goal_Planner.py"
                }
            ]
        },
        {
            "id": "ai_features",
            "title": "AI Features",
            "pages": ["08_AI_Coach.py", "10_Explainable_AI.py"],
            "items": [
                {
                    "title": "AI Coach",
                    "url": f"/AI_Coach{uid_param}",
                    "icon": "fa-solid fa-robot",
                    "desc": "Personalized financial guidance powered by AI",
                    "page": "08_AI_Coach.py"
                },
                {
                    "title": "Explainable AI",
                    "url": f"/Explainable_AI{uid_param}",
                    "icon": "fa-solid fa-brain",
                    "desc": "Understand why AI generated each prediction",
                    "page": "10_Explainable_AI.py"
                }
            ]
        }
    ]

    # ── Build Center Navigation HTML ──────────────────────────────────────
    center_nav_html = []
    
    # 1. Dashboard
    dash_active = " active" if active_page == "app.py" else ""
    dash_url = "#" if active_page == "app.py" else f"/?nav=dashboard{f'&uid={twin.user_id}' if twin else ''}"
    brand_url = "#" if active_page == "app.py" else f"/?nav=dashboard{f'&uid={twin.user_id}' if twin else ''}"
    center_nav_html.append(f"""
        <div class="fintwin-nav-item">
            <a href="{dash_url}" target="_self" class="fintwin-nav-link{dash_active}">
                <i class="fa-solid fa-table-cells-large" style="font-size:0.95rem;"></i>
                <span>Dashboard</span>
            </a>
        </div>
    """)

    # 2. Categories with Dropdowns
    for cat in nav_categories:
        cat_active = " active" if active_page in cat["pages"] else ""
        items_html = []
        for itm in cat["items"]:
            itm_active = " active-item" if active_page == itm["page"] else ""
            items_html.append(f"""
                <a href="{itm['url']}" target="_self" class="fintwin-dropdown-item{itm_active}">
                    <div class="fintwin-item-icon"><i class="{itm['icon']}"></i></div>
                    <div class="fintwin-item-text">
                        <div class="fintwin-item-title">{itm['title']}</div>
                        <div class="fintwin-item-desc">{itm['desc']}</div>
                    </div>
                </a>
            """)

        center_nav_html.append(f"""
            <div class="fintwin-nav-item">
                <div class="fintwin-nav-link{cat_active}">
                    <span>{cat['title']}</span>
                    <i class="fa-solid fa-chevron-down nav-chevron"></i>
                </div>
                <div class="fintwin-dropdown">
                    <div class="fintwin-dropdown-header">{cat['title']}</div>
                    {''.join(items_html)}
                </div>
            </div>
        """)

    # ── Build Right User/Auth Area HTML ───────────────────────────────────
    if twin:
        name_parts = [p for p in twin.name.strip().split() if p]
        if len(name_parts) >= 2:
            initials = f"{name_parts[0][0]}{name_parts[-1][0]}".upper()
            first_name = name_parts[0]
        elif len(name_parts) == 1 and name_parts[0]:
            initials = name_parts[0][:2].upper()
            first_name = name_parts[0]
        else:
            initials = "FT"
            first_name = "User"

        # Financial Status Summary for User Dropdown
        has_profile_data = (twin.total_income > 0 or twin.net_worth != 0 or twin.bank_savings > 0 or twin.basic_expenses > 0)
        health_widget_html = ""
        if has_profile_data:
            try:
                score_data = HealthScoreEngine(twin).compute_overall_health_score()
                h_score = score_data["overall_score"]
                h_grade = score_data["financial_grade"]
                grade_color = "#10B981" if h_score >= 75 else ("#F59E0B" if h_score >= 55 else "#EF4444")
                grade_bg = "rgba(16, 185, 129, 0.15)" if h_score >= 75 else ("rgba(245, 158, 11, 0.15)" if h_score >= 55 else "rgba(239, 68, 68, 0.15)")
                health_widget_html = f"""
                    <div class="fintwin-user-health-box">
                        <div class="fintwin-health-row">
                            <span class="fintwin-health-label">FINANCIAL STATUS</span>
                            <span class="fintwin-health-badge" style="background:{grade_bg}; color:{grade_color};">{h_grade}</span>
                        </div>
                        <div class="fintwin-health-row" style="margin-bottom:0.4rem;">
                            <span class="fintwin-health-score-val">{h_score} <span style="font-size:0.75rem; font-weight:500; color:#64748B;">/ 100</span></span>
                            <span style="font-size:0.78rem; font-weight:600; color:{grade_color};">{h_grade}</span>
                        </div>
                        <div class="fintwin-health-bar">
                            <div class="fintwin-health-fill" style="width:{h_score}%; background:{grade_color};"></div>
                        </div>
                    </div>
                """
            except Exception:
                health_widget_html = ""

        from utils.avatar import render_avatar_html
        twin_avatar_id = getattr(twin, "avatar_id", "avatar_01")
        header_avatar_html = render_avatar_html(twin_avatar_id, size=28, class_name="fintwin-user-avatar-img")
        card_avatar_html = render_avatar_html(twin_avatar_id, size=38, class_name="fintwin-user-card-avatar-img")

        right_nav_html = f"""
            <div class="fintwin-nav-item user-item">
                <div class="fintwin-user-trigger">
                    <div class="fintwin-user-avatar">{header_avatar_html}</div>
                    <span class="fintwin-user-name">{_he(first_name)}</span>
                    <i class="fa-solid fa-chevron-down nav-chevron" style="font-size:0.65rem; color:#94A3B8;"></i>
                </div>
                <div class="fintwin-dropdown fintwin-dropdown-user">
                    <div class="fintwin-user-card">
                        <div class="fintwin-user-card-avatar">{card_avatar_html}</div>
                        <div class="fintwin-user-card-meta">
                            <div class="fintwin-user-card-name">{_he(twin.name)}</div>
                            <div class="fintwin-user-card-tag"><i class="fa-solid fa-circle-check"></i> Digital Twin Active</div>
                        </div>
                    </div>
                    {health_widget_html}
                    <a href="/Settings{uid_param}" target="_self" class="fintwin-dropdown-item">
                        <div class="fintwin-item-icon"><i class="fa-solid fa-gear"></i></div>
                        <div class="fintwin-item-text">
                            <div class="fintwin-item-title">Settings & Privacy</div>
                            <div class="fintwin-item-desc">Security, password and account settings</div>
                        </div>
                    </a>
                    <div class="fintwin-dropdown-divider"></div>
                    <a href="?action=logout" target="_self" class="fintwin-dropdown-item">
                        <div class="fintwin-item-icon" style="background:rgba(239, 68, 68, 0.14); color:#EF4444;"><i class="fa-solid fa-arrow-right-from-bracket"></i></div>
                        <div class="fintwin-item-text">
                            <div class="fintwin-item-title" style="color:#F87171;">Log Out</div>
                            <div class="fintwin-item-desc">Securely end your session</div>
                        </div>
                    </a>
                </div>
            </div>
        """
    else:
        right_nav_html = """
            <div class="fintwin-auth-btn-placeholder" style="width:130px; height:38px; display:inline-block;"></div>
        """

    # ── Build Mobile Navigation Drawer HTML ───────────────────────────────
    mobile_sections_html = [
        f"""
        <div class="fintwin-mobile-section">
            <div class="fintwin-mobile-section-title">Overview</div>
            <a href="{dash_url}" target="_self" class="fintwin-mobile-link{' active-item' if active_page == 'app.py' else ''}">
                <i class="fa-solid fa-table-cells-large"></i>
                <span>Dashboard</span>
            </a>
        </div>
        """
    ]

    for cat in nav_categories:
        cat_links = []
        for itm in cat["items"]:
            cat_links.append(f"""
                <a href="{itm['url']}" target="_self" class="fintwin-mobile-link{' active-item' if active_page == itm['page'] else ''}">
                    <i class="{itm['icon']}"></i>
                    <span>{itm['title']}</span>
                </a>
            """)
        mobile_sections_html.append(f"""
            <div class="fintwin-mobile-section">
                <div class="fintwin-mobile-section-title">{cat['title']}</div>
                {''.join(cat_links)}
            </div>
        """)

    if twin:
        mobile_avatar_html = render_avatar_html(twin_avatar_id, size=20, class_name="fintwin-mobile-avatar-img")
        mobile_sections_html.append(f"""
            <div class="fintwin-mobile-section">
                <div class="fintwin-mobile-section-title" style="display:flex; align-items:center; gap:8px;">
                    {mobile_avatar_html}
                    <span>Account ({_he(twin.name)})</span>
                </div>
                <a href="/Settings{uid_param}" target="_self" class="fintwin-mobile-link">
                    <i class="fa-solid fa-gear"></i>
                    <span>Settings & Privacy</span>
                </a>
                <a href="?action=logout" target="_self" class="fintwin-mobile-link logout-item" style="color:#F87171;">
                    <i class="fa-solid fa-arrow-right-from-bracket"></i>
                    <span>Log Out</span>
                </a>
            </div>
        """)
    else:
        mobile_sections_html.append("""
            <div class="fintwin-mobile-section">
                <div class="fintwin-mobile-section-title">Account</div>
                <button type="button" class="fintwin-mobile-link" id="fintwin-mobile-login-register-btn">
                    <i class="fa-solid fa-arrow-right-to-bracket"></i>
                    <span>Login/Register</span>
                </button>
            </div>
        """)

    # ── Render Top Navigation Bar ─────────────────────────────────────────
    topbar_html = f"""
        <nav class="fintwin-topbar" id="fintwin-top-navigation">
            <div class="fintwin-topbar-inner">
                <a href="{brand_url}" target="_self" class="fintwin-brand-link" title="FinTwin AI — Financial Digital Twin">
                    {brand_svg}
                </a>
                <div class="fintwin-nav-center">
                    {''.join(center_nav_html)}
                </div>
                <div class="fintwin-nav-right">
                    {right_nav_html}
                    <div class="fintwin-mobile-nav">
                        <details class="fintwin-mobile-menu">
                            <summary class="fintwin-mobile-toggle" aria-label="Toggle navigation menu">
                                <i class="fa-solid fa-bars"></i>
                            </summary>
                            <div class="fintwin-mobile-drawer">
                                {''.join(mobile_sections_html)}
                            </div>
                        </details>
                    </div>
                </div>
            </div>
        </nav>
    """
    st.markdown(_clean_html(topbar_html), unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        function setupHandlers() {
            var mobLogin = doc.getElementById('fintwin-mobile-login-register-btn');
            if (mobLogin && !mobLogin.dataset.loginBound) {
                mobLogin.dataset.loginBound = "1";
                mobLogin.addEventListener('click', function(e) {
                    e.preventDefault();
                    var b = doc.querySelector('.st-key-fintwin_top_auth_btn button');
                    if (b) b.click();
                });
            }
        }
        setupHandlers();
        setInterval(setupHandlers, 300);
    })();
    </script>
    """, height=0, width=0)

    # Render fixed topbar auth button for guest users (handles modal trigger via WebSocket without page reload)
    if not twin:
        if st.button("Login/Register", key="fintwin_top_auth_btn", type="primary"):
            login_modal()

    # Render floating FinBot chatbot (grounded in active twin)
    render_chatbot(twin)
    return twin


# Backward compatibility alias: all existing pages and tests calling render_sidebar_user_selector
# will seamlessly call render_top_navigation without breaking contract or imports.
render_sidebar_user_selector = render_top_navigation
