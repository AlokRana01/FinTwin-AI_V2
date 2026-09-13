"""
Floating AI Chatbot Widget  (FinBot)
─────────────────────────────────────────────────────────────────────────────
A Gemini-powered assistant that floats bottom-right on every page of the app,
with a large, expressive animated "face" orb (idle blink + glance + breathing
glow, in the spirit of modern AI-assistant launchers). Clicking it opens a
roomy chat panel. Logged-in users get answers grounded in their own Financial
Digital Twin data (income, spending, investments, debt, goals) plus a digest
from the app's own rule-based AI Coach engine, so FinBot's advice matches
what the rest of the app already tells them. A daily message limit is
enforced per user (persisted in the DB) to protect API usage.

Wire-up: call render_chatbot(twin) once per page, after the user/twin has
been resolved (see utils/session.py -> render_sidebar_user_selector()).

API key resolution order:
  1. st.secrets["GEMINI_API_KEY"]           (.streamlit/secrets.toml)
  2. environment variable GEMINI_API_KEY
"""

from __future__ import annotations

import datetime
import re
from html import escape as _he
from typing import Optional, Any, Dict, List
import streamlit as st

from database.db_manager import DBManager
from utils.security import sanitize_chat_message
from utils.agent_chat_adapter import (
    AgentChatAdapter,
    ChatOrchestrationRequest,
    DEFAULT_CHAT_ADAPTER,
)
from agents.orchestrator import OrchestratorAgent
from agents.schemas import AgentResult, AgentStatus


def _format_inline_markdown(text: str) -> str:
    """Escapes HTML and applies inline bold, italic, code formatting."""
    escaped = _he(text)
    # Bold: **text** or __text__
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong style='color: #F8FAFC; font-weight: 600;'>\1</strong>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<strong style='color: #F8FAFC; font-weight: 600;'>\1</strong>", escaped)
    # Italic: *text* or _text_ (single asterisk or underscore)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em style='color: #94A3B8;'>\1</em>", escaped)
    escaped = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<em style='color: #94A3B8;'>\1</em>", escaped)
    # Inline code: `text`
    escaped = re.sub(r"`(.+?)`", r"<code style='background: rgba(255,255,255,0.08); padding: 1px 5px; border-radius: 4px; font-size: 0.86em; color: #38BDF8;'>\1</code>", escaped)
    # Remove any stray unparsed markdown markers
    escaped = escaped.replace("**", "").replace("###", "").replace("##", "")
    return escaped


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _is_table_delimiter(line: str) -> bool:
    s = line.strip()
    if not _is_table_row(s):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return len(cells) > 0 and all(re.match(r"^:?-+:?$", c) for c in cells if c)


def _render_table_html(table_rows: list[str]) -> str:
    cleaned_rows = [r.strip() for r in table_rows if r.strip() and _is_table_row(r)]
    if not cleaned_rows:
        return ""

    has_delimiter = False
    delimiter_idx = -1

    for idx, row in enumerate(cleaned_rows):
        if _is_table_delimiter(row):
            has_delimiter = True
            delimiter_idx = idx
            break

    html = [
        '<div style="overflow-x: auto; margin: 10px 0 12px 0; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.12); background: rgba(15, 23, 42, 0.7);">'
        '<table style="width: 100%; border-collapse: collapse; font-family: \'Inter\', sans-serif; font-size: 0.83rem; text-align: left;">'
    ]

    if has_delimiter and delimiter_idx > 0:
        header_cells = [c.strip() for c in cleaned_rows[0].strip("|").split("|")]
        html.append('<thead><tr style="background: rgba(30, 41, 59, 0.95); border-bottom: 1px solid rgba(255, 255, 255, 0.15);">')
        for c in header_cells:
            formatted_c = _format_inline_markdown(c)
            html.append(f'<th style="padding: 8px 10px; color: #F8FAFC; font-weight: 700; white-space: nowrap;">{formatted_c}</th>')
        html.append('</tr></thead><tbody>')

        body_rows = [r for idx, r in enumerate(cleaned_rows) if idx != 0 and idx != delimiter_idx]
        for r_idx, b_row in enumerate(body_rows):
            b_cells = [c.strip() for c in b_row.strip("|").split("|")]
            bg = "background: rgba(255, 255, 255, 0.025);" if r_idx % 2 == 1 else ""
            html.append(f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); {bg}">')
            for c in b_cells:
                formatted_c = _format_inline_markdown(c)
                html.append(f'<td style="padding: 7px 10px; color: #CBD5E1; vertical-align: top; line-height: 1.45;">{formatted_c}</td>')
            html.append('</tr>')
        html.append('</tbody>')
    else:
        html.append('<tbody>')
        for r_idx, row in enumerate(cleaned_rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            bg = "background: rgba(255, 255, 255, 0.025);" if r_idx % 2 == 1 else ""
            html.append(f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); {bg}">')
            for c in cells:
                formatted_c = _format_inline_markdown(c)
                html.append(f'<td style="padding: 7px 10px; color: #CBD5E1; vertical-align: top; line-height: 1.45;">{formatted_c}</td>')
            html.append('</tr>')
        html.append('</tbody>')

    html.append('</table></div>')
    return "".join(html)


def _format_chat_markdown(text: str) -> str:
    """Converts Markdown (headers, bold, italic, bullet lists, numbered lists, tables, HRs)
    into clean, formatted HTML while escaping any unsafe user/model input."""
    if not text:
        return ""

    raw_lines = text.strip().split("\n")
    html_lines = []
    in_ul = False
    in_ol = False

    i = 0
    n = len(raw_lines)

    while i < n:
        raw_line = raw_lines[i]
        line = raw_line.strip()

        # Check for table block
        if _is_table_row(line):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False

            table_lines = []
            while i < n:
                curr_line = raw_lines[i].strip()
                if _is_table_row(curr_line):
                    table_lines.append(curr_line)
                    i += 1
                elif not curr_line and i + 1 < n and _is_table_row(raw_lines[i + 1].strip()):
                    # Empty line inside table
                    i += 1
                else:
                    break
            html_lines.append(_render_table_html(table_lines))
            continue

        if not line:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            html_lines.append("<div style='height: 6px;'></div>")
            i += 1
            continue

        # Check for Horizontal Rules: ---, ***, ___
        if re.match(r"^(\-{3,}|\*{3,}|_{3,})$", line):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            html_lines.append("<div style='height: 1px; background: rgba(255, 255, 255, 0.1); margin: 12px 0;'></div>")
            i += 1
            continue

        # Check for headings: ### or ## or #
        h_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h_match:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            level = len(h_match.group(1))
            h_content = _format_inline_markdown(h_match.group(2))
            font_size = "1rem" if level <= 3 else "0.92rem"
            html_lines.append(
                f'<div style="font-family: \'Space Grotesk\', sans-serif; font-weight: 600; font-size: {font_size}; color: #F8FAFC; margin: 10px 0 4px 0;">{h_content}</div>'
            )
            i += 1
            continue

        # Check for unordered list item: * or -
        ul_match = re.match(r"^[\*\-]\s+(.*)$", line)
        if ul_match:
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append("<ul style='margin: 4px 0 6px 0; padding-left: 18px; list-style-type: disc;'>")
                in_ul = True
            item_content = _format_inline_markdown(ul_match.group(1))
            html_lines.append(f"<li style='margin-bottom: 5px; color: #CBD5E1; line-height: 1.5;'>{item_content}</li>")
            i += 1
            continue

        # Check for ordered list item: 1. 2. etc.
        ol_match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if ol_match:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append("<ol style='margin: 4px 0 6px 0; padding-left: 18px;'>")
                in_ol = True
            item_content = _format_inline_markdown(ol_match.group(2))
            html_lines.append(f"<li style='margin-bottom: 5px; color: #CBD5E1; line-height: 1.5;'>{item_content}</li>")
            i += 1
            continue

        # Normal text line
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

        content = _format_inline_markdown(line)
        html_lines.append(f"<p style='margin: 0 0 6px 0; color: #CBD5E1; line-height: 1.55;'>{content}</p>")
        i += 1

    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")

    return "".join(html_lines)


def _safe_html(text: str) -> str:
    """HTML-escape user-supplied text and convert newlines to <br> tags."""
    return _he(text).replace("\n", "<br>")

# ══════════════════════════════════════════════════════════════════════════
# Session & Usage Configuration
# ══════════════════════════════════════════════════════════════════════════

HISTORY_KEY = "ftw_chat_history"     # list[{"role": "user"|"model", "text": str}]
OPEN_KEY = "ftw_chat_open"           # bool
ANON_USAGE_KEY = "ftw_anon_usage"    # {"date": iso, "count": int}  (guest fallback, session-only)

DAILY_LIMIT_LOGGED_IN = 40
DAILY_LIMIT_GUEST = 12
MAX_HISTORY = 50  # Maximum stored turns to prevent memory exhaustion and excessive context


# ══════════════════════════════════════════════════════════════════════════
# Daily Usage Tracking & Rate Limiting
# ══════════════════════════════════════════════════════════════════════════

def _usage_today(twin) -> tuple[int, int]:
    """Returns (used_today, daily_limit) for the current user (or guest).
    Employs session-level caching to eliminate redundant SQLite SELECT queries
    on passive page reruns while preserving authoritative DB checks on send."""
    today = datetime.date.today().isoformat()
    if twin is not None:
        cached = st.session_state.get("ftw_cached_usage")
        if (
            cached is not None
            and isinstance(cached, tuple)
            and len(cached) == 3
            and cached[0] == twin.user_id
            and cached[1] == today
        ):
            return cached[2], DAILY_LIMIT_LOGGED_IN

        used = DBManager.get_chat_usage_today(twin.user_id)
        st.session_state["ftw_cached_usage"] = (twin.user_id, today, used)
        return used, DAILY_LIMIT_LOGGED_IN

    rec = st.session_state.get(ANON_USAGE_KEY)
    if not rec or rec.get("date") != today:
        rec = {"date": today, "count": 0}
        st.session_state[ANON_USAGE_KEY] = rec
    return rec["count"], DAILY_LIMIT_GUEST


def _try_consume_quota(twin) -> tuple[bool, int]:
    """Atomically checks and consumes one message from the daily quota.
    Returns (True, new_used_count) if quota was available and successfully consumed.
    Returns (False, current_used_count) if quota was exhausted; zero consumption occurs.
    Updates the session cache with the authoritative count."""
    today = datetime.date.today().isoformat()
    if twin is not None:
        allowed, count = DBManager.check_and_consume_quota(twin.user_id, DAILY_LIMIT_LOGGED_IN)
        st.session_state["ftw_cached_usage"] = (twin.user_id, today, count)
        return allowed, count

    # Guest user (session-scoped)
    rec = st.session_state.get(ANON_USAGE_KEY)
    if not rec or rec.get("date") != today:
        rec = {"date": today, "count": 0}
    if rec["count"] >= DAILY_LIMIT_GUEST:
        st.session_state[ANON_USAGE_KEY] = rec
        return False, rec["count"]
    rec["count"] += 1
    st.session_state[ANON_USAGE_KEY] = rec
    return True, rec["count"]


def _record_usage(twin) -> int:
    """Increments and returns the new used-today count authoritatively."""
    allowed, count = _try_consume_quota(twin)
    return count


# ══════════════════════════════════════════════════════════════════════════
# CSS (shared, injected once per page run)
# ══════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown("""
    <style>
    /* ---- Professional Fintech Floating AI Coach Launcher Orb ---- */
    .ftw-orb-wrap {
        position: fixed;
        bottom: 26px;
        right: 26px;
        width: 60px;
        height: 60px;
        z-index: 999998;
        pointer-events: none;
    }
    .ftw-orb {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 26px rgba(37, 99, 235, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.1);
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
    }
    .ftw-orb-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
    }
    .ftw-badge {
        position: absolute;
        bottom: 2px;
        right: 2px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #10B981;
        border: 2px solid #0F172A;
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
    }
    .ftw-ping {
        position: absolute;
        inset: -2px;
        border-radius: 50%;
        border: 1.5px solid rgba(16, 185, 129, 0.6);
        animation: ftw-ping 2.5s cubic-bezier(0, 0.5, 0.5, 1) infinite;
    }
    @keyframes ftw-ping {
        0% { transform: scale(0.9); opacity: 0.7; }
        100% { transform: scale(1.35); opacity: 0; }
    }

    /* ---- Invisible clickable button placed exactly over the launcher orb ---- */
    .st-key-ftw_toggle_btn {
        position: fixed !important;
        bottom: 26px !important;
        right: 26px !important;
        width: 60px !important;
        height: 60px !important;
        z-index: 999999 !important;
    }
    .st-key-ftw_toggle_btn > div { height: 100%; }
    .st-key-ftw_toggle_btn button {
        width: 100% !important;
        height: 100% !important;
        border-radius: 50% !important;
        background: transparent !important;
        border: none !important;
        opacity: 0 !important;
        cursor: pointer !important;
        padding: 0 !important;
    }

    /* ---- Chat panel: Institutional Dark Fintech, right-anchored ---- */
    .st-key-ftw_panel {
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        width: 32vw !important;
        min-width: 440px !important;
        max-width: 530px !important;
        height: 78vh !important;
        max-height: 820px !important;
        z-index: 999999 !important;
        background: rgba(15, 23, 42, 0.97) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.06) !important;
        padding: 16px 18px 12px 18px !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        animation: ftw-pop 0.26s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @media (max-width: 900px) {
        .st-key-ftw_panel {
            width: calc(100vw - 32px) !important;
            min-width: 0 !important;
            right: 16px !important;
            bottom: 16px !important;
            height: 80vh !important;
            max-height: calc(100vh - 32px) !important;
        }
        .ftw-orb-wrap, .st-key-ftw_toggle_btn { right: 16px !important; bottom: 16px !important; }
    }
    @keyframes ftw-pop {
        0% { opacity: 0; transform: translateY(16px) scale(0.95); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Clean custom scrollbar */
    .ftw-msgs::-webkit-scrollbar,
    .ftw-empty-card::-webkit-scrollbar {
        width: 5px;
    }
    .ftw-msgs::-webkit-scrollbar-track,
    .ftw-empty-card::-webkit-scrollbar-track {
        background: transparent;
    }
    .ftw-msgs::-webkit-scrollbar-thumb,
    .ftw-empty-card::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 99px;
    }
    .ftw-msgs::-webkit-scrollbar-thumb:hover,
    .ftw-empty-card::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.22);
    }

    /* ---- Header Layout & Vertical Alignment ---- */
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 12px !important;
        padding-bottom: 10px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        height: 100% !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-child(1) {
        justify-content: flex-start !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-child(2) {
        justify-content: center !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:nth-child(3) {
        justify-content: flex-end !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stVerticalBlock"] {
        height: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:first-child [data-testid="stVerticalBlock"] {
        justify-content: flex-start !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] {
        justify-content: flex-end !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] {
        height: 100% !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border-bottom: none !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-end !important;
        gap: 6px !important;
    }
    .st-key-ftw_panel [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
        width: auto !important;
        flex: 0 0 auto !important;
    }

    .ftw-panel-title {
        display: flex;
        align-items: center;
        gap: 10px;
        height: 38px;
    }
    .ftw-mini-orb {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        flex-shrink: 0;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
    }
    .ftw-mini-badge {
        position: absolute;
        bottom: 0px;
        right: 0px;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #10B981;
        border: 1.5px solid #0F172A;
    }
    @keyframes ftw-dot-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }
    .ftw-pulse-dot {
        display: inline-block;
        color: #10B981;
        text-shadow: 0 0 6px #10B981;
        animation: ftw-dot-pulse 2s ease-in-out infinite;
        margin-right: 3px;
    }
    .ftw-title-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #F8FAFC;
        font-size: 1.02rem;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: 0;
    }
    .ftw-intel-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.62rem;
        font-weight: 600;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 999px;
        padding: 1px 7px;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
        gap: 3px;
        letter-spacing: 0.02em;
        line-height: 1.3;
    }
    .ftw-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 4px;
        line-height: 1;
        margin-top: 2px;
    }
    .ftw-usage-pill {
        font-family: 'Inter', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        color: #94A3B8;
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        padding: 0 12px;
        height: 28px;
        white-space: nowrap;
        text-align: center;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        line-height: 28px;
        margin: 0 !important;
    }
    .ftw-usage-pill.low { color: #F59E0B; border-color: rgba(245, 158, 11, 0.3); }
    .ftw-usage-pill.zero { color: #EF4444; border-color: rgba(239, 68, 68, 0.3); }

    .st-key-ftw_close_btn,
    .st-key-ftw_clear_hdr_btn {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 32px !important;
    }
    .st-key-ftw_close_btn button,
    .st-key-ftw_clear_hdr_btn button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
    }
    .st-key-ftw_close_btn button:hover,
    .st-key-ftw_clear_hdr_btn button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    .st-key-ftw_close_btn button [data-testid="stIconMaterial"],
    .st-key-ftw_clear_hdr_btn button [data-testid="stIconMaterial"],
    .st-key-ftw_close_btn button span,
    .st-key-ftw_clear_hdr_btn button span {
        color: #F8FAFC !important;
        font-size: 1.15rem !important;
    }

    /* ---- Messages Feed ---- */
    .ftw-msgs {
        flex: 1 1 auto !important;
        overflow-y: auto !important;
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 6px 4px 10px 2px;
        margin-bottom: 6px;
        max-height: calc(78vh - 165px) !important;
    }
    @media (min-height: 1000px) {
        .ftw-msgs {
            max-height: 600px !important;
        }
    }
    @media (max-width: 900px) {
        .ftw-msgs {
            max-height: calc(80vh - 165px) !important;
        }
    }

    /* ---- Professional Chat Bubbles ---- */
    .ftw-bubble {
        max-width: 94%;
        padding: 12px 16px;
        border-radius: 14px;
        font-family: 'Inter', sans-serif;
        font-size: 0.89rem;
        line-height: 1.55;
        white-space: normal;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        animation: ftw-bubble-in 0.22s ease-out;
    }
    @keyframes ftw-bubble-in {
        0% { opacity: 0; transform: translateY(6px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .ftw-bubble.user {
        align-self: flex-end;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border-bottom-right-radius: 4px;
        font-weight: 500;
        max-width: 85%;
    }
    .ftw-bubble.model {
        align-self: flex-start;
        background: rgba(30, 41, 59, 0.85) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    }
    .ftw-bubble table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
    }

    /* ---- Bubble Footer & Copy Action ---- */
    .ftw-bubble-footer {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .ftw-copy-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        cursor: pointer;
        transition: all 0.15s ease;
        user-select: none;
    }
    .ftw-copy-btn:hover {
        color: #F8FAFC;
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .ftw-copy-btn .ftw-check-icon {
        display: none;
    }
    .ftw-copy-btn.copied {
        color: #10B981 !important;
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
    }
    .ftw-copy-btn.copied .ftw-copy-icon {
        display: none;
    }
    .ftw-copy-btn.copied .ftw-check-icon {
        display: inline-block;
    }

    /* ---- Structured Empty State (Balanced Onboarding & Suggestion Chips) ---- */
    .ftw-empty-card {
        flex: 1 1 auto;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 8px 4px 6px 4px;
        max-height: calc(78vh - 165px);
    }
    .ftw-welcome-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.12);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 999px;
        padding: 4px 12px;
        color: #60A5FA;
        font-family: 'Inter', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 10px;
        width: fit-content;
    }
    .ftw-welcome-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .ftw-welcome-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.83rem;
        color: #94A3B8;
        line-height: 1.45;
        margin: 0 0 10px 0;
    }
    .ftw-suggestion-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Suggestion Buttons in Streamlit */
    div[class*="st-key-ftw_sugg_"] {
        margin-bottom: 5px !important;
    }
    div[class*="st-key-ftw_sugg_"] button {
        background: rgba(30, 41, 59, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #CBD5E1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 7px 12px !important;
        height: 36px !important;
        min-height: 36px !important;
        justify-content: flex-start !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    div[class*="st-key-ftw_sugg_"] button:hover {
        background: rgba(59, 130, 246, 0.15) !important;
        border-color: rgba(59, 130, 246, 0.45) !important;
        color: #60A5FA !important;
        transform: translateX(2px) !important;
    }

    /* ---- Typing & Limit Banners ---- */
    .ftw-limit-banner {
        font-family: 'Inter', sans-serif;
        font-size: 0.80rem;
        color: #F59E0B;
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 6px;
        line-height: 1.4;
    }

    /* ---- Input Form & Controls ---- */
    .st-key-ftw_panel form {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-top: 6px !important;
        margin-bottom: 2px !important;
    }
    .st-key-ftw_panel input {
        background-color: rgba(30, 41, 59, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        padding: 9px 14px !important;
        height: 40px !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease !important;
    }
    .st-key-ftw_panel input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
        background-color: rgba(30, 41, 59, 0.98) !important;
    }
    .st-key-ftw_panel input::placeholder,
    .st-key-ftw_panel input::-webkit-input-placeholder,
    .st-key-ftw_panel input::-moz-placeholder,
    .st-key-ftw_panel input:-ms-input-placeholder {
        color: #64748B !important;
        font-size: 0.88rem !important;
        opacity: 1 !important;
    }
    .st-key-ftw_panel [data-baseweb="input"] {
        border: none !important;
        background: transparent !important;
    }

    /* Hide Streamlit form input instructions and character count overlay */
    .st-key-ftw_panel [data-testid="stInputInstructions"],
    .st-key-ftw_panel [data-testid="InputInstructions"],
    .st-key-ftw_panel small,
    .st-key-ftw_panel .st-emotion-cache-16idsys,
    .st-key-ftw_panel div:has(> [data-testid="stInputInstructions"]) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Form Submit Button */
    .st-key-ftw_panel button[type="submit"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.18s ease !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    .st-key-ftw_panel button[type="submit"]:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.38) !important;
    }
    .st-key-ftw_panel button[type="submit"]:active {
        transform: translateY(1px) !important;
    }

    /* Trust Disclaimer */
    .ftw-disclaimer {
        font-family: 'Inter', sans-serif;
        font-size: 0.70rem;
        color: #64748B;
        text-align: center;
        margin-top: 3px;
        margin-bottom: 0px;
        line-height: 1.3;
    }

    /* Flex structure to prevent bottom clipping */
    .st-key-ftw_panel > div,
    .st-key-ftw_panel [data-testid="stVerticalBlock"] {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div {
        flex-shrink: 0 !important;
    }
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div:has(.ftw-msgs),
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div:has(.ftw-msgs) > div,
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div:has(.ftw-msgs) [data-testid="stMarkdownContainer"],
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div:has(.ftw-empty-card),
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div:has(.ftw-empty-card) > div,
    .st-key-ftw_panel [data-testid="stVerticalBlock"] > div:has(.ftw-empty-card) [data-testid="stMarkdownContainer"] {
        flex: 1 1 auto !important;
        flex-shrink: 1 !important;
        overflow: hidden !important;
    }
    </style>
    <script>
    if (typeof window.ftwCopy === 'undefined') {
        window.ftwCopy = function(btn) {
            try {
                const bubble = btn.closest('.ftw-bubble');
                if (!bubble) return;
                const source = bubble.querySelector('.ftw-copy-source');
                if (!source) return;
                const text = source.textContent;

                const onDone = () => {
                    const lbl = btn.querySelector('.ftw-copy-text');
                    btn.classList.add('copied');
                    if (lbl) lbl.textContent = 'Copied!';
                    setTimeout(() => {
                        btn.classList.remove('copied');
                        if (lbl) lbl.textContent = 'Copy';
                    }, 2000);
                };

                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).then(onDone).catch(() => fallback(text, onDone));
                } else {
                    fallback(text, onDone);
                }

                function fallback(txt, cb) {
                    const t = document.createElement('textarea');
                    t.value = txt;
                    t.style.position = 'fixed';
                    t.style.left = '-999999px';
                    document.body.appendChild(t);
                    t.focus();
                    t.select();
                    try { document.execCommand('copy'); } catch(e) {}
                    document.body.removeChild(t);
                    cb();
                }
            } catch(e) {
                console.error(e);
            }
        };
    }
    </script>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Responses & Multi-Agent Formatting Helpers
# ══════════════════════════════════════════════════════════════════════════

GREETING_RESPONSE = (
    "Hello! I am FinBot, your AI financial assistant at FinTwin AI. "
    "I can analyze your financial health, evaluate debt & EMI risks, forecast future savings & net worth, "
    "track financial goals, and simulate what-if life decisions (such as buying a car, job loss, or salary hikes). "
    "How can I help you today?"
)

APP_GUIDANCE_RESPONSE = (
    "FinTwin AI is a comprehensive financial intelligence platform. Here is what I can help you with:\n\n"
    "• **Financial Health & Diagnostics**: Check your 0–100 health score, pillar breakdown, and key risk factors.\n"
    "• **Risk & Behavioral Insights**: Analyze debt burden, emergency fund adequacy, and spending leaks.\n"
    "• **Forecasts & Projections**: Project savings and net worth growth over 6 to 60 months.\n"
    "• **Goal Planning**: Assess target feasibility and determine required monthly SIP contributions.\n"
    "• **Scenario Simulation**: Test what-if events (buying a car, salary changes, home loans, marriage, or career breaks).\n\n"
    "Ask any financial question or test a scenario to get started!"
)


def _format_agent_result_to_text(result: AgentResult) -> str:
    """Formats an AgentResult from OrchestratorAgent/FinancialCoachAgent into clean user-facing markdown."""
    if result.status == AgentStatus.FAILED:
        if result.errors:
            return f"I encountered an issue processing your request: {'; '.join(result.errors)}"
        return "I was unable to analyze that financial request at this time. Please try again."

    # 1. Check for natural language explanation from LLMService via FinancialCoachAgent
    if result.authoritative_data and result.authoritative_data.metrics:
        llm_exp = result.authoritative_data.metrics.get("llm_explanation")
        if llm_exp and isinstance(llm_exp, str) and llm_exp.strip():
            return llm_exp.strip()

    # 2. Check for [AI Explanation] insight
    for ins in result.insights:
        if ins.startswith("[AI Explanation]"):
            return ins.replace("[AI Explanation] ", "").strip()

    # 3. Fallback to deterministic insights and summary
    parts = []
    if result.insights:
        parts.extend(result.insights)
    elif result.authoritative_data and result.authoritative_data.metrics:
        summary = result.authoritative_data.metrics.get("summary")
        if summary:
            parts.append(summary)

    if parts:
        return "\n\n".join(parts)

    return "Financial analysis completed."


# ══════════════════════════════════════════════════════════════════════════
# Single Source of Truth Message Handler (Multi-Agent Integration)
# ══════════════════════════════════════════════════════════════════════════

def _process_user_message(
    msg_text: str,
    twin,
    remaining: int = 1,
    adapter: Optional[AgentChatAdapter] = None,
    orchestrator: Optional[OrchestratorAgent] = None,
):
    """Processes message submission from either manual text input or suggestion chips.
    Routes financial queries authoritatively through AgentChatAdapter and OrchestratorAgent,
    while enforcing atomic database quota reservation, session history, and 50-turn bounds."""
    if not msg_text or not msg_text.strip() or remaining <= 0:
        return

    safe_msg = sanitize_chat_message(msg_text.strip())
    if not safe_msg:
        return

    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []

    # 1. Authoritative atomic check-and-consume quota reservation BEFORE dispatch
    allowed, used_count = _try_consume_quota(twin)
    if not allowed:
        limit = DAILY_LIMIT_LOGGED_IN if twin is not None else DAILY_LIMIT_GUEST
        who = "You've" if twin is not None else "This browser session has"
        quota_msg = (
            f"{who} reached today's limit of {limit} FinBot messages. "
            f"{'Please come back tomorrow.' if twin is not None else 'Log in for a higher daily limit, or come back tomorrow.'}"
        )
        st.session_state[HISTORY_KEY].append({"role": "user", "text": safe_msg})
        st.session_state[HISTORY_KEY].append({"role": "model", "text": quota_msg})
        if len(st.session_state[HISTORY_KEY]) > MAX_HISTORY:
            st.session_state[HISTORY_KEY] = st.session_state[HISTORY_KEY][-MAX_HISTORY:]
        st.rerun()
        return

    # Append user message
    st.session_state[HISTORY_KEY].append({"role": "user", "text": safe_msg})

    # 2. Construct immutable orchestration request via AgentChatAdapter
    chat_adapter = adapter or DEFAULT_CHAT_ADAPTER
    request = chat_adapter.build_request(safe_msg, twin=twin)

    # 3. Handle clarifications, non-financial queries, or execute multi-agent orchestration
    reply: str
    with st.spinner("FinBot is analyzing..."):
        if request.requires_clarification and request.clarification_prompt:
            reply = request.clarification_prompt
        elif not request.is_financial_query:
            if request.query_category == "GREETING":
                reply = GREETING_RESPONSE
            elif request.query_category == "APP_GUIDANCE":
                reply = APP_GUIDANCE_RESPONSE
            else:
                reply = "I am your personal finance assistant. Ask me about your financial health, savings, debt, goals, or what-if scenarios!"
        else:
            try:
                orch = orchestrator or OrchestratorAgent()
                result = orch.run(
                    intent=request.intent or "overview",
                    state=request.financial_state,
                    request_id=request.request_id,
                    scenario_name=request.scenario_name,
                    scenario_params=request.scenario_params,
                    goal_input=request.goal_params,
                )
                reply = _format_agent_result_to_text(result)
            except Exception as e:
                reply = "I'm sorry, an error occurred while processing your financial analysis. Please try again."

    # 4. Append assistant reply and enforce history limits
    st.session_state[HISTORY_KEY].append({"role": "model", "text": reply})
    if len(st.session_state[HISTORY_KEY]) > MAX_HISTORY:
        st.session_state[HISTORY_KEY] = st.session_state[HISTORY_KEY][-MAX_HISTORY:]

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════

def render_chatbot(twin=None):
    """Renders the floating FinBot widget. Call once per page."""
    # Check if the user has changed to clear chat history from previous user
    curr_user_id = twin.user_id if twin is not None else "guest"
    last_user_id = st.session_state.get("ftw_last_user_id")
    if last_user_id is not None and last_user_id != curr_user_id:
        st.session_state[HISTORY_KEY] = []
        st.session_state[OPEN_KEY] = False
        st.session_state.pop("ftw_cached_usage", None)
    st.session_state["ftw_last_user_id"] = curr_user_id

    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if OPEN_KEY not in st.session_state:
        st.session_state[OPEN_KEY] = False

    _inject_css()
    used, limit = _usage_today(twin)
    remaining = max(0, limit - used)

    if not st.session_state[OPEN_KEY]:
        st.markdown("""
        <div class="ftw-orb-wrap" title="AI Financial Coach">
            <div class="ftw-orb">
                <div class="ftw-ping"></div>
                <div class="ftw-orb-icon">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
                    </svg>
                </div>
                <div class="ftw-badge"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(" ", key="ftw_toggle_btn", help="Open AI Financial Assistant"):
            st.session_state[OPEN_KEY] = True
            st.rerun()
        return

    # ── Open panel ──────────────────────────────────────────────────────
    with st.container(key="ftw_panel"):
        history = st.session_state[HISTORY_KEY]

        header_l, header_m, header_r = st.columns([0.56, 0.26, 0.18], vertical_alignment="center")
        with header_l:
            st.markdown("""
            <div class="ftw-panel-title">
                <div class="ftw-mini-orb">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
                    </svg>
                    <div class="ftw-mini-badge"></div>
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <div class="ftw-title-text">FinBot</div>
                        <span class="ftw-intel-badge"><i class="fa-solid fa-sparkles" style="font-size: 0.55rem;"></i> FinTwin Intelligence</span>
                    </div>
                    <div class="ftw-subtitle"><span class="ftw-pulse-dot">●</span> Online</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with header_m:
            pill_class = "zero" if remaining == 0 else ("low" if remaining <= max(3, limit // 5) else "")
            st.markdown(
                f'<div class="ftw-usage-pill {pill_class}" title="{remaining} messages remaining today">{used}/{limit} today</div>',
                unsafe_allow_html=True,
            )
        with header_r:
            if history:
                act_c1, act_c2 = st.columns([1, 1], gap="small", vertical_alignment="center")
                with act_c1:
                    if st.button("", icon=":material/delete:", key="ftw_clear_hdr_btn", help="Clear conversation"):
                        st.session_state[HISTORY_KEY] = []
                        st.rerun()
                with act_c2:
                    if st.button("", icon=":material/close:", key="ftw_close_btn", help="Close Assistant"):
                        st.session_state[OPEN_KEY] = False
                        st.rerun()
            else:
                if st.button("", icon=":material/close:", key="ftw_close_btn", help="Close Assistant"):
                    st.session_state[OPEN_KEY] = False
                    st.rerun()

        if not history:
            user_first_name = _he(twin.name.split()[0]) if (twin is not None and getattr(twin, "name", None)) else None
            greeting_title = f"Hi {user_first_name} \U0001f44b" if user_first_name else "Hi there \U0001f44b"
            greeting_desc = (
                "I'm your FinTwin financial assistant. Ask me about your savings, spending, "
                "investments, debt, taxes, or financial goals."
            )
            st.markdown(f"""
            <div class="ftw-empty-card">
                <h4 class="ftw-welcome-title">{greeting_title}</h4>
                <p class="ftw-welcome-desc">{greeting_desc}</p>
                <div class="ftw-suggestion-label">
                    <i class="fa-regular fa-lightbulb"></i> Suggested Questions
                </div>
            </div>
            """, unsafe_allow_html=True)

            suggestions = [
                "Analyze my savings & budget",
                "How is my spending?",
                "Review my investments",
                "How am I doing toward my goals?",
            ]
            for idx, prompt_text in enumerate(suggestions):
                if st.button(prompt_text, key=f"ftw_sugg_{idx}", use_container_width=True, disabled=(remaining == 0)):
                    _process_user_message(prompt_text, twin, remaining)
        else:
            bubble_items = []
            for m in history:
                if m["role"] == "user":
                    bubble_items.append(f'<div class="ftw-bubble user">{_format_inline_markdown(m["text"])}</div>')
                else:
                    raw_text_escaped = _he(m["text"])
                    formatted_content = _format_chat_markdown(m["text"])
                    bubble_items.append(
                        f'<div class="ftw-bubble model">'
                        f'{formatted_content}'
                        f'<div class="ftw-bubble-footer">'
                        f'<button class="ftw-copy-btn" type="button" onclick="ftwCopy(this)" title="Copy response">'
                        f'<svg class="ftw-copy-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                        f'<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>'
                        f'<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>'
                        f'</svg>'
                        f'<svg class="ftw-check-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                        f'<polyline points="20 6 9 17 4 12"></polyline>'
                        f'</svg>'
                        f'<span class="ftw-copy-text">Copy</span>'
                        f'</button>'
                        f'<span class="ftw-copy-source" style="display:none;">{raw_text_escaped}</span>'
                        f'</div>'
                        f'</div>'
                    )
            bubbles_html = "".join(bubble_items)
            st.markdown(f'<div class="ftw-msgs">{bubbles_html}</div>', unsafe_allow_html=True)

        if remaining == 0:
            who = "You've" if twin is not None else "This browser session has"
            st.markdown(
                f'<div class="ftw-limit-banner"><i class="fa-solid fa-circle-exclamation"></i> {who} reached today\u2019s limit of '
                f'{limit} FinBot messages. '
                f'{"Please come back tomorrow." if twin is not None else "Log in for a higher daily limit, or come back tomorrow."}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.form("ftw_input_form", clear_on_submit=True, border=False):
            c1, c2 = st.columns([0.83, 0.17], vertical_alignment="center")
            with c1:
                user_msg = st.text_input(
                    "Message", key="ftw_user_msg", label_visibility="collapsed",
                    placeholder="Ask FinBot about your finances...",
                    disabled=(remaining == 0),
                )
            with c2:
                sent = st.form_submit_button(
                    "\u27a4", use_container_width=True, disabled=(remaining == 0),
                    help="Send message"
                )

        st.markdown("""
        <div class="ftw-disclaimer">
            <i class="fa-solid fa-shield-halved" style="font-size: 0.68rem; margin-right: 3px; opacity: 0.7;"></i>
            FinBot can make mistakes. Verify important financial decisions.
        </div>
        """, unsafe_allow_html=True)

        if sent and user_msg.strip() and remaining > 0:
            _process_user_message(user_msg.strip(), twin, remaining)
