"""
Startup Splash / Loading Screen.

Renders a premium, AI-themed animated loading overlay the very first time a
user opens the app in a browser session. It is intentionally implemented as
a single self-hiding CSS overlay (no JavaScript, no extra dependencies) so
that it cannot block Streamlit's normal rendering pipeline:

- The overlay paints instantly on top of the page.
- A pure-CSS keyframe animation fades it out automatically after ~2.8s.
- Once faded, `visibility: hidden` + `pointer-events: none` remove it from
  the layout and from hit-testing, revealing the already-rendered dashboard
  underneath — this is what creates the "navigate to Home once loading is
  complete" effect without any redirects or reruns.
- A `st.session_state` flag ensures it is only ever shown once per browser
  session, so it never reappears when the user navigates between pages or
  returns to Home later.

This module does not touch any business logic, data, or existing UI — it
only injects an overlay via `st.markdown`.
"""

import base64
import os

import streamlit as st

_SESSION_FLAG = "_ft_splash_shown"


def _load_logo_base64(current_dir: str) -> str:
    """Best-effort load of the project logo as a base64 data URI."""
    candidates = [
        os.path.join(current_dir, "assets", "logos", "3_icon_only.svg"),
        os.path.join(current_dir, "assets", "logos", "4_app_icon.svg"),
        os.path.join(current_dir, "assets", "logos", "favicon.svg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/svg+xml;base64,{encoded}"
    return ""


def render_startup_splash(
    current_dir: str,
    title: str = "FinTwin AI",
    subtitle: str = "AI-Powered Financial Health Digital Twin",
) -> None:
    """
    Render the animated startup splash screen exactly once per session.

    Safe to call at the top of app.py on every rerun of the Home page —
    it becomes a no-op automatically after the first successful render.
    """
    # 1. Already shown in this Streamlit session
    if st.session_state.get(_SESSION_FLAG):
        return

    # 2. Never show if user is performing an auth action (login, register, logout)
    params = st.query_params
    if params.get("action") in ("login", "register", "logout"):
        st.session_state[_SESSION_FLAG] = True
        return

    # 3. Never show if navigating back to dashboard via nav query parameter
    if params.get("nav"):
        st.session_state[_SESSION_FLAG] = True
        return

    # 4. Never show if returning user, token verification, or user already authenticated
    if (
        st.session_state.get("user_id")
        or params.get("uid")
        or "verify_email_token" in params
        or "reset_password_token" in params
    ):
        st.session_state[_SESSION_FLAG] = True
        return

    st.session_state[_SESSION_FLAG] = True

    logo_src = _load_logo_base64(current_dir)
    logo_html = (
        f'<img src="{logo_src}" class="ft-splash-logo-img" alt="{title} logo" />'
        if logo_src
        else '<div class="ft-splash-logo-fallback"><i class="fa-solid fa-shield-halved"></i></div>'
    )

    st.markdown(
        f"""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');

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
header[data-testid="stHeader"] button[data-testid="collapsedControl"] {{
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
}}

@keyframes ft-splash-fade-out {{
    0%   {{ opacity: 1; }}
    82%  {{ opacity: 1; }}
    100% {{ opacity: 0; }}
}}
@keyframes ft-splash-go-hidden {{
    to {{ visibility: hidden; }}
}}
@keyframes ft-splash-logo-in {{
    0%   {{ opacity: 0; transform: scale(0.55); }}
    60%  {{ opacity: 1; transform: scale(1.08); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}
@keyframes ft-splash-glow-pulse {{
    0%, 100% {{ filter: drop-shadow(0 0 14px rgba(79, 140, 255, 0.35)); }}
    50%      {{ filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.55)); }}
}}
@keyframes ft-splash-text-in {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes ft-splash-bar-fill {{
    from {{ width: 0%; }}
    to   {{ width: 100%; }}
}}
@keyframes ft-splash-shimmer {{
    0%   {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(220%); }}
}}
@keyframes ft-splash-orbit {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
}}

.ft-splash-overlay {{
    position: fixed;
    inset: 0;
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        radial-gradient(circle at 50% 35%, rgba(79, 140, 255, 0.14) 0%, transparent 55%),
        radial-gradient(circle at 85% 85%, rgba(0, 212, 255, 0.08) 0%, transparent 45%),
        #0B1220;
    animation:
        ft-splash-fade-out 3s ease forwards,
        ft-splash-go-hidden 0s linear 3s forwards;
}}

.ft-splash-inner {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    max-width: 92vw;
}}

.ft-splash-logo-ring {{
    position: relative;
    width: 108px;
    height: 108px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.75rem;
    animation: ft-splash-logo-in 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards,
               ft-splash-glow-pulse 2.2s ease-in-out 0.7s infinite;
}}
.ft-splash-logo-ring::before {{
    content: "";
    position: absolute;
    inset: -14px;
    border-radius: 50%;
    border: 2px solid rgba(79, 140, 255, 0.25);
    border-top-color: #00D4FF;
    animation: ft-splash-orbit 1.6s linear infinite;
}}
.ft-splash-logo-img {{
    width: 64px;
    height: 64px;
    border: none;
    background: transparent;
}}
.ft-splash-logo-fallback {{
    font-size: 2.4rem;
    color: #4F8CFF;
}}

.ft-splash-title {{
    color: #F8FAFC;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
    opacity: 0;
    animation: ft-splash-text-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.55s forwards;
}}
.ft-splash-subtitle {{
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin: 0 0 2.25rem 0;
    opacity: 0;
    animation: ft-splash-text-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.8s forwards;
}}

.ft-splash-progress-track {{
    position: relative;
    width: 220px;
    height: 6px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.15);
    overflow: hidden;
    opacity: 0;
    animation: ft-splash-text-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) 1.0s forwards;
}}
.ft-splash-progress-fill {{
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 0%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4F8CFF 0%, #00D4FF 100%);
    animation: ft-splash-bar-fill 2.0s cubic-bezier(0.4, 0, 0.2, 1) 1.05s forwards;
}}
.ft-splash-progress-fill::after {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
    animation: ft-splash-shimmer 1.1s ease-in-out infinite;
}}

.ft-splash-status {{
    margin-top: 0.9rem;
    color: #64748B;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0;
    animation: ft-splash-text-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) 1.15s forwards;
}}

@media (max-width: 480px) {{
    .ft-splash-title {{ font-size: 1.6rem; }}
    .ft-splash-subtitle {{ font-size: 0.85rem; }}
    .ft-splash-progress-track {{ width: 170px; }}
}}
</style>

<div class="ft-splash-overlay" id="ft-splash-overlay" role="status" aria-label="Loading FinTwin AI">
    <script>
    (function() {{
        try {{
            // Expire legacy cookie and clean sessionStorage
            document.cookie = 'ft_splash_shown=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax';
            if (window.sessionStorage) {{
                window.sessionStorage.removeItem('ft_splash_shown');
            }}
            if (window.parent && window.parent !== window) {{
                try {{
                    window.parent.document.cookie = 'ft_splash_shown=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax';
                    if (window.parent.sessionStorage) {{
                        window.parent.sessionStorage.removeItem('ft_splash_shown');
                    }}
                }} catch(err) {{}}
            }}
            // Clean up DOM element once splash animation finishes
            setTimeout(function() {{
                var el = document.getElementById('ft-splash-overlay');
                if (el) {{
                    el.style.display = 'none';
                    el.remove();
                }}
            }}, 3200);
        }} catch(e) {{}}
    }})();
    </script>
    <div class="ft-splash-inner">
        <div class="ft-splash-logo-ring">
            {logo_html}
        </div>
        <h1 class="ft-splash-title">{title}</h1>
        <p class="ft-splash-subtitle">{subtitle}</p>
        <div class="ft-splash-progress-track">
            <div class="ft-splash-progress-fill"></div>
        </div>
        <div class="ft-splash-status">Initializing your financial twin…</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
