"""
Central Avatar Resolver and Component
FinTwin AI — 10 Profile Avatar Selection System

Centralized manager for profile avatars:
- Scans and resolves avatar image assets from assets/avatars/
- Maximum 10 avatar choices supported
- No gender categorization or labeling
- Accessible labels ("Profile avatar 1", "Profile avatar 2", etc.)
- In-memory base64 caching for zero-latency UI rendering
- Safe fallbacks for missing/invalid avatar IDs or existing users
- Web UI usage only (strictly excluded from PDF reports)
"""

import os
import re
import base64
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import BASE_DIR

logger = logging.getLogger(__name__)

AVATAR_DIR = BASE_DIR / "assets" / "avatars"
DEFAULT_AVATAR_ID = "avatar_01"
MAX_AVATARS = 10


def _natural_sort_key(s: str):
    """Sort strings containing numbers in natural human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def get_available_avatars() -> List[Dict[str, Any]]:
    """
    Scans the assets/avatars/ directory and returns up to 10 available profile avatars.
    Avatars are treated strictly as neutral profile avatar choices with no gender categorization.
    """
    if not AVATAR_DIR.exists() or not AVATAR_DIR.is_dir():
        return []

    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_files.extend(AVATAR_DIR.glob(ext))

    # Sort naturally by filename
    image_files.sort(key=lambda p: _natural_sort_key(p.name))

    # Enforce maximum 10 avatars
    image_files = image_files[:MAX_AVATARS]

    avatars = []
    for idx, img_path in enumerate(image_files, start=1):
        avatar_id = img_path.stem  # e.g., "avatar_01"
        avatars.append({
            "id": avatar_id,
            "filename": img_path.name,
            "path": img_path,
            "label": f"Profile avatar {idx}",
            "index": idx,
        })

    return avatars


def is_valid_avatar_id(avatar_id: Optional[str]) -> bool:
    """Checks whether the given avatar_id matches an available avatar asset."""
    if not avatar_id:
        return False
    available_ids = {a["id"] for a in get_available_avatars()}
    return avatar_id in available_ids


def resolve_avatar_path(avatar_id: Optional[str]) -> Path:
    """
    Resolves an avatar_id to its absolute filesystem path.
    Safely falls back to DEFAULT_AVATAR_ID if avatar_id is None, empty, or not found.
    """
    avatars = get_available_avatars()
    avatar_map = {a["id"]: a["path"] for a in avatars}

    if avatar_id and avatar_id in avatar_map:
        return avatar_map[avatar_id]

    # Fallback to default avatar
    if DEFAULT_AVATAR_ID in avatar_map:
        return avatar_map[DEFAULT_AVATAR_ID]

    # If default not in map, pick the first available if any exist
    if avatars:
        return avatars[0]["path"]

    # Absolute fallback (even if dir is empty)
    return AVATAR_DIR / f"{DEFAULT_AVATAR_ID}.png"


@lru_cache(maxsize=32)
def resolve_avatar_base64(avatar_id: Optional[str]) -> str:
    """
    Returns an in-memory cached base64 data URI (data:image/png;base64,...) for the avatar.
    Provides instant, zero-latency rendering across Streamlit HTML components.
    """
    path = resolve_avatar_path(avatar_id)
    if not path.exists():
        # Fallback to default avatar path
        path = AVATAR_DIR / f"{DEFAULT_AVATAR_ID}.png"
        if not path.exists():
            return ""

    try:
        with open(path, "rb") as f:
            data = f.read()
            mime = "image/png"
            if path.suffix.lower() in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif path.suffix.lower() == ".webp":
                mime = "image/webp"
            encoded = base64.b64encode(data).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    except Exception as e:
        logger.error(f"Error reading avatar asset {path}: {e}")
        return ""


def render_avatar_html(
    avatar_id: Optional[str],
    size: int = 28,
    class_name: str = "fintwin-user-avatar-img",
    alt: Optional[str] = None,
    style: str = "",
) -> str:
    """
    Renders an HTML <img> element for the avatar.
    Used for web UI only (top navigation, header, cards, greetings).
    """
    b64 = resolve_avatar_base64(avatar_id)
    if not alt:
        avatars = {a["id"]: a["label"] for a in get_available_avatars()}
        alt = avatars.get(avatar_id or "", "Profile avatar")

    if not b64:
        # Safe fallback HTML element if image cannot be loaded
        return (
            f'<div class="{class_name}" style="width:{size}px; height:{size}px; border-radius:50%; '
            f'background: linear-gradient(135deg, #2563EB, #1D4ED8); border: 1.5px solid rgba(255,255,255,0.2); '
            f'display:inline-flex; align-items:center; justify-content:center; color:#FFF; font-weight:700; '
            f'font-size:{max(10, size//3)}px; vertical-align:middle; {style}">'
            f'AI</div>'
        )

    return (
        f'<img src="{b64}" class="{class_name}" alt="{alt}" '
        f'style="width:{size}px; height:{size}px; min-width:{size}px; min-height:{size}px; border-radius:50%; '
        f'object-fit:cover; border:1.5px solid rgba(255,255,255,0.25); '
        f'box-shadow:0 2px 6px rgba(0,0,0,0.3); display:inline-block; vertical-align:middle; {style}" />'
    )


def render_avatar_selector(
    key: str = "register_avatar_selector",
    selected_id: Optional[str] = None,
    label: str = "Choose your profile avatar",
    help_text: Optional[str] = None,
) -> Optional[str]:
    """
    Renders the FinTwin AI 10-avatar selection grid using a styled Streamlit radio widget.
    Works inside st.form as well as standard pages.
    Displays up to 10 choices in a responsive grid with hover effects and checkmark badge.
    """
    import streamlit as st

    avatars = get_available_avatars()
    if not avatars:
        st.warning("No profile avatars currently available.")
        return None

    avatar_ids = [a["id"] for a in avatars]
    avatar_labels = {a["id"]: a["label"] for a in avatars}

    # Determine default selection index
    default_index = None
    if selected_id and selected_id in avatar_ids:
        default_index = avatar_ids.index(selected_id)

    # Generate CSS rules for mapping each radio option to its corresponding avatar image
    css_rules = []
    for idx, a in enumerate(avatars, start=1):
        b64 = resolve_avatar_base64(a["id"])
        css_rules.append(f"""
            .st-key-{key} div[role="radiogroup"] > label:nth-child({idx}) [data-testid="stMarkdownContainer"]::before {{
                background-image: url('{b64}') !important;
            }}
        """)

    custom_css = f"""
    <style>
    /* FinTwin AI — Avatar Selection System Grid Styles */
    .st-key-{key} div[role="radiogroup"] {{
        display: grid !important;
        grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
        gap: 12px 10px !important;
        justify-items: center !important;
        align-items: center !important;
        padding: 8px 4px 14px 4px !important;
        width: 100% !important;
    }}

    @media (max-width: 540px) {{
        .st-key-{key} div[role="radiogroup"] {{
            grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
            gap: 8px 6px !important;
        }}
    }}

    /* Radio item container */
    .st-key-{key} div[role="radiogroup"] > label {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
        cursor: pointer !important;
        margin: 0 !important;
        padding: 4px !important;
        background: transparent !important;
        border-radius: 50% !important;
        transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    /* Hide the native radio button dot/circle */
    .st-key-{key} div[role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}

    /* Avatar circle container inside radio */
    .st-key-{key} div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0 !important; /* Hide text label visually while retaining accessibility */
        line-height: 0 !important;
        color: transparent !important;
    }}

    .st-key-{key} div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {{
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* Avatar image pseudo-element */
    .st-key-{key} div[role="radiogroup"] > label [data-testid="stMarkdownContainer"]::before {{
        content: '' !important;
        display: block !important;
        width: 52px !important;
        height: 52px !important;
        border-radius: 50% !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        border: 2px solid rgba(255, 255, 255, 0.16) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-sizing: border-box !important;
    }}

    @media (max-width: 540px) {{
        .st-key-{key} div[role="radiogroup"] > label [data-testid="stMarkdownContainer"]::before {{
            width: 44px !important;
            height: 44px !important;
        }}
    }}

    /* Hover & Focus states */
    .st-key-{key} div[role="radiogroup"] > label:hover {{
        transform: translateY(-2px) scale(1.04) !important;
    }}

    .st-key-{key} div[role="radiogroup"] > label:hover [data-testid="stMarkdownContainer"]::before {{
        border-color: rgba(79, 140, 255, 0.6) !important;
        box-shadow: 0 0 12px rgba(79, 140, 255, 0.45) !important;
    }}

    /* Active / Selected state with highlight and checkmark */
    .st-key-{key} div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"]::before {{
        border: 2.5px solid #00D4FF !important;
        box-shadow: 0 0 0 3px rgba(79, 140, 255, 0.35), 0 0 16px rgba(0, 212, 255, 0.55) !important;
        transform: scale(1.02) !important;
    }}

    /* Checkmark badge */
    .st-key-{key} div[role="radiogroup"] > label:has(input:checked)::after {{
        content: '✓' !important;
        position: absolute !important;
        bottom: 2px !important;
        right: 2px !important;
        width: 18px !important;
        height: 18px !important;
        background: #22C55E !important;
        color: #FFFFFF !important;
        font-size: 11px !important;
        font-weight: 900 !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
        border: 1.5px solid #0F172A !important;
        z-index: 2 !important;
        line-height: 1 !important;
        text-align: center !important;
    }}

    {''.join(css_rules)}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    choice = st.radio(
        label=label,
        options=avatar_ids,
        index=default_index,
        format_func=lambda aid: avatar_labels.get(aid, "Profile avatar"),
        key=key,
        help=help_text,
        horizontal=True,
    )

    return choice
