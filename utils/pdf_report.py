"""
utils/pdf_report.py
====================
Professional, branded PDF report engine for FinTwin AI.

This module is purely a PRESENTATION layer. It never recalculates or derives
financial data — every value, table and chart it renders is supplied by the
caller (already computed elsewhere in the app: Digital Twin engine, ML
models, forecasting engine, etc.). This keeps the report generator decoupled
from business logic, so existing calculations remain completely untouched.

Usage
-----
    from utils.pdf_report import FinTwinPDFReport

    report = FinTwinPDFReport(
        report_title="Digital Twin Profile Report",
        user_id=twin.user_id,
        report_id=f"DT-{twin.user_id}",
    )
    report.add_cover_page(user_name="Alok", report_type="Digital Twin")
    report.add_kpi_summary_row([
        {"label": "Net Worth", "value": "Rs. 12,50,000", "status": "positive"},
        {"label": "Health Score", "value": "78 / 100", "status": "neutral"},
    ])
    report.add_section_divider("Income Analysis")
    report.add_dataframe_table(df, title="Income Breakdown")
    report.add_plotly_figure(fig, caption="Monthly Cash Flow")
    report.add_callout("Your savings rate is above the 70th percentile.", style="success")
    pdf_bytes = report.build()

    st.download_button("Download PDF Report", data=pdf_bytes,
                        file_name=report.suggested_filename("digital_twin"),
                        mime="application/pdf")
"""

from __future__ import annotations

import io
import os
import re
import html
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Optional vector logo support (SVG -> ReportLab drawing, no Cairo required)
# --------------------------------------------------------------------------- #
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    _SVG_AVAILABLE = True
except ImportError:  # pragma: no cover - svglib is an optional dependency
    _SVG_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Brand palette (matches the FinTwin AI Streamlit theme in utils/ui_components.py)
# --------------------------------------------------------------------------- #
BRAND_PRIMARY      = colors.HexColor("#4F8CFF")
BRAND_CYAN         = colors.HexColor("#00D4FF")
BRAND_GREEN        = colors.HexColor("#22C55E")
BRAND_AMBER        = colors.HexColor("#F59E0B")
BRAND_RED          = colors.HexColor("#EF4444")
BRAND_DARK         = colors.HexColor("#0F172A")
BRAND_NAVY         = colors.HexColor("#1E293B")
BRAND_SLATE        = colors.HexColor("#334155")
BRAND_GREY         = colors.HexColor("#64748B")
BRAND_LIGHT_ROW    = colors.HexColor("#F1F5F9")
BRAND_BORDER       = colors.HexColor("#CBD5E1")
BRAND_WHITE        = colors.HexColor("#FFFFFF")
BRAND_COVER_BG     = colors.HexColor("#0F172A")
BRAND_COVER_ACCENT = colors.HexColor("#00D4FF")

# Callout background tints
CALLOUT_INFO_BG      = colors.HexColor("#EFF6FF")
CALLOUT_INFO_BORDER  = colors.HexColor("#3B82F6")
CALLOUT_WARN_BG      = colors.HexColor("#FFFBEB")
CALLOUT_WARN_BORDER  = colors.HexColor("#F59E0B")
CALLOUT_OK_BG        = colors.HexColor("#F0FDF4")
CALLOUT_OK_BORDER    = colors.HexColor("#22C55E")
CALLOUT_DANGER_BG    = colors.HexColor("#FEF2F2")
CALLOUT_DANGER_BORDER= colors.HexColor("#EF4444")

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
LOGO_PATH = os.path.join(_ASSETS_DIR, "logos", "8_transparent.svg")

COMPANY_NAME = "FinTwin AI"

DISCLAIMER_TEXT = (
    "Disclaimer: This report is generated using Artificial Intelligence and predictive "
    "models. AI-generated insights may contain inaccuracies or omissions. Please verify "
    "all important financial information before making any financial or investment "
    "decisions."
)

PAGE_SIZE = A4
PAGE_W, PAGE_H = PAGE_SIZE
MARGIN = 16 * mm
HEADER_RESERVE = 26 * mm
FOOTER_RESERVE = 14 * mm
CONTENT_WIDTH = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------- #
# Header / Footer drawing helpers
# --------------------------------------------------------------------------- #
def _draw_logo(c: pdfcanvas.Canvas, x: float, y_baseline: float, target_h: float = 8.5 * mm):
    """Draw the FinTwin AI logo (vector SVG if available, else a text wordmark)."""
    if _SVG_AVAILABLE and os.path.exists(LOGO_PATH):
        try:
            drawing = svg2rlg(LOGO_PATH)
            scale = target_h / float(drawing.height)
            c.saveState()
            c.translate(x, y_baseline)
            c.scale(scale, scale)
            renderPDF.draw(drawing, c, 0, 0)
            c.restoreState()
            return drawing.width * scale
        except Exception:
            pass
    # Fallback: styled text wordmark
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(BRAND_PRIMARY)
    c.drawString(x, y_baseline + 1, "FinTwin")
    c.setFillColor(BRAND_CYAN)
    c.drawString(x + c.stringWidth("FinTwin", "Helvetica-Bold", 13), y_baseline + 1, " AI")
    return 52


def _draw_header(c: pdfcanvas.Canvas, ctx: Dict[str, Any]):
    c.saveState()
    top = PAGE_H - MARGIN

    # Subtle header background
    c.setFillColor(colors.HexColor("#F8FAFC"))
    c.rect(MARGIN - 2, top - 14 * mm, PAGE_W - 2 * MARGIN + 4, 14 * mm, fill=1, stroke=0)

    _draw_logo(c, MARGIN + 2, top - 10 * mm)

    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColor(BRAND_DARK)
    c.drawCentredString(PAGE_W / 2, top - 6 * mm, ctx["report_title"])

    c.setFont("Helvetica", 7.5)
    c.setFillColor(BRAND_GREY)
    meta_bits = [f"Generated: {ctx['generated_at']}"]
    if ctx.get("report_id"):
        meta_bits.append(f"ID: {ctx['report_id']}")
    c.drawRightString(PAGE_W - MARGIN, top - 5.5 * mm, "   |   ".join(meta_bits))

    c.setFont("Helvetica", 7.5)
    c.setFillColor(BRAND_GREY)
    c.drawRightString(PAGE_W - MARGIN, top - 10 * mm, f"Page {ctx['page_num']} of {ctx['total_pages']}")

    # Two-tone accent rule
    c.setStrokeColor(BRAND_PRIMARY)
    c.setLineWidth(2.0)
    c.line(MARGIN, top - 14 * mm, PAGE_W / 2, top - 14 * mm)
    c.setStrokeColor(BRAND_CYAN)
    c.setLineWidth(2.0)
    c.line(PAGE_W / 2, top - 14 * mm, PAGE_W - MARGIN, top - 14 * mm)
    c.restoreState()


def _draw_footer(c: pdfcanvas.Canvas, ctx: Dict[str, Any]):
    c.saveState()
    y_rule = MARGIN + 5 * mm
    y_text = MARGIN + 1.5 * mm

    c.setStrokeColor(BRAND_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, y_rule, PAGE_W - MARGIN, y_rule)

    c.setFont("Helvetica", 7.0)
    c.setFillColor(BRAND_GREY)
    c.drawString(MARGIN, y_text, f"{COMPANY_NAME}  *  AI-Powered Financial Intelligence  *  Confidential")
    c.drawCentredString(PAGE_W / 2, y_text, f"Page {ctx['page_num']} of {ctx['total_pages']}")
    c.drawRightString(PAGE_W - MARGIN, y_text, "For personal use only")
    c.restoreState()


class _NumberedCanvas(pdfcanvas.Canvas):
    """Buffers every page so the true 'Page X of Y' total can be rendered."""

    def __init__(self, *args, report_ctx: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self._report_ctx = report_ctx
        self._saved_states: List[dict] = []

    def showPage(self):
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_states)
        for state in self._saved_states:
            self.__dict__.update(state)
            ctx = dict(self._report_ctx)
            ctx["page_num"] = self._pageNumber
            ctx["total_pages"] = total_pages
            _draw_header(self, ctx)
            _draw_footer(self, ctx)
            super().showPage()
        super().save()


# --------------------------------------------------------------------------- #
# Cover page drawing (full-page canvas)
# --------------------------------------------------------------------------- #
def _wrap_text_simple(text: str, max_chars: int = 40) -> List[str]:
    """Simple word-wrap for canvas drawString usage."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _draw_cover_page(
    c: pdfcanvas.Canvas,
    report_title: str,
    report_type: str,
    user_name: str,
    report_id: str,
    generated_at: str,
    subtitle: str = "",
):
    """Draw a full-page branded cover on the current canvas page."""
    c.saveState()

    # Dark navy background
    c.setFillColor(BRAND_COVER_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Top cyan accent bar
    c.setFillColor(BRAND_COVER_ACCENT)
    c.rect(0, PAGE_H - 7 * mm, PAGE_W, 7 * mm, fill=1, stroke=0)

    # Logo wordmark top-left
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(BRAND_WHITE)
    c.drawString(MARGIN, PAGE_H - 18 * mm, "FinTwin")
    logo_w = c.stringWidth("FinTwin", "Helvetica-Bold", 16)
    c.setFillColor(BRAND_COVER_ACCENT)
    c.drawString(MARGIN + logo_w, PAGE_H - 18 * mm, " AI")

    # Report type badge (small caps, cyan)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(BRAND_COVER_ACCENT)
    c.drawString(MARGIN, PAGE_H * 0.62, report_type.upper())

    # Thin cyan underline
    c.setStrokeColor(BRAND_COVER_ACCENT)
    c.setLineWidth(1.5)
    c.line(MARGIN, PAGE_H * 0.615, MARGIN + 40 * mm, PAGE_H * 0.615)

    # Main report title
    title_lines = _wrap_text_simple(report_title, max_chars=32)
    y_title = PAGE_H * 0.58
    for line in title_lines:
        c.setFont("Helvetica-Bold", 26)
        c.setFillColor(BRAND_WHITE)
        c.drawString(MARGIN, y_title, line)
        y_title -= 30

    # Subtitle
    if subtitle:
        c.setFont("Helvetica", 11.5)
        c.setFillColor(colors.HexColor("#94A3B8"))
        sub_lines = _wrap_text_simple(subtitle, max_chars=55)
        y_sub = y_title - 6
        for line in sub_lines:
            c.drawString(MARGIN, y_sub, line)
            y_sub -= 15

    # Horizontal divider
    div_y = PAGE_H * 0.35
    c.setStrokeColor(colors.HexColor("#1E3A5F"))
    c.setLineWidth(1)
    c.line(MARGIN, div_y, PAGE_W - MARGIN, div_y)

    # Metadata block
    meta_y = div_y - 10 * mm
    meta_items = [
        ("Prepared For", user_name or "FinTwin AI User"),
        ("Report ID", report_id or "---"),
        ("Generated On", generated_at),
        ("Classification", "Confidential -- Personal Use Only"),
    ]
    for label, value in meta_items:
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#64748B"))
        c.drawString(MARGIN, meta_y, label.upper())
        c.setFont("Helvetica", 9.5)
        c.setFillColor(colors.HexColor("#E2E8F0"))
        c.drawString(MARGIN + 44 * mm, meta_y, str(value))
        meta_y -= 9 * mm

    # Bottom tagline
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawCentredString(
        PAGE_W / 2, MARGIN + 5 * mm,
        "AI-Powered Financial Intelligence  *  FinTwin AI"
    )

    # Bottom accent bar
    c.setFillColor(BRAND_PRIMARY)
    c.rect(0, 0, PAGE_W, 3 * mm, fill=1, stroke=0)

    c.restoreState()


# --------------------------------------------------------------------------- #
# Paragraph / table styles
# --------------------------------------------------------------------------- #
def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "H1": ParagraphStyle(
            "FT_H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, textColor=BRAND_DARK, spaceBefore=10, spaceAfter=5,
            borderPadding=0,
        ),
        "H2": ParagraphStyle(
            "FT_H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.5, textColor=BRAND_PRIMARY, spaceBefore=7, spaceAfter=3,
        ),
        "Body": ParagraphStyle(
            "FT_Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.3, leading=13.5, textColor=BRAND_DARK, spaceAfter=4,
        ),
        "Caption": ParagraphStyle(
            "FT_Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=8, leading=11, textColor=BRAND_GREY, alignment=TA_CENTER,
            spaceBefore=2, spaceAfter=10,
        ),
        "CoverTitle": ParagraphStyle(
            "FT_CoverTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=28, textColor=BRAND_WHITE, alignment=TA_LEFT, spaceAfter=8,
        ),
        "CoverSub": ParagraphStyle(
            "FT_CoverSub", parent=base["BodyText"], fontName="Helvetica",
            fontSize=12, textColor=colors.HexColor("#94A3B8"), alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "CoverMeta": ParagraphStyle(
            "FT_CoverMeta", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.5, textColor=colors.HexColor("#CBD5E1"), alignment=TA_LEFT,
            spaceAfter=3,
        ),
        "Disclaimer": ParagraphStyle(
            "FT_Disclaimer", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=7.8, leading=11, textColor=BRAND_GREY, spaceBefore=16,
            borderColor=BRAND_BORDER, borderWidth=0.6, borderPadding=8,
            backColor=BRAND_LIGHT_ROW,
        ),
        "TableCell": ParagraphStyle(
            "FT_TableCell", fontName="Helvetica", fontSize=8.3, leading=10.5,
            textColor=BRAND_DARK,
        ),
        "TableCellRight": ParagraphStyle(
            "FT_TableCellRight", fontName="Helvetica", fontSize=8.3, leading=10.5,
            textColor=BRAND_DARK, alignment=TA_RIGHT,
        ),
        "TableHeader": ParagraphStyle(
            "FT_TableHeader", fontName="Helvetica-Bold", fontSize=8.5, leading=10.5,
            textColor=colors.white,
        ),
        "KPILabel": ParagraphStyle(
            "FT_KPILabel", fontName="Helvetica", fontSize=7.8, leading=10,
            textColor=BRAND_GREY, alignment=TA_LEFT,
        ),
        "KPIValue": ParagraphStyle(
            "FT_KPIValue", fontName="Helvetica-Bold", fontSize=13, leading=16,
            textColor=BRAND_DARK, alignment=TA_LEFT,
        ),
        "KPIValuePos": ParagraphStyle(
            "FT_KPIValuePos", fontName="Helvetica-Bold", fontSize=13, leading=16,
            textColor=BRAND_GREEN, alignment=TA_LEFT,
        ),
        "KPIValueNeg": ParagraphStyle(
            "FT_KPIValueNeg", fontName="Helvetica-Bold", fontSize=13, leading=16,
            textColor=BRAND_RED, alignment=TA_LEFT,
        ),
        "CalloutText": ParagraphStyle(
            "FT_CalloutText", fontName="Helvetica", fontSize=9, leading=13,
            textColor=BRAND_DARK, spaceBefore=0, spaceAfter=0,
        ),
        "SectionDividerText": ParagraphStyle(
            "FT_SectionDivider", fontName="Helvetica-Bold", fontSize=11,
            textColor=BRAND_DARK, leading=14,
        ),
        "TOCEntry": ParagraphStyle(
            "FT_TOCEntry", fontName="Helvetica", fontSize=9.5, leading=14,
            textColor=BRAND_DARK, leftIndent=10, spaceAfter=2,
        ),
        "ExecSummary": ParagraphStyle(
            "FT_ExecSummary", fontName="Helvetica", fontSize=9.5, leading=13.5,
            textColor=BRAND_NAVY, spaceBefore=4, spaceAfter=6,
            borderColor=BRAND_PRIMARY, borderWidth=0.8, borderPadding=8,
            backColor=colors.HexColor("#EFF6FF"),
        ),
    }
    return styles


def _clean_text_for_pdf(text: str, is_html: bool = True) -> str:
    if text is None:
        return ""
    text = str(text)

    # Replace Font Awesome / HTML icon tags before escaping
    text = re.sub(r'<i[^>]*arrow-trend-up[^>]*>.*?</i>', '(+) ', text, flags=re.IGNORECASE)
    text = re.sub(r'<i[^>]*arrow-trend-down[^>]*>.*?</i>', '(-) ', text, flags=re.IGNORECASE)
    text = re.sub(r'<i[^>]*circle-check[^>]*>.*?</i>', '[OK] ', text, flags=re.IGNORECASE)
    text = re.sub(r'<i[^>]*triangle-exclamation[^>]*>.*?</i>', '[!] ', text, flags=re.IGNORECASE)
    text = re.sub(r'<i\s+class=[^>]*>.*?</i>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:span|div|p|small|section|aside|header|strong)[^>]*>', '', text, flags=re.IGNORECASE)

    # Convert markdown bold (**) and italic (*) to HTML tags
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)

    # Convert unicode symbols & emojis to PDF-safe equivalents
    text = text.replace("\u20b9", "Rs. ")
    text = text.replace("\U0001f7e2", "+ ")
    text = text.replace("\U0001f534", "- ")
    text = text.replace("\u26a0\ufe0f", "[!] ")
    text = text.replace("<strong>", "<b>").replace("</strong>", "</b>")

    if is_html:
        safe = html.escape(text)
        # Restore allowed ReportLab tag markups
        safe = safe.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
        safe = safe.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
        safe = safe.replace("&lt;u&gt;", "<u>").replace("&lt;/u&gt;", "</u>")
        safe = safe.replace("&lt;br/&gt;", "<br/>").replace("&lt;br&gt;", "<br/>")
        safe = safe.replace("\n", "<br/>")
        return safe
    else:
        return html.escape(text)


def _fmt_cell(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "\u2014"
    if isinstance(val, float):
        return f"{val:,.2f}"
    return _clean_text_for_pdf(val)


def _convert_to_light_mode(fig):
    import copy
    fig_light = copy.deepcopy(fig)

    # Apply white template
    try:
        fig_light.update_layout(template="plotly_white")
    except Exception:
        pass

    # Make background transparent to match PDF background
    fig_light.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    # Update global layout font to dark slate
    if fig_light.layout.font:
        if isinstance(fig_light.layout.font, dict):
            fig_light.layout.font["color"] = "#1E293B"
        else:
            fig_light.layout.font.color = "#1E293B"
    else:
        fig_light.layout.font = dict(color="#1E293B")

    # Update title font to dark slate
    if fig_light.layout.title:
        title = fig_light.layout.title
        if isinstance(title, dict):
            if "font" in title and title["font"]:
                if isinstance(title["font"], dict):
                    title["font"]["color"] = "#0F172A"
                else:
                    title["font"].color = "#0F172A"
            else:
                title["font"] = dict(color="#0F172A")
        else:
            if hasattr(title, "font") and title.font:
                title.font.color = "#0F172A"
            else:
                title.font = dict(color="#0F172A")

    # Update X and Y axes for cartesian charts
    try:
        fig_light.update_xaxes(
            linecolor="#CBD5E1",
            gridcolor="#F1F5F9",
            title_font=dict(color="#0F172A"),
            tickfont=dict(color="#334155")
        )
        fig_light.update_yaxes(
            linecolor="#CBD5E1",
            gridcolor="#F1F5F9",
            title_font=dict(color="#0F172A"),
            tickfont=dict(color="#334155")
        )
    except Exception:
        pass

    # Update indicator (gauge) traces
    try:
        fig_light.update_traces(
            selector=dict(type="indicator"),
            title_font=dict(color="#0F172A"),
            gauge_bgcolor="#F1F5F9",
            gauge_bordercolor="#CBD5E1",
            gauge_axis_tickcolor="#334155",
            gauge_axis_tickfont=dict(color="#334155"),
        )
    except Exception:
        pass

    return fig_light


class FinTwinPDFReport:
    """Builds a single professional, multi-page A4 PDF report."""

    def __init__(
        self,
        report_title: str,
        user_id: Optional[Union[str, int]] = None,
        report_id: Optional[str] = None,
        subtitle: Optional[str] = None,
    ):
        self.report_title = report_title
        self.user_id = user_id
        self._cover_drawn = False
        self._cover_params: Optional[Dict[str, Any]] = None

        # Obfuscate user_id in report_id if it contains database user keys
        import hashlib
        if report_id:
            if user_id and isinstance(user_id, str) and user_id.startswith("user_"):
                anon_suffix = hashlib.md5(user_id.encode()).hexdigest()[:8].upper()
                self.report_id = report_id.replace(user_id, anon_suffix)
            else:
                self.report_id = report_id
        else:
            if user_id and isinstance(user_id, str) and user_id.startswith("user_"):
                anon_suffix = hashlib.md5(user_id.encode()).hexdigest()[:8].upper()
                self.report_id = f"FT-{anon_suffix}"
            else:
                self.report_id = f"FT-{user_id}" if user_id is not None else None

        self.subtitle = subtitle
        self.generated_at = datetime.now()
        self.styles = _build_styles()
        self.story: List[Any] = []
        # Build the default inline cover (overridden if add_cover_page() is called)
        self._build_inline_cover()

    # ------------------------------------------------------------------ #
    # Inline cover block (minimal fallback when add_cover_page not called)
    # ------------------------------------------------------------------ #
    def _build_inline_cover(self):
        s = self.styles
        self.story.append(Spacer(1, 4 * mm))
        self.story.append(Paragraph(html.escape(self.report_title), s["H1"]))
        if self.subtitle:
            self.story.append(Paragraph(html.escape(self.subtitle), s["Body"]))
        meta = f"Generated on {self.generated_at.strftime('%d %b %Y, %H:%M')}"
        if self.report_id:
            meta += f"   |   Report ID: {self.report_id}"
        self.story.append(Paragraph(meta, s["Body"]))
        self.story.append(Spacer(1, 3 * mm))
        # Divider
        divider = Table([[""]], colWidths=[CONTENT_WIDTH], rowHeights=[1.4])
        divider.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BRAND_PRIMARY)]))
        self.story.append(divider)
        self.story.append(Spacer(1, 5 * mm))

    # ------------------------------------------------------------------ #
    # NEW: Full-page branded cover page
    # ------------------------------------------------------------------ #
    def add_cover_page(
        self,
        user_name: str = "",
        report_type: str = "Financial Intelligence Report",
    ):
        """
        Register a full-page branded cover page.
        Must be called right after __init__, before adding any content.
        Clears the default inline cover and replaces it with a full-page design.
        """
        self.story.clear()
        self._cover_params = {
            "user_name": user_name,
            "report_type": report_type,
            "report_title": self.report_title,
            "subtitle": self.subtitle or "",
            "report_id": self.report_id or "",
            "generated_at": self.generated_at.strftime("%d %B %Y, %H:%M"),
        }
        self._cover_drawn = True

    # ------------------------------------------------------------------ #
    # NEW: Table of Contents
    # ------------------------------------------------------------------ #
    def add_table_of_contents(self, sections: List[str]):
        """
        Render a clean numbered table of contents.
        Call after add_cover_page(), before any section content.
        """
        self.add_section_divider("Table of Contents")
        toc_rows = []
        for i, section in enumerate(sections, start=1):
            toc_rows.append([
                Paragraph(f"{i}.", self.styles["TOCEntry"]),
                Paragraph(html.escape(str(section)), self.styles["TOCEntry"]),
            ])
        if toc_rows:
            t = Table(toc_rows, colWidths=[12 * mm, CONTENT_WIDTH - 12 * mm], hAlign="LEFT")
            t.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, BRAND_BORDER),
            ]))
            self.story.append(t)
        self.story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------ #
    # NEW: Section Divider (branded major section header bar)
    # ------------------------------------------------------------------ #
    def add_section_divider(self, title: str, icon_char: str = ""):
        """
        Full-width branded section divider.
        Use for major sections; use add_section() for sub-sections.
        """
        display_title = f"{icon_char}  {title}" if icon_char else title
        label = Paragraph(html.escape(display_title), self.styles["SectionDividerText"])
        t = Table([[label]], colWidths=[CONTENT_WIDTH])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBEFORE", (0, 0), (0, -1), 4, BRAND_PRIMARY),
            ("LINEABOVE", (0, 0), (-1, 0), 0.4, BRAND_BORDER),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
        ]))
        self.story.append(Spacer(1, 4 * mm))
        self.story.append(t)
        self.story.append(Spacer(1, 3 * mm))

    # ------------------------------------------------------------------ #
    # NEW: KPI Summary Row (horizontal mini-cards)
    # ------------------------------------------------------------------ #
    def add_kpi_summary_row(self, kpis: List[Dict[str, Any]]):
        """
        Render a horizontal row of up to 4 KPI mini-cards.
        Each dict: {"label": str, "value": str, "status": "positive"|"negative"|"neutral"}
        """
        if not kpis:
            return
        kpis = kpis[:4]
        n = len(kpis)
        card_w = CONTENT_WIDTH / n

        card_contents = []
        for kpi in kpis:
            status = kpi.get("status", "neutral")
            if status == "positive":
                val_style = self.styles["KPIValuePos"]
            elif status == "negative":
                val_style = self.styles["KPIValueNeg"]
            else:
                val_style = self.styles["KPIValue"]

            label_p = Paragraph(html.escape(str(kpi.get("label", ""))), self.styles["KPILabel"])
            value_p = Paragraph(html.escape(str(kpi.get("value", "\u2014"))), val_style)
            inner = Table([[label_p], [value_p]], colWidths=[card_w - 10 * mm])
            inner.setStyle(TableStyle([
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            card_contents.append(inner)

        outer = Table([card_contents], colWidths=[card_w] * n)
        style_cmds = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_WHITE),
            ("BOX", (0, 0), (-1, -1), 0.5, BRAND_BORDER),
            ("LINEAFTER", (0, 0), (-2, -1), 0.5, BRAND_BORDER),
        ]
        for i in range(n):
            style_cmds.append(("LINEBEFORE", (i, 0), (i, 0), 3, BRAND_PRIMARY))
        outer.setStyle(TableStyle(style_cmds))
        self.story.append(outer)
        self.story.append(Spacer(1, 5 * mm))

    # ------------------------------------------------------------------ #
    # NEW: Executive Summary block
    # ------------------------------------------------------------------ #
    def add_executive_summary(self, summary_text: str, kpis: Optional[Dict[str, Any]] = None):
        """
        Renders a styled executive summary block with optional KPI grid.
        summary_text: prose paragraph summarizing the section.
        kpis: optional dict of {label: value} for a 2-column grid below.
        """
        if summary_text and str(summary_text).strip():
            safe = _clean_text_for_pdf(summary_text)
            try:
                self.story.append(Paragraph(safe, self.styles["ExecSummary"]))
            except Exception:
                plain = html.escape(re.sub(r'<[^>]+>', '', str(summary_text)))
                self.story.append(Paragraph(plain, self.styles["ExecSummary"]))
            self.story.append(Spacer(1, 3 * mm))
        if kpis:
            self.add_key_value_grid(kpis)

    # ------------------------------------------------------------------ #
    # NEW: Callout Box
    # ------------------------------------------------------------------ #
    def add_callout(self, text: str, style: str = "info", title: str = ""):
        """
        Render a colored callout box for key insights or alerts.
        style: "info" | "warning" | "success" | "danger"
        """
        if not text or not str(text).strip():
            return

        style_map = {
            "info":    (CALLOUT_INFO_BG,    CALLOUT_INFO_BORDER,    "Info"),
            "warning": (CALLOUT_WARN_BG,    CALLOUT_WARN_BORDER,    "Note"),
            "success": (CALLOUT_OK_BG,      CALLOUT_OK_BORDER,      ""),
            "danger":  (CALLOUT_DANGER_BG,  CALLOUT_DANGER_BORDER,  "Alert"),
        }
        bg, border_color, default_label = style_map.get(style, style_map["info"])
        display_title = title or default_label

        safe_text = _clean_text_for_pdf(str(text))
        prefix = f"<b>{html.escape(display_title)}:</b> " if display_title else ""
        try:
            callout_p = Paragraph(prefix + safe_text, self.styles["CalloutText"])
        except Exception:
            callout_p = Paragraph(html.escape(re.sub(r'<[^>]+>', '', str(text))), self.styles["CalloutText"])

        t = Table([[callout_p]], colWidths=[CONTENT_WIDTH])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("LINEBEFORE", (0, 0), (0, -1), 4, border_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 4 * mm))

    # ------------------------------------------------------------------ #
    # Content builders
    # ------------------------------------------------------------------ #
    def add_section(self, title: str, level: int = 1):
        """Add a section/sub-section heading."""
        style = self.styles["H1"] if level == 1 else self.styles["H2"]
        clean_title = _clean_text_for_pdf(title, is_html=False)
        self.story.append(Paragraph(clean_title, style))

    def add_paragraph(self, text: str):
        if text is None or str(text).strip() == "":
            return
        safe = _clean_text_for_pdf(text)
        try:
            self.story.append(Paragraph(safe, self.styles["Body"]))
        except Exception:
            plain = html.escape(re.sub(r'<[^>]+>', '', str(text)))
            self.story.append(Paragraph(plain, self.styles["Body"]))

    def add_spacer(self, height_mm: float = 3):
        self.story.append(Spacer(1, height_mm * mm))

    def add_page_break(self):
        self.story.append(PageBreak())

    def add_key_value_grid(self, data: Dict[str, Any], columns: int = 2):
        """Render a dict as a clean card-style labeled grid with accent borders."""
        if not data:
            return
        items = [(str(k).replace("_", " ").title(), _fmt_cell(v)) for k, v in data.items()]
        rows = []
        for i in range(0, len(items), columns):
            row = []
            for k, v in items[i: i + columns]:
                row.extend([k, v])
            while len(row) < columns * 2:
                row.extend(["", ""])
            rows.append(row)

        col_w = CONTENT_WIDTH / (columns * 2)
        widths = []
        for _ in range(columns):
            widths.extend([col_w * 0.82, col_w * 1.18])

        cell_style_label = ParagraphStyle(
            "kv_label", fontName="Helvetica", fontSize=8.2, textColor=BRAND_GREY
        )
        cell_style_val = ParagraphStyle(
            "kv_val", fontName="Helvetica-Bold", fontSize=9.5, textColor=BRAND_DARK
        )
        table_rows = []
        for row in rows:
            styled = []
            for idx, cell in enumerate(row):
                style = cell_style_label if idx % 2 == 0 else cell_style_val
                styled.append(Paragraph(str(cell), style) if cell != "" else "")
            table_rows.append(styled)

        t = Table(table_rows, colWidths=widths, hAlign="LEFT")
        style_cmds = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_WHITE),
        ]
        for i in range(0, len(table_rows), 2):
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), BRAND_LIGHT_ROW))
        for col in range(0, columns * 2, 2):
            style_cmds.append(("LINEBEFORE", (col, 0), (col, -1), 2.5, BRAND_PRIMARY))
        t.setStyle(TableStyle(style_cmds))
        self.story.append(t)
        self.story.append(Spacer(1, 5 * mm))

    def add_dataframe_table(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        max_col_width_chars: int = 28,
    ):
        """Render a DataFrame as a professionally styled, auto-wrapping table
        that automatically continues across pages when it is long."""
        if df is None or df.empty:
            if title:
                self.add_section(title, level=2)
            self.add_paragraph("No data available for this section.")
            return

        if title:
            self.add_section(title, level=2)

        df = df.copy()
        df = df.dropna(axis=1, how="all")
        if df.empty:
            self.add_paragraph("No data available for this section.")
            return

        headers = [str(c).replace("_", " ").title() for c in df.columns]
        header_style = self.styles["TableHeader"]
        cell_style = self.styles["TableCell"]
        cell_right = self.styles["TableCellRight"]

        # Detect numeric columns for right-alignment
        numeric_cols = set()
        for ci, col in enumerate(df.columns):
            try:
                if pd.to_numeric(df[col].dropna(), errors="coerce").notna().mean() > 0.7:
                    numeric_cols.add(ci)
            except Exception:
                pass

        data_rows = [[Paragraph(h, header_style) for h in headers]]
        for _, row in df.iterrows():
            data_row = []
            for ci, v in enumerate(row.tolist()):
                st = cell_right if ci in numeric_cols else cell_style
                data_row.append(Paragraph(_fmt_cell(v), st))
            data_rows.append(data_row)

        n_cols = len(headers)

        col_widths_chars = []
        for col_idx in range(n_cols):
            max_len = len(headers[col_idx])
            for _, row in df.iterrows():
                val = row.iloc[col_idx]
                formatted_val = _fmt_cell(val)
                clean_val = re.sub(r'<[^>]*>', '', formatted_val)
                max_len = max(max_len, len(clean_val))
            max_len = max(min(max_len, max_col_width_chars), 6)
            col_widths_chars.append(max_len)

        total_chars = sum(col_widths_chars)
        widths = [(w / total_chars) * CONTENT_WIDTH for w in col_widths_chars]

        t = Table(data_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, BRAND_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BRAND_WHITE, BRAND_LIGHT_ROW]),
        ]
        for ci in numeric_cols:
            style_cmds.append(("ALIGN", (ci, 1), (ci, -1), "RIGHT"))

        t.setStyle(TableStyle(style_cmds))
        self.story.append(t)
        self.story.append(Spacer(1, 5 * mm))

    def add_image_bytes(
        self,
        image_bytes: bytes,
        caption: Optional[str] = None,
        max_height_mm: float = 90,
    ):
        """Center an already-rendered image (PNG bytes), scaled to fit the
        page width/height without stretching, with an optional caption."""
        if not image_bytes:
            return
        try:
            img_reader = RLImage(io.BytesIO(image_bytes))
            iw, ih = img_reader.imageWidth, img_reader.imageHeight
            if not iw or not ih:
                return
            aspect = ih / float(iw)
            draw_w = CONTENT_WIDTH
            draw_h = draw_w * aspect
            max_h = max_height_mm * mm
            if draw_h > max_h:
                draw_h = max_h
                draw_w = draw_h / aspect
            img = RLImage(io.BytesIO(image_bytes), width=draw_w, height=draw_h)
            img.hAlign = "CENTER"
            block = [img]
            if caption:
                block.append(Paragraph(html.escape(caption), self.styles["Caption"]))
            self.story.append(KeepTogether(block))
        except Exception:
            self.add_paragraph(f"[Chart unavailable: {caption or 'image failed to render'}]")

    def add_plotly_figure(self, fig, caption: Optional[str] = None, scale: float = 2.0):
        """Render a Plotly go.Figure to high-resolution PNG (via kaleido) and
        embed it. Fails gracefully if kaleido/plotly image export is unavailable."""
        if fig is None:
            return
        try:
            fig_light = _convert_to_light_mode(fig)
            img_bytes = fig_light.to_image(format="png", scale=scale, engine="kaleido")
            self.add_image_bytes(img_bytes, caption=caption)
        except Exception as e:
            self.add_paragraph(
                f"[Chart '{caption or 'chart'}' could not be rendered: {e}]"
            )

    def add_matplotlib_figure(self, fig, caption: Optional[str] = None, dpi: int = 200):
        """Render a Matplotlib figure to high-resolution PNG and embed it."""
        if fig is None:
            return
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            self.add_image_bytes(buf.read(), caption=caption)
        except Exception as e:
            self.add_paragraph(
                f"[Chart '{caption or 'chart'}' could not be rendered: {e}]"
            )

    def add_disclaimer(self):
        """Append the mandatory AI disclaimer as the final block of content."""
        self.story.append(Spacer(1, 6 * mm))
        self.story.append(Paragraph(html.escape(DISCLAIMER_TEXT), self.styles["Disclaimer"]))

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self) -> bytes:
        """Finalize the story and render the complete PDF, returning it as bytes."""
        if not self.story or DISCLAIMER_TEXT not in getattr(
            self.story[-1], "text", ""
        ):
            self.add_disclaimer()

        buf = io.BytesIO()
        ctx = {
            "report_title": self.report_title,
            "generated_at": self.generated_at.strftime("%d %b %Y, %H:%M"),
            "report_id": self.report_id,
        }

        def _make_canvas(*args, **kwargs):
            return _NumberedCanvas(*args, report_ctx=ctx, **kwargs)

        doc = BaseDocTemplate(
            buf,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN + HEADER_RESERVE,
            bottomMargin=MARGIN + FOOTER_RESERVE,
            title=self.report_title,
            author="",
            creator="",
            subject="",
        )
        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="main",
        )
        doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

        if self._cover_drawn and self._cover_params:
            p = self._cover_params

            class _CoverNumberedCanvas(_NumberedCanvas):
                def save(self_inner):
                    total_pages = len(self_inner._saved_states)
                    first = True
                    for state in self_inner._saved_states:
                        self_inner.__dict__.update(state)
                        ctx_page = dict(self_inner._report_ctx)
                        ctx_page["page_num"] = self_inner._pageNumber
                        ctx_page["total_pages"] = total_pages
                        if first:
                            _draw_cover_page(
                                self_inner,
                                report_title=p["report_title"],
                                report_type=p["report_type"],
                                user_name=p["user_name"],
                                report_id=p["report_id"],
                                generated_at=p["generated_at"],
                                subtitle=p["subtitle"],
                            )
                            first = False
                        else:
                            _draw_header(self_inner, ctx_page)
                            _draw_footer(self_inner, ctx_page)
                        pdfcanvas.Canvas.showPage(self_inner)
                    pdfcanvas.Canvas.save(self_inner)

            # PageBreak ensures cover is page 1, content starts page 2
            story_with_break = [PageBreak()] + self.story
            doc.build(
                story_with_break,
                canvasmaker=lambda *a, **k: _CoverNumberedCanvas(*a, report_ctx=ctx, **k)
            )
        else:
            doc.build(self.story, canvasmaker=_make_canvas)

        return buf.getvalue()

    def suggested_filename(self, prefix: str) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d_%H-%M-%S")
        safe_prefix = "".join(c for c in prefix if c.isalnum() or c in ("_", "-")) or "report"
        return f"FinTwinAI_{safe_prefix}_{ts}.pdf"
