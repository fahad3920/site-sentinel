"""Weekly safety + progress report: a PDF and copy-paste-ready WhatsApp/email text.

Privacy-first: the report references annotated thumbnails only, never names or
faces. Text payloads are plain and short so a site supervisor can forward them.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import config


@dataclass
class WeekSummary:
    site_name: str
    period_start: str
    period_end: str
    inspections: int
    average_score: float
    worst_score: int
    total_violations: int
    top_issues: List[str]
    progress_percent: Optional[float] = None
    progress_confirmed: bool = False
    highlights: Optional[List[str]] = None


def _whatsapp_text(s: WeekSummary) -> str:
    lines = [
        f"*Site Sentinel weekly report - {s.site_name}*",
        f"{s.period_start} to {s.period_end}",
        "",
        f"Safety score (avg): {s.average_score:.0f}/100  |  worst day: {s.worst_score}/100",
        f"Inspections: {s.inspections}  |  violations flagged: {s.total_violations}",
    ]
    if s.top_issues:
        lines.append("")
        lines.append("Top issues:")
        for issue in s.top_issues:
            lines.append(f"- {issue}")
    if s.progress_percent is not None:
        tag = "confirmed" if s.progress_confirmed else "estimate, needs confirmation"
        lines.append("")
        lines.append(f"Progress: {s.progress_percent:.0f}% ({tag})")
    lines.append("")
    lines.append("_Estimate only. Please confirm with a site supervisor._")
    return "\n".join(lines)


def _email_text(s: WeekSummary) -> str:
    body = _whatsapp_text(s).replace("*", "").replace("_", "")
    return f"Subject: Weekly Site Safety Report - {s.site_name}\n\n{body}\n"


def build_texts(s: WeekSummary) -> dict:
    return {"whatsapp": _whatsapp_text(s), "email": _email_text(s)}


def build_pdf(s: WeekSummary, out_path: Path, thumbnail_paths: Optional[List[Path]] = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    orange = colors.HexColor("#EA580C")
    dark = colors.HexColor("#0B1220")
    story = []

    story.append(Paragraph(f"<font color='#EA580C'><b>Site Sentinel</b></font> "
                           f"weekly report", Paragraph and _para_style(16)))
    story.append(Paragraph(f"{s.site_name} &nbsp;|&nbsp; {s.period_start} to {s.period_end}",
                           _para_style(10, muted=True)))
    story.append(Spacer(1, 8 * mm))

    data = [
        ["Average safety score", f"{s.average_score:.0f}/100"],
        ["Worst day", f"{s.worst_score}/100"],
        ["Inspections", str(s.inspections)],
        ["Violations flagged", str(s.total_violations)],
    ]
    if s.progress_percent is not None:
        tag = "confirmed" if s.progress_confirmed else "estimate"
        data.append(["Progress", f"{s.progress_percent:.0f}% ({tag})"])
    t = Table(data, colWidths=[70 * mm, 90 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), dark),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("TEXTCOLOR", (1, 0), (1, -1), orange),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#1F2937")),
    ]))
    story.append(t)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Top issues", _para_style(12)))
    for issue in s.top_issues or ["None recorded"]:
        story.append(Paragraph(f"&bull; {issue}", _para_style(10)))
    story.append(Spacer(1, 6 * mm))

    if thumbnail_paths:
        story.append(Paragraph("Annotated inspections", _para_style(12)))
        row = []
        for p in thumbnail_paths[:3]:
            p = Path(p)
            if p.exists():
                try:
                    row.append(Image(str(p), width=52 * mm, height=36 * mm))
                except Exception:
                    pass
        if row:
            story.append(Table([row]))
            story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "<i>Site Sentinel provides an estimate to support supervisors. "
        "It is not a legal compliance measurement. Faces are blurred by default.</i>",
        _para_style(9, muted=True)))

    doc.build(story)
    return out_path


def _para_style(size: int, muted: bool = False):
    from reportlab.lib.styles import ParagraphStyle
    return ParagraphStyle(
        name=f"ss{size}{int(muted)}",
        fontName="Helvetica",
        fontSize=size,
        leading=size + 4,
        textColor=("#6B7280" if muted else "#111827"),
        spaceAfter=3,
    )
