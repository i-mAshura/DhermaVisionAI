
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

from datetime import datetime


# ── Colour palette ────────────────────────────────────────
NAVY    = colors.HexColor("#0f172a")
SLATE   = colors.HexColor("#1e293b")
BORDER  = colors.HexColor("#334155")
ACCENT  = colors.HexColor("#38bdf8")
WHITE   = colors.white
LIGHT   = colors.HexColor("#cbd5e1")
MUTED   = colors.HexColor("#64748b")
RED     = colors.HexColor("#ef4444")
AMBER   = colors.HexColor("#f59e0b")
GREEN   = colors.HexColor("#22c55e")


def _risk_color(risk_label: str):
    if "High" in risk_label:
        return RED
    if "Moderate" in risk_label:
        return AMBER
    return GREEN


def _bar(prob: float, width: float = 3.2 * inch, height: float = 0.18 * inch):
    """Return a mini Table that acts as a filled progress bar."""
    filled = max(prob / 100, 0.01)
    empty  = 1 - filled
    data   = [["", ""]]
    col_w  = [width * filled, width * empty]
    tbl    = Table(data, colWidths=col_w, rowHeights=[height])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), ACCENT),
        ("BACKGROUND", (1, 0), (1, 0), BORDER),
        ("LINEABOVE",  (0, 0), (-1, 0), 0, colors.transparent),
        ("LINEBELOW",  (0, 0), (-1, 0), 0, colors.transparent),
        ("LINEBEFORE", (0, 0), (0,  0), 0, colors.transparent),
        ("LINEAFTER",  (-1, 0), (-1, 0), 0, colors.transparent),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tbl


def create_pdf_report(
    pdf_path: str,
    prediction: str,
    full_name: str,
    confidence: float,
    risk_label: str,
    uploaded_image_path: str,
    heatmap_path: str,
    probabilities: dict,
    filename: str = "N/A",
) -> str:

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    # ── Custom styles ──────────────────────────────────────
    SS = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=SS["Title"],
        fontSize=22,
        textColor=WHITE,
        spaceAfter=2,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    sub_style = ParagraphStyle(
        "SubTitle",
        parent=SS["Normal"],
        fontSize=10,
        textColor=MUTED,
        spaceAfter=6,
        fontName="Helvetica",
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=SS["Heading2"],
        fontSize=11,
        textColor=ACCENT,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=SS["Normal"],
        fontSize=9,
        textColor=LIGHT,
        fontName="Helvetica",
        leading=14,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=SS["Normal"],
        fontSize=8,
        textColor=MUTED,
        fontName="Helvetica",
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=SS["Normal"],
        fontSize=8,
        textColor=MUTED,
        fontName="Helvetica-Oblique",
        alignment=TA_CENTER,
        leading=12,
    )

    elements = []

    # ── Header banner ──────────────────────────────────────
    header_data = [[
        Paragraph("🩺 DermaVision", title_style),
    ]]
    header_tbl = Table(header_data, colWidths=[doc.width])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [8]),
    ]))
    elements.append(header_tbl)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Clinical AI Analysis Report", sub_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}   |   File: {filename}",
        sub_style,
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=10))

    # ── Diagnosis summary card ─────────────────────────────
    elements.append(Paragraph("Diagnosis", section_style))

    rc = _risk_color(risk_label)
    diag_data = [
        [
            Paragraph("Predicted Condition", label_style),
            Paragraph("Confidence", label_style),
            Paragraph("Risk Level", label_style),
        ],
        [
            Paragraph(f"<b>{full_name} ({prediction})</b>", body_style),
            Paragraph(f"<b>{confidence*100:.2f}%</b>", body_style),
            Paragraph(f"<b>{risk_label}</b>", body_style),
        ],
    ]
    diag_tbl = Table(diag_data, colWidths=[doc.width * 0.45, doc.width * 0.25, doc.width * 0.30])
    diag_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), SLATE),
        ("BACKGROUND",    (0, 1), (-1, 1), NAVY),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",     (2, 1), (2, 1), rc),
        ("ROUNDEDCORNERS", [6]),
    ]))
    elements.append(diag_tbl)
    elements.append(Spacer(1, 8))

    # ── Disease information ────────────────────────────────
    elements.append(Paragraph("About This Condition", section_style))
    from utils.disease_info import DISEASE_INFO
    info_text = DISEASE_INFO.get(prediction, "No information available.")
    info_text = " ".join(info_text.split())  # collapse whitespace
    elements.append(Paragraph(info_text, body_style))

    # ── Images side by side ───────────────────────────────
    elements.append(Paragraph("Dermoscopic Image & Grad-CAM", section_style))

    img_w = (doc.width - 0.3 * inch) / 2
    img_h = img_w

    img_row = [[
        PDFImage(uploaded_image_path, width=img_w, height=img_h),
        PDFImage(heatmap_path,        width=img_w, height=img_h),
    ]]
    lbl_row = [[
        Paragraph("Uploaded Image", label_style),
        Paragraph("Grad-CAM Heatmap (warmer = higher attention)", label_style),
    ]]
    img_tbl = Table(img_row + lbl_row, colWidths=[img_w + 0.1 * inch, img_w + 0.1 * inch])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
    ]))
    elements.append(img_tbl)

    # ── Probability table with inline bars ────────────────
    elements.append(Paragraph("Class Probabilities", section_style))

    prob_header = [
        Paragraph("Disease", label_style),
        Paragraph("Probability", label_style),
        Paragraph("Distribution", label_style),
    ]
    prob_rows = [prob_header]
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for disease, prob in sorted_probs:
        row_style = body_style if disease != prediction else ParagraphStyle(
            "HighlightRow", parent=body_style,
            textColor=ACCENT, fontName="Helvetica-Bold",
        )
        prob_rows.append([
            Paragraph(disease, row_style),
            Paragraph(f"{prob:.2f}%", row_style),
            _bar(prob),
        ])

    bar_col_w = 3.2 * inch
    prob_tbl = Table(
        prob_rows,
        colWidths=[1.2 * inch, 1.0 * inch, bar_col_w],
    )
    prob_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), SLATE),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [NAVY, SLATE]),
    ])
    prob_tbl.setStyle(prob_style)
    elements.append(prob_tbl)

    # ── Disclaimer ─────────────────────────────────────────
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))
    elements.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model (DermaVision v2.0) and is "
        "intended for educational and research purposes only. It is not a medical device "
        "and should not be used as a substitute for professional dermatological evaluation. "
        "Always consult a qualified healthcare professional for medical advice.",
        disclaimer_style,
    ))

    # ── Build ──────────────────────────────────────────────
    def _bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(elements, onFirstPage=_bg, onLaterPages=_bg)

    return pdf_path
    