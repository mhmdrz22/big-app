"""PDF report generator for stakeholder briefings."""
from __future__ import annotations

import io
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register Persian font
pdfmetrics.registerFont(TTFont("Amiri", "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Amiri-Bold", "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"))


def build_stakeholder_pdf(summary: dict, drift_report: dict, metrics: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 30 * mm

    def line(text, size=12, font="Amiri", dy=8 * mm):
        nonlocal y
        c.setFont(font, size)
        c.drawRightString(width - 20 * mm, y, text)
        y -= dy

    def h1(text):
        nonlocal y
        y -= 5 * mm
        c.setFont("Amiri-Bold", 18)
        c.drawRightString(width - 20 * mm, y, text)
        y -= 12 * mm

    h1("گزارش مدیران آرام‌نگار")
    line(f"تاریخ تولید: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", size=10)

    h1("شاخص‌های کلیدی")
    store = summary.get("store", {})
    line(f"تعداد ارزیابی‌ها: {store.get('assessment_count', 0)}")
    line(f"تعداد بازخوردها: {store.get('feedback_count', 0)}")
    line(f"تعداد مدل‌های ثبت‌شده: {store.get('model_count', 0)}")

    h1("وضعیت مدل")
    line(f"مدل فعلی: {metrics.get('selected_model', 'نامشخص')}")
    best = next((c for c in metrics.get("candidates", []) if c["model"] == metrics.get("selected_model")), {})
    line(f"F1 Score: {best.get('f1', 0):.3f}")
    line(f"ROC AUC: {best.get('roc_auc', 0):.3f}")
    line(f"Accuracy: {best.get('accuracy', 0):.3f}")

    h1("رانش داده")
    line(f"سطح رانش: {drift_report.get('drift_level', 'نامشخص')}")
    line(f"تعداد ارزیابی‌های اخیر: {drift_report.get('population_size', 0)}")

    h1("توصیه")
    if drift_report.get("drift_level") == "critical":
        line("⚠️ رانش بحرانی است. بازآموزی فوری توصیه می‌شود.")
    elif drift_report.get("drift_level") == "warning":
        line("⚠️ رانش هشدار است. بررسی دستی توصیه می‌شود.")
    else:
        line("✅ مدل پایدار است. ادامه‌ی جمع‌آوری بازخورد توصیه می‌شود.")

    c.showPage()
    c.save()
    return buffer.getvalue()
