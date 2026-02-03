"""Export functions for downloading data and reports."""

import io
import tempfile
from typing import List, Tuple

import pandas as pd
import streamlit as st

from .statistics import cp_cpk_summary, imr_spc_points


def download_plot_button(fig, filename: str) -> None:
    """Create download button for plot PNG."""
    try:
        img_bytes = fig.to_image(format="png", scale=3)
    except Exception:
        img_bytes = None

    if img_bytes:
        st.download_button(
            "下載圖表 (PNG)",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    else:
        st.caption("PNG 下載需要 kaleido，請確認已安裝")


def download_excel_button(df: pd.DataFrame, filename: str) -> None:
    """Create download button for Excel data."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button(
        "下載資料 (Excel)",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def download_stats_excel(stats: pd.DataFrame, filename: str) -> None:
    """Create download button for statistics summary Excel."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        stats.to_excel(writer, index=False, sheet_name="summary")
    st.download_button(
        "下載統計摘要 (Excel)",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def download_quality_reports(df: pd.DataFrame) -> None:
    """Create download button for CP/CPK + SPC quality reports."""
    cp_cpk = cp_cpk_summary(df)
    spc_summary, spc_points = imr_spc_points(df)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        cp_cpk.to_excel(writer, index=False, sheet_name="cp_cpk_summary")
        spc_summary.to_excel(writer, index=False, sheet_name="spc_imr_summary")
        spc_points.to_excel(writer, index=False, sheet_name="spc_imr_points")
    st.download_button(
        "下載 CP/CPK + SPC 報表 (Excel)",
        data=buffer.getvalue(),
        file_name="quality_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def build_report_html(stats: pd.DataFrame, figures: list) -> str:
    """Build HTML report with statistics and figures.

    Args:
        stats: Statistics summary DataFrame
        figures: List of tuples (title, base64_image)

    Returns:
        HTML string for the report
    """
    table_html = stats.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    images_html = "\n".join(
        [
            f"<h3>{title}</h3><img style='max-width:100%;' src='data:image/png;base64,{img_b64}'/>"
            for title, img_b64 in figures
        ]
    )
    return f"""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>Boxplot Report</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 20px; }}
img {{ margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>盒鬚圖報表</h1>
<h2>統計摘要</h2>
{table_html}
<h2>圖表</h2>
{images_html}
</body>
</html>
"""


def generate_pdf_report(
    stats: pd.DataFrame,
    cpk_df: pd.DataFrame,
    figures: List[Tuple[str, bytes]],
    title: str = "品質分析報表",
) -> bytes:
    """Generate PDF report with statistics and figures.

    Args:
        stats: Statistics summary DataFrame
        cpk_df: Cpk analysis DataFrame
        figures: List of tuples (title, png_bytes)
        title: Report title

    Returns:
        PDF file as bytes
    """
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.cell(0, 10, title, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()

    # Page 1: Statistics Summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Statistics Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Create stats table
    pdf.set_font("Helvetica", "", 8)
    cols = ["dimension", "count", "mean", "std", "min", "max", "nominal", "upper", "lower", "out_of_spec"]
    col_widths = [50, 15, 25, 25, 25, 25, 25, 25, 25, 20]

    # Header
    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Helvetica", "B", 8)
    for col, w in zip(cols, col_widths):
        pdf.cell(w, 7, col, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 7)
    for _, row in stats.iterrows():
        for col, w in zip(cols, col_widths):
            val = row.get(col, "")
            if isinstance(val, float):
                text = f"{val:.4f}" if pd.notna(val) else "N/A"
            else:
                text = str(val)[:20]  # Truncate long dimension names
            pdf.cell(w, 6, text, border=1, align="C")
        pdf.ln()

    # Page 2: Cpk Summary
    if not cpk_df.empty:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Cpk Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Cpk table
        cpk_cols = ["dimension", "count", "mean", "std", "USL", "LSL", "Cp", "Cpk", "rating"]
        cpk_widths = [50, 15, 30, 30, 30, 30, 25, 25, 25]

        # Header
        pdf.set_fill_color(200, 200, 200)
        pdf.set_font("Helvetica", "B", 8)
        for col, w in zip(cpk_cols, cpk_widths):
            pdf.cell(w, 7, col, border=1, fill=True, align="C")
        pdf.ln()

        # Data rows with color coding
        pdf.set_font("Helvetica", "", 7)
        for _, row in cpk_df.iterrows():
            color = row.get("color", "gray")
            if color == "green":
                pdf.set_fill_color(46, 204, 113)
            elif color == "yellow":
                pdf.set_fill_color(241, 196, 15)
            elif color == "red":
                pdf.set_fill_color(231, 76, 60)
            else:
                pdf.set_fill_color(255, 255, 255)

            for col, w in zip(cpk_cols, cpk_widths):
                val = row.get(col, "")
                if isinstance(val, float):
                    text = f"{val:.4f}" if pd.notna(val) else "N/A"
                else:
                    text = str(val)[:20]
                fill = col in ["Cpk", "rating"]
                pdf.cell(w, 6, text, border=1, align="C", fill=fill)
            pdf.ln()

    # Chart pages
    for chart_title, img_bytes in figures:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, chart_title, new_x="LMARGIN", new_y="NEXT")

        # Save image to temp file and add to PDF
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp.flush()
            # A4 Landscape: 297mm x 210mm, with margins
            pdf.image(tmp.name, x=10, y=30, w=277)

    return pdf.output()


def download_pdf_report_button(
    stats: pd.DataFrame,
    cpk_df: pd.DataFrame,
    figures: List[Tuple[str, bytes]],
    filename: str = "quality_report.pdf",
) -> None:
    """Create download button for PDF report.

    Args:
        stats: Statistics summary DataFrame
        cpk_df: Cpk analysis DataFrame
        figures: List of tuples (title, png_bytes)
        filename: Output filename
    """
    try:
        pdf_bytes = generate_pdf_report(stats, cpk_df, figures)
        st.download_button(
            "下載完整報表 (PDF)",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )
    except ImportError:
        st.warning("PDF 產生需要 fpdf2，請執行: pip install fpdf2")
    except Exception as e:
        st.error(f"PDF 產生失敗: {e}")
