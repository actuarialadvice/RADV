#!/usr/bin/env python3
"""
Append diagnosis-heavy addendum pages to every existing PDF in ./pdfs
so LLM evidence/suspect tasks show up during review.

Usage examples:
  python scripts/enrich_existing_pdfs.py --per-pdf-pages 2 --only-inventory --overwrite
  python scripts/enrich_existing_pdfs.py --per-pdf-pages 2
  python scripts/enrich_existing_pdfs.py --terms "atrial fibrillation" "CKD stage 3" COPD pneumonia "Type 2 diabetes"
"""
import argparse
import random
from pathlib import Path
from datetime import datetime

import pandas as pd

# Prefer pypdf; fallback to PyPDF2
try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    try:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore
    except Exception:
        raise SystemExit("Please install pypdf or PyPDF2")

# ReportLab for making the addendum pages
try:
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
except Exception:
    raise SystemExit("Please install reportlab: pip install reportlab")

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
PDF_DIR  = APP_ROOT / "pdfs"
INV_CSV  = DATA_DIR / "chart_inventory.csv"

DEFAULT_TERMS = [
    # Pulmonary / ID
    "COPD", "emphysema", "chronic bronchitis", "pneumonia", "asthma", "OSA",
    "tobacco use", "smoker", "inhaler", "tiotropium", "albuterol",
    # Endocrine / Metabolic
    "Type 2 diabetes", "hyperglycemia", "A1c", "insulin", "metformin",
    "obesity", "BMI 36", "hyperlipidemia",
    # Cardio
    "congestive heart failure", "CHF", "HFrEF", "HFpEF", "atrial fibrillation",
    "hypertension", "CAD", "MI", "statin therapy",
    # Renal / Heme
    "chronic kidney disease", "CKD stage 3", "proteinuria", "anemia", "iron deficiency",
    # Neuropsych
    "depression", "anxiety", "dementia",
    # HIV & others
    "HIV", "antiretroviral", "viral load", "CD4 count",
]

def load_inventory_paths() -> set[str]:
    if not INV_CSV.exists():
        return set()
    try:
        df = pd.read_csv(INV_CSV)
    except Exception:
        return set()
    if "pdf_path" not in df.columns:
        return set()
    return set(str(Path(p).resolve()) for p in df["pdf_path"].dropna().tolist())

def make_addendum_pdf(out_path: Path, header: str, terms: list[str], pages: int = 1):
    """Create a small PDF with diagnosis-rich content."""
    c = rl_canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    left = 0.75 * inch
    top  = height - 0.75 * inch

    # Build multi-page text payload
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks = [
        header,
        f"Addendum generated: {now}",
        "",
        "ASSESSMENT / PLAN HIGHLIGHTS:",
        *[f" - {t}" for t in terms],
        "",
        "Clinical rationale:",
        "  Symptoms reported include dyspnea, fatigue, and reduced exercise tolerance.",
        "  Medications reconciled (see MAR). Diagnostic testing and follow-up arranged.",
        "",
        "Note: This addendum is synthetic for LLM testing and contains rich terminology.",
        "-"*90,
    ]
    full_text = "\n".join(blocks + [""]*120)  # filler lines to fill pages

    lines = full_text.split("\n")
    lines_per_page = 50
    i = 0
    for _ in range(pages):
        y = top
        c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "ADDENDUM: CLINICAL NOTE (LLM TEST)"); y -= 16
        c.setFont("Helvetica", 10)
        while i < len(lines):
            c.drawString(left, y, lines[i][:110])
            y -= 12
            i += 1
            if y < 0.75*inch:
                break
        c.showPage()
    c.save()

def append_addendum(original_pdf: Path, addendum_pdf: Path, overwrite=True):
    """Append the addendum PDF pages to the existing PDF."""
    reader = PdfReader(str(original_pdf))
    writer = PdfWriter()
    # copy original pages
    for p in reader.pages:
        writer.add_page(p)
    # append addendum pages
    add_r = PdfReader(str(addendum_pdf))
    for p in add_r.pages:
        writer.add_page(p)
    # write to temp then replace
    tmp_path = original_pdf.with_suffix(".tmp.pdf")
    with open(tmp_path, "wb") as f:
        writer.write(f)
    if overwrite:
        tmp_path.replace(original_pdf)
    else:
        new_path = original_pdf.with_name(original_pdf.stem + "_enriched.pdf")
        tmp_path.replace(new_path)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pdf-pages", type=int, default=2, help="Addendum pages to append to each PDF")
    ap.add_argument("--terms", nargs="*", default=None, help="Explicit terms to embed (otherwise use defaults/random)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-inventory", action="store_true",
                    help="Only enrich PDFs referenced in data/chart_inventory.csv")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite original PDFs (default). If not set, write *_enriched.pdf copies.")
    args = ap.parse_args()

    random.seed(args.seed)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Choose which PDFs to touch
    if args.only_inventory:
        inv_paths = load_inventory_paths()
        pdfs = [Path(p) for p in inv_paths if Path(p).exists()]
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found to enrich under:", PDF_DIR)
        return

    # Build a reasonable term set per file
    base_terms = args.terms or DEFAULT_TERMS
    created = 0
    tmp_addendum = PDF_DIR / "_addendum_tmp.pdf"

    for pdf in pdfs:
        # vary terms a bit per file
        k = min(12, max(8, len(base_terms)//3))
        terms = sorted(set(random.sample(base_terms, k=k)))
        header = f"Enrichment addendum for file: {pdf.name}"
        make_addendum_pdf(tmp_addendum, header, terms, pages=args.per_pdf_pages)
        append_addendum(pdf, tmp_addendum, overwrite=args.overwrite or True)
        created += 1
        print(f"[ok] appended {args.per_pdf_pages} page(s) to {pdf.name} with terms: {', '.join(terms[:6])}…")

    # cleanup
    if tmp_addendum.exists():
        try:
            tmp_addendum.unlink()
        except Exception:
            pass

    print(f"\nEnriched {created} PDF(s).")
    if args.only_inventory:
        print("Scope: PDFs referenced in data/chart_inventory.csv")
    else:
        print("Scope: all PDFs in ./pdfs")

if __name__ == "__main__":
    main()
