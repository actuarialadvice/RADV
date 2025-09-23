# services/pdf.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import streamlit as st

# Try optional deps used for rendering/highlighting
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

# Minimal valid single-page PDF (fallback if PyMuPDF/Pillow are unavailable)
MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj <<>> endobj\n"
    b"2 0 obj << /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] >> endobj\n"
    b"3 0 obj << /Type /Pages /Kids [2 0 R] /Count 1 >> endobj\n"
    b"4 0 obj << /Type /Catalog /Pages 3 0 R >> endobj\n"
    b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000045 00000 n \n0000000108 00000 n \n0000000166 00000 n \n"
    b"trailer << /Size 5 /Root 4 0 R >>\nstartxref\n220\n%%EOF\n"
)

def ensure_valid_pdf(pathlike) -> None:
    """
    Ensure there is a valid, single-page PDF at the provided path.
    If the file exists, do nothing. If not, attempt to create a simple
    one-page PDF that says 'Demo Chart: <filename>' using PyMuPDF.
    If PyMuPDF is not available, write a tiny minimal PDF byte string.
    """
    try:
        p = Path(pathlike)
    except Exception:
        p = Path(str(pathlike))

    if p.exists():
        return

    p.parent.mkdir(parents=True, exist_ok=True)

    # Try with PyMuPDF for a nicer demo page
    if fitz is not None:
        try:
            doc = fitz.open()
            page = doc.new_page()
            text = f"Demo Chart: {p.name}"
            page.insert_text((72, 72), text, fontsize=16)
            doc.save(str(p))
            doc.close()
            return
        except Exception:
            pass

    # Fallback: write minimal valid PDF bytes
    try:
        p.write_bytes(MINIMAL_PDF_BYTES)
    except Exception:
        # Last resort: ignore—caller should handle missing file errors
        pass

def read_pdf_pages(pdf_path: str) -> List[str]:
    """
    Return a list of extracted text strings, one per page.
    Empty strings indicate pages without a text layer (e.g., scanned PDFs).
    """
    if fitz is None:
        return []
    texts: List[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                txt = page.get_text("text") or ""
                texts.append(txt)
    except Exception:
        return []
    return texts

def read_pdf_text(pdf_path: str) -> str:
    """Return all extracted text concatenated across pages."""
    parts = read_pdf_pages(pdf_path)
    return "\n".join(parts) if parts else ""

def pdf_num_pages(pdf_path: str) -> int:
    if fitz is None:
        return 0
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception:
        return 0

def _search_rects(page, term: str):
    """
    Find on-page rectangles for a search term.
    Attempts case-insensitive search where supported.
    """
    term = (term or "").strip()
    if not term:
        return []
    try:
        flags = getattr(fitz, "TEXT_IGNORECASE", 0)
        return page.search_for(term, flags=flags) or []
    except Exception:
        # Fallback: case-sensitive; try word-wise intersection as a last resort.
        rects = page.search_for(term) or []
        if rects:
            return rects
        words = [w for w in re.split(r"\s+", term) if w]
        if not words:
            return []
        last = None
        for w in words:
            wr = page.search_for(w) or []
            as_tuples = set(map(tuple, wr))
            last = as_tuples if last is None else (last & as_tuples)
        return [fitz.Rect(*r) for r in (last or [])]

def _draw_highlights(img: Image.Image, rects, scale: float, rgba_fill, rgba_outline):
    if Image is None or ImageDraw is None:
        return img
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for r in rects:
        x0, y0 = int(r.x0 * scale), int(r.y0 * scale)
        x1, y1 = int(r.x1 * scale), int(r.y1 * scale)
        draw.rectangle([x0, y0, x1, y1], fill=rgba_fill, outline=rgba_outline, width=3)
    img.alpha_composite(overlay)
    return img

def show_pdf_inline(
    pdf_path: str,
    page_num_1based: int,
    primary_terms: list[str],
    offset_terms: list[str],
    key_suffix: str = "",
):
    """
    Render a single PDF page with highlight overlays.
    - Primary terms (yellow)
    - Offset terms (blue)
    Uses Streamlit's width='stretch' (replaces deprecated use_container_width).
    """
    if fitz is None or Image is None:
        st.warning("PDF rendering dependencies missing. Install 'pymupdf' and 'pillow'.")
        return

    primary_terms = [t for t in (primary_terms or []) if str(t).strip()]
    offset_terms = [t for t in (offset_terms or []) if str(t).strip()]

    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                st.warning("Empty PDF.")
                return
            page_index = max(0, min((page_num_1based or 1) - 1, len(doc) - 1))
            page = doc[page_index]

            scale = 2.0
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")

            prim_rects = []
            off_rects = []
            for t in primary_terms:
                prim_rects.extend(_search_rects(page, t))
            for t in offset_terms:
                off_rects.extend(_search_rects(page, t))

            img = _draw_highlights(img, prim_rects, scale, rgba_fill=(255, 243, 176, 110), rgba_outline=(231, 210, 122, 200))
            img = _draw_highlights(img, off_rects, scale, rgba_fill=(176, 208, 255, 110), rgba_outline=(142, 197, 255, 200))

            st.image(img, caption=f"Page {page_index + 1}", width="stretch")
            st.caption(f"Highlights — primary: {len(prim_rects)}, offsets: {len(off_rects)}")
    except Exception as e:
        st.warning(f"Unable to render PDF page: {e}")