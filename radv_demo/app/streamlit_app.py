# app/streamlit_app.py

import os
import re
import json
import base64
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# --- AgGrid ↔ Streamlit compatibility shim ---
# Some versions of st-aggrid reference: streamlit.components.v1.components.MarshallComponentException
# Newer Streamlit builds don't expose that attribute path. Provide a minimal shim so AgGrid won't crash.
try:
    import importlib
    _scv1 = importlib.import_module("streamlit.components.v1")
    if not hasattr(_scv1, "components"):
        class _Compat:
            class MarshallComponentException(Exception):
                pass
        setattr(_scv1, "components", _Compat)
except Exception:
    # Don't block app if anything goes wrong here
    pass

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="RADV Chart Validation Workbench",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- LIGHT UI POLISH ----------
st.markdown("""
/* Sidebar action buttons: full-width, left-aligned, green */
.stSidebar .stButton > button {
  width: 100%;
  display: flex !important;
  justify-content: flex-start !important;
  text-align: left !important;
  background: #10B981 !important;        /* emerald-500 */
  color: #ffffff !important;
  border: 1px solid #059669 !important;   /* emerald-600 */
  border-radius: 10px !important;
}
.stSidebar .stButton > button:hover {
  filter: brightness(0.96);
}
<style>
:root {
  --brand:#0F766E; /* teal-700 */
  --brand-soft:#ECFDF5;
  --muted:#f6f8fb;
  --line:#e5e7eb;
}
.block-container { max-width: 1500px !important; padding-top: .6rem; }
h1, h2, h3 { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
.stButton>button[kind="primary"] { background: var(--brand); color: #fff; border-radius: 10px; border: 1px solid #0b5e58; }
.stButton>button:hover { filter: brightness(0.95); }
.legend-sticky { position: sticky; top: 0; z-index: 2; background: #fff; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; display:inline-block; }
.badge { display:inline-block; padding:.18rem .5rem; border-radius:999px; font-size:.78rem; border:1px solid #eaeaea; background:#f8f9fb; color:#333; }
.badge.green { background:#dcfce7; border-color:#86efac; color:#065f46; }
.badge.orange { background:#ffedd5; border-color:#fdba74; color:#7c2d12; }
.note-box { white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas; max-height: 72vh; overflow:auto; border:1px solid #eee; padding:12px; border-radius:10px; }
.ag-header-cell { justify-content: center; }
.ag-cell { display:flex; align-items:center; }
.center-text { justify-content:center !important; }
.left-text { justify-content:flex-start !important; }
.card { border:1px solid var(--line); padding:10px 12px; border-radius:12px; background:#fff; }
.kpis { background:var(--muted); border:1px dashed var(--line); border-radius:12px; padding:8px 12px; }
a.small { font-size: .85rem; color: var(--brand) !important; text-decoration:none; }

/* Status pills */
.pill { display:inline-block; padding:.22rem .65rem; border-radius:999px; font-size:.80rem; font-weight:600; border:1px solid var(--line); }
.pill--ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }        /* green */
.pill--pending { background:#fff1f2; color:#9f1239; border-color:#fecdd3; }   /* red */

/* Upper table spacing */
.table-row { padding:6px 0; border-bottom:1px solid var(--line); }
.table-header { padding:6px 0; border-bottom:2px solid var(--line); }
</style>
""", unsafe_allow_html=True)

# ---------- OPTIONAL VIEWERS / GRIDS ----------
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    HAS_AGGRID = False

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Inline highlighting renderer
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    fitz = None
    HAS_FITZ = False

# Optional BigQuery
try:
    from google.cloud import bigquery
    HAS_BQ = True
except Exception:
    HAS_BQ = False

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
PDF_DIR  = APP_ROOT / "pdfs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if Path(path).exists() else pd.DataFrame()

def load_icd_mapping_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader for ICD→HCC crosswalk where the last column (label) may contain unquoted commas.
    We split each line on the first 3 commas only so the remainder is the label.
    Also normalizes header 'hcc lable' → 'hcc_label' and replaces commas in labels with hyphens.
    """
    if not Path(path).exists():
        return pd.DataFrame(columns=["icd10", "model", "hcc", "hcc_label"])

    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip()
        if not header:
            return pd.DataFrame(columns=["icd10", "model", "hcc", "hcc_label"])

        # Normalize header
        h = [p.strip() for p in header.split(",", 3)]
        if len(h) < 4:
            # If header is malformed, force expected names
            cols = ["icd10", "model", "hcc", "hcc_label"]
        else:
            # Map legacy 'hcc lable' → 'hcc_label'
            h[3] = "hcc_label" if h[3].lower().replace(" ", "") in ("hcclable", "hcc_label") else h[3]
            cols = [h[0] or "icd10", h[1] or "model", h[2] or "hcc", h[3] or "hcc_label"]

        for ln in f:
            ln = ln.rstrip("\r\n")
            if not ln.strip():
                continue
            parts = [p.strip() for p in ln.split(",", 3)]
            if len(parts) < 4:
                # skip malformed row
                continue
            icd10, model, hcc, hcc_label = parts[0], parts[1], parts[2], parts[3]
            # Replace internal commas in label with hyphens (display-safe)
            hcc_label = hcc_label.replace(",", " - ")
            rows.append({"icd10": icd10, "model": model, "hcc": hcc, "hcc_label": hcc_label})

    df = pd.DataFrame(rows, columns=["icd10", "model", "hcc", "hcc_label"])
    return df
    
# ---------- DATA LOAD ----------
radv_sample = load_csv(DATA_DIR / "radv_sample.csv")
chart_inventory = load_csv(DATA_DIR / "chart_inventory.csv")
coded_status_path = DATA_DIR / "radv_coding_status.csv"
coded_status = load_csv(coded_status_path)
audit_log_path = DATA_DIR / "audit_log.csv"

# Use project-local crosswalk if provided; else fallback, always use robust loader
user_mapping_path = Path(DATA_DIR / "icd10_to_hcc_v24_v28.csv")
icd_to_hcc_path = user_mapping_path if user_mapping_path.exists() else (DATA_DIR / "icd_to_hcc_fallback.csv")
icd_to_hcc = load_icd_mapping_csv(icd_to_hcc_path)
# Minimal hints for demo highlighting (replace with real LLM later)
hints_file = DATA_DIR / "llm_hints.json"
if not hints_file.exists():
    hints_file.write_text(json.dumps({
        "J44.1":{"primary_terms":["COPD","exacerbation","J44.1","wheezing","dyspnea","prednisone","tiotropium","albuterol"],
                 "offset_terms":["pneumonia","bronchitis","J44.0","tobacco","smoker","emphysema","PFT","FEV1/FVC"]},
        "E11.9":{"primary_terms":["Type 2 diabetes","E11.9","A1c","metformin","diet","exercise"],
                 "offset_terms":["neuropathy","retinopathy","nephropathy","CKD","insulin","hyperglycemia"]},
        "B20":{"primary_terms":["HIV","B20","antiretroviral","ART","viral load","CD4","tenofovir","emtricitabine","dolutegravir"],
               "offset_terms":["opportunistic infection","pneumocystis","toxoplasmosis","candidiasis"]}
    }, indent=2))
LLM_HINTS = json.loads(hints_file.read_text())

# Simple keyword→ICD demo map for LLM suggestions
TERM_TO_ICD = {
    "bronchitis": "J40",
    "pneumonia": "J18.9",
    "emphysema": "J43.9",
    "COPD": "J44.9",
    "diabetes": "E11.9",
    "HIV": "B20",
}

# ---------- SCHEMA NORMALIZERS ----------
CODED_STATUS_SCHEMA = {
    "member_id": "", "chart_id": "", "icd10": "", "model": "",
    "hcc": "", "hcc_label": "",
    "primary_coder_initials": "", "primary_decision": "", "primary_notes": "", "primary_timestamp": "", "primary_no_evidence": False,
    "secondary_coder_initials": "", "secondary_decision": "", "secondary_notes": "", "secondary_timestamp": "",
    "final_coder_initials": "", "final_decision": "", "final_notes": "", "final_timestamp": "",
    "dos": "", "npi": "", "suspect_offsets": "",
    "evidence_found": "", "confidence_level": "", "evidence_pages": "",
    "dos_start": "", "dos_end": "", "user_added_findings": ""
}

CHART_INVENTORY_SCHEMA = {
    "member_id": "", "chart_id": "", "provider": "", "npi": "",
    "start_dos": "", "end_dos": "", "pdf_path": "", "radv_diags": "",
    "llm_scanned": False, "coded_status": "", "dq_flag": ""
}

def ensure_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{k: v for k, v in schema.items()}]).iloc[0:0].copy()
    for col, default in schema.items():
        if col not in df.columns:
            df[col] = default
    df = df[[c for c in schema.keys() if c in df.columns] + [c for c in df.columns if c not in schema]]
    return df

# Normalize loaded dataframes
chart_inventory = ensure_columns(chart_inventory, CHART_INVENTORY_SCHEMA)
coded_status = ensure_columns(coded_status, CODED_STATUS_SCHEMA)

# ---------- HELPERS ----------
def nlp_scan_text(text: str, icd: str):
    hints = LLM_HINTS.get(icd, {})
    prim = hints.get("primary_terms", [])
    offs = hints.get("offset_terms", [])
    hits_primary = sorted({t for t in prim if t.lower() in text.lower()})
    hits_offsets = sorted({t for t in offs if t.lower() in text.lower()})
    return hits_primary, hits_offsets

def read_pdf_text(pdf_path: str) -> str:
    if PyPDF2 is None:
        return "(PyPDF2 not installed)"
    try:
        text = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        return f"(Unable to read PDF: {e})"

def pdf_num_pages(pdf_path: str) -> int:
    if PyPDF2 is None:
        return 0
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return 0

# Small helper for rendering status pills
def pill(text: str, kind: str) -> str:
    """Return a small HTML pill. kind: 'ok' (green) or 'pending' (red)."""
    return f'<span class="pill pill--{kind}">{text}</span>'

def render_pdf_page_image(pdf_path: str, page_index: int, primary_terms: List[str], offset_terms: List[str]) -> bytes:
    """
    Render a single PDF page (0-based) to PNG with translucent highlights for matched terms.
    Uses PyMuPDF; if unavailable, returns empty bytes.
    """
    if not HAS_FITZ:
        return b""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return b""
        page_index = max(0, min(page_index, len(doc) - 1))
        page = doc.load_page(page_index)

        rects_to_paint = []
        terms = [(t, "primary") for t in primary_terms if t] + [(t, "offset") for t in offset_terms if t]
        text_flags = getattr(fitz, "TEXT_DEHYPHENATE", 0)

        for term, kind in terms:
            try:
                quads = page.search_for(term, flags=text_flags)
            except Exception:
                quads = []
            for r in quads:
                rects_to_paint.append((r, kind))

        mat = fitz.Matrix(1, 1)
        if rects_to_paint:
            shape = page.new_shape()
            for r, kind in rects_to_paint:
                if kind == "primary":
                    fill = (1, 1, 0)   # yellow
                    alpha = 0.35
                    stroke = (0.9, 0.85, 0.3)
                else:
                    fill = (0.65, 0.82, 1.0)  # blue-ish
                    alpha = 0.35
                    stroke = (0.35, 0.55, 0.9)
                shape.draw_rect(r)
                shape.finish(color=stroke, fill=fill, fill_opacity=alpha, width=0.5)
            shape.commit()
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return b""

def show_pdf_inline(pdf_path: str, page_num_1based: int, primary_terms: List[str], offset_terms: List[str], width: int = 980, height: int = 760, key_suffix: str = ""):
    """
    Display a single page image with highlights. Avoids long data-URIs and works offline.
    """
    page_idx = max(0, page_num_1based - 1)
    img = render_pdf_page_image(pdf_path, page_idx, primary_terms, offset_terms)
    if img:
        st.image(img, caption=f"Page {page_num_1based}", use_container_width=True)
    else:
        try:
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            components.html(
                f'<iframe src="data:application/pdf;base64,{b64}#page={page_num_1based}&zoom=page-width" style="width:{width}px;height:{height}px;border:none;"></iframe>',
                height=height,
            )
        except Exception as e:
            st.warning(f"Unable to render PDF inline: {e}")

def persist_coded_status(df: pd.DataFrame):
    df = ensure_columns(df, CODED_STATUS_SCHEMA)
    df.to_csv(coded_status_path, index=False)

def persist_audit(action: str, who: str, member_id: str, chart_id: str, n_rows: int):
    ts = datetime.now().isoformat(timespec="seconds")
    rec = pd.DataFrame([{
        "timestamp": ts, "actor": who, "action": action,
        "member_id": member_id, "chart_id": chart_id, "rows_affected": n_rows
    }])
    mode = "a" if audit_log_path.exists() else "w"
    header = not audit_log_path.exists()
    rec.to_csv(audit_log_path, mode=mode, header=header, index=False)

def sanitize_df(df: pd.DataFrame):
    return df.reset_index(drop=True) if (df is not None and not df.empty) else df

def save_chart_inventory(df: pd.DataFrame):
    df = ensure_columns(df, CHART_INVENTORY_SCHEMA)
    df.to_csv(DATA_DIR / "chart_inventory.csv", index=False)

# ---------- PREFILL / SUBMISSION HELPERS ----------
def prior_values_for_task(member_id: str, chart_id: str, icd: str, role: str):
    try:
        m = (
            (coded_status["member_id"] == member_id) &
            (coded_status["chart_id"] == chart_id) &
            (coded_status["icd10"] == icd) &
            (coded_status["model"] == "both")
        )
        if coded_status[m].empty:
            return {"ev": False, "pages": "", f"notes_{role}": ""}
        row = coded_status[m].iloc[-1]
        ev_global = str(row.get("evidence_found", "")).lower() == "true"
        pages = str(row.get("evidence_pages", "") or "")
        role_notes = str(row.get(f"{role}_notes", "") or "")
        role_decision = str(row.get(f"{role}_decision", "") or "")
        ev_for_role = ev_global or (role_decision == "validated")
        return {"ev": ev_for_role, "pages": pages, f"notes_{role}": role_notes}
    except Exception:
        return {"ev": False, "pages": "", f"notes_{role}": ""}

def has_submission(member_id: str, chart_id: str, icd: str, role: str) -> bool:
    try:
        m = (
            (coded_status["member_id"] == member_id) &
            (coded_status["chart_id"] == chart_id) &
            (coded_status["icd10"] == icd) &
            (coded_status["model"] == "both")
        )
        if coded_status[m].empty:
            return False
        row = coded_status[m].iloc[-1]
        ts = str(row.get(f"{role}_timestamp", "") or "")
        dec = str(row.get(f"{role}_decision", "") or "")
        return (ts != "") or (dec != "")
    except Exception:
        return False

def submit_task(member_id: str, chart_id: str, icd: str, role: str, decision_payload: dict):
    global coded_status
    now = datetime.now().isoformat(timespec="seconds")

    if coded_status is None or coded_status.empty:
        coded_status = pd.DataFrame([{k: v for k, v in CODED_STATUS_SCHEMA.items()}]).iloc[0:0].copy()

    mask = (
        (coded_status["member_id"] == member_id) &
        (coded_status["chart_id"] == chart_id) &
        (coded_status["icd10"] == icd) &
        (coded_status["model"] == "both")
    )
    if coded_status[mask].empty:
        new_row = {k: "" for k in CODED_STATUS_SCHEMA.keys()}
        new_row.update({"member_id": member_id, "chart_id": chart_id, "icd10": icd, "model": "both"})
        coded_status = pd.concat([coded_status, pd.DataFrame([new_row])], ignore_index=True)
        mask = (
            (coded_status["member_id"] == member_id) &
            (coded_status["chart_id"] == chart_id) &
            (coded_status["icd10"] == icd) &
            (coded_status["model"] == "both")
        )

    # copy static fields
    for fld in ["hcc", "hcc_label", "dos_start", "dos_end", "npi", "evidence_pages"]:
        if fld in decision_payload:
            coded_status.loc[mask, fld] = decision_payload[fld]
    # evidence flags + DOS convenience
    coded_status.loc[mask, "evidence_found"] = "True" if decision_payload.get("evidence_found", False) else "False"
    ds = decision_payload.get("dos_start", "")
    de = decision_payload.get("dos_end", "")
    coded_status.loc[mask, "dos"] = ds if ds == de else f"{ds} - {de}"

    # role-specific fields
    if role == "primary":
        coded_status.loc[mask, "primary_coder_initials"] = st.session_state.get("coder_initials","")
        coded_status.loc[mask, "primary_notes"] = decision_payload.get("notes","")
        coded_status.loc[mask, "primary_timestamp"] = now
        coded_status.loc[mask, "primary_decision"] = decision_payload.get("decision","")
        coded_status.loc[mask, "primary_no_evidence"] = not decision_payload.get("evidence_found", False)
    elif role == "secondary":
        coded_status.loc[mask, "secondary_coder_initials"] = st.session_state.get("coder_initials","")
        coded_status.loc[mask, "secondary_notes"] = decision_payload.get("notes","")
        coded_status.loc[mask, "secondary_timestamp"] = now
        coded_status.loc[mask, "secondary_decision"] = decision_payload.get("decision","")
    elif role == "final":
        coded_status.loc[mask, "final_coder_initials"] = st.session_state.get("coder_initials","")
        coded_status.loc[mask, "final_notes"] = decision_payload.get("notes","")
        coded_status.loc[mask, "final_timestamp"] = now
        coded_status.loc[mask, "final_decision"] = decision_payload.get("decision","")

    coded_status = ensure_columns(coded_status, CODED_STATUS_SCHEMA)
    persist_coded_status(coded_status)
    persist_audit("submit_task", st.session_state.get("coder_initials",""), member_id, chart_id, 1)

# ---------- GRID RENDERER (used in other tabs) ----------
def render_grid(
    df: pd.DataFrame,
    col_defs=None,
    height=360,
    row_height=36,
    download_name=None,
    grid_key: Optional[str] = None,
):    
    df = sanitize_df(df)
    if df is None:
        df = pd.DataFrame()
    if not HAS_AGGRID or df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filter=True, sortable=True, resizable=True, wrapText=False, autoHeight=False, minWidth=110)
        if col_defs:
            for c, opts in col_defs.items():
                gb.configure_column(
                    c,
                    header_name=opts.get("header_name", c),
                    width=opts.get("width"),
                    maxWidth=opts.get("maxWidth"),
                    minWidth=opts.get("minWidth", 90),
                    cellClass=opts.get("cellClass", "center-text"),
                    headerClass=opts.get("headerClass", ""),
                    wrapText=opts.get("wrapText", False),
                    autoHeight=opts.get("autoHeight", False),
                )
        gb.configure_grid_options(domLayout="normal", ensureDomOrder=True, rowHeight=row_height)
        AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.NO_UPDATE,
            theme="balham",
            height=height,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            key=grid_key or f"ag_{download_name or 'grid'}",)
    if download_name and not df.empty:
        st.download_button(
            f"Download {download_name}.csv",
            data=df.to_csv(index=False).encode(),
            file_name=f"{download_name}.csv",
            mime="text/csv",
            key=f"dl_{download_name}"
        )

# ---- Demo seeding ----
def seed_demo():
    rs_path = DATA_DIR / "radv_sample.csv"
    if (not rs_path.exists()) or load_csv(rs_path).empty:
        rs = pd.DataFrame([
            {"member_id":"M001","icd10_submitted":"J44.1","dx_summary":"COPD with acute exacerbation"},
            {"member_id":"M002","icd10_submitted":"E11.9","dx_summary":"Type 2 diabetes mellitus without complications"},
            {"member_id":"M003","icd10_submitted":"B20","dx_summary":"HIV disease"}
        ])
        rs.to_csv(rs_path, index=False)

    ci_path = DATA_DIR / "chart_inventory.csv"
    if (not ci_path.exists()) or load_csv(ci_path).empty:
        rows = []
        providers = [("General Pulmonary Clinic","1223456789"),
                     ("Sunset Primary Care","1098765432"),
                     ("City Hospital Inpatient","1456789012"),
                     ("Northside Specialty Group","1789012345")]
        pdf_path = str(next((PDF_DIR.glob("*.pdf")), PDF_DIR / "CHART0010.pdf"))
        if not Path(pdf_path).exists():
            (PDF_DIR / "CHART0010.pdf").write_bytes(b"%PDF-1.4\n% demo placeholder\n")
            pdf_path = str(PDF_DIR / "CHART0010.pdf")
        for m_idx, m in enumerate(["M001","M002","M003"], start=1):
            for i in range(1,5):
                prov, npi = providers[i-1]
                rows.append({
                    "member_id": m,
                    "chart_id": f"{m}-CH{i:02d}",
                    "provider": prov, "npi": npi,
                    "start_dos": f"2024-0{(i%9)+1}-0{(m_idx%7)+1}",
                    "end_dos":   f"2024-0{(i%9)+1}-1{(m_idx%7)+1}",
                    "pdf_path": pdf_path,
                    "llm_scanned": False,
                    "coded_status": "",
                    "radv_diags": ""
                })
        ci = pd.DataFrame(rows)
        sample = load_csv(DATA_DIR / "radv_sample.csv")
        icd_map = dict(sample[["member_id","icd10_submitted"]].values) if not sample.empty else {}
        ci["radv_diags"] = ci["member_id"].map(icd_map).fillna("")
        ensure_columns(ci, CHART_INVENTORY_SCHEMA).to_csv(ci_path, index=False)

# ---------- OPTIONAL: BigQuery I/O ----------
def load_from_bigquery():
    if not HAS_BQ:
        st.warning("google-cloud-bigquery not installed.")
        return None, None
    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    if not project or not dataset:
        st.warning("Set env BQ_PROJECT and BQ_DATASET to enable BigQuery load.")
        return None, None
    client = bigquery.Client(project=project)
    def _read(table):
        return client.query(f"SELECT * FROM `{project}.{dataset}.{table}`").to_dataframe()
    try:
        return _read("radv_sample"), _read("chart_inventory")
    except Exception as e:
        st.error(f"BigQuery load failed: {e}")
        return None, None

def save_to_bigquery(df: pd.DataFrame, table: str, schema=None, write_mode="WRITE_TRUNCATE"):
    if not HAS_BQ:
        st.warning("google-cloud-bigquery not installed.")
        return
    project = os.environ.get("BQ_PROJECT")
    dataset = os.environ.get("BQ_DATASET")
    if not project or not dataset:
        st.warning("Set env BQ_PROJECT and BQ_DATASET to enable BigQuery save.")
        return
    client = bigquery.Client(project=project)
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_mode,
        autodetect=(schema is None),
        schema=schema or []
    )
    table_id = f"{project}.{dataset}.{table}"
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

# ---------- SIDEBAR ----------
# ---------- SIDEBAR ----------
st.sidebar.header("Coder Profile")
st.session_state["coder_initials"] = st.sidebar.text_input(
    "Your initials", value=st.session_state.get("coder_initials", "PC")
)
role = st.sidebar.selectbox("Your role", ["primary", "secondary", "final"])
st.sidebar.markdown("---")

# Make BQ buttons green (primary) and give them unique keys
bq_load = st.sidebar.button(
    "Load RADV Sample → BigQuery",
    type="primary",
    key="btn_bq_load",
    help="Load radv_sample & chart_inventory if configured",
)
bq_save = st.sidebar.button(
    "Save Coding Status → BigQuery",
    type="primary",
    key="btn_bq_save",
    help="Save coded_status table to BQ (table: coded_status)",
)

st.sidebar.caption("Yellow = likely evidence · Blue = suspects/offsets")
st.sidebar.markdown(
    '<span class="badge">Demo only – do not use with real PHI.</span>',
    unsafe_allow_html=True,
)

if bq_load:
    rs, ci = load_from_bigquery()
    if rs is not None and ci is not None:
        radv_sample = rs
        chart_inventory = ensure_columns(ci, CHART_INVENTORY_SCHEMA)
        st.sidebar.success("Loaded from BigQuery.")

if bq_save:
    df_bq = ensure_columns(load_csv(coded_status_path), CODED_STATUS_SCHEMA)
    if df_bq.empty:
        st.sidebar.info("Nothing to save yet.")
    else:
        try:
            save_to_bigquery(df_bq, "coded_status")
            st.sidebar.success("Saved to BigQuery table coded_status.")
        except Exception as e:
            st.sidebar.error(f"Save failed: {e}")

# ---------- TITLE ----------
st.title("RADV Chart Validation Workbench")
st.caption("Professional reviewer UI • v24/v28 aware • coder/QA trail")

# ---------- TABS ----------
tab_sample, tab_inventory, tab_validate, tab_progress, tab_output, tab_charts, tab_reference = st.tabs([
    "RADV sample", "Chart inventory", "RADV validation", "Progress", "RADV output", "Charts", "Reference"
])

# ==================== RADV SAMPLE ====================
with tab_sample:
    counts = chart_inventory.groupby("member_id")["chart_id"].nunique().rename("charts_count") if not chart_inventory.empty else pd.Series([], dtype=int)
    sample = radv_sample.copy()
    sample = sample.merge(counts, on="member_id", how="left").fillna({"charts_count": 0})

    if not icd_to_hcc.empty and "icd10" in icd_to_hcc.columns and "icd10_submitted" in sample.columns:
        v24_map = dict(icd_to_hcc[icd_to_hcc["model"].str.lower()=="v24"][["icd10","hcc"]].values)
        v28_map = dict(icd_to_hcc[icd_to_hcc["model"].str.lower()=="v28"][["icd10","hcc"]].values)
        sample["v24_hcc"] = sample["icd10_submitted"].map(v24_map)
        sample["v28_hcc"] = sample["icd10_submitted"].map(v28_map)

    cols = [c for c in ["member_id","icd10_submitted","dx_summary","v24_hcc","v28_hcc","charts_count"] if c in sample.columns]
    col_defs = {
        "member_id": {"width": 140, "cellClass": "center-text"},
        "icd10_submitted": {"width": 140, "cellClass": "center-text"},
        "dx_summary": {"width": 460, "wrapText": True, "autoHeight": True, "cellClass": "left-text"},
        "v24_hcc": {"width": 120, "cellClass": "center-text"},
        "v28_hcc": {"width": 120, "cellClass": "center-text"},
        "charts_count": {"width": 140, "cellClass": "center-text", "header_name": "Charts"},
    }

    st.subheader("CMS sample")
    render_grid(sample[cols], col_defs=col_defs, height=240, download_name="radv_sample", grid_key="grid_sample")
# ==================== CHART INVENTORY ====================
with tab_inventory:
    left, right = st.columns([6,1])
    with left:
        st.subheader("All charts")
    with right:
        run_all = st.button("Run NLP on ALL charts", type="primary", help="Scan all charts for evidence & offsets")

    if "start_dos" not in chart_inventory.columns:
        chart_inventory["start_dos"] = chart_inventory.get("dos","")
    if "end_dos" not in chart_inventory.columns:
        chart_inventory["end_dos"] = chart_inventory.get("dos","")
    if "radv_diags" not in chart_inventory.columns:
        icd_map = dict(radv_sample[["member_id","icd10_submitted"]].values) if not radv_sample.empty else {}
        chart_inventory["radv_diags"] = chart_inventory["member_id"].map(icd_map).fillna("")
    if "llm_scanned" not in chart_inventory.columns:
        chart_inventory["llm_scanned"] = False
    if "coded_status" not in chart_inventory.columns:
        chart_inventory["coded_status"] = ""

    # DQ warnings (data quality)
    chart_inventory["dq_flag"] = chart_inventory.apply(
        lambda r: "; ".join([m for m in [
            "Missing NPI" if not str(r.get("npi","")).strip() else "",
            "Missing Start DOS" if not str(r.get("start_dos","")).strip() else "",
            "Missing End DOS" if not str(r.get("end_dos","")).strip() else "",
        ] if m]), axis=1
    )

    if run_all and not chart_inventory.empty:
        updated = chart_inventory.copy()
        for idx, row in updated.iterrows():
            pdf = row["pdf_path"]
            icds = [s.strip() for s in str(row.get("radv_diags","")).split(";") if s.strip()]
            focus_icd = icds[0] if icds else ""
            text = read_pdf_text(pdf)
            _prim, _off = nlp_scan_text(text, focus_icd)
            updated.loc[idx, "llm_scanned"] = True
        chart_inventory = ensure_columns(updated, CHART_INVENTORY_SCHEMA)
        save_chart_inventory(chart_inventory)
        st.success("NLP scan complete for all charts.")

    # Filters row
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        f_member = st.selectbox("Filter by member (optional)", ["(All)"] + sorted(chart_inventory["member_id"].dropna().unique().tolist()) if not chart_inventory.empty else ["(All)"])
    with c2:
        f_provider = st.text_input("Filter provider (contains)", "")
    with c3:
        f_llm = st.selectbox("LLM scanned", ["(All)","Yes","No"])

    # Apply filters
    view = chart_inventory.copy()
    if not view.empty:
        if f_member != "(All)":
            view = view[view["member_id"]==f_member]
        if f_provider.strip():
            view = view[view["provider"].astype(str).str.contains(f_provider.strip(), case=False, na=False)]
        if f_llm == "Yes":
            view = view[view["llm_scanned"]==True]
        elif f_llm == "No":
            view = view[view["llm_scanned"]==False]

    cols = [c for c in ["member_id","chart_id","provider","npi","start_dos","end_dos","llm_scanned","coded_status","dq_flag"] if c in view.columns]
    col_defs = {
        "member_id":   {"width": 110},
        "chart_id":    {"width": 110},
        "provider":    {"width": 250, "cellClass": "left-text"},
        "npi":         {"width": 120},
        "start_dos":   {"width": 120},
        "end_dos":     {"width": 120},
        "llm_scanned": {"width": 120},
        #"coded_status":{"width": 130},
        "dq_flag":     {"width": 220, "cellClass": "left-text", "header_name": "DQ warnings"},
    }
    render_grid(view[cols] if not view.empty else view, col_defs=col_defs, height=420, row_height=34, download_name="chart_inventory", grid_key="grid_inventory")
# ==================== RADV VALIDATION ====================
# ==================== RADV VALIDATION ====================
with tab_validate:
    st.subheader("RADV validation")

    # Build per-chart v24/v28 HCCs for display (optional)
    inv_full = chart_inventory.copy()
    if not inv_full.empty and not icd_to_hcc.empty:
        def hcc_join(diags, model):
            codes = [s.strip() for s in str(diags).split(";") if s.strip()]
            if not codes: return ""
            out = []
            for c in codes:
                rec = icd_to_hcc[(icd_to_hcc["icd10"]==c) & (icd_to_hcc["model"].str.lower()==model)]
                if not rec.empty:
                    out.append(str(rec.iloc[0]["hcc"]))
            return ";".join(sorted(set(out)))
        inv_full["v24_hccs"] = inv_full["radv_diags"].apply(lambda d: hcc_join(d, "v24"))
        inv_full["v28_hccs"] = inv_full["radv_diags"].apply(lambda d: hcc_join(d, "v28"))

    members = sorted(radv_sample["member_id"].dropna().unique().tolist())
    if not members:
        st.info("No members found.")
    else:
        member_id = st.selectbox("Member", members, key="val_member")

        # Current member's charts
        inv = inv_full[inv_full["member_id"] == member_id].copy()

        # Helper: has anything been saved for this chart?
        def chart_reviewed(mid: str, cid: str) -> bool:
            cs = coded_status[(coded_status["member_id"] == mid) & (coded_status["chart_id"] == cid)]
            if not cs.empty:
                return True
            # also respect an explicit chart_inventory status if present
            row = chart_inventory[
                (chart_inventory["member_id"] == mid) & (chart_inventory["chart_id"] == cid)
            ]
            return str(row["coded_status"].iloc[0]).strip().lower() == "reviewed" if not row.empty else False

        # -------- Upper "table" with headers + Review buttons --------
        if inv.empty:
            st.info("No charts for this member.")
        else:
            st.markdown("**Charts for member**")

            # Header row
            h1, h2, h3, h4, h5, h6 = st.columns([1.3, 3.2, 1.2, 2.2, 1.1, 1.1])
            with h1: st.markdown('<div class="table-header"><strong>Chart</strong></div>', unsafe_allow_html=True)
            with h2: st.markdown('<div class="table-header"><strong>Provider</strong></div>', unsafe_allow_html=True)
            with h3: st.markdown('<div class="table-header"><strong>NPI</strong></div>', unsafe_allow_html=True)
            with h4: st.markdown('<div class="table-header"><strong>DOS</strong></div>', unsafe_allow_html=True)
            with h5: st.markdown('<div class="table-header"><strong>Status</strong></div>', unsafe_allow_html=True)
            with h6: st.markdown('<div class="table-header"><strong>Action</strong></div>', unsafe_allow_html=True)

            # Rows
            for _, row in inv.iterrows():
                c1, c2, c3, c4, c5, c6 = st.columns([1.3, 3.2, 1.2, 2.2, 1.1, 1.1])

                with c1:
                    st.markdown(f'<div class="table-row"><strong>{row["chart_id"]}</strong></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="table-row">{str(row.get("provider", ""))}</div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="table-row">{str(row.get("npi", ""))}</div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="table-row">{row.get("start_dos","")} → {row.get("end_dos","")}</div>', unsafe_allow_html=True)

                is_reviewed = chart_reviewed(member_id, row["chart_id"])
                status_html = pill("Reviewed", "ok") if is_reviewed else pill("Pending", "pending")
                with c5:
                    st.markdown(f'<div class="table-row">{status_html}</div>', unsafe_allow_html=True)

                with c6:
                    # green primary Review button
                    if st.button("Review", key=f"review_btn_{member_id}_{row['chart_id']}", type="primary"):
                        st.session_state["val_chart"] = row["chart_id"]
                        st.session_state["review_open"] = True
                        st.rerun()

        # If a chart was chosen (via button), open the workbench
        if not inv.empty:
            target_chart = st.session_state.get("val_chart", inv["chart_id"].iloc[0])

            if st.session_state.get("review_open"):
                selected_chart = inv[inv["chart_id"] == target_chart].iloc[0]

                # Layout: left viewer, right tasks
                left, right = st.columns([8, 4], gap="large")

                # -------- Left: image-based viewer with highlights --------
                with left:
                    st.markdown("**Chart viewer**")
                    num_pages = pdf_num_pages(selected_chart["pdf_path"])
                    page_key = f"page_{selected_chart['chart_id']}"
                    current_page = st.session_state.get(page_key, 1)
                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max(num_pages, 1),
                        value=current_page,
                        step=1,
                        help=f"Total pages: {num_pages or 'unknown'}",
                        key=f"numinput_{selected_chart['chart_id']}",
                    )
                    st.session_state[page_key] = page

                    # highlight terms are driven by the active task tab
                    prim_terms = st.session_state.get(f"prim_terms_{selected_chart['chart_id']}", [])
                    off_terms = st.session_state.get(f"off_terms_{selected_chart['chart_id']}", [])
                    show_pdf_inline(
                        selected_chart["pdf_path"],
                        page_num_1based=page,
                        primary_terms=prim_terms,
                        offset_terms=off_terms,
                        key_suffix=selected_chart["chart_id"],
                    )

                    st.markdown(
                        '<div class="legend-sticky">'
                        '<span class="badge" style="background:#fff3b0;border-color:#e7d27a">Evidence</span> '
                        '<span class="badge" style="background:#cfe8ff;border-color:#8ec5ff">Suspects/Offsets</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                # -------- Right: Tasks (tabs) --------
                with right:
                    st.markdown("**Tasks**")

                    # Pull NPI/DOS from the selected chart (moved to upper table; no inputs here)
                    pre_npi = str(selected_chart.get("npi", ""))
                    pre_start = str(selected_chart.get("start_dos", ""))
                    pre_end = str(selected_chart.get("end_dos", ""))

                    # helpers to get both-model mappings
                    def hcc_for(icd, model):
                        recs = icd_to_hcc[(icd_to_hcc["icd10"] == icd) & (icd_to_hcc["model"].str.lower() == model)]
                        if recs.empty:
                            return None, None
                        r = recs.iloc[0]
                        return str(r["hcc"]), str(r.get("hcc_label", "") or "")

                    def both_models_line(icd):
                        h24, lbl24 = hcc_for(icd, "v24")
                        h28, lbl28 = hcc_for(icd, "v28")
                        label = lbl24 or lbl28 or ""
                        text = f"ICD **{icd}** → V24HCC **{h24 or '—'}** / V28HCC **{h28 or '—'}**"
                        if label:
                            text += f" — {label}"
                        return text, h24, h28, label

                    # Simple LLM candidates (keyword→ICD demo)
                    def llm_candidates(text: str):
                        cands = []
                        low = text.lower()
                        for term, icd in TERM_TO_ICD.items():
                            if term.lower() in low:
                                cands.append({"term": term, "icd10": icd})
                        seen, out = set(), []
                        for c in cands:
                            k = (c["term"], c["icd10"])
                            if k not in seen:
                                out.append(c)
                                seen.add(k)
                        return out

                    raw_text = read_pdf_text(selected_chart["pdf_path"])
                    icds_for_chart = [s.strip() for s in str(selected_chart.get("radv_diags", "")).split(";") if s.strip()]
                    llm_sugs = llm_candidates(raw_text)

                    # Tabs: RADV tasks, each LLM suggestion its own tab, and Add task
                    t_labels = ["RADV tasks"] + [f"LLM: {s['term']}" for s in llm_sugs] + ["Add task"]
                    tabs = st.tabs(t_labels)

                    def set_terms(prim, off):
                        st.session_state[f"prim_terms_{selected_chart['chart_id']}"] = prim
                        st.session_state[f"off_terms_{selected_chart['chart_id']}"] = off

                    save_queue = []

                    # ---- RADV tasks (first tab)
                    with tabs[0]:
                        # When the user views the RADV tab, auto-highlight the first RADV ICD terms
                        if icds_for_chart:
                            first_icd = icds_for_chart[0]
                            hints = LLM_HINTS.get(first_icd, {})
                            set_terms(hints.get("primary_terms", []), hints.get("offset_terms", []))

                        if not icds_for_chart:
                            st.info("No RADV ICDs listed for this chart.")
                        for icd in icds_for_chart:
                            hints = LLM_HINTS.get(icd, {})
                            prim = hints.get("primary_terms", [])
                            offs = hints.get("offset_terms", [])
                            line, h24, h28, lbl = both_models_line(icd)
                            with st.container(border=True):
                                st.caption(line)
                                # auto-focus highlight when interacting
                                if st.button("Focus highlights", key=f"focus_{selected_chart['chart_id']}_{icd}"):
                                    set_terms(prim, offs)
                                    st.rerun()
                                ev = st.checkbox("Evidence present", key=f"ev_{selected_chart['chart_id']}_{icd}")
                                pages = st.text_input("Evidence page(s)", key=f"pg_{selected_chart['chart_id']}_{icd}")
                                try:
                                    with st.popover("Add notes", use_container_width=True, key=f"pop_{selected_chart['chart_id']}_{icd}"):
                                        notes = st.text_area("Notes", height=100, key=f"nt_{selected_chart['chart_id']}_{icd}")
                                except Exception:
                                    with st.expander("Add notes", expanded=False):
                                        notes = st.text_area("Notes", height=100, key=f"nt_{selected_chart['chart_id']}_{icd}")

                                # Queue a row
                                save_queue.append({
                                    "icd": icd, "model": "both",
                                    "hcc": f"V24:{h24 or ''}/V28:{h28 or ''}", "hcc_label": lbl,
                                    "evidence_found": ev, "pages": pages, "notes": notes,
                                    "dos_start": pre_start, "dos_end": pre_end, "npi": pre_npi,
                                    "decision": "validated" if ev else ""
                                })

                    # ---- Each LLM suggestion gets its own tab
                    for i, sug in enumerate(llm_sugs, start=1):
                        with tabs[i]:
                            # auto-highlight this term when tab opens
                            set_terms([sug["term"]], [])
                            h24, lbl24 = hcc_for(sug["icd10"], "v24")
                            h28, lbl28 = hcc_for(sug["icd10"], "v28")
                            label = lbl24 or lbl28 or ""
                            st.caption(
                                f"Term **{sug['term']}** → ICD **{sug['icd10']}** → "
                                f"V24HCC **{h24 or '—'}** / V28HCC **{h28 or '—'}**{(' — ' + label) if label else ''}"
                            )
                            ev = st.checkbox("Evidence present", key=f"ev_llm_{selected_chart['chart_id']}_{i}")
                            pages = st.text_input("Evidence page(s)", key=f"pg_llm_{selected_chart['chart_id']}_{i}")
                            try:
                                with st.popover("Add notes", use_container_width=True, key=f"pop_llm_{selected_chart['chart_id']}_{i}"):
                                    notes = st.text_area("Notes", height=100, key=f"nt_llm_{selected_chart['chart_id']}_{i}")
                            except Exception:
                                with st.expander("Add notes", expanded=False):
                                    notes = st.text_area("Notes", height=100, key=f"nt_llm_{selected_chart['chart_id']}_{i}")

                            save_queue.append({
                                "icd": sug["icd10"], "model": "both",
                                "hcc": f"V24:{h24 or ''}/V28:{h28 or ''}", "hcc_label": label,
                                "evidence_found": ev, "pages": pages, "notes": notes,
                                "dos_start": pre_start, "dos_end": pre_end, "npi": pre_npi,
                                "decision": "validated" if ev else ""
                            })

                    # ---- Add task (last tab)
                    with tabs[-1]:
                        c1, c2 = st.columns([2, 1])
                        add_icd = c1.text_input("ICD-10 code (e.g., J44.1)", key=f"add_icd_{selected_chart['chart_id']}")
                        add_pages = c1.text_input("Evidence page(s)", key=f"add_pages_{selected_chart['chart_id']}")
                        try:
                            with st.popover("Add notes", use_container_width=True, key=f"add_pop_{selected_chart['chart_id']}"):
                                add_notes = st.text_area("Notes", height=100, key=f"add_notes_{selected_chart['chart_id']}")
                        except Exception:
                            with st.expander("Add notes", expanded=False):
                                add_notes = st.text_area("Notes", height=100, key=f"add_notes_{selected_chart['chart_id']}")
                        if c2.button("Add to this chart", key=f"add_btn_{selected_chart['chart_id']}") and add_icd.strip():
                            h24, lbl24 = hcc_for(add_icd.strip(), "v24")
                            h28, lbl28 = hcc_for(add_icd.strip(), "v28")
                            label = lbl24 or lbl28 or ""
                            save_queue.append({
                                "icd": add_icd.strip(), "model": "both",
                                "hcc": f"V24:{h24 or ''}/V28:{h28 or ''}", "hcc_label": label,
                                "evidence_found": True, "pages": add_pages, "notes": add_notes,
                                "dos_start": pre_start, "dos_end": pre_end, "npi": pre_npi,
                                "decision": "validated"
                            })
                            st.success("Added. Scroll down and Submit to save it.")

                    # ---- Submit / Save & Next ----
                    cA, cB = st.columns([1, 1])
                    submit = cA.button("Submit", key=f"save_{selected_chart['chart_id']}")
                    save_next = cB.button("Submit & Next chart", key=f"savenext_{selected_chart['chart_id']}")

                    if submit or save_next:
                        now = datetime.now().isoformat(timespec="seconds")
                        if coded_status is None or coded_status.empty:
                            coded_status = pd.DataFrame([{k: v for k, v in CODED_STATUS_SCHEMA.items()}]).iloc[0:0].copy()

                        rows_affected = 0
                        for entry in save_queue:
                            mask = (
                                (coded_status["member_id"] == member_id)
                                & (coded_status["chart_id"] == selected_chart["chart_id"])
                                & (coded_status["icd10"] == entry["icd"])
                                & (coded_status["model"] == entry["model"])
                            )
                            if coded_status[mask].empty:
                                new_row = {k: "" for k in CODED_STATUS_SCHEMA.keys()}
                                new_row.update(
                                    {
                                        "member_id": member_id,
                                        "chart_id": selected_chart["chart_id"],
                                        "icd10": entry["icd"],
                                        "model": entry["model"],
                                        "hcc": entry.get("hcc", ""),
                                        "hcc_label": entry.get("hcc_label", ""),
                                    }
                                )
                                coded_status = pd.concat([coded_status, pd.DataFrame([new_row])], ignore_index=True)
                                mask = (
                                    (coded_status["member_id"] == member_id)
                                    & (coded_status["chart_id"] == selected_chart["chart_id"])
                                    & (coded_status["icd10"] == entry["icd"])
                                    & (coded_status["model"] == entry["model"])
                                )

                            ts = now
                            if role == "primary":
                                coded_status.loc[mask, "primary_coder_initials"] = st.session_state.get("coder_initials", "")
                                coded_status.loc[mask, "primary_decision"] = entry["decision"]
                                coded_status.loc[mask, "primary_notes"] = entry.get("notes", "")
                                coded_status.loc[mask, "primary_timestamp"] = ts
                                coded_status.loc[mask, "primary_no_evidence"] = not entry["evidence_found"]
                            elif role == "secondary":
                                coded_status.loc[mask, "secondary_coder_initials"] = st.session_state.get("coder_initials", "")
                                coded_status.loc[mask, "secondary_decision"] = entry["decision"]
                                coded_status.loc[mask, "secondary_notes"] = entry.get("notes", "")
                                coded_status.loc[mask, "secondary_timestamp"] = ts
                            elif role == "final":
                                coded_status.loc[mask, "final_coder_initials"] = st.session_state.get("coder_initials", "")
                                coded_status.loc[mask, "final_decision"] = entry["decision"]
                                coded_status.loc[mask, "final_notes"] = entry.get("notes", "")
                                coded_status.loc[mask, "final_timestamp"] = ts

                            coded_status.loc[mask, "dos_start"] = entry["dos_start"]
                            coded_status.loc[mask, "dos_end"] = entry["dos_end"]
                            coded_status.loc[mask, "npi"] = entry["npi"]
                            coded_status.loc[mask, "evidence_pages"] = entry["pages"]
                            coded_status.loc[mask, "evidence_found"] = "True" if entry["evidence_found"] else "False"
                            coded_status.loc[mask, "dos"] = (
                                entry["dos_start"] if entry["dos_start"] == entry["dos_end"] else f"{entry['dos_start']} - {entry['dos_end']}"
                            )
                            rows_affected += 1

                        coded_status = ensure_columns(coded_status, CODED_STATUS_SCHEMA)
                        persist_coded_status(coded_status)
                        persist_audit("save_decisions", st.session_state.get("coder_initials", ""), member_id, selected_chart["chart_id"], rows_affected)

                        # Mark this chart as Reviewed in chart_inventory so the upper table shows it
                        chart_inventory.loc[
                            (chart_inventory["member_id"] == member_id) & (chart_inventory["chart_id"] == selected_chart["chart_id"]),
                            "coded_status",
                        ] = "Reviewed"
                        save_chart_inventory(chart_inventory)

                        st.success("Submitted.")

                        if save_next:
                            ids = inv["chart_id"].tolist()
                            if selected_chart["chart_id"] in ids:
                                i = ids.index(selected_chart["chart_id"])
                                nxt = ids[(i + 1) % len(ids)]
                                st.session_state["val_chart"] = nxt
                                st.rerun()

# ==================== PROGRESS ====================
with tab_progress:
    st.subheader("Throughput & signoffs")

    df = ensure_columns(load_csv(coded_status_path), CODED_STATUS_SCHEMA)

    base_charts = chart_inventory[["member_id", "chart_id"]].drop_duplicates()
    if base_charts.empty and not df.empty:
        base_charts = df[["member_id", "chart_id"]].drop_duplicates()

    if base_charts.empty:
        st.info("No charts found.")
    else:
        def first_nonempty(s: pd.Series) -> str:
            vals = [str(x).strip() for x in s.tolist() if str(x).strip()]
            return vals[0] if vals else ""

        def cols_for(role: str):
            return (
                f"{role}_coder_initials",
                f"{role}_decision",
                f"{role}_timestamp",
            )

        def role_summary(g: pd.DataFrame, role: str, other_inits: list[str]):
            """Per-role status for this chart, enforcing unique initials."""
            init_col, dec_col, ts_col = cols_for(role)

            initials = first_nonempty(g[init_col]) if (init_col in g.columns and not g.empty) else ""
            has_ts  = any(str(x).strip() for x in g[ts_col]) if (ts_col in g.columns and not g.empty) else False
            role_has_any_decision_rows = any(str(x).strip() for x in g[dec_col]) if (dec_col in g.columns and not g.empty) else False
            has_activity = bool(initials or has_ts or role_has_any_decision_rows)

            # Duplicate initials across roles? Flag + treat as not reviewed.
            is_dup_initials = initials and (initials in {i for i in other_inits if i})

            # Evidence only if THIS role has at least one validated
            evidence = False
            if dec_col in g.columns and not g.empty:
                evidence = any(str(x).strip().lower() == "validated" for x in g[dec_col])

            if is_dup_initials:
                status_text, status_emoji = "Duplicate initials", "⚠️"
                reviewed = False
            elif not has_activity:
                status_text, status_emoji = "Pending", "⏳"
                reviewed = False
            else:
                reviewed = True
                if evidence:
                    status_text, status_emoji = "Evidence", "🟢"
                else:
                    status_text, status_emoji = "No evidence", "🔴"

            return {
                "initials": initials or "—",
                "reviewed": reviewed,
                "evidence": evidence,
                "dup": is_dup_initials,
                "status_text": status_text,
                "status_emoji": status_emoji,
            }

        rows = []
        for _, r in base_charts.sort_values(["member_id", "chart_id"]).iterrows():
            mid, cid = r["member_id"], r["chart_id"]
            g = df[(df["member_id"] == mid) & (df["chart_id"] == cid)]

            # Grab first-known initials per role (may be empty); used for duplicate checks
            p_init = first_nonempty(g["primary_coder_initials"]) if "primary_coder_initials" in g.columns else ""
            s_init = first_nonempty(g["secondary_coder_initials"]) if "secondary_coder_initials" in g.columns else ""
            f_init = first_nonempty(g["final_coder_initials"]) if "final_coder_initials" in g.columns else ""

            p = role_summary(g, "primary",   other_inits=[s_init, f_init])
            s = role_summary(g, "secondary", other_inits=[p_init, f_init])
            f = role_summary(g, "final",     other_inits=[p_init, s_init])

            # Consensus among roles that genuinely reviewed (dup initials do not count)
            reviewed_roles = [r for r in [p, s, f] if r["reviewed"]]
            if len(reviewed_roles) == 0:
                consensus = ("⏳", "Pending")
            else:
                ev_vals = [r["evidence"] for r in reviewed_roles]
                if all(ev_vals):
                    consensus = ("🟢", "Evidence")
                elif not any(ev_vals):
                    consensus = ("🔴", "No evidence")
                else:
                    consensus = ("⚠️", "Conflict")

            rows.append({
                "member_id": mid,
                "chart_id": cid,
                "Primary": p["initials"],
                "Primary status": f"{p['status_emoji']} {p['status_text']}",
                "Secondary": s["initials"],
                "Secondary status": f"{s['status_emoji']} {s['status_text']}",
                "Final": f["initials"],
                "Final status": f"{f['status_emoji']} {f['status_text']}",
                "Consensus": f"{consensus[0]} {consensus[1]}",
            })

        progress_view = pd.DataFrame(rows)

        # Metrics: count reviewed charts by role (unique initials & non-pending)
        def reviewed_count(status_col: str) -> int:
            return progress_view.loc[
                ~progress_view[status_col].str.contains("Pending|Duplicate initials", case=False, na=False),
                ["member_id", "chart_id"]
            ].drop_duplicates().shape[0]

        unique_charts_total = base_charts.shape[0]
        p_count = reviewed_count("Primary status")
        s_count = reviewed_count("Secondary status")
        f_count = reviewed_count("Final status")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total charts", unique_charts_total)
        with c2: st.metric("Primary reviewed", p_count)
        with c3: st.metric("Secondary reviewed", s_count)
        with c4: st.metric("Final reviewed", f_count)

        st.caption("Legend: 🟢 Evidence · 🔴 No evidence · ⏳ Pending · ⚠️ Conflict / Duplicate")

        col_defs = {
            "member_id":        {"width": 130},
            "chart_id":         {"width": 120},
            "Primary":          {"width": 120},
            "Primary status":   {"width": 170},
            "Secondary":        {"width": 120},
            "Secondary status": {"width": 170},
            "Final":            {"width": 120},
            "Final status":     {"width": 170},
            "Consensus":        {"width": 140},
        }
        render_grid(progress_view, col_defs=col_defs, height=340, row_height=34, download_name=None, grid_key="grid_progress")
# ==================== RADV OUTPUT ====================
with tab_output:
    st.subheader("RADV output summary")
    df = ensure_columns(load_csv(coded_status_path), CODED_STATUS_SCHEMA)
    if df.empty:
        st.info("No decisions saved.")
    else:
        df["support_flag"] = (
            (df["primary_decision"] == "validated")
            | (df["secondary_decision"] == "validated")
            | (df["final_decision"] == "validated")
            | (df["evidence_found"].astype(str).str.lower() == "true")
        )

        # ---- FIXED AGGREGATION (no misaligned masks) ----
        charts_with_support = (
            df.loc[df["support_flag"], ["member_id", "chart_id"]]
              .drop_duplicates()
              .groupby("member_id")["chart_id"]
              .apply(lambda s: sorted(s.unique().tolist()))
              .rename("charts_with_support")
        )

        support_count = (
            df.groupby("member_id")["support_flag"]
              .sum()
              .rename("support_count")
        )

        agg = (
            support_count.to_frame()
            .join(charts_with_support, how="left")
            .reset_index()
        )
        agg["charts_with_support"] = agg["charts_with_support"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        # --------------------------------------------------

        base = radv_sample.copy()
        if not icd_to_hcc.empty and "icd10" in icd_to_hcc.columns and "icd10_submitted" in base.columns:
            v24_map = dict(icd_to_hcc[icd_to_hcc["model"].str.lower() == "v24"][["icd10", "hcc"]].values)
            v28_map = dict(icd_to_hcc[icd_to_hcc["model"].str.lower() == "v28"][["icd10", "hcc"]].values)
            base["v24_hcc"] = base["icd10_submitted"].map(v24_map)
            base["v28_hcc"] = base["icd10_submitted"].map(v28_map)

        out = (
            base.merge(agg[["member_id", "support_count"]], on="member_id", how="left")
                .fillna({"support_count": 0})
        )

        out_cols = [c for c in ["member_id", "icd10_submitted", "dx_summary", "v24_hcc", "v28_hcc", "support_count"] if c in out.columns]
        col_defs = {
            "member_id": {"width": 130},
            "icd10_submitted": {"width": 140},
            "dx_summary": {"width": 460, "wrapText": True, "autoHeight": True, "cellClass": "left-text"},
            "v24_hcc": {"width": 120},
            "v28_hcc": {"width": 120},
            "support_count": {"width": 160, "header_name": "Charts w/ support"},
        }
        render_grid(sanitize_df(out[out_cols]), col_defs=col_defs, height=230, download_name="radv_output", grid_key="grid_output")
        st.write("Charts with supporting evidence (by member):")
        for _, row in agg.iterrows():
            st.markdown(f"**{row['member_id']}** — {row['charts_with_support']}")

        if coded_status_path.exists():
            st.download_button(
                "Download coding_status.csv",
                data=open(coded_status_path, "rb").read(),
                file_name="radv_coding_status.csv",
                mime="text/csv",
                key="dl_coded_status"
            )

# ==================== REFERENCE ====================
with tab_reference:
    st.subheader("ICD-10 → HCC crosswalk (v24/v28)")
    st.caption("Reference only — load CMS official mapping here in production.")
    df_ref = icd_to_hcc if not icd_to_hcc.empty else load_csv(DATA_DIR / "icd_to_hcc_fallback.csv")
    render_grid(sanitize_df(df_ref), height=360, download_name="icd10_to_hcc", grid_key="grid_reference")
st.caption("Demo only – do not use with real PHI.")