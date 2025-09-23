import os, json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from services import data as D
from services import llm as L
from services import pdf as P
from services import ui as U

# ---- Safe LLM wrappers (fallback to local heuristics if L is unavailable) ----
def _safe_pick_provider():
    try:
        return L.pick_llm_provider()
    except Exception:
        return "local"

def _safe_parse_llm(v):
    # Prefer services.llm.parse_llm_suggestions
    try:
        return L.parse_llm_suggestions(v)
    except Exception:
        pass
    # Local permissive JSON parse
    try:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict) and "term" in x and "icd10" in x]
        if isinstance(v, str) and v.strip():
            data = json.loads(v)
            return [x for x in data if isinstance(x, dict) and "term" in x and "icd10" in x]
    except Exception:
        pass
    return []

def _safe_get_candidates(text: str, provider: str):
    """
    Ask services.llm.get_candidates(text, provider) for ICD suggestions.
    Falls back to simple keyword map if the LLM path errors out.
    """
    try:
        c = L.get_candidates(text, provider)
        if isinstance(c, list):
            return c
    except Exception:
        pass
    # Local fallback (only if L fails)
    TERM_TO_ICD = {
        "bronchitis": "J40",
        "pneumonia": "J18.9",
        "emphysema": "J43.9",
        "copd": "J44.9",
        "diabetes": "E11.9",
        "hiv": "B20",
    }
    low = (text or "").lower()
    out, seen = [], set()
    for term, icd in TERM_TO_ICD.items():
        if term in low and (term, icd) not in seen:
            out.append({"term": term, "icd10": icd})
            seen.add((term, icd))
    return out

def _safe_get_task_terms(text: str, icd_list: list[str], provider: str):
    """
    Ask services.llm.get_task_terms(text, icd_list, provider) for highlight terms per ICD.
    Falls back to llm_hints.json if the LLM path errors out.
    """
    try:
        m = L.get_task_terms(text, icd_list, provider)
        if isinstance(m, dict):
            return m
    except Exception:
        pass
    # Fallback to static hints in this app if L fails
    try:
        hints_path = D.DATA_DIR / "llm_hints.json"
        hints = json.loads(hints_path.read_text()) if hints_path.exists() else {}
    except Exception:
        hints = {}
    out = {}
    for icd in (icd_list or []):
        h = hints.get(icd, {})
        out[icd] = {
            "primary_terms": h.get("primary_terms", []),
            "offset_terms": h.get("offset_terms", []),
        }
    return out

def _coarse_best_page(pages: list[str], terms: list[str]) -> int:
    """Return 1-based best page index by simple term frequency."""
    if not pages:
        return 1
    if not terms:
        return 1
    terms = [t for t in terms if t]
    if not terms:
        return 1
    scores = []
    for p in pages:
        low = (p or "").lower()
        scores.append(sum(low.count(t.lower()) for t in terms))
    if not any(scores):
        return 1
    return 1 + scores.index(max(scores))

def _safe_locate_evidence(pdf_path: str, icd_list: list[str], llm_sugs: list[dict], dx_summary: str, provider: str):
    """
    Try services.llm.locate_evidence(...) to pick best page + terms.
    Accepts flexible return shapes and falls back to heuristic when needed.
    Returns dict: {"page_1based": int, "primary_terms": list[str], "offset_terms": list[str]}
    """
    pages = P.read_pdf_pages(pdf_path)
    # 1) Try LLM-driven locator if available
    try:
        # Try several likely call signatures defensively
        result = None
        try:
            result = L.locate_evidence(pages=pages, icd_list=icd_list, dx_summary=dx_summary, provider=provider)
        except TypeError:
            try:
                full_text = "\n".join(pages)
                result = L.locate_evidence(full_text, icd_list, provider)
            except Exception:
                pass
        if isinstance(result, dict):
            # normalize keys: best_page (1-based) OR page (0-based)
            page_1 = None
            if "best_page" in result:
                page_1 = int(result.get("best_page") or 1)
            elif "page" in result:
                page_1 = int(result.get("page") or 0) + 1
            prim = [str(x) for x in (result.get("primary_terms") or []) if str(x).strip()]
            offs = [str(x) for x in (result.get("offset_terms") or []) if str(x).strip()]
            if page_1 is None or page_1 < 1:
                page_1 = _coarse_best_page(pages, prim or [])
            return {"page_1based": page_1, "primary_terms": prim, "offset_terms": offs}
        if isinstance(result, (list, tuple)) and len(result) >= 3:
            page0, prim, offs = result[0], list(result[1] or []), list(result[2] or [])
            return {"page_1based": int(page0) + 1, "primary_terms": prim, "offset_terms": offs}
    except Exception:
        pass

    # 2) Heuristic fallback
    # Prefer the first RADV ICD's hint terms; else first LLM suggestion's term
    target_prim, target_off = [], []
    hints = {}
    try:
        hints = json.loads((D.DATA_DIR / "llm_hints.json").read_text())
    except Exception:
        hints = {}
    if icd_list:
        icd0 = icd_list[0]
        h = hints.get(icd0, {})
        target_prim = h.get("primary_terms", []) or [icd0]
        target_off = h.get("offset_terms", [])
    elif llm_sugs:
        target_prim = [llm_sugs[0].get("term", "")]
        target_off = []
    page_1 = _coarse_best_page(pages, target_prim)
    return {"page_1based": page_1, "primary_terms": target_prim, "offset_terms": target_off}

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="RADV Chart Validation Workbench", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ---------- LIGHT UI POLISH ----------
st.markdown("""
<style>
.stSidebar .stButton > button { width: 100%; display: flex !important; justify-content:flex-start !important; text-align:left !important;
 background:#10B981 !important; color:#fff !important; border:1px solid #059669 !important; border-radius:10px !important; }
.stSidebar .stButton > button:hover { filter: brightness(0.96); }
:root { --brand:#0F766E; --brand-soft:#ECFDF5; --muted:#f6f8fb; --line:#e5e7eb; }
.block-container { max-width:1500px !important; padding-top:.6rem; }
h1,h2,h3 { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
.stButton>button[kind="primary"] { background: var(--brand); color:#fff; border-radius:10px; border:1px solid #0b5e58; }
.stButton>button:hover { filter:brightness(0.95); }
.legend-sticky { position:sticky; top:0; z-index:2; background:#fff; padding:6px 8px; border:1px solid var(--line); border-radius:8px; display:inline-block; }
.badge { display:inline-block; padding:.18rem .5rem; border-radius:999px; font-size:.78rem; border:1px solid #eaeaea; background:#f8f9fb; color:#333; }
.pill { display:inline-block; padding:.22rem .65rem; border-radius:999px; font-size:.80rem; font-weight:600; border:1px solid var(--line); }
.pill--ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
.pill--pending { background:#fff1f2; color:#9f1239; border-color:#fecdd3; }
.table-row { padding:6px 0; border-bottom:1px solid var(--line); }
.table-header { padding:6px 0; border-bottom:2px solid var(--line); }
.center-text { justify-content:center !important; }
.left-text { justify-content:flex-start !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Load/Seed ----------
demo_pdf = D.PDF_DIR / "CHART0010.pdf"
P.ensure_valid_pdf(demo_pdf)
D.seed_demo(demo_pdf)

radv_sample = D.load_csv(D.DATA_DIR / "radv_sample.csv")
chart_inventory = D.ensure_columns(D.load_csv(D.DATA_DIR / "chart_inventory.csv"), D.CHART_INVENTORY_SCHEMA)
coded_status_path = D.coded_status_path
coded_status = D.ensure_columns(D.load_csv(coded_status_path), D.CODED_STATUS_SCHEMA)

user_mapping_path = D.DATA_DIR / "icd10_to_hcc_v24_v28.csv"
icd_to_hcc_path = user_mapping_path if user_mapping_path.exists() else (D.DATA_DIR / "icd_to_hcc_fallback.csv")
icd_to_hcc = D.load_icd_mapping_csv(icd_to_hcc_path)

# hints file (used only as fallback if LLM can't produce phrases)
hints_file = D.DATA_DIR / "llm_hints.json"
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

# ---------- Sidebar ----------
st.sidebar.header("Coder Profile")
st.session_state["coder_initials"] = st.sidebar.text_input("Your initials", value=st.session_state.get("coder_initials", "PC"))
role = st.sidebar.selectbox("Your role", ["primary", "secondary", "final"])
st.sidebar.markdown("---")

# LLM provider (detected) + user toggle to avoid hangs
LLM_PROVIDER_DETECTED = _safe_pick_provider()
enable_llm = st.sidebar.checkbox(
    "Enable LLM calls (Vertex/Gemini/OpenAI)", value=True,
    help="If off, the app uses local demo rules only."
)
LLM_PROVIDER = LLM_PROVIDER_DETECTED if enable_llm else "local"
_label_map = {
    "vertex": "Vertex AI (ADC/Service Account)",
    "gemini_api": "Gemini API key",
    "openai": "OpenAI API key",
    "local": "Demo rules",
}
label_detected = _label_map[LLM_PROVIDER_DETECTED]
label_effective = _label_map[LLM_PROVIDER]
status_txt = "Enabled" if (enable_llm and LLM_PROVIDER != "local") else "Disabled (using demo rules)"
st.sidebar.info(f"LLM provider (detected): **{label_detected}** — Status: **{status_txt}**")

# BQ buttons
bq_load = st.sidebar.button("Load RADV Sample → BigQuery", type="primary", key="btn_bq_load")
bq_save = st.sidebar.button("Save Coding Status → BigQuery", type="primary", key="btn_bq_save")
st.sidebar.caption("Yellow = likely evidence · Blue = suspects/offsets")
st.sidebar.markdown('<span class="badge">Demo only – do not use with real PHI.</span>', unsafe_allow_html=True)

if bq_load:
    rs, ci, err = D.load_from_bigquery(st)
    if err:
        st.sidebar.error(err)
    elif rs is not None and ci is not None:
        radv_sample = rs
        chart_inventory = D.ensure_columns(ci, D.CHART_INVENTORY_SCHEMA)
        st.sidebar.success("Loaded from BigQuery.")

if bq_save:
    df_bq = D.ensure_columns(D.load_csv(coded_status_path), D.CODED_STATUS_SCHEMA)
    if df_bq.empty:
        st.sidebar.info("Nothing to save yet.")
    else:
        err = D.save_to_bigquery(st, df_bq, "coded_status")
        st.sidebar.success("Saved to BigQuery table coded_status.") if not err else st.sidebar.error(err)

# ---------- Title ----------
st.title("RADV Chart Validation Workbench")
st.caption("Professional reviewer UI • v24/v28 aware • coder/QA trail")

# ---------- Tabs ----------
tab_sample, tab_inventory, tab_validate, tab_progress, tab_output, tab_reference = st.tabs(
    ["RADV sample", "Chart inventory", "RADV validation", "Progress", "RADV output", "Reference"]
)

# ==================== RADV SAMPLE ====================
with tab_sample:
    counts = chart_inventory.groupby("member_id")["chart_id"].nunique().rename("charts_count") if not chart_inventory.empty else pd.Series([], dtype=int)
    sample = radv_sample.copy().merge(counts, on="member_id", how="left").fillna({"charts_count": 0})

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
    U.render_grid(sample[cols], col_defs=col_defs, height=240, download_name="radv_sample", grid_key="grid_sample")

# ==================== CHART INVENTORY ====================
with tab_inventory:
    left, right = st.columns([6,1])
    with left: st.subheader("All charts")
    with right: run_all = st.button("Run NLP on ALL charts", type="primary")

    # DQ flags
    if not chart_inventory.empty:
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
            pdf = row.get("pdf_path","")
            pages_text = P.read_pdf_pages(pdf)
            full_text = "\n".join(pages_text) if pages_text else P.read_pdf_text(pdf)

            # Add dx_summary to give the LLM signal even if the demo PDF is sparse
            dx = ""
            try:
                dx = radv_sample.loc[radv_sample["member_id"] == row.get("member_id"), "dx_summary"].iloc[0]
            except Exception:
                pass
            combined = (dx + "\n\n" + full_text)[:8000] if dx else (full_text or "")

            sugs = _safe_get_candidates(combined, LLM_PROVIDER)
            updated.loc[idx, "llm_scanned"] = True
            updated.loc[idx, "llm_suggestions"] = json.dumps(sugs)

        chart_inventory = D.ensure_columns(updated, D.CHART_INVENTORY_SCHEMA)
        D.save_chart_inventory(chart_inventory)
        st.success(f"NLP scan complete for all charts. Provider: {label_effective}")
   
    # Filters
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        f_member = st.selectbox("Filter by member", ["(All)"] + sorted(chart_inventory["member_id"].dropna().unique().tolist()) if not chart_inventory.empty else ["(All)"])
    with c2:
        f_provider = st.text_input("Filter provider (contains)", "")
    with c3:
        f_llm = st.selectbox("LLM scanned", ["(All)","Yes","No"])

    view = chart_inventory.copy()
    if not view.empty:
        if f_member != "(All)": view = view[view["member_id"]==f_member]
        if f_provider.strip():  view = view[view["provider"].astype(str).str.contains(f_provider.strip(), case=False, na=False)]
        if f_llm == "Yes":      view = view[view["llm_scanned"]==True]
        elif f_llm == "No":     view = view[view["llm_scanned"]==False]

    cols = [c for c in ["member_id","chart_id","provider","npi","start_dos","end_dos","llm_scanned","dq_flag"] if c in view.columns]
    col_defs = {
        "member_id": {"width": 110},
        "chart_id":  {"width": 110},
        "provider":  {"width": 250, "cellClass": "left-text"},
        "npi":       {"width": 120},
        "start_dos": {"width": 120},
        "end_dos":   {"width": 120},
        "llm_scanned":{"width": 120},
        "dq_flag":   {"width": 220, "cellClass": "left-text", "header_name": "DQ warnings"},
    }
    U.render_grid(view[cols] if not view.empty else view, col_defs=col_defs, height=420, row_height=34, download_name="chart_inventory", grid_key="grid_inventory")

# ==================== RADV VALIDATION ====================
with tab_validate:
    st.subheader("RADV validation")

    # enrich HCC per chart for display
    inv_full = chart_inventory.copy()
    if not inv_full.empty and not icd_to_hcc.empty:
        def hcc_join(diags, model):
            codes = [s.strip() for s in str(diags).split(";") if s.strip()]
            if not codes: return ""
            out = []
            for c in codes:
                rec = icd_to_hcc[(icd_to_hcc["icd10"]==c) & (icd_to_hcc["model"].str.lower()==model)]
                if not rec.empty: out.append(str(rec.iloc[0]["hcc"]))
            return ";".join(sorted(set(out)))
        inv_full["v24_hccs"] = inv_full["radv_diags"].apply(lambda d: hcc_join(d, "v24"))
        inv_full["v28_hccs"] = inv_full["radv_diags"].apply(lambda d: hcc_join(d, "v28"))

    members = sorted(radv_sample["member_id"].dropna().unique().tolist())
    if not members:
        st.info("No members found.")
    else:
        member_id = st.selectbox("Member", members, key="val_member")
        inv = inv_full[inv_full["member_id"] == member_id].copy()

        def chart_reviewed(mid, cid):
            cs = coded_status[(coded_status["member_id"] == mid) & (coded_status["chart_id"] == cid)]
            if not cs.empty: return True
            row = chart_inventory[(chart_inventory["member_id"] == mid) & (chart_inventory["chart_id"] == cid)]
            return str(row["coded_status"].iloc[0]).strip().lower() == "reviewed" if not row.empty else False

        if inv.empty:
            st.info("No charts for this member.")
        else:
            st.markdown("**Charts for member**")
            h1, h2, h3, h4, h5, h6 = st.columns([1.3,3.2,1.2,2.2,1.1,1.1])
            with h1: st.markdown('<div class="table-header"><strong>Chart</strong></div>', unsafe_allow_html=True)
            with h2: st.markdown('<div class="table-header"><strong>Provider</strong></div>', unsafe_allow_html=True)
            with h3: st.markdown('<div class="table-header"><strong>NPI</strong></div>', unsafe_allow_html=True)
            with h4: st.markdown('<div class="table-header"><strong>DOS</strong></div>', unsafe_allow_html=True)
            with h5: st.markdown('<div class="table-header"><strong>Status</strong></div>', unsafe_allow_html=True)
            with h6: st.markdown('<div class="table-header"><strong>Action</strong></div>', unsafe_allow_html=True)

            for _, row in inv.iterrows():
                c1, c2, c3, c4, c5, c6 = st.columns([1.3,3.2,1.2,2.2,1.1,1.1])
                with c1: st.markdown(f'<div class="table-row"><strong>{row["chart_id"]}</strong></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="table-row">{str(row.get("provider",""))}</div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="table-row">{str(row.get("npi",""))}</div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="table-row">{row.get("start_dos","")} → {row.get("end_dos","")}</div>', unsafe_allow_html=True)
                status_html = U.pill("Reviewed","ok") if chart_reviewed(member_id, row["chart_id"]) else U.pill("Pending","pending")
                with c5: st.markdown(f'<div class="table-row">{status_html}</div>', unsafe_allow_html=True)
                with c6:
                    if st.button("Review", key=f"review_btn_{member_id}_{row['chart_id']}", type="primary"):
                        st.session_state["val_chart"] = row["chart_id"]
                        st.session_state["review_open"] = True
                        st.rerun()

        if not inv.empty and st.session_state.get("review_open"):
            selected_chart = inv[inv["chart_id"] == st.session_state.get("val_chart", inv["chart_id"].iloc[0])].iloc[0]

            # --- LLM terms + focus selector (before viewer)
            # Pull text & dx_summary to strengthen sparse PDFs
            pages_list = P.read_pdf_pages(selected_chart["pdf_path"])
            raw_text = "\n".join(pages_list) if pages_list else P.read_pdf_text(selected_chart["pdf_path"])
            dx = ""
            try:
                dx = radv_sample.loc[radv_sample["member_id"] == member_id, "dx_summary"].iloc[0]
            except Exception:
                pass
            combined_text = ((dx + "\n\n" + (raw_text or "")) if dx else (raw_text or ""))[:8000]

            # Detect missing text layer (scanned PDFs) so coder isn’t confused
            pages_list = P.read_pdf_pages(selected_chart["pdf_path"])
            if not any(p.strip() for p in pages_list):
                st.warning("This PDF appears to have **no selectable text** (likely scanned). "
                    "Highlights won’t work until the document is OCR’d. "
                    "You can still view pages, but search/highlighting will be empty.")

# use pages_list later for best-page selection

            icds_for_chart = [s.strip() for s in str(selected_chart.get("radv_diags","")).split(";") if s.strip()]

            llm_sugs = _safe_parse_llm(selected_chart.get("llm_suggestions",""))
            if not llm_sugs:
                llm_sugs = _safe_get_candidates(combined_text, LLM_PROVIDER)

            terms_map = _safe_get_task_terms(combined_text, icds_for_chart, LLM_PROVIDER)

            # --- Auto-focus: set highlight terms and best page once per chart selection
            chart_key = selected_chart["chart_id"]
            prim_key = f"prim_terms_{chart_key}"
            off_key  = f"off_terms_{chart_key}"
            page_key = f"page_{chart_key}"

            if not st.session_state.get(prim_key) and not st.session_state.get(off_key):
                # Ask LLM to pick page/terms (with graceful fallback)
                loc = _safe_locate_evidence(
                    pdf_path=selected_chart["pdf_path"],
                    icd_list=icds_for_chart,
                    llm_sugs=llm_sugs,
                    dx_summary=dx,
                    provider=LLM_PROVIDER
                )
                st.session_state[prim_key] = loc.get("primary_terms", [])
                st.session_state[off_key]  = loc.get("offset_terms", [])
                st.session_state[page_key] = int(loc.get("page_1based", 1))

            left, right = st.columns([8,4], gap="large")

            # -------- Left: viewer --------
            with left:
                st.markdown("**Chart viewer**")
                total_pages = P.pdf_num_pages(selected_chart["pdf_path"])
                current_page = st.session_state.get(page_key, 1)
                page = st.number_input("Page", min_value=1, max_value=max(total_pages,1), value=current_page, step=1,
                                       help=f"Total pages: {total_pages or 'unknown'}",
                                       key=f"numinput_{selected_chart['chart_id']}")
                st.session_state[page_key] = page
                prim_terms = st.session_state.get(prim_key, [])
                off_terms  = st.session_state.get(off_key, [])
                P.show_pdf_inline(selected_chart["pdf_path"], page, prim_terms, off_terms, key_suffix=selected_chart["chart_id"])
                st.markdown('<div class="legend-sticky"><span class="badge" style="background:#fff3b0;border-color:#e7d27a">Evidence</span> '
                            '<span class="badge" style="background:#cfe8ff;border-color:#8ec5ff">Suspects/Offsets</span></div>', unsafe_allow_html=True)

            # -------- Right: tasks --------
            with right:
                st.markdown("**Tasks**")

                def hcc_for(icd, model):
                    recs = icd_to_hcc[(icd_to_hcc["icd10"]==icd) & (icd_to_hcc["model"].str.lower()==model)]
                    if recs.empty: return None, None
                    r = recs.iloc[0]
                    return str(r["hcc"]), str(r.get("hcc_label","") or "")

                def both_models_line(icd):
                    h24,lbl24 = hcc_for(icd,"v24")
                    h28,lbl28 = hcc_for(icd,"v28")
                    label = lbl24 or lbl28 or ""
                    text = f"ICD **{icd}** → V24HCC **{h24 or '—'}** / V28HCC **{h28 or '—'}**"
                    if label: text += f" — {label}"
                    return text, h24, h28, label

                pre_npi   = str(selected_chart.get("npi",""))
                pre_start = str(selected_chart.get("start_dos",""))
                pre_end   = str(selected_chart.get("end_dos",""))

                st.caption(f"LLM suggestions ({label_effective}): {len(llm_sugs)}")
                t_labels = ["RADV tasks"] + [f"LLM: {s['term']}" for s in llm_sugs] + ["Add task"]
                tabs = st.tabs(t_labels)

                save_queue = []

                with st.expander("LLM results / debug", expanded=False):
                    st.write(f"Provider: **{label_effective}** | Enabled: **{enable_llm and LLM_PROVIDER!='local'}**")
                    st.caption("Candidates (term → ICD10):")
                    st.json(llm_sugs if llm_sugs else [])
                    st.caption("Task terms (per RADV ICD):")
                    st.json(terms_map if isinstance(terms_map, dict) else {})

                    if st.button("Re-run LLM on this chart", key=f"rerun_llm_{selected_chart['chart_id']}"):
                        llm_sugs = _safe_get_candidates(combined_text, LLM_PROVIDER)
                        terms_map = _safe_get_task_terms(combined_text, icds_for_chart, LLM_PROVIDER)
                        # persist suggestions back to inventory so they show in inventory tab too
                        chart_inventory.loc[
                            (chart_inventory["member_id"]==member_id) &
                            (chart_inventory["chart_id"]==selected_chart["chart_id"]),
                            "llm_suggestions"
                        ] = json.dumps(llm_sugs)
                        D.save_chart_inventory(chart_inventory)
                        # update highlights to first available suggestion/ICD
                        target_terms = []
                        if icds_for_chart:
                            lm = terms_map.get(icds_for_chart[0], {})
                            target_terms = lm.get("primary_terms", [])
                        elif llm_sugs:
                            target_terms = [llm_sugs[0]["term"]]
                        st.session_state[f"prim_terms_{selected_chart['chart_id']}"] = target_terms
                        st.session_state[f"off_terms_{selected_chart['chart_id']}"]  = []
                        # jump to best page
                        if pages_list and target_terms:
                            scores = [sum(p.lower().count(t.lower()) for t in target_terms) for p in pages_list]
                            best = 1 + (scores.index(max(scores)) if any(scores) else 0)
                            st.session_state[f"page_{selected_chart['chart_id']}"] = best
                        st.experimental_rerun()

                # ---- RADV tasks
                with tabs[0]:
                    if not icds_for_chart:
                        st.info("No RADV ICDs listed for this chart.")
                    for icd in icds_for_chart:
                        lm = terms_map.get(icd, {})
                        prim = (lm.get("primary_terms") or LLM_HINTS.get(icd,{}).get("primary_terms",[]))
                        offs = (lm.get("offset_terms")  or LLM_HINTS.get(icd,{}).get("offset_terms",[]))
                        line, h24, h28, lbl = both_models_line(icd)
                        with st.container(border=True):
                            st.caption(line)
                            if st.button("Focus highlights", key=f"focus_{selected_chart['chart_id']}_{icd}"):
                                st.session_state[prim_key] = prim
                                st.session_state[off_key]  = offs
                                # Jump to best page by these terms
                                st.session_state[page_key] = _coarse_best_page(P.read_pdf_pages(selected_chart["pdf_path"]), prim or [])
                                st.rerun()

                            ev    = st.checkbox("Evidence present", key=f"ev_{selected_chart['chart_id']}_{icd}")
                            pages = st.text_input("Evidence page(s)", key=f"pg_{selected_chart['chart_id']}_{icd}")
                            notes = st.text_area("Notes", height=80, key=f"nt_{selected_chart['chart_id']}_{icd}")
                            save_queue.append({
                                "icd": icd, "model": "both",
                                "hcc": f"V24:{h24 or ''}/V28:{h28 or ''}", "hcc_label": lbl,
                                "evidence_found": ev, "pages": pages, "notes": notes,
                                "dos_start": pre_start, "dos_end": pre_end, "npi": pre_npi,
                                "decision": "validated" if ev else ""
                            })

                # ---- LLM suggestions
                for i, sug in enumerate(llm_sugs, start=1):
                    with tabs[i]:
                        h24,lbl24 = hcc_for(sug["icd10"],"v24")
                        h28,lbl28 = hcc_for(sug["icd10"],"v28")
                        label = lbl24 or lbl28 or ""
                        st.caption(f"Term **{sug['term']}** → ICD **{sug['icd10']}** → V24HCC **{h24 or '—'}** / V28HCC **{h28 or '—'}**{(' — ' + label) if label else ''}")
                        if st.button("Focus highlights", key=f"focus_llm_{selected_chart['chart_id']}_{i}"):
                            st.session_state[prim_key] = [sug["term"]]
                            st.session_state[off_key]  = []
                            st.session_state[page_key] = _coarse_best_page(P.read_pdf_pages(selected_chart["pdf_path"]), [sug["term"]])
                            st.rerun()                       

                        ev    = st.checkbox("Evidence present", key=f"ev_llm_{selected_chart['chart_id']}_{i}")
                        pages = st.text_input("Evidence page(s)", key=f"pg_llm_{selected_chart['chart_id']}_{i}")
                        notes = st.text_area("Notes", height=80, key=f"nt_llm_{selected_chart['chart_id']}_{i}")
                        save_queue.append({
                            "icd": sug["icd10"], "model": "both",
                            "hcc": f"V24:{h24 or ''}/V28:{h28 or ''}", "hcc_label": label,
                            "evidence_found": ev, "pages": pages, "notes": notes,
                            "dos_start": pre_start, "dos_end": pre_end, "npi": pre_npi,
                            "decision": "validated" if ev else ""
                        })

                # ---- Add task
                with tabs[-1]:
                    c1, c2 = st.columns([2,1])
                    add_icd   = c1.text_input("ICD-10 code (e.g., J44.1)", key=f"add_icd_{selected_chart['chart_id']}")
                    add_pages = c1.text_input("Evidence page(s)", key=f"add_pages_{selected_chart['chart_id']}")
                    add_notes = c1.text_area("Notes", height=80, key=f"add_notes_{selected_chart['chart_id']}")
                    if c2.button("Add to this chart", key=f"add_btn_{selected_chart['chart_id']}") and add_icd.strip():
                        h24,lbl24 = hcc_for(add_icd.strip(),"v24")
                        h28,lbl28 = hcc_for(add_icd.strip(),"v28")
                        label = lbl24 or lbl28 or ""
                        save_queue.append({
                            "icd": add_icd.strip(), "model": "both",
                            "hcc": f"V24:{h24 or ''}/V28:{h28 or ''}", "hcc_label": label,
                            "evidence_found": True, "pages": add_pages, "notes": add_notes,
                            "dos_start": pre_start, "dos_end": pre_end, "npi": pre_npi,
                            "decision": "validated"
                        })
                        st.success("Added. Scroll down and Submit to save it.")

                # ---- Submit
                cA, cB = st.columns(2)
                submit   = cA.button("Submit", key=f"save_{selected_chart['chart_id']}")
                save_next= cB.button("Submit & Next chart", key=f"savenext_{selected_chart['chart_id']}")

                if submit or save_next:
                    now = datetime.now().isoformat(timespec="seconds")
                    if coded_status is None or coded_status.empty:
                        coded_status = pd.DataFrame([{k:v for k,v in D.CODED_STATUS_SCHEMA.items()}]).iloc[0:0].copy()
                    rows_affected = 0
                    for entry in save_queue:
                        mask = (
                            (coded_status["member_id"] == member_id) &
                            (coded_status["chart_id"]  == selected_chart["chart_id"]) &
                            (coded_status["icd10"]     == entry["icd"]) &
                            (coded_status["model"]     == entry["model"])
                        )
                        if coded_status[mask].empty:
                            new_row = {k:"" for k in D.CODED_STATUS_SCHEMA.keys()}
                            new_row.update({
                                "member_id": member_id, "chart_id": selected_chart["chart_id"],
                                "icd10": entry["icd"], "model": entry["model"],
                                "hcc": entry.get("hcc",""), "hcc_label": entry.get("hcc_label",""),
                            })
                            coded_status = pd.concat([coded_status, pd.DataFrame([new_row])], ignore_index=True)
                            mask = (
                                (coded_status["member_id"] == member_id) &
                                (coded_status["chart_id"]  == selected_chart["chart_id"]) &
                                (coded_status["icd10"]     == entry["icd"]) &
                                (coded_status["model"]     == entry["model"])
                            )
                        ts = now
                        if role == "primary":
                            coded_status.loc[mask, "primary_coder_initials"] = st.session_state.get("coder_initials","")
                            coded_status.loc[mask, "primary_decision"] = entry["decision"]
                            coded_status.loc[mask, "primary_notes"] = entry.get("notes","")
                            coded_status.loc[mask, "primary_timestamp"] = ts
                            coded_status.loc[mask, "primary_no_evidence"] = not entry["evidence_found"]
                        elif role == "secondary":
                            coded_status.loc[mask, "secondary_coder_initials"] = st.session_state.get("coder_initials","")
                            coded_status.loc[mask, "secondary_decision"] = entry["decision"]
                            coded_status.loc[mask, "secondary_notes"] = entry.get("notes","")
                            coded_status.loc[mask, "secondary_timestamp"] = ts
                        elif role == "final":
                            coded_status.loc[mask, "final_coder_initials"] = st.session_state.get("coder_initials","")
                            coded_status.loc[mask, "final_decision"] = entry["decision"]
                            coded_status.loc[mask, "final_notes"] = entry.get("notes","")
                            coded_status.loc[mask, "final_timestamp"] = ts
                        coded_status.loc[mask, "dos_start"] = entry["dos_start"]
                        coded_status.loc[mask, "dos_end"]   = entry["dos_end"]
                        coded_status.loc[mask, "npi"]       = entry["npi"]
                        coded_status.loc[mask, "evidence_pages"] = entry["pages"]
                        coded_status.loc[mask, "evidence_found"] = "True" if entry["evidence_found"] else "False"
                        coded_status.loc[mask, "dos"] = entry["dos_start"] if entry["dos_start"] == entry["dos_end"] else f"{entry['dos_start']} - {entry['dos_end']}"
                        rows_affected += 1
                    D.persist_coded_status(coded_status)
                    D.persist_audit("save_decisions", st.session_state.get("coder_initials",""), member_id, selected_chart["chart_id"], rows_affected)
                    chart_inventory.loc[
                        (chart_inventory["member_id"]==member_id) & (chart_inventory["chart_id"]==selected_chart["chart_id"]),
                        "coded_status"
                    ] = "Reviewed"
                    D.save_chart_inventory(chart_inventory)
                    st.success("Submitted.")
                    if save_next:
                        ids = inv["chart_id"].tolist()
                        if selected_chart["chart_id"] in ids:
                            i = ids.index(selected_chart["chart_id"])
                            nxt = ids[(i+1) % len(ids)]
                            st.session_state["val_chart"] = nxt
                            st.rerun()

# ==================== PROGRESS ====================
with tab_progress:
    st.subheader("Throughput & signoffs")
    df = D.ensure_columns(D.load_csv(coded_status_path), D.CODED_STATUS_SCHEMA)
    base_charts = chart_inventory[["member_id","chart_id"]].drop_duplicates()
    if base_charts.empty and not df.empty:
        base_charts = df[["member_id","chart_id"]].drop_duplicates()
    if base_charts.empty:
        st.info("No charts found.")
    else:
        def first_nonempty(s: pd.Series) -> str:
            vals = [str(x).strip() for x in s.tolist() if str(x).strip()]
            return vals[0] if vals else ""
        def cols_for(r: str): return (f"{r}_coder_initials", f"{r}_decision", f"{r}_timestamp")
        def role_summary(g: pd.DataFrame, role: str, other_inits: list[str]):
            init_col, dec_col, ts_col = cols_for(role)
            initials = first_nonempty(g[init_col]) if (init_col in g.columns and not g.empty) else ""
            has_ts  = any(str(x).strip() for x in g[ts_col]) if (ts_col in g.columns and not g.empty) else False
            has_dec = any(str(x).strip() for x in g[dec_col]) if (dec_col in g.columns and not g.empty) else False
            has_activity = bool(initials or has_ts or has_dec)
            is_dup_initials = initials and (initials in {i for i in other_inits if i})
            evidence = any(str(x).strip().lower() == "validated" for x in g.get(dec_col, []))
            if is_dup_initials: return {"initials": initials or "—","reviewed": False,"evidence": False,"status_text": "Duplicate initials","status_emoji":"⚠️"}
            if not has_activity: return {"initials": initials or "—","reviewed": False,"evidence": False,"status_text": "Pending","status_emoji":"⏳"}
            return {"initials": initials or "—","reviewed": True,"evidence": evidence,"status_text": ("Evidence" if evidence else "No evidence"),"status_emoji": ("🟢" if evidence else "🔴")}
        rows = []
        for _, r in base_charts.sort_values(["member_id","chart_id"]).iterrows():
            mid, cid = r["member_id"], r["chart_id"]
            g = df[(df["member_id"] == mid) & (df["chart_id"] == cid)]
            p_init = first_nonempty(g.get("primary_coder_initials", pd.Series(dtype=str)))
            s_init = first_nonempty(g.get("secondary_coder_initials", pd.Series(dtype=str)))
            f_init = first_nonempty(g.get("final_coder_initials", pd.Series(dtype=str)))
            p = role_summary(g, "primary",   other_inits=[s_init, f_init])
            s = role_summary(g, "secondary", other_inits=[p_init, f_init])
            f = role_summary(g, "final",     other_inits=[p_init, s_init])
            reviewed_roles = [x for x in [p,s,f] if x["reviewed"]]
            if len(reviewed_roles)==0: consensus=("⏳","Pending")
            else:
                ev_vals = [x["evidence"] for x in reviewed_roles]
                consensus = ("🟢","Evidence") if all(ev_vals) else (("🔴","No evidence") if not any(ev_vals) else ("⚠️","Conflict"))
            rows.append({
                "member_id": mid, "chart_id": cid,
                "Primary": p["initials"], "Primary status": f"{p['status_emoji']} {p['status_text']}",
                "Secondary": s["initials"], "Secondary status": f"{s['status_emoji']} {s['status_text']}",
                "Final": f["initials"], "Final status": f"{f['status_emoji']} {f['status_text']}",
                "Consensus": f"{consensus[0]} {consensus[1]}",
            })
        progress_view = pd.DataFrame(rows)
        col_defs = {
            "member_id":{"width":130},"chart_id":{"width":120},"Primary":{"width":120},"Primary status":{"width":170},
            "Secondary":{"width":120},"Secondary status":{"width":170},"Final":{"width":120},"Final status":{"width":170},"Consensus":{"width":140},
        }
        U.render_grid(progress_view, col_defs=col_defs, height=340, row_height=34, grid_key="grid_progress")

# ==================== RADV OUTPUT ====================
with tab_reference:
    st.subheader("ICD-10 → HCC crosswalk (v24/v28)")
    df_ref = icd_to_hcc if not icd_to_hcc.empty else D.load_csv(D.DATA_DIR / "icd_to_hcc_fallback.csv")
    U.render_grid(U.sanitize_df(df_ref), height=360, download_name="icd10_to_hcc", grid_key="grid_reference")

st.caption("Demo only – do not use with real PHI.")