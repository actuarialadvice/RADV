import os, json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd

# Optional BigQuery
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    HAS_GCP = True
except Exception:
    bigquery = None
    service_account = None
    HAS_GCP = False

# Paths
APP_ROOT = Path(__file__).resolve().parents[2]  # .../radv_demo
DATA_DIR = APP_ROOT / "data"
PDF_DIR  = APP_ROOT / "pdfs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Schemas (copied from your app)
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
    "llm_scanned": False, "llm_suggestions": "", "coded_status": "", "dq_flag": ""
}

def ensure_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{k: v for k, v in schema.items()}]).iloc[0:0].copy()
    for col, default in schema.items():
        if col not in df.columns:
            df[col] = default
    df = df[[c for c in schema.keys() if c in df.columns] + [c for c in df.columns if c not in schema]]
    return df

def load_csv(path, **kwargs):
    p = Path(path)
    return pd.read_csv(p, **kwargs) if p.exists() else pd.DataFrame()

def load_icd_mapping_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["icd10", "model", "hcc", "hcc_label"])
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip()
        if not header: return pd.DataFrame(columns=["icd10","model","hcc","hcc_label"])
        h = [p.strip() for p in header.split(",", 3)]
        if len(h) < 4:
            cols = ["icd10", "model", "hcc", "hcc_label"]
        else:
            h[3] = "hcc_label" if h[3].lower().replace(" ","") in ("hcclable","hcc_label") else h[3]
            cols = [h[0] or "icd10", h[1] or "model", h[2] or "hcc", h[3] or "hcc_label"]
        for ln in f:
            ln = ln.rstrip("\r\n")
            if not ln.strip(): continue
            parts = [p.strip() for p in ln.split(",", 3)]
            if len(parts) < 4: continue
            icd10, model, hcc, hcc_label = parts[0], parts[1], parts[2], parts[3]
            hcc_label = hcc_label.replace(",", " - ")
            rows.append({"icd10": icd10, "model": model, "hcc": hcc, "hcc_label": hcc_label})
    return pd.DataFrame(rows, columns=["icd10","model","hcc","hcc_label"])

# storage paths
coded_status_path = DATA_DIR / "radv_coding_status.csv"
audit_log_path    = DATA_DIR / "audit_log.csv"

def save_chart_inventory(df: pd.DataFrame):
    ensure_columns(df, CHART_INVENTORY_SCHEMA).to_csv(DATA_DIR / "chart_inventory.csv", index=False)

def persist_coded_status(df: pd.DataFrame):
    ensure_columns(df, CODED_STATUS_SCHEMA).to_csv(coded_status_path, index=False)

def persist_audit(action: str, who: str, member_id: str, chart_id: str, n_rows: int):
    ts = datetime.now().isoformat(timespec="seconds")
    rec = pd.DataFrame([{
        "timestamp": ts, "actor": who, "action": action,
        "member_id": member_id, "chart_id": chart_id, "rows_affected": n_rows
    }])
    mode = "a" if audit_log_path.exists() else "w"
    header = not audit_log_path.exists()
    rec.to_csv(audit_log_path, mode=mode, header=header, index=False)

# GCP auth
def _read_sa_from_secrets(st):
    try:
        if "gcp_service_account" in st.secrets:
            sa = st.secrets["gcp_service_account"]
            if isinstance(sa, str):
                sa = json.loads(sa)
        elif "gcp_service_account_json" in st.secrets:
            sa = json.loads(st.secrets["gcp_service_account_json"])
        else:
            return None, None, None
        project = st.secrets.get("gcp_project") or sa.get("project_id")
        location = st.secrets.get("gcp_location", "us-central1")
        creds = service_account.Credentials.from_service_account_info(sa)
        return creds, project, location
    except Exception:
        return None, None, None

def _gcp_project_location(st) -> Tuple[Optional[str], str]:
    project = os.getenv("VERTEXAI_PROJECT") or os.getenv("BQ_PROJECT") or st.secrets.get("gcp_project", None)
    location = os.getenv("VERTEXAI_LOCATION") or st.secrets.get("gcp_location", "us-central1") or "us-central1"
    return project, location

def get_bq_client(st):
    if not HAS_GCP: return None
    sa_creds, sa_proj, _ = _read_sa_from_secrets(st)
    env_proj, _ = _gcp_project_location(st)
    try:
        if sa_creds:
            return bigquery.Client(project=sa_proj or env_proj, credentials=sa_creds)
        return bigquery.Client(project=env_proj)
    except Exception:
        return None

def load_from_bigquery(st):
    client = get_bq_client(st)
    if not client:
        return None, None, "BigQuery client unavailable. Check GCP credentials."
    project = os.environ.get("BQ_PROJECT") or _gcp_project_location(st)[0]
    dataset = os.environ.get("BQ_DATASET") or "radv_demo"
    if not project:
        return None, None, "Set BQ_PROJECT (or gcp_project in secrets)."
    def _read(table):
        return client.query(f"SELECT * FROM `{project}.{dataset}.{table}`").to_dataframe()
    try:
        return _read("radv_sample"), _read("chart_inventory"), None
    except Exception as e:
        return None, None, f"BigQuery load failed: {e}"

def save_to_bigquery(st, df: pd.DataFrame, table: str, schema=None, write_mode="WRITE_TRUNCATE"):
    client = get_bq_client(st)
    if not client: return "BigQuery client unavailable. Check GCP credentials."
    project = os.environ.get("BQ_PROJECT") or _gcp_project_location(st)[0]
    dataset = os.environ.get("BQ_DATASET") or "radv_demo"
    if not project: return "Set BQ_PROJECT (or gcp_project in secrets)."
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_mode,
        autodetect=(schema is None),
        schema=schema or []
    )
    table_id = f"{project}.{dataset}.{table}"
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    return None

# tiny CSV helpers
def sanitize_df(df: pd.DataFrame):
    return df.reset_index(drop=True) if (df is not None and not df.empty) else df

# seed demo data
def seed_demo(pdf_path: Path):
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
        providers = [("General Pulmonary Clinic","1223456789"),
                     ("Sunset Primary Care","1098765432"),
                     ("City Hospital Inpatient","1456789012"),
                     ("Northside Specialty Group","1789012345")]
        rows = []
        for m_idx, m in enumerate(["M001","M002","M003"], start=1):
            for i in range(1,5):
                prov, npi = providers[i-1]
                rows.append({
                    "member_id": m,
                    "chart_id": f"{m}-CH{i:02d}",
                    "provider": prov, "npi": npi,
                    "start_dos": f"2024-0{(i%9)+1}-0{(m_idx%7)+1}",
                    "end_dos":   f"2024-0{(i%9)+1}-1{(m_idx%7)+1}",
                    "pdf_path": str(pdf_path),
                    "llm_scanned": False,
                    "coded_status": "",
                    "radv_diags": ""
                })
        ci = pd.DataFrame(rows)
        sample = load_csv(DATA_DIR / "radv_sample.csv")
        icd_map = dict(sample[["member_id","icd10_submitted"]].values) if not sample.empty else {}
        ci["radv_diags"] = ci["member_id"].map(icd_map).fillna("")
        ensure_columns(ci, CHART_INVENTORY_SCHEMA).to_csv(ci_path, index=False)