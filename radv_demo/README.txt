
# RADV Chart Validation Workbench (Demo)

**What you have**

- RADV sample list: `/mnt/data/radv_demo/data/radv_sample.csv`
- Chart inventory (12 PDFs): `/mnt/data/radv_demo/data/chart_inventory.csv` and folder `/mnt/data/radv_demo/pdfs`
- Fallback ICD→HCC mapping for 3 demo ICDs: `/mnt/data/radv_demo/data/icd_to_hcc_fallback.csv`
- Coder status log (initially empty): `/mnt/data/radv_demo/data/radv_coding_status.csv`
- Streamlit app: `/mnt/data/radv_demo/app/streamlit_app.py`

**Quickstart**

```bash
pip install streamlit pandas matplotlib PyPDF2 reportlab google-cloud-bigquery
streamlit run /mnt/data/radv_demo/app/streamlit_app.py
```

**Features**
- Tab 1: RADV Sample — shows dummy members and crosswalk (v24/v28).
- Tab 2: Chart Inventory — lists charts, runs a lightweight NLP ("LLM") scan to pre-highlight likely evidence and offsets.
- Tab 3: RADV Validation — select member & chart, view PDF (link) and extracted text with color-coded highlights. Record decisions at primary/secondary/final stages, with notes, DOS, NPI, and suspected offsets.
- Tab 4: Progress — live counters and a chart for progress by stage.

**BigQuery (optional)**
- If you set `GOOGLE_APPLICATION_CREDENTIALS` and modify the app to query your BigQuery tables, you can replace CSVs with live data. The current app includes a `USE_BIGQUERY` flag and imports; add your dataset/table IDs as needed.

**Custom ICD→HCC mapping**
- If a file exists at `/mnt/data/icd10_to_hcc_v24_v28.csv` with columns `icd10,model,hcc,hcc_label`, the app will use it automatically. Otherwise, it falls back to the bundled demo mapping for J44.1, E11.9, and B20.

**Important**
- This is a demo. Replace the PDFs and CSVs with your actual data pipeline, integrate your LLM service for smarter highlighting, and enforce authentication, audit trails, and PHI protections before any real-world use.
