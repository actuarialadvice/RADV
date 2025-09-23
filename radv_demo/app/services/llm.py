# services/llm.py
from __future__ import annotations
import os, json, re
from typing import List, Dict, Any, Tuple

# --- Provider SDKs (optional) ---
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    _HAS_VERTEX = True
except Exception:
    vertexai = None
    GenerativeModel = None
    _HAS_VERTEX = False

try:
    import google.generativeai as genai
    _HAS_GEMINI_API = True
except Exception:
    genai = None
    _HAS_GEMINI_API = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    openai = None
    _HAS_OPENAI = False


# ----------------- Normalizers -----------------
_ICD_NONALNUM = re.compile(r"[^A-Za-z0-9]+")

def canon_icd(code: str) -> str:
    """Uppercase ICD-10 and strip dots and other non-alnum: 'J44.1' → 'J441'."""
    if not code:
        return ""
    return _ICD_NONALNUM.sub("", str(code).upper())

def pretty_icd(code: str) -> str:
    """Add a dot after the 3rd char if none present (best-effort pretty form)."""
    if not code:
        return ""
    c = code.upper()
    if "." in c or len(c) < 4:
        return c
    return c[:3] + "." + c[3:]


# ----------------- JSON parsing -----------------
def _parse_json_forgiving(raw: str) -> Any:
    if not raw:
        return None
    # direct
    try:
        return json.loads(raw)
    except Exception:
        pass
    # first json-looking block
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


# ----------------- Provider picking -----------------
def _vertex_available() -> bool:
    if not _HAS_VERTEX:
        return False
    # Only check that project/location can be inferred; actual init may happen lazily
    project = os.getenv("VERTEXAI_PROJECT") or os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEXAI_LOCATION") or "us-central1"
    return bool(project and location)

def pick_llm_provider() -> str:
    """Prefer Vertex (ADC/SA), then Gemini API, then OpenAI, else local."""
    if _vertex_available():
        return "vertex"
    if _HAS_GEMINI_API and os.getenv("GOOGLE_API_KEY"):
        return "gemini_api"
    if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "local"


# ----------------- Core prompts -----------------
CANDIDATES_SYSTEM = (
    "You are a RADV coding assistant. "
    "From the chart text, suggest up to 6 plausible ICD-10 codes that might be clinically supported. "
    "Return ONLY JSON array, each item: {\"term\": <short trigger phrase>, \"icd10\": <ICD without dot>}.\n"
    "- IMPORTANT: The 'icd10' MUST be uppercase and have NO dot (e.g., 'J441', not 'J44.1').\n"
    "- The 'term' should be a short trigger phrase (diagnosis wording, key medication, lab threshold, imaging finding).\n"
    "- If nothing is supportable, return []."
)

TASK_TERMS_SYSTEM = (
    "You are a RADV coding specialist. You will be given chart text and a list of target ICD-10 codes. "
    "For EACH target code, produce two lists of short phrases likely to appear in notes as evidence OR offset:\n"
    "  - primary_terms: direct supporting phrases (diagnosis wording, A/P lines, problem list wording, hallmark meds, labs with thresholds, imaging findings)\n"
    "  - offset_terms: competing/offset diagnoses, mimics, or exclusions.\n"
    "Return ONLY a JSON object keyed by the ICD code **WITHOUT dot** (uppercase), e.g., 'J441'.\n"
    "Each value must be: {\"primary_terms\": [.. up to 12 ..], \"offset_terms\": [.. up to 8 ..]}.\n"
    "If chart text is sparse, still return typical phrases a coder would look for.\n"
    "Do not include explanations; just JSON."
)

def _make_task_terms_user(text: str, icd_list: List[str]) -> str:
    # Normalize targets: remove dots for keys we expect back
    targets = [canon_icd(x) for x in (icd_list or []) if x]
    # Provide the dotted and undotted forms for grounding within the content (as comments in prompt body).
    dotted = [pretty_icd(x) for x in targets]
    body = (
        "CHART_TEXT (truncated as needed):\n"
        f"{text[:12000]}\n\n"
        f"TARGETS (NO-DOT): {targets}\n"
        f"TARGETS (pretty): {dotted}\n"
    )
    return body

def _make_candidates_user(text: str) -> str:
    return f"CHART_TEXT (truncated as needed):\n{text[:12000]}"


# ----------------- Vertex calls -----------------
def _vertex_generate(prompt: str, model_name: str = None) -> Any:
    if not _vertex_available():
        return None
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        # Lazy init
        project = os.getenv("VERTEXAI_PROJECT") or os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEXAI_LOCATION", "us-central1")
        vertexai.init(project=project, location=location)
        model = GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 768},
        )
        return getattr(resp, "text", "") or ""
    except Exception:
        return None


# ----------------- Public API -----------------
def parse_llm_suggestions(v: Any) -> List[Dict[str, str]]:
    """Accept JSON/list/string → normalize to [{'term','icd10'}] with icd10 NO DOT."""
    if v is None:
        return []
    try:
        if isinstance(v, list):
            arr = v
        elif isinstance(v, str):
            obj = _parse_json_forgiving(v) if v.strip() else None
            if isinstance(obj, list):
                arr = obj
            else:
                return []
        else:
            return []
        out, seen = [], set()
        for it in arr:
            if not isinstance(it, dict):
                continue
            term = str(it.get("term", "")).strip()
            icd  = canon_icd(str(it.get("icd10", "")).strip())
            if term and icd and (term, icd) not in seen:
                out.append({"term": term, "icd10": icd})
                seen.add((term, icd))
        return out
    except Exception:
        return []


def get_candidates(text: str, provider: str) -> List[Dict[str, str]]:
    """Return [{'term','icd10'}] with icd10 normalized (no dot)."""
    provider = provider or pick_llm_provider()
    if provider == "vertex":
        raw = _vertex_generate(f"{CANDIDATES_SYSTEM}\n\n{_make_candidates_user(text)}")
        parsed = parse_llm_suggestions(raw)
        return parsed
    elif provider == "gemini_api" and _HAS_GEMINI_API and os.getenv("GOOGLE_API_KEY"):
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            resp = genai.GenerativeModel(model_name).generate_content(f"{CANDIDATES_SYSTEM}\n\n{_make_candidates_user(text)}")
            return parse_llm_suggestions(getattr(resp, "text", "") or "")
        except Exception:
            return []
    elif provider == "openai" and _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            msg = [
                {"role": "system", "content": CANDIDATES_SYSTEM},
                {"role": "user", "content": _make_candidates_user(text)},
            ]
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=msg, temperature=0.2, max_tokens=800)
            return parse_llm_suggestions(r["choices"][0]["message"]["content"])
        except Exception:
            return []
    # local fallback (only if all above fail)
    TERM_TO_ICD = {"bronchitis": "J40", "pneumonia": "J189", "emphysema": "J439", "copd": "J449", "diabetes": "E119", "hiv": "B20"}
    low = (text or "").lower()
    out, seen = [], set()
    for term, icd in TERM_TO_ICD.items():
        if term in low and (term, icd) not in seen:
            out.append({"term": term, "icd10": icd})
            seen.add((term, icd))
    return out


def get_task_terms(text: str, icd_list: List[str], provider: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns { 'J441': {'primary_terms': [...], 'offset_terms': [...]}, ... }
    Keys are normalized (no dot).
    """
    targets = [canon_icd(x) for x in (icd_list or []) if x]
    if not targets:
        return {}

    provider = provider or pick_llm_provider()
    if provider == "vertex":
        raw = _vertex_generate(f"{TASK_TERMS_SYSTEM}\n\n{_make_task_terms_user(text, icd_list)}")
    elif provider == "gemini_api" and _HAS_GEMINI_API and os.getenv("GOOGLE_API_KEY"):
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            resp = genai.GenerativeModel(model_name).generate_content(f"{TASK_TERMS_SYSTEM}\n\n{_make_task_terms_user(text, icd_list)}")
            raw = getattr(resp, "text", "") or ""
        except Exception:
            raw = ""
    elif provider == "openai" and _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            msg = [
                {"role": "system", "content": TASK_TERMS_SYSTEM},
                {"role": "user", "content": _make_task_terms_user(text, icd_list)},
            ]
            r = openai.ChatCompletion.create(model="gpt-4o-mini", messages=msg, temperature=0.2, max_tokens=1000)
            raw = r["choices"][0]["message"]["content"]
        except Exception:
            raw = ""
    else:
        raw = ""

    obj = _parse_json_forgiving(raw) if raw else None
    out: Dict[str, Dict[str, List[str]]] = {}

    # Accept either an object keyed by ICDs or a list of entries.
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        # list of {icd10_primary, primary_terms, offset_terms}
        items = []
        for entry in obj:
            if isinstance(entry, dict):
                k = entry.get("icd10") or entry.get("code") or entry.get("icd") or ""
                items.append((k, {"primary_terms": entry.get("primary_terms", []), "offset_terms": entry.get("offset_terms", [])}))
    else:
        items = []

    for k, v in items:
        nk = canon_icd(k)
        if not nk:
            continue
        prim = [str(x).strip() for x in (v.get("primary_terms") or []) if str(x).strip()]
        off  = [str(x).strip() for x in (v.get("offset_terms")  or []) if str(x).strip()]
        out[nk] = {"primary_terms": prim[:12], "offset_terms": off[:8]}

    # Ensure every requested target exists, fill missing with empty arrays
    for t in targets:
        out.setdefault(t, {"primary_terms": [], "offset_terms": []})

    return out