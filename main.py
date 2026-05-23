"""
AltheaRare — FastAPI Backend
=============================
Runs on Railway. Phone app sends symptoms + genes here.
This server searches SQLite, builds Gemini prompt, returns diagnosis.

Start locally:  uvicorn main:app --reload --port 8000
"""

import os, re, json, sqlite3, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AltheaRare API", version="1.0.0")

# Allow requests from anywhere (your phone app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
)
DB_PATH = os.environ.get("DB_PATH", "rare_diseases.db")


# ── Request/Response models ──────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symptoms:      str
    gene_variants: str

class AnalyzeResponse(BaseModel):
    success:    bool
    diagnosis:  str
    candidates: list  # top DB matches returned for transparency
    error:      str = ""


# ── DB search ────────────────────────────────────────────────────

def search_database(symptoms: str, genes: str, top_n: int = 10) -> list:
    """
    Search SQLite for diseases matching symptoms or genes.
    Gene match = 3x weight, symptom match = 1x weight.
    """
    if not os.path.exists(DB_PATH):
        return []

    sym_tokens = [t for t in symptoms.lower().split() if len(t) > 3]
    gen_tokens = [t for t in genes.upper().split()    if len(t) > 1]
    # also split on comma/semicolon
    for sep in [",", ";"]:
        sym_tokens += [t.strip() for t in symptoms.lower().split(sep) if len(t.strip()) > 3]
        gen_tokens += [t.strip() for t in genes.upper().split(sep)    if len(t.strip()) > 1]
    sym_tokens = list(set(sym_tokens))
    gen_tokens = list(set(gen_tokens))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    cur.execute("SELECT * FROM diseases")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for row in rows:
        d   = dict(row)
        ds  = (d.get("hpo_symptoms", "") or "").lower()
        dg  = (d.get("all_genes",    "") or "").upper()
        score = 0
        for g in gen_tokens:
            if g and g in dg:
                score += 3
        for s in sym_tokens:
            if s and s in ds:
                score += 1
        if score > 0:
            d["_score"] = score
            scored.append(d)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_n]


def format_candidates(candidates: list) -> str:
    if not candidates:
        return "No direct matches found in local database."
    lines = []
    for i, d in enumerate(candidates, 1):
        syms  = "; ".join((d.get("hpo_symptoms") or "").split(";")[:8])
        vars_ = " | ".join((d.get("variants_hgvs") or "").split("|")[:3])
        lines.append(f"""
CANDIDATE {i}: {d.get('disease_name','?')} ({d.get('orpha_code','?')})
  ICD-10      : {d.get('icd10_code','N/A')   or 'N/A'}
  Prevalence  : {d.get('prevalence','N/A')   or 'N/A'}
  Age of onset: {d.get('age_of_onset','N/A') or 'N/A'}
  Genes       : {d.get('all_genes','N/A')    or 'N/A'}
  Variants    : {vars_ or 'N/A'}
  Symptoms    : {syms  or 'N/A'}""")
    return "\n---".join(lines)


# ── Gemini call ──────────────────────────────────────────────────

async def call_gemini(symptoms: str, genes: str, db_context: str) -> str:
    prompt = f"""You are AltheaRare, a clinical AI for rare disease diagnosis.
You have access to a curated database of 11,456 rare diseases (Orphanet, HPO, ClinVar).

Use the DATABASE MATCHES below as PRIMARY evidence, then apply medical knowledge to refine.

DATABASE MATCHES:
{db_context}

PATIENT DATA:
Symptoms / Phenotypes: {symptoms}
Gene Variants: {genes}

Respond ONLY in this format:

━━━━━━━━━━━━━━━━━━━━━━━
ALTHEARARE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━

HPO MAPPING:
[each symptom → HP:XXXXXXX — Label]

TOP DIFFERENTIAL DIAGNOSES:

1. [Disease Name] — Confidence: X%
   Orphanet: ORPHA:XXXXX | ICD-10: XXX
   Symptom match: [which symptoms matched]
   Gene evidence: [supporting genes/variants]
   Prevalence: [how rare]
   Treatment: [standard of care]
   Next steps: [confirmatory tests]

[Up to 5 diseases, ranked by confidence]

⚠️ Clinical decision support only. Physician confirmation required."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GEMINI_URL, json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"Gemini error: {r.text[:300]}")

    data = r.json()
    text = data.get("candidates", [{}])[0] \
               .get("content", {}) \
               .get("parts", [{}])[0] \
               .get("text", "")
    if not text:
        raise HTTPException(status_code=502, detail="Empty Gemini response")
    return text


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "AltheaRare API running",
            "db_exists": os.path.exists(DB_PATH)}

@app.get("/health")
def health():
    db_ok = os.path.exists(DB_PATH)
    count = 0
    if db_ok:
        try:
            conn  = sqlite3.connect(DB_PATH)
            count = conn.execute("SELECT COUNT(*) FROM diseases").fetchone()[0]
            conn.close()
        except Exception:
            db_ok = False
    return {"status": "ok", "db_ok": db_ok, "disease_count": count}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not req.symptoms.strip() and not req.gene_variants.strip():
        raise HTTPException(status_code=400, detail="Symptoms or genes required")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server")

    # 1. Search local DB
    candidates   = search_database(req.symptoms, req.gene_variants)
    db_context   = format_candidates(candidates)

    # 2. Call Gemini with DB context
    diagnosis = await call_gemini(req.symptoms, req.gene_variants, db_context)

    # 3. Return result + candidates for transparency
    safe_candidates = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in candidates
    ]
    return AnalyzeResponse(
        success=True,
        diagnosis=diagnosis,
        candidates=safe_candidates,
    )