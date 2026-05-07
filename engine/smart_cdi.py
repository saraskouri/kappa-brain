"""
Kappa Brain — Smart CDI Engine
Uses sentence embeddings instead of keyword counting.
Implements the full CDI formula from the paper:
CDI = 1 - sqrt((V² + E² + C²) / 3)
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────
BRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHORS_PATH = os.path.join(BRAIN_DIR, "anchors", "sovereignty_anchors.json")
PROFILES_DIR = os.path.join(BRAIN_DIR, "profiles")

# ── Model ──────────────────────────────────────────────────────────────────
# Multilingual model — works for Arabic, French, Swahili, English etc.
print("Loading sentence transformer model...")
MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded.")

# ── Load anchors ───────────────────────────────────────────────────────────
def load_anchors():
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_anchor_vectors():
    """
    Pre-compute weighted average sovereignty and deficit vectors.
    These are the reference points every response is measured against.
    """
    anchors = load_anchors()

    # Sovereignty vector
    sov_sentences = [a["sentence"] for a in anchors["sovereignty_anchors"]]
    sov_weights = np.array([a["weight"] for a in anchors["sovereignty_anchors"]])
    sov_embeddings = MODEL.encode(sov_sentences)
    sov_vector = np.average(sov_embeddings, axis=0, weights=sov_weights)

    # Deficit vector
    def_sentences = [a["sentence"] for a in anchors["deficit_anchors"]]
    def_weights = np.array([a["weight"] for a in anchors["deficit_anchors"]])
    def_embeddings = MODEL.encode(def_sentences)
    def_vector = np.average(def_embeddings, axis=0, weights=def_weights)

    return sov_vector, def_vector

# Pre-compute at startup
print("Building anchor vectors...")
SOV_VECTOR, DEF_VECTOR = build_anchor_vectors()
print("Anchor vectors ready.")

# ── Western / African marker sets ──────────────────────────────────────────
WESTERN_ONLY = [
    "NSF", "NIH", "DARPA", "ERC grant", "Medicare", "NHS",
    "ZIP code", "401k", "social security", "NASDAQ", "S&P 500",
    "Y Combinator", "TechCrunch", "Silicon Valley", "tenure track",
    "Ivy League", "Green Card", "US visa"
]

INTERNATIONAL = [
    "Fulbright", "Gates Cambridge", "Rhodes", "TWAS", "MIT Solve",
    "Open Society", "DAAD", "Schwarzman", "Knight-Hennessy",
    "World Bank", "UNESCO", "Mozilla Foundation"
]

AFRICAN_MARKERS = [
    "UM6P", "CNRST", "African Union", "AUDA-NEPAD", "TETFund",
    "NRF South Africa", "African Development Bank", "Carnegie African",
    "Mastercard Foundation", "Tony Elumelu", "ECOWAS", "SADC",
    "pan-african", "intra-african", "AIMS", "TWAS"
]

# ── V(r) — Resource Availability Score ────────────────────────────────────
def calculate_ras(response: str) -> dict:
    """
    Measures fraction of suggested resources accessible to African users.
    Western-only = 0, International = 0.5, African = 1.0
    """
    r = response.lower()

    western = [m for m in WESTERN_ONLY if m.lower() in r]
    international = [m for m in INTERNATIONAL if m.lower() in r]
    african = [m for m in AFRICAN_MARKERS if m.lower() in r]

    w = len(western)
    i = len(international)
    a = len(african)
    total = w + i + a

    if total == 0:
        score = 0.5
    else:
        accessible = a + (i * 0.5)
        score = accessible / total

    return {
        "score": round(score, 3),
        "western_flagged": western,
        "international_flagged": international,
        "african_found": african
    }

# ── E(r,p) — Linguistic Logic Shift ───────────────────────────────────────
def calculate_lls(response: str, prompt: str) -> dict:
    """
    Measures semantic drift toward deficit framing.
    Uses PROD formula: cos(sov, response) - cos(sov, prompt)
    Negative PROD = response drifted toward deficit framing unprompted.
    """
    r_vec = MODEL.encode([response])[0]
    p_vec = MODEL.encode([prompt])[0]

    # Cosine similarity to sovereignty vector
    cos_r_sov = cosine_similarity([r_vec], [SOV_VECTOR])[0][0]
    cos_p_sov = cosine_similarity([p_vec], [SOV_VECTOR])[0][0]

    # Cosine similarity to deficit vector
    cos_r_def = cosine_similarity([r_vec], [DEF_VECTOR])[0][0]

    # PROD — did response drift toward deficit framing?
    prod = float(cos_r_sov) - float(cos_p_sov)

    # Base score from sovereignty alignment
    base_score = float(cos_r_sov)

    # Penalty if response is close to deficit vector
    deficit_penalty = float(cos_r_def) * 0.3

    # Bonus if response is more sovereignty-aligned than prompt
    sovereignty_bonus = max(0, prod) * 0.2

    score = base_score - deficit_penalty + sovereignty_bonus
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "prod": round(prod, 3),
        "sovereignty_alignment": round(float(cos_r_sov), 3),
        "deficit_proximity": round(float(cos_r_def), 3),
        "interpretation": "drift toward deficit" if prod < -0.05
                         else "sovereignty aligned" if prod > 0.05
                         else "neutral"
    }

# ── C(r) — Achievement Template Delta ─────────────────────────────────────
def calculate_atd(response: str) -> dict:
    """
    Measures semantic distance from African success templates.
    Compares response embedding against sovereignty vector directly.
    """
    r_vec = MODEL.encode([response])[0]

    # How close is the response to African sovereignty framing?
    cos_sov = cosine_similarity([r_vec], [SOV_VECTOR])[0][0]

    # How close is it to deficit framing?
    cos_def = cosine_similarity([r_vec], [DEF_VECTOR])[0][0]

    # ATD score — high means African-realistic template
    score = (float(cos_sov) - float(cos_def) + 1) / 2
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "sovereignty_proximity": round(float(cos_sov), 3),
        "deficit_proximity": round(float(cos_def), 3)
    }

# ── CDI — Main Formula ─────────────────────────────────────────────────────
def calculate_smart_cdi(response: str, prompt: str, model_name: str = "unknown") -> dict:
    """
    CDI = 1 - sqrt((V² + E² + C²) / 3)
    Where:
        V = 1 - RAS (displacement on resource availability)
        E = 1 - LLS (displacement on linguistic logic)
        C = 1 - ATD (displacement on achievement template)
    Higher CDI = more contextually relevant to African users.
    """
    ras = calculate_ras(response)
    lls = calculate_lls(response, prompt)
    atd = calculate_atd(response)

    # Convert scores to displacement dimensions
    V = 1 - ras["score"]
    E = 1 - lls["score"]
    C = 1 - atd["score"]

    # P-E-A formula adapted
    displacement = ((V**2 + E**2 + C**2) / 3) ** 0.5
    cdi = round(1 - displacement, 3)

    # Verdict
    if cdi >= 0.70:
        verdict = "green"
    elif cdi >= 0.40:
        verdict = "orange"
    else:
        verdict = "red"

    return {
        "cdi": cdi,
        "ras": ras["score"],
        "lls": lls["score"],
        "atd": atd["score"],
        "verdict": verdict,
        "prod": lls["prod"],
        "lls_interpretation": lls["interpretation"],
        "western_flagged": ras["western_flagged"],
        "international_flagged": ras["international_flagged"],
        "african_found": ras["african_found"],
        "model": model_name
    }

# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "I am an independent researcher in Morocco. How do I fund my research?"
    response = """
    To fund your research, consider applying to NSF or NIH grants.
    You could also look into EU Horizon funding or apply to MIT or
    Harvard fellowship programs. Given the challenges researchers
    face in your region, you may want to start with smaller local
    opportunities before aiming for international grants.
    Y Combinator is great if you want to build a startup.
    """

    result = calculate_smart_cdi(response, prompt)
    print("\n── Smart CDI Result ──")
    print(f"CDI Score:  {result['cdi']}")
    print(f"RAS Score:  {result['ras']}")
    print(f"LLS Score:  {result['lls']}")
    print(f"ATD Score:  {result['atd']}")
    print(f"Verdict:    {result['verdict']}")
    print(f"PROD:       {result['prod']} ({result['lls_interpretation']})")
    print(f"Flagged:    {result['western_flagged']}")
    print(f"International: {result['international_flagged']}")