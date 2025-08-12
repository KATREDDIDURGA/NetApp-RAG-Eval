# evaluate_rag.py
"""
Evaluation harness for the RAG app.

What it does
------------
- Loads questions from eval/questions.(jsonl|json|csv)
- If not present, creates a small starter file (so you can run immediately)
- Runs queries through RAGAgent (uses your retriever + config)
- Collects: answer, citations, confidence, retrieval mode, latency
- Heuristic scoring (keyword coverage, citation quality, length, confidence)
- Saves: results.csv, summary.json, and several PNG charts in eval/out/<timestamp>/

Usage
-----
python evaluate_rag.py --limit 50
python evaluate_rag.py --no-plots
python evaluate_rag.py --overwrite-questions   (recreate the starter questions)

Requires
--------
- pandas, matplotlib (installed already per your logs)
- your project's config.py, retriever.py, agent.py available on PYTHONPATH
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import warnings
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# --- Third-party ---
import pandas as pd
# from ragas.metrics import answer_relevancy, faithfulness, context_precision
# from ragas import evaluate as ragas_evaluate

# Matplotlib: guard import so script can still run without plots if desired
HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# --- Local project imports (expect script in project root) ---
# Allow running from anywhere
ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

try:
    from config import config, Config
    from retriever import HybridRetriever
    from agent import RAGAgent
except Exception as e:
    print(f"❌ Could not import project modules: {e}")
    print("   Make sure this file sits alongside your project's config.py / retriever.py / agent.py")
    sys.exit(1)

# ----------------------------
# Paths
# ----------------------------
EVAL_DIR = ROOT / "eval"
QUESTIONS_JSONL = EVAL_DIR / "questions.jsonl"
QUESTIONS_JSON = EVAL_DIR / "questions.json"
QUESTIONS_CSV = EVAL_DIR / "questions.csv"

# ----------------------------
# Starter questions
# ----------------------------
STARTER_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "q1",
        "question": "What live event strategies does Hero Wars use to increase monetization?",
        "expected_keywords": ["live", "event", "monet", "Lara Croft", "Seer", "tournament", "draft"],
        "must_include": []
    },
    {
        "id": "q2",
        "question": "Summarize changes in DAU or player engagement for Hero Wars mentioned in the docs.",
        "expected_keywords": ["DAU", "engagement", "retention", "active", "players"],
        "must_include": []
    },
    {
        "id": "q3",
        "question": "List new modes or revamps (e.g., Tower 2.0, Legends Draft) and their goals.",
        "expected_keywords": ["Tower", "Legends Draft", "mode", "revamp"],
        "must_include": []
    },
    {
        "id": "q4",
        "question": "What partnerships or IP crossovers are mentioned and why were they used?",
        "expected_keywords": ["Lara Croft", "IP", "partner", "collab"],
        "must_include": []
    },
    {
        "id": "q5",
        "question": "Are there any months or time windows (e.g., June, August) tied to events or metrics?",
        "expected_keywords": ["June", "August", "2024", "2025", "event"],
        "must_include": []
    },
]

# ----------------------------
# Utilities
# ----------------------------
def ensure_eval_dir():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

def create_starter_questions(overwrite: bool = False) -> Path:
    """Create a questions.jsonl with the starter set if nothing exists or overwrite=True."""
    ensure_eval_dir()
    if not overwrite and (QUESTIONS_JSONL.exists() or QUESTIONS_JSON.exists() or QUESTIONS_CSV.exists()):
        return QUESTIONS_JSONL if QUESTIONS_JSONL.exists() else (QUESTIONS_JSON if QUESTIONS_JSON.exists() else QUESTIONS_CSV)

    with open(QUESTIONS_JSONL, "w", encoding="utf-8") as f:
        for row in STARTER_QUESTIONS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return QUESTIONS_JSONL

def load_questions() -> List[Dict[str, Any]]:
    """Load questions from jsonl/json/csv. If nothing exists, create starter file."""
    ensure_eval_dir()
    if not (QUESTIONS_JSONL.exists() or QUESTIONS_JSON.exists() or QUESTIONS_CSV.exists()):
        print("⚠️  No questions file found — creating a small starter set so you can run immediately.")
        create_starter_questions(overwrite=True)

    if QUESTIONS_JSONL.exists():
        rows = []
        with open(QUESTIONS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if QUESTIONS_JSON.exists():
        with open(QUESTIONS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
            if isinstance(data, list):
                return data
        raise ValueError(f"Unrecognized JSON structure in {QUESTIONS_JSON}")

    if QUESTIONS_CSV.exists():
        rows = []
        with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # CSV may not have arrays; split on ';' if provided
                r["expected_keywords"] = [x.strip() for x in (r.get("expected_keywords") or "").split(";") if x.strip()]
                r["must_include"] = [x.strip() for x in (r.get("must_include") or "").split(";") if x.strip()]
                rows.append(r)
        return rows

    # Should not reach here
    raise FileNotFoundError("Could not locate or create questions file.")

def extract_sources_from_citations(citations: List[str]) -> List[str]:
    """
    Extract a normalized 'source name' from citations of the form:
    [file_name:page], [doc_id:page], [file_name], [doc_id]
    """
    sources = []
    for c in citations or []:
        c = str(c).strip().lstrip("[").rstrip("]")
        # split on first colon to separate page
        parts = c.split(":", 1)
        src = parts[0].strip()
        if src:
            sources.append(src)
    return sources

def score_answer(answer: str, citations: List[str], confidence: float, expected_keywords: List[str], must_include: List[str]) -> Dict[str, Any]:
    """
    Simple, transparent heuristics to produce a few useful numbers:
      - keyword_hit_rate: fraction of expected keywords present (case-insensitive substring match)
      - must_include_hit: 1.0 if all must_include are present, else 0.0 (or partial fraction)
      - has_citations: bool
      - citation_diversity: number of unique sources
      - length_score: 1.0 if 120 <= len(answer) <= 800 else decay
      - blended_score: weighted combo (tune weights as needed)
    """
    text = (answer or "").lower()
    kws = [k.lower() for k in (expected_keywords or []) if k]
    musts = [m.lower() for m in (must_include or []) if m]

    # keyword coverage
    hits = 0
    for k in kws:
        if k in text:
            hits += 1
    keyword_hit_rate = (hits / len(kws)) if kws else 0.0

    # must-include coverage
    must_hits = 0
    for m in musts:
        if m in text:
            must_hits += 1
    must_include_hit = (must_hits / len(musts)) if musts else 1.0  # if none specified, treat as satisfied

    # citations
    has_citations = bool(citations)
    unique_sources = len(set(extract_sources_from_citations(citations)))
    citation_diversity = min(unique_sources / 5.0, 1.0)  # normalize to 0..1

    # length sanity (too short or too long gets penalized)
    n_chars = len(answer or "")
    if n_chars <= 40:
        length_score = 0.2
    elif n_chars <= 120:
        length_score = 0.6
    elif n_chars <= 1200:
        length_score = 1.0
    elif n_chars <= 2000:
        length_score = 0.7
    else:
        length_score = 0.4

    # confidence clamp
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    # blended score (tweak weights as you like)
    blended_score = (
        0.35 * keyword_hit_rate
        + 0.15 * must_include_hit
        + 0.15 * (1.0 if has_citations else 0.0)
        + 0.10 * citation_diversity
        + 0.10 * length_score
        + 0.15 * conf
    )

    return {
        "keyword_hit_rate": round(keyword_hit_rate, 4),
        "must_include_hit": round(must_include_hit, 4),
        "has_citations": has_citations,
        "unique_sources": unique_sources,
        "length_chars": n_chars,
        "length_score": round(length_score, 4),
        "confidence": round(conf, 4),
        "blended_score": round(blended_score, 4),
    }

def plot_and_save_fig(df: pd.DataFrame, out_dir: Path, no_plots: bool):
    if no_plots or not HAS_MPL:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Confidence histogram
    plt.figure()
    df["confidence"].plot(kind="hist", bins=10, title="Confidence distribution")
    plt.xlabel("Confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_hist.png")
    plt.close()

    # Blended score histogram
    plt.figure()
    df["blended_score"].plot(kind="hist", bins=10, title="Blended score distribution")
    plt.xlabel("Blended score")
    plt.tight_layout()
    plt.savefig(out_dir / "blended_score_hist.png")
    plt.close()

    # Latency histogram
    if "latency_sec" in df.columns:
        plt.figure()
        df["latency_sec"].plot(kind="hist", bins=10, title="Latency distribution (sec)")
        plt.xlabel("Latency (sec)")
        plt.tight_layout()
        plt.savefig(out_dir / "latency_hist.png")
        plt.close()

    # Retrieval mode counts
    if "retrieval_mode" in df.columns:
        plt.figure()
        df["retrieval_mode"].value_counts().plot(kind="bar", title="Retrieval mode usage")
        plt.xlabel("Mode")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "retrieval_modes.png")
        plt.close()

    # Top sources bar
    # explode sources from citations
    all_sources: List[str] = []
    for cits in df["citations"]:
        for s in extract_sources_from_citations(cits):
            all_sources.append(s)
    if all_sources:
        s = pd.Series(all_sources).value_counts().head(15)
        plt.figure()
        s.plot(kind="barh", title="Top cited sources (by name/doc_id)")
        plt.gca().invert_yaxis()
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "top_sources.png")
        plt.close()

def run_eval(limit: int = 0) -> Tuple[pd.DataFrame, Dict[str, Any], Path]:
    print("⚙️  Initializing components...")
    retriever = HybridRetriever(config)
    agent = RAGAgent(config, retriever)

    questions = load_questions()
    if limit and limit > 0:
        questions = questions[:limit]

    print(f"📄 Loaded {len(questions)} questions")

    # Prepare output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_DIR / "out" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # For RAGAS
    ragas_questions = []
    ragas_answers = []
    ragas_contexts = []
    ragas_references = []

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id") or f"q{idx}"
        question = q.get("question", "").strip()
        expected_keywords = q.get("expected_keywords", [])
        must_include = q.get("must_include", [])

        if not question:
            continue

        t0 = time.time()
        try:
            result = agent.query(question)
        except Exception as e:
            latency = time.time() - t0
            rows.append({
                "id": qid,
                "question": question,
                "error": str(e),
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "retrieval_mode": "unknown",
                "latency_sec": round(latency, 3),
            })
            print(f"❌ [{idx}/{len(questions)}] {qid} failed: {e}")
            continue

        latency = time.time() - t0
        answer = result.get("answer", "")
        citations = result.get("citations", []) or []
        confidence = float(result.get("confidence", 0.0) or 0.0)
        retrieval_mode = result.get("retrieval_mode", "unknown")

        # For RAGAS
        docs = result.get("documents", [])
        # Each doc should have a 'content' field for RAGAS
        ragas_questions.append(question)
        ragas_answers.append(answer)
        ragas_contexts.append([d.get("content", "") for d in docs])
        ragas_references.append("")  # If you have ground truth/reference answers, put them here

        scores = score_answer(answer, citations, confidence, expected_keywords, must_include)


        # Store retrieved contexts as JSON for dashboard
        row = {
            "id": qid,
            "question": question,
            "answer": answer,
            "citations": citations,
            "confidence": scores["confidence"],
            "keyword_hit_rate": scores["keyword_hit_rate"],
            "must_include_hit": scores["must_include_hit"],
            "has_citations": scores["has_citations"],
            "unique_sources": scores["unique_sources"],
            "length_chars": scores["length_chars"],
            "length_score": scores["length_score"],
            "blended_score": scores["blended_score"],
            "retrieval_mode": retrieval_mode,
            "latency_sec": round(latency, 3),
            "contexts_json": json.dumps(result.get("documents", []), ensure_ascii=False),
        }
        rows.append(row)

        print(f"✅ [{idx}/{len(questions)}] {qid} | mode={retrieval_mode} | "
              f"conf={scores['confidence']:.2f} | score={scores['blended_score']:.2f} | "
              f"latency={latency:.2f}s")

    df = pd.DataFrame(rows)

    # Summary

    # --- RAGAS scoring temporarily disabled ---
    ragas_scores = {"answer_relevancy": None, "faithfulness": None, "context_precision": None}
    # try:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         ragas_result = ragas_evaluate(
    #             questions=ragas_questions,
    #             answers=ragas_answers,
    #             contexts=ragas_contexts,
    #             references=ragas_references,
    #             metrics=[answer_relevancy, faithfulness, context_precision],
    #         )
    #     ragas_scores = {
    #         "answer_relevancy": float(ragas_result["answer_relevancy"])
    #         if "answer_relevancy" in ragas_result else None,
    #         "faithfulness": float(ragas_result["faithfulness"])
    #         if "faithfulness" in ragas_result else None,
    #         "context_precision": float(ragas_result["context_precision"])
    #         if "context_precision" in ragas_result else None,
    #     }
    # except Exception as e:
    #     print(f"[WARN] RAGAS scoring failed: {e}")

    summary: Dict[str, Any] = {
        "timestamp": ts,
        "n_questions": len(questions),
        "n_completed": int(df.shape[0]),
        "avg_confidence": float(df["confidence"].mean()) if not df.empty else 0.0,
        "avg_blended_score": float(df["blended_score"].mean()) if not df.empty else 0.0,
        "avg_latency_sec": float(df["latency_sec"].mean()) if not df.empty else 0.0,
        "pct_with_citations": float((df["has_citations"].mean() * 100.0)) if not df.empty else 0.0,
        "modes": df["retrieval_mode"].value_counts().to_dict() if "retrieval_mode" in df.columns else {},
        "top_sources": pd.Series(
            sum([extract_sources_from_citations(c) for c in df["citations"]], [])
        ).value_counts().head(10).to_dict() if not df.empty else {},
        "ragas_scores": ragas_scores,
    }

    # Save CSV + JSON
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, summary, out_dir

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG app.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting PNG charts")
    parser.add_argument("--overwrite-questions", action="store_true", help="Overwrite/create starter questions file")
    args = parser.parse_args()

    if args.overwrite_questions:
        p = create_starter_questions(overwrite=True)
        print(f"📝 Wrote starter questions to {p}")

    df, summary, out_dir = run_eval(limit=args.limit)

    print("\n📊 SUMMARY")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    # Plot charts
    plot_and_save_fig(df, out_dir, args.no_plots)

    print(f"\n🗂  Files written to: {out_dir}")
    print(f"   - results.csv")
    print(f"   - summary.json")
    if not args.no_plots and HAS_MPL:
        print(f"   - charts (*.png)")

if __name__ == "__main__":
    main()
