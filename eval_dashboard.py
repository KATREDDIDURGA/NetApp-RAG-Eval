# eval_dashboard.py
# Streamlit dashboard for inspecting RAG evaluation results
# Run: streamlit run eval_dashboard.py

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# ------------- Page setup -------------
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

# ------------- Styles -------------
st.markdown(
    """
    <style>
      .metric-small { font-size: 0.9rem; color: #666; }
      .codebox {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.9rem;
        background: #0b1020;
        color: #e6edf3;
        border: 1px solid #1f2a44;
        border-radius: 8px;
        padding: 12px 14px;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .soft {
        color: #666; font-size: 0.9rem;
      }
      .info-card {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------- Paths -------------
ROOT = Path(__file__).parent
EVAL_OUT_DIR = ROOT / "eval" / "out"

def get_latest_results_csv() -> Path | None:
    if not EVAL_OUT_DIR.exists():
        return None
    subdirs = [d for d in EVAL_OUT_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda d: d.name)
    results_csv = latest / "results.csv"
    return results_csv if results_csv.exists() else None

REPORT_PATH = ROOT / "eval_report.csv"
SUMMARY_PATH = ROOT / "eval_summary.json"

st.title("📊 RAG Evaluation Dashboard")

# ------------- Load data -------------
def load_report(path: Path) -> pd.DataFrame:
    # Try eval_report.csv first
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read {path.name}: {e}")
            return pd.DataFrame()
        # Coerce expected columns
        for col in ["confidence", "latency_sec", "context_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # If not found, try latest results.csv in eval/out/
    latest_results = get_latest_results_csv()
    if latest_results is not None:
        st.info(f"Using latest results: {latest_results.relative_to(ROOT)}")
        try:
            df = pd.read_csv(latest_results)
        except Exception as e:
            st.error(f"Failed to read {latest_results}: {e}")
            return pd.DataFrame()
        for col in ["confidence", "latency_sec", "context_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    st.warning(f"`{path.name}` not found and no results found in eval/out/. Generate it by running `python evaluate_rag.py`.")
    return pd.DataFrame()

def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return {}

df = load_report(REPORT_PATH)
summary = load_summary(SUMMARY_PATH)

if df.empty:
    st.stop()

# ------------- Sidebar filters -------------
with st.sidebar:
    st.header("Filters")
    q_search = st.text_input("Search question text", value="")
    # Mode filter — now includes 'keyboard' as an option
    modes = sorted(set(df["retrieval_mode"].dropna().unique().tolist() + ["keyboard"])) if "retrieval_mode" in df.columns else ["keyboard"]
    selected_modes = st.multiselect("Retrieval mode", options=modes, default=modes)

    # Confidence filter
    if "confidence" in df.columns:
        min_conf, max_conf = float(np.nanmin(df["confidence"])), float(np.nanmax(df["confidence"]))
        conf_range = st.slider(
            "Confidence range",
            min_value=0.0,
            max_value=1.0,
            value=(round(min_conf, 2) if not np.isnan(min_conf) else 0.0,
                   round(max_conf, 2) if not np.isnan(max_conf) else 1.0),
            step=0.01
        )
    else:
        conf_range = (0.0, 1.0)

    # Latency filter
    if "latency_sec" in df.columns:
        lat_min, lat_max = float(np.nanmin(df["latency_sec"])), float(np.nanmax(df["latency_sec"]))
        lat_range = st.slider(
            "Latency (sec) range",
            min_value=0.0,
            max_value=max(1.0, round(lat_max, 1) if not np.isnan(lat_max) else 10.0),
            value=(0.0, max(1.0, round(lat_max, 1) if not np.isnan(lat_max) else 10.0)),
            step=0.1
        )
    else:
        lat_range = (0.0, 9999.0)

# Apply filters
fdf = df.copy()
if q_search.strip():
    fdf = fdf[fdf["question"].astype(str).str.contains(q_search, case=False, na=False)] if "question" in fdf.columns else fdf
if "retrieval_mode" in fdf.columns and selected_modes:
    fdf = fdf[fdf["retrieval_mode"].isin(selected_modes)]
if "confidence" in fdf.columns:
    fdf = fdf[(fdf["confidence"] >= conf_range[0]) & (fdf["confidence"] <= conf_range[1])]
if "latency_sec" in fdf.columns:
    fdf = fdf[(fdf["latency_sec"] >= lat_range[0]) & (fdf["latency_sec"] <= lat_range[1])]

# ------------- KPIs -------------
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Questions", len(fdf))
with c2:
    avg_conf = float(np.nanmean(fdf["confidence"])) if "confidence" in fdf.columns and len(fdf) else 0.0
    st.metric("Avg Confidence", f"{avg_conf:.2f}")
with c3:
    avg_lat = float(np.nanmean(fdf["latency_sec"])) if "latency_sec" in fdf.columns and len(fdf) else 0.0
    st.metric("Avg Latency (s)", f"{avg_lat:.2f}")
with c4:
    pass

if "retrieval_mode" in fdf.columns:
    st.caption("Mode Breakdown")
    st.bar_chart(fdf["retrieval_mode"].value_counts())

# RAGAS summary
if (
    "ragas_scores" in summary and isinstance(summary["ragas_scores"], dict)
    and any(summary["ragas_scores"].get(k) is not None for k in ("answer_relevancy", "faithfulness", "context_precision"))
):
    rs = summary["ragas_scores"]
    st.caption("RAGAS (from summary)")
    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("Answer Relevancy", "—" if rs.get("answer_relevancy") is None else f"{rs['answer_relevancy']:.3f}")
    with c6:
        st.metric("Faithfulness", "—" if rs.get("faithfulness") is None else f"{rs['faithfulness']:.3f}")
    with c7:
        st.metric("Context Precision", "—" if rs.get("context_precision") is None else f"{rs['context_precision']:.3f}")

st.markdown("---")

# ------------- Per-question table -------------
st.subheader("Per-question Results")
preferred_cols = [
    "idx", "question", "retrieval_mode", "context_count", "latency_sec", "confidence",
    "ragas_answer_relevancy", "ragas_faithfulness", "ragas_context_precision", "citations", "contexts_preview"
]
cols = [c for c in preferred_cols if c in fdf.columns]
if not cols:
    cols = list(fdf.columns)
if "confidence" in fdf.columns:
    show_df = fdf.sort_values(by=["confidence"], ascending=False)
else:
    show_df = fdf.copy()
st.dataframe(show_df[cols], use_container_width=True, hide_index=True)

# ------------- Drill-down -------------
st.markdown("### 🧾 Inspect a Row")
if len(fdf) == 0:
    st.info("No rows after filters.")
else:
    sel_idx = st.number_input("Pick row number from table above", min_value=1, max_value=len(show_df), value=1, step=1)
    selected = show_df.iloc[sel_idx - 1]

    st.markdown("#### Question")
    st.write(selected.get("question", ""))

    st.markdown("#### Answer")
    st.markdown(f"<div class='codebox'>{(selected.get('answer','') or '').replace('<','&lt;').replace('>','&gt;')}</div>", unsafe_allow_html=True)

    # 📚 Sources / Confidence / Mode / Response Time block
    sources = selected.get("citations", "—") if pd.notna(selected.get("citations", "")) else "—"
    conf_val = f"{selected.get('confidence', 0):.2f}" if pd.notna(selected.get("confidence", None)) else "—"
    mode_val = selected.get("retrieval_mode", "—")
    latency_val = f"{selected.get('latency_sec', 0):.2f}s" if pd.notna(selected.get("latency_sec", None)) else "—"

    st.markdown(
        f"""
        <div class='info-card'>
            <p>📚 <strong>Sources:</strong> {sources}</p>
            <p><strong>Confidence:</strong> {conf_val}</p>
            <p><strong>Mode:</strong> {mode_val}</p>
            <p><strong>Response Time:</strong> {latency_val}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Citations")
    st.write(sources)

    ctxs: List[Dict[str, Any]] = []
    if "contexts_json" in selected and isinstance(selected["contexts_json"], str) and selected["contexts_json"].strip():
        try:
            ctxs = json.loads(selected["contexts_json"])
        except Exception:
            ctxs = []
    if ctxs:
        st.markdown("#### Retrieved Contexts")
        st.write(f"Count: {len(ctxs)}")
        for i, c in enumerate(ctxs, start=1):
            header = f"Context #{i}"
            sub = []
            if c.get("file_name"):
                sub.append(c["file_name"])
            elif c.get("doc_id"):
                sub.append(str(c["doc_id"]))
            if c.get("score") is not None:
                sub.append(f"score={round(float(c['score']), 4)}")
            if sub:
                header += "  •  " + "  •  ".join(sub)
            with st.expander(header, expanded=False):
                meta = {k: c.get(k) for k in ["doc_id", "file_name", "chunk_id", "score"]}
                st.json(meta)
                st.markdown(f"<div class='codebox'>{(c.get('content','') or '').replace('<','&lt;').replace('>','&gt;')}</div>", unsafe_allow_html=True)

# ------------- Exports -------------
st.markdown("---")
st.subheader("Export")
csv_name = "eval_report_filtered.csv"
csv_bytes = fdf.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name=csv_name, mime="text/csv")
colA, colB = st.columns(2)
with colA:
    if REPORT_PATH.exists():
        st.download_button("⬇️ Download full eval_report.csv", data=REPORT_PATH.read_bytes(), file_name="eval_report.csv", mime="text/csv")
with colB:
    if SUMMARY_PATH.exists():
        st.download_button("⬇️ Download eval_summary.json", data=SUMMARY_PATH.read_bytes(), file_name="eval_summary.json", mime="application/json")
