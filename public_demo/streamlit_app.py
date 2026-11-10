"""Public saved-results viewer. No provider SDKs, credentials, or live queries."""
import json
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
st.set_page_config(page_title="FinClarity", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp {background:#f7f9fc;}
.block-container {max-width:1150px;padding-top:2rem;}
.fin-banner {background:linear-gradient(115deg,#102e3b,#17675e);padding:30px;
border-radius:18px;color:white;margin-bottom:18px;}
.fin-banner h1 {color:white;margin:0;} .fin-banner p {margin-bottom:0;}
</style><div class="fin-banner"><h1>FinClarity</h1>
<p>Financial results with evidence you can inspect</p></div>""", unsafe_allow_html=True)
st.info("Public saved-results demo · Select a recorded question to inspect its original answers. "
        "This app does not run live retrieval, generation, uploads, or provider calls.")

data = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))["sources"]
allowed_urls = {source["url"] for source in sources}
questions = list(dict.fromkeys(run["question"] for run in data["runs"]))
labels = {"A": "A · Fixed chunks", "B": "B · Semantic chunks",
          "C": "C · Semantic chunks + Pinecone reranking"}


def show_answer(run):
    result = run["result"]
    st.caption("Recorded status: " + result["status"].replace("_", " "))
    st.markdown(result["answer"])
    if not result["citations"]:
        st.caption("No supporting citations were returned for this recorded response.")
    for number, citation in enumerate(result["citations"], 1):
        with st.expander(f"Source {number} · {citation['source_title']}"):
            st.write("Section: " + citation["section"])
            st.code(citation["evidence_id"], language=None)
            st.write("Supporting quotation")
            for quote in citation["quotes"]:
                st.text(quote)
            with st.expander("Surrounding retrieved evidence"):
                for span in citation["retrieved_spans"]:
                    st.text(span)
            if citation["source_url"] in allowed_urls:
                st.link_button("Open official Microsoft report", citation["source_url"])


ask_tab, compare_tab, results_tab, sources_tab = st.tabs(
    ["Ask documents", "Compare A/B/C", "Measured results", "Source documents"])

with ask_tab:
    st.subheader("Explore a saved financial question")
    question = st.selectbox("Choose a recorded question", questions, key="saved_question")
    method = st.selectbox("Answer method", list(labels), format_func=labels.get, key="saved_method")
    selected = [r for r in data["runs"] if r["question"] == question]
    show_answer(next(r for r in selected if r["experiment"] == method))

with compare_tab:
    st.subheader("Same question, three retrieval strategies")
    st.write(question)
    for column, method in zip(st.columns(3), labels):
        with column:
            st.markdown("**" + labels[method] + "**")
            show_answer(next(r for r in selected if r["experiment"] == method))
    st.caption("A/B/C are the original saved executions. C reranks B's same ten candidates "
               "using Pinecone bge-reranker-v2-m3.")

with results_tab:
    st.subheader("Original Microsoft held-out evaluation")
    rows = []
    for method in labels:
        scores = data["summary"][method]
        rows.append({"Method": labels[method],
                     "Requested answers correct": f"{scores['requested_answer_correct']}/10",
                     "Strict citation support": f"{scores['strict_citation_support_pass']}/10",
                     "Unsupported refusals": f"{scores['correct_refusals']}/2",
                     "RAGAS faithfulness": round(scores["faithfulness"], 3)})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.write("36 original runs: ten answerable and two unsupported questions per method. "
             "Correctness and citation support were assessed through AI-assisted source inspection, "
             "not independent human validation. Faithfulness means cover ten answerable cases per method.")
    st.warning("Known limitation: C used annual diluted EPS of $13.64 for a Q4 question whose "
               "answer was $3.65. Its quotation was genuine, but the period was wrong; the automated "
               "faithfulness judge missed this. The original answer is retained.")
    st.write("A had the strongest requested-answer correctness on this small test. "
             "These results do not establish accuracy for arbitrary uploads or new questions.")
    st.caption("Original evaluation freeze: " + data["freeze_id"])

with sources_tab:
    st.subheader("Microsoft source documents")
    st.write("The experiment searched selected financial sections of these two English HTML publications. "
             "Links open the official reports; this hosted demo contains the saved cited passages.")
    for source in sources:
        st.markdown("**" + source["title"] + "**")
        st.write("Owner: " + source["owner"])
        st.link_button("Open " + source["document_id"], source["url"])
    st.caption("Historical FY2025 snapshots with manual freshness; no live prices or scheduled refresh.")

st.divider()
st.caption("Original pipeline: Nebius embeddings/generation, Pinecone dense retrieval and BGE reranking, "
           "and LangGraph orchestration. This separate public viewer reads local saved JSON only.")
