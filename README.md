# FinClarity Financial RAG

**An evaluated financial RAG pipeline with traceable evidence, controlled retrieval experiments, and documented failure analysis.**

**Stack:** Python 3.12 · LangChain · LangGraph · Nebius · Pinecone · Streamlit · RAGAS

## Project overview

FinClarity helps business analysts answer period-specific financial questions and inspect the source evidence. The original experiment uses Microsoft's FY2025 annual report and Q4 earnings release to measure how chunking and reranking affect answer correctness and citation support.

This repository includes the original pipeline source and a public Streamlit viewer of saved results. The viewer requires no API keys; deployment is pending.

## What the current application does

- **Public demo:** explores 12 recorded questions, compares 36 A/B/C answers, expands quotations and surrounding evidence, and links to official Microsoft reports. It displays saved results without live inference or uploads.
- **Broader local application:** supports public/synthetic PDF, DOCX, XLSX and CSV collections, document readers, cited questions, and separate User/Admin access. That upload/account interface is outside this minimal repository; the included source covers the original Microsoft RAG pipeline.

## Architecture

```text
Microsoft reports → Cleaned evidence → Fixed / semantic chunks → Embeddings → Pinecone
Question → LangGraph routing → Retrieval → Optional BGE reranking → Generation → Citation validation
```

## Engineering highlights

- **Traceable ingestion:** source checksums, stable evidence IDs, and preserved table headers, currency units, and fiscal periods across 430 evidence blocks.
- **Retrieval engineering:** 110 fixed and 138 sentence-distance semantic chunks; 4,096-dimensional hosted embeddings; versioned Pinecone namespaces and validated vector/source mappings.
- **Orchestration and validation:** a five-node LangGraph flow, structured answers, evidence-ID checks, exact supporting quotations, and explicit insufficient-evidence handling.
- **Cost controls:** embedding/answer caching, bounded batching, token budgets, and persistent usage reservations. Application estimates are not provider billing caps.
- **Controlled evaluation:** 36 original A/B/C runs with the same generator, prompt, and context ceiling; C reranks B's identical ten candidates. Failures are retained rather than replaced.

## Results

Ten answerable and two unsupported questions per method:

| Method | Correct answers | Strict citation support | Correct refusals | RAGAS faithfulness |
| --- | --- | --- | --- | --- |
| A Fixed | 10/10 | 9/10 | 2/2 | 1.000 |
| B Semantic | 9/10 | 8/10 | 2/2 | 0.967 |
| C Semantic + reranking | 8/10 | 7/10 | 2/2 | 0.971 |

Correctness and citation support used AI-assisted source inspection, not independent human validation. Faithfulness means cover ten answerable cases per method. This small test supports descriptive findings, not accuracy claims for new documents.

**Engineering takeaway:** reranking added complexity without improving overall correctness here. C cited annual EPS of **$13.64** for a quarterly question whose answer was **$3.65**; the automated faithfulness judge missed the period mismatch. Valid quotations require fiscal-period checks too. The original error remains visible.

## Run the public demo

With Python 3.12 and uv, run from the repository root:

```powershell
uv venv public_demo/.venv --python 3.12
uv pip install --python public_demo/.venv/Scripts/python.exe -r public_demo/requirements.txt
public_demo/.venv/Scripts/python.exe -m streamlit run public_demo/streamlit_app.py
```

On macOS/Linux, replace `public_demo/.venv/Scripts/python.exe` with `public_demo/.venv/bin/python`.

**Streamlit Community Cloud:** select `public_demo/streamlit_app.py`, Python 3.12, and leave secrets empty. The entry-point-local requirements file isolates the demo from the heavier pipeline dependencies.

## Repository guide

| Path | Purpose |
| --- | --- |
| `public_demo/` | Runnable saved-results viewer and required display data |
| `financial_rag/` | Original ingestion, chunking, embedding, vector-store, generation, and LangGraph modules |
| `config/` and `prompts/` | Original retrieval settings and grounded-answer instructions |
| `data/sources.json` | Official source URLs, provenance, scope, and checksums |
| `pyproject.toml` and `uv.lock` | Original pipeline dependency declaration and lock |

Pipeline models: Nebius **Qwen3-Embedding-8B** and **Qwen3-30B-A3B-Instruct-2507**; reranking: Pinecone **bge-reranker-v2-m3** (`pinecone_rerank`).

The pipeline source is included for inspection. Full pipeline execution additionally needs source/build artifacts, tokenizer files, private provider configuration, and credit readiness; those assets and the evaluation harness are outside this minimal repository. The saved viewer is independently runnable. Credentials, accounts, private data, and local caches are excluded.

## Limitations and future work

**Current limitations:** the small Microsoft evaluation and AI-assisted judgments do not establish accuracy on new documents. Exact citation checks can miss wrong-period or category errors. The public viewer cannot process uploads or answer new questions; the broader local app supports searchable documents only, with no OCR, scheduled refresh, or production identity service.

**Proposed next steps:** add fiscal-period and financial-category validation, evaluate new collections with independently reviewed questions, and test hybrid retrieval and table-aware calculations against the existing baselines. Production deployment of the full app would also require managed authentication, durable storage, and stronger tenant isolation. These improvements are not implemented or measured here.

