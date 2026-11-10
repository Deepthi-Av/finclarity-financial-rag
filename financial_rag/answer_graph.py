"""A small LangGraph RAG workflow; no tools are exposed to the answer model."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing
from importlib.metadata import version
import json
import math
import re
import sqlite3
import time
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from financial_rag.answer_services import AnswerServices, GroundedAnswer
from financial_rag.embeddings import NebiusEmbeddings, ServiceError, query_input
from financial_rag.sources import ROOT, sha256, write_json
from financial_rag.vector_store import PineconeStore, load_chunk_build


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def request_counts(root):
    counts = {"embedding": 0, "generation": 0, "rerank": 0}
    for filename in ["embeddings.sqlite3", "answer_usage.sqlite3"]:
        path = root / "artifacts" / filename
        if not path.exists():
            continue
        with closing(sqlite3.connect(path)) as connection:
            if filename == "embeddings.sqlite3":
                counts["embedding"] = connection.execute("SELECT count(*) FROM calls").fetchone()[0]
            else:
                for kind, count in connection.execute("SELECT kind,count(*) FROM calls GROUP BY kind"):
                    counts[kind] = count
    return counts


def route_question(question):
    if not question.strip():
        raise ServiceError("Enter a financial question.")
    lower = question.lower()
    if re.search(r"\b(stock|share) price\b", lower) and re.search(r"\b(now|current|currently|live|today|latest)\b", lower):
        return "insufficient_evidence"
    metric = re.search(r"\b(revenue|net income|operating income|earnings per share|eps|cash flow)\b", lower)
    period = re.search(r"\b(20\d{2}|fy\s*\d{2,4}|q[1-4]|annual|quarter\w*|year\w*)\b", lower)
    if metric and not period:
        return "needs_clarification"
    return "retrieve"


def context_passage(chunk):
    # Expose only the exact spans present in this chunk, never a whole parent block
    # when only part of it was retrieved. The shared evidence ID remains stable.
    return {"chunk_id": chunk["chunk_id"], "source_title": chunk["source_title"],
            "source_url": chunk["source_url"], "section": chunk["section"],
            "evidence": [{"evidence_id": s["evidence_id"], "text": s["text"]} for s in chunk["segments"]]}


def select_context(ranked, counter, config):
    selected, passages = [], []
    for candidate in ranked[:config["final_passages"]]:
        passage = context_passage(candidate)
        tentative = json.dumps(passages + [passage], ensure_ascii=False)
        if counter.count(tentative) > config["context_token_cap"]:
            break
        selected.append(candidate)
        passages.append(passage)
    if not selected:
        raise ServiceError("No complete passage fits the shared context budget.")
    return selected, passages, counter.count(json.dumps(passages, ensure_ascii=False))


SAFE_MESSAGES = {
    "insufficient_evidence": "The selected Microsoft FY2025 documents do not contain enough evidence to answer this question. They cannot provide live prices or actual results for later periods.",
    "needs_clarification": "Do you mean Microsoft's FY2025 full-year results or its FY2025 fourth-quarter results? Please specify the fiscal period.",
}


def validate_answer(raw, selected):
    try:
        answer = GroundedAnswer.model_validate(raw)
    except Exception:
        raise ServiceError("Answer failed the structured response schema; withheld.") from None
    if answer.status != "answered":
        if answer.statements:
            raise ServiceError("A non-answer must not contain factual answer statements.")
        return {"status": answer.status, "answer": SAFE_MESSAGES[answer.status], "citations": []}
    if not answer.statements:
        raise ServiceError("An answered response must contain cited statements.")
    allowed = {}
    for chunk in selected:
        for segment in chunk["segments"]:
            allowed.setdefault(segment["evidence_id"], []).append((segment["text"], chunk))
    rendered, citations = [], {}
    for statement in answer.statements:
        # Citation syntax and links are produced by the app, never model text.
        if re.search(r"\[|\]|https?://", statement.text):
            raise ServiceError("Statement contains unmanaged citation/link syntax; withheld.")
        ids = []
        for citation in statement.citations:
            matches = [(text, chunk) for text, chunk in allowed.get(citation.evidence_id, []) if citation.quote in text]
            if not matches:
                raise ServiceError("Citation ID or exact supporting quotation is absent from selected evidence; answer withheld.")
            text, chunk = matches[0]
            item = citations.setdefault(citation.evidence_id, {
                "evidence_id": citation.evidence_id, "source_title": chunk["source_title"],
                "source_url": chunk["source_url"], "section": chunk["section"], "quotes": [], "retrieved_spans": []})
            if citation.quote not in item["quotes"]:
                item["quotes"].append(citation.quote)
            if text not in item["retrieved_spans"]:
                item["retrieved_spans"].append(text)
            ids.append(citation.evidence_id)
        rendered.append(statement.text + " " + " ".join(f"[{identifier}]" for identifier in dict.fromkeys(ids)))
    return {"status": "answered", "answer": "\n\n".join(rendered), "citations": list(citations.values())}


class QueryState(TypedDict, total=False):
    question: str
    experiment: str
    route: str
    candidates: list
    ranked: list
    selected: list
    passages: list
    context_tokens: int
    raw_answer: dict
    result: dict
    retrieval_cache_hit: bool
    generation_cache_hit: bool
    rerank_cache_hit: bool
    generation_usage: dict
    rerank_usage: dict
    candidate_fingerprint: str


class AnswerPipeline:
    def __init__(self, root=ROOT):
        self.root = root
        self.strategies, self.embedding_config, self.chunk_report, self.counter, self.build_id = load_chunk_build(root)
        self.config = json.loads((root / "config/answers.json").read_text(encoding="utf-8"))
        self.system = (root / "prompts/answer_system.txt").read_text(encoding="utf-8")
        self.schema = GroundedAnswer.model_json_schema()
        self.config_hash, self.prompt_hash = fingerprint(self.config), sha256(self.system.encode())
        self.schema_hash = fingerprint(self.schema)
        self.embedding = self.store = self.services = None
        self.timings, self.current_stage = {}, None
        graph = StateGraph(QueryState)
        for name in ["route", "retrieve", "select", "generate", "validate"]:
            graph.add_node(name, self.timed(name, getattr(self, name)))
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", lambda state: state["route"],
            {"retrieve": "retrieve", "insufficient_evidence": "validate", "needs_clarification": "validate"})
        graph.add_edge("retrieve", "select")
        graph.add_edge("select", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", END)
        self.graph = graph.compile()

    def timed(self, name, function):
        def invoke(state):
            self.current_stage = name
            start = time.perf_counter()
            try:
                update = function(state)
                self.latest_state.update(update)
                return update
            finally:
                self.timings[name] = time.perf_counter() - start
        return invoke

    def hosted(self):
        if self.services is None:
            self.services = AnswerServices(self.config, self.root)
        return self.services

    def cache(self, kind, key, call):
        path = self.root / "artifacts/answer_cache" / kind / (fingerprint(key) + ".json")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["key"] != key or fingerprint(record["value"]) != record["value_sha256"]:
                raise ServiceError("Answer cache integrity mismatch.")
            return record["value"], True
        value = call()
        write_json(path, {"key": key, "value": value, "value_sha256": fingerprint(value)})
        return value, False

    def route(self, state):
        question = state["question"]
        if len(question) > self.config["question_character_cap"]:
            raise ServiceError("Question exceeds the 1,000-character application limit.")
        route = route_question(question)
        return {"route": route, **({"raw_answer": {"status": route, "statements": []}, "selected": []} if route != "retrieve" else {})}

    def retrieve(self, state):
        strategy = "fixed" if state["experiment"] == "A" else "semantic"
        chunks = self.strategies[strategy]
        key = {"build_id": self.build_id, "strategy": strategy, "question": state["question"],
               "query_instruction": self.embedding_config["query_instruction"], "top_k": self.config["candidate_count"]}
        def request():
            if self.embedding is None:
                self.embedding = NebiusEmbeddings(self.counter, self.embedding_config, self.root)
            if self.store is None:
                self.store = PineconeStore(self.root)
            text = query_input(state["question"], self.embedding_config)
            vector = self.embedding.embed([text], purpose="answer_query")[text]
            matches = self.store.query(f"{self.store.prefix}-{strategy}-{self.build_id}", vector, chunks, self.config["candidate_count"])
            return [{"chunk_id": m["id"], "dense_score": m["score"]} for m in matches]
        records, hit = self.cache("retrieval", key, request)
        lookup = {c["chunk_id"]: c for c in chunks}
        if len(records) != self.config["candidate_count"] or len({r["chunk_id"] for r in records}) != len(records):
            raise ServiceError("Cached retrieval has an invalid candidate count.")
        if any(r["chunk_id"] not in lookup or not math.isfinite(r["dense_score"]) for r in records):
            raise ServiceError("Cached retrieval contains invalid IDs or scores.")
        return {"candidates": [{**lookup[r["chunk_id"]], "dense_score": r["dense_score"]} for r in records],
                "retrieval_cache_hit": hit, "candidate_fingerprint": fingerprint(records)}

    def select(self, state):
        ranked, hit, usage = state["candidates"], False, {}
        if state["experiment"] == "C":
            key = {"question": state["question"], "model": self.config["rerank_model"],
                   "candidate_hash": fingerprint(ranked), "protocol": "rerank-v1"}
            (ranked, usage), hit = self.cache("rerank", key, lambda: self.hosted().rerank(state["question"], ranked))
            # Validate cached ranked records against originals as well.
            lookup = {c["chunk_id"]: c for c in state["candidates"]}
            if len(ranked) != len(lookup) or {c["chunk_id"] for c in ranked} != set(lookup):
                raise ServiceError("Cached reranking IDs do not match candidates.")
            for c in ranked:
                if {k: v for k, v in c.items() if k not in ("rerank_score", "dense_position")} != lookup[c["chunk_id"]]:
                    raise ServiceError("Cached reranker changed a source passage.")
        selected, passages, tokens = select_context(ranked, self.counter, self.config)
        return {"ranked": ranked, "selected": selected, "passages": passages, "context_tokens": tokens,
                "rerank_cache_hit": hit, "rerank_usage": usage}

    def generate(self, state):
        user = json.dumps({"question": state["question"], "evidence_passages": state["passages"],
                           "required_json_schema": self.schema}, ensure_ascii=False)
        key = {"config_hash": self.config_hash, "prompt_hash": self.prompt_hash, "schema_hash": self.schema_hash,
               "user": user}
        (raw, usage), hit = self.cache("generation", key, lambda: self.hosted().generate(self.system, user))
        return {"raw_answer": raw, "generation_usage": usage, "generation_cache_hit": hit}

    def validate(self, state):
        return {"result": validate_answer(state["raw_answer"], state["selected"])}

    def run(self, question, experiment="C", *, question_id=None, split="interactive"):
        if experiment not in ("A", "B", "C"):
            raise ServiceError("Choose A (fixed), B (semantic), or C (semantic with Pinecone reranking).")
        self.timings, self.current_stage = {}, None
        start = time.perf_counter()
        before_requests = request_counts(self.root)
        pinecone_usage_start = len(self.store.usage) if self.store is not None and isinstance(self.store.usage, list) else 0
        state = {"question": question.strip(), "experiment": experiment}
        self.latest_state = dict(state)
        record = {"run_id": str(uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "question_id": question_id, "split": split, "build_id": self.build_id,
                  "corpus_sha256": self.chunk_report["corpus_sha256"], "answer_config": self.config,
                  "config_sha256": self.config_hash, "prompt_sha256": self.prompt_hash, "schema_sha256": self.schema_hash,
                  "rerank_method": "pinecone_rerank" if experiment == "C" else "none",
                  "library_versions": {name: version(name) for name in ["langgraph", "langchain-core", "langchain-openai", "pydantic"]}}
        try:
            state = self.graph.invoke(state)
            record["success"] = True
        except Exception as error:
            state = self.latest_state
            record.update(success=False, failed_stage=self.current_stage,
                          error=str(error) if isinstance(error, ServiceError) else f"{type(error).__name__}; sensitive details omitted.")
            state["result"] = {"status": "error", "answer": "The request could not be completed. No unchecked answer is shown.", "citations": []}
        record.update(state)
        record["rerank_applied"] = experiment == "C" and "ranked" in state
        record["elapsed_seconds"] = time.perf_counter() - start
        record["stage_seconds"] = self.timings
        after_requests = request_counts(self.root)
        record["new_request_reservations"] = {kind: after_requests[kind] - before_requests[kind] for kind in before_requests}
        record["usage_note"] = "generation_usage and rerank_usage describe the cached or original response; new_request_reservations counts new attempts this run, including uncertain failures."
        if self.store is not None and isinstance(self.store.usage, list):
            record["pinecone_read_usage_this_run"] = self.store.usage[pinecone_usage_start:]
        record["citation_validation_scope"] = "Schema, ID membership, exact quote presence, and statement citation coverage; semantic entailment/numerical correctness require evaluation."
        if self.services:
            record["cumulative_answer_service_usage"] = self.services.ledger.summary(self.config)
        path = self.root / "artifacts/runs" / (record["run_id"] + ".json")
        write_json(path, record)
        return record

    def close(self):
        for resource in (self.services, self.embedding, self.store):
            if resource:
                resource.close()
