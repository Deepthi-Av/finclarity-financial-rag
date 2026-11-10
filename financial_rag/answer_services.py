"""Bounded hosted ranking/generation, with durable usage reservations."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import sqlite3
import ssl

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from financial_rag.embeddings import ServiceError, require_credit_policy
from financial_rag.sources import ROOT


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(min_length=1, max_length=180)
    quote: str = Field(min_length=12, max_length=2000)


class Statement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1200)
    citations: list[Citation] = Field(min_length=1, max_length=5)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence", "needs_clarification"]
    statements: list[Statement] = Field(max_length=5)


class UsageLedger:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls (
          id INTEGER PRIMARY KEY, timestamp_utc TEXT, kind TEXT, reserved INTEGER,
          status TEXT, input_tokens INTEGER, output_tokens INTEGER, rerank_units REAL)""")

    def reserve(self, kind, reserved, config):
        # The write lock makes check-and-reserve atomic across local processes.
        try:
            self.db.execute("BEGIN IMMEDIATE")
            count, tokens = self.db.execute(
                "SELECT count(*), coalesce(sum(max(reserved,coalesce(input_tokens,0)+coalesce(output_tokens,0))),0) FROM calls WHERE kind=?", (kind,)
            ).fetchone()
            if count >= config[f"{kind}_call_budget"] or (kind == "generation" and tokens + reserved > config["generation_total_token_budget"]):
                raise ServiceError(f"Local {kind} budget exhausted; no provider request made.")
            row = self.db.execute("INSERT INTO calls(timestamp_utc,kind,reserved,status) VALUES(?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), kind, reserved, "pending"))
            self.db.commit()
            return row.lastrowid
        except Exception:
            self.db.rollback()
            raise

    def finish(self, identifier, *, status, input_tokens=None, output_tokens=None, rerank_units=None):
        self.db.execute("UPDATE calls SET status=?,input_tokens=?,output_tokens=?,rerank_units=? WHERE id=?",
                        (status, input_tokens, output_tokens, rerank_units, identifier))
        self.db.commit()

    def summary(self, config):
        rows = self.db.execute("SELECT kind,status,input_tokens,output_tokens,rerank_units FROM calls").fetchall()
        inputs = sum(r[2] or 0 for r in rows)
        outputs = sum(r[3] or 0 for r in rows)
        return {"generation_requests": sum(r[0] == "generation" for r in rows),
                "rerank_requests": sum(r[0] == "rerank" for r in rows),
                "uncertain_or_failed_requests": sum(r[1] != "success" for r in rows),
                "reported_input_tokens": inputs, "reported_output_tokens": outputs,
                "reported_rerank_units": sum(r[4] or 0 for r in rows),
                "known_rate_generation_estimate_usd": (inputs * config["input_usd_per_million"] + outputs * config["output_usd_per_million"]) / 1e6,
                "note": "Local answer-service ledger only; excludes embeddings, setup, Pinecone read costs and unknown usage. Not a billing cap."}

    def close(self):
        self.db.close()


def map_rerank(payload, candidates):
    rows = payload.get("data", [])
    if len(rows) != len(candidates):
        raise ServiceError("Reranker did not return all candidates.")
    seen, ranked, previous = set(), [], float("inf")
    for row in rows:
        index, score = row.get("index"), row.get("score")
        if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
            raise ServiceError("Reranker returned duplicate or invalid input positions.")
        candidate = candidates[index]
        if row.get("document") != {"id": candidate["chunk_id"], "text": candidate["text"]}:
            raise ServiceError("Reranker document ID/text does not match its input position.")
        if type(score) not in (int, float) or not math.isfinite(score) or score > previous:
            raise ServiceError("Reranker scores are invalid or out of order.")
        seen.add(index)
        previous = score
        ranked.append({**candidate, "rerank_score": score, "dense_position": index})
    return ranked


class AnswerServices:
    def __init__(self, config, root=ROOT):
        require_credit_policy(root)
        if (os.getenv("NEBIUS_BASE_URL", "").rstrip("/") != "https://api.tokenfactory.nebius.com/v1"
                or os.getenv("NEBIUS_CHAT_MODEL") != config["chat_model"]
                or config["chat_model"] != "Qwen/Qwen3-30B-A3B-Instruct-2507"
                or os.getenv("RERANK_PROVIDER") != "pinecone_rerank"
                or os.getenv("PINECONE_RERANK_MODEL") != config["rerank_model"]
                or config["rerank_model"] != "bge-reranker-v2-m3"):
            raise ServiceError("Answer services require the verified providers and models.")
        if not os.getenv("NEBIUS_API_KEY") or not os.getenv("PINECONE_API_KEY"):
            raise ServiceError("Configure both keys locally.")
        self.config = config
        self.ledger = UsageLedger(root / "artifacts/answer_usage.sqlite3")
        self.client = httpx.Client(verify=ssl.create_default_context(), timeout=55, follow_redirects=False)
        # This project does not send question/evidence traces to a third-party tracing service.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        self.llm = ChatOpenAI(model=config["chat_model"], base_url="https://api.tokenfactory.nebius.com/v1",
            api_key=os.environ["NEBIUS_API_KEY"], temperature=0,
            max_tokens=config["generation_output_token_cap"], max_retries=0, timeout=55,
            http_client=self.client, extra_body={"service_tier": "default"})

    def rerank(self, question, candidates):
        identifier = self.ledger.reserve("rerank", 0, self.config)
        try:
            response = self.client.post("https://api.pinecone.io/rerank",
                headers={"Api-Key": os.environ["PINECONE_API_KEY"], "X-Pinecone-Api-Version": "2025-10"},
                json={"model": self.config["rerank_model"], "query": question,
                      "documents": [{"id": c["chunk_id"], "text": c["text"]} for c in candidates],
                      "top_n": len(candidates), "return_documents": True, "parameters": {"truncate": "NONE"}})
            if response.status_code != 200:
                raise ServiceError(f"Pinecone reranking returned HTTP {response.status_code}; no dense fallback or automatic retry.")
            payload = response.json()
            ranked = map_rerank(payload, candidates)
            units = payload.get("usage", {}).get("rerank_units")
            if type(units) not in (int, float) or not math.isfinite(units) or units < 0:
                raise ServiceError("Reranking usage is missing or invalid.")
            self.ledger.finish(identifier, status="success", rerank_units=units)
            return ranked, {"rerank_units": units}
        except Exception as error:
            self.ledger.finish(identifier, status="uncertain_or_failed")
            if isinstance(error, ServiceError):
                raise
            raise ServiceError(f"Reranking failed ({type(error).__name__}); sensitive details omitted.") from None

    def generate(self, system, user):
        # A UTF-8 byte bound plus overhead is deliberately more conservative than
        # estimating this chat model with the embedding model's tokenizer.
        bound = len((system + user).encode("utf-8")) + 256
        if bound > self.config["generation_input_byte_cap"]:
            raise ServiceError("Generation input exceeds the conservative byte limit; no truncation or request.")
        identifier = self.ledger.reserve("generation", bound + self.config["generation_output_token_cap"], self.config)
        usage = {}
        try:
            result = self.llm.with_structured_output(GroundedAnswer, method="json_mode", include_raw=True).invoke(
                [SystemMessage(content=system), HumanMessage(content=user)])
            usage = result["raw"].usage_metadata or {}
            inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
            if type(inputs) is not int or type(outputs) is not int or min(inputs, outputs) < 0:
                raise ServiceError("Generation token usage is missing or invalid.")
            self.ledger.finish(identifier, status="success", input_tokens=inputs, output_tokens=outputs)
            if result["raw"].response_metadata.get("finish_reason") != "stop" or result.get("parsing_error") or result.get("parsed") is None:
                raise ServiceError("Generation was truncated or failed the response schema; answer withheld.")
            return result["parsed"].model_dump(), usage
        except Exception as error:
            self.ledger.finish(identifier, status="uncertain_or_failed", input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
            if isinstance(error, ServiceError):
                raise
            raise ServiceError(f"Generation failed ({type(error).__name__}); sensitive details omitted; no automatic paid retry.") from None

    def close(self):
        self.client.close()
        self.ledger.close()
