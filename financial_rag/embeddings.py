"""Guarded Nebius embeddings with a persistent, model-specific SQLite cache."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import ssl

from dotenv import load_dotenv
import httpx
import numpy as np

from financial_rag.sources import ROOT, sha256


class ServiceError(RuntimeError):
    """A deliberately secret-free error suitable for CLI output."""


def require_credit_policy(root: Path = ROOT):
    load_dotenv(root / ".env", override=False)
    if os.getenv("ALLOW_BILLABLE_CALLS", "false").lower() != "true":
        raise ServiceError("Provider calls are disabled by ALLOW_BILLABLE_CALLS.")
    if os.getenv("SPEND_POLICY") != "credits_only" or os.getenv("MAX_ADDITIONAL_SPEND_USD") != "0":
        raise ServiceError("The agreed credits-only/zero-additional-spend settings are required.")
    record = root / "artifacts/account_readiness.json"
    if not record.exists():
        raise ServiceError("Record private credit readiness before enabling provider calls; see SETUP.md.")
    account = json.loads(record.read_text())
    if account.get("nebius_auto_top_up_disabled") is not True or account.get("nebius_balance_usd", 0) < 1 or account.get("pinecone_plan") != "Starter":
        raise ServiceError("Confirmed positive credits, disabled top-up, and Pinecone Starter are required.")


def checked_vector(values, dimension):
    vector = np.asarray(values, dtype="<f4")
    if vector.shape != (dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ServiceError("Embedding dimension, finite-value, or nonzero-vector validation failed.")
    return vector


class EmbeddingCache:
    def __init__(self, path: Path, model: str, dimension: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.model, self.dimension = model, dimension
        self.connection.executescript('''
          CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY, timestamp_utc TEXT, purpose TEXT,
            reserved_tokens INTEGER, reported_tokens INTEGER, status TEXT);
        ''')

    def key(self, text):
        return sha256(json.dumps([self.model, self.dimension, "float/default/plain-input-v1", text], ensure_ascii=False).encode())

    def get(self, text):
        row = self.connection.execute("SELECT vector FROM embeddings WHERE cache_key=?", (self.key(text),)).fetchone()
        return None if row is None else checked_vector(np.frombuffer(row[0], dtype="<f4"), self.dimension)

    def put(self, text, vector):
        self.connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (self.key(text), checked_vector(vector, self.dimension).tobytes()))

    def usage(self):
        rows = self.connection.execute("SELECT reserved_tokens,reported_tokens,status FROM calls").fetchall()
        return {
            "successful_requests": sum(row[2] == "success" for row in rows),
            "reported_input_tokens": sum((row[1] or 0) for row in rows if row[2] == "success"),
            "uncertain_or_failed_requests": sum(row[2] != "success" for row in rows),
            "conservative_budget_tokens": sum(max(row[0], row[1] or 0) for row in rows),
        }

    def reserve(self, tokens, purpose, budget):
        if self.usage()["conservative_budget_tokens"] + tokens > budget:
            raise ServiceError("Embedding build token budget would be exceeded; stopped before the request.")
        cursor = self.connection.execute("INSERT INTO calls(timestamp_utc,purpose,reserved_tokens,status) VALUES (?,?,?,?)", (datetime.now(timezone.utc).isoformat(), purpose, tokens, "pending"))
        self.connection.commit()
        return cursor.lastrowid

    def close(self):
        self.connection.close()


class NebiusEmbeddings:
    def __init__(self, counter, config, root: Path = ROOT, transport=None):
        require_credit_policy(root)
        if os.getenv("NEBIUS_BASE_URL", "").rstrip("/") != "https://api.tokenfactory.nebius.com/v1":
            raise ServiceError("Embedding credentials may only be sent to the official Nebius endpoint.")
        if config["embedding_model"] != "Qwen/Qwen3-Embedding-8B" or config["embedding_dimension"] != 4096:
            raise ServiceError("This bounded build is validated for the selected Qwen embedding model/dimension.")
        if os.getenv("NEBIUS_EMBED_MODEL") != config["embedding_model"] or not os.getenv("NEBIUS_API_KEY"):
            raise ServiceError("Configure the selected Nebius embedding model and key locally.")
        if config["embedding_input_token_cap"] > 8192:
            raise ServiceError("Keep the conservative application input cap at or below 8,192 tokens.")
        self.counter, self.config = counter, config
        self.cache = EmbeddingCache(root / "artifacts/embeddings.sqlite3", config["embedding_model"], config["embedding_dimension"])
        self.client = httpx.Client(verify=ssl.create_default_context(), timeout=90, follow_redirects=False, transport=transport)
        self.cache_hits = 0

    def embed(self, texts: list[str], *, purpose: str):
        unique = list(dict.fromkeys(texts))
        results, missing = {}, []
        for text in unique:
            count = self.counter.count(text) + 8  # reserve space for provider special tokens
            if not text.strip() or count > self.config["embedding_input_token_cap"]:
                raise ServiceError("Empty or over-limit embedding input; no silent truncation.")
            cached = self.cache.get(text)
            if cached is None:
                missing.append((text, count))
            else:
                results[text] = cached
                self.cache_hits += 1
        batch, batch_tokens = [], 0
        batches = []
        for text, count in missing:
            if batch and (len(batch) >= self.config["embedding_batch_size"] or batch_tokens + count > self.config["embedding_batch_token_cap"]):
                batches.append((batch, batch_tokens))
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += count
        if batch:
            batches.append((batch, batch_tokens))
        for index, (batch, reserved) in enumerate(batches):
            request_id = self.cache.reserve(reserved, purpose, self.config["embedding_build_token_budget"])
            try:
                response = self.client.post("https://api.tokenfactory.nebius.com/v1/embeddings", headers={"Authorization": "Bearer " + os.environ["NEBIUS_API_KEY"]}, json={
                    "model": self.config["embedding_model"], "input": batch,
                    "encoding_format": "float", "service_tier": "default",
                })
                if response.status_code != 200:
                    raise ServiceError(f"Embedding request returned HTTP {response.status_code}; response omitted; no automatic paid retry.")
                payload = response.json()
                data = sorted(payload["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise ServiceError("Embedding response indices do not map to every input exactly once.")
                vectors = [checked_vector(item["embedding"], self.config["embedding_dimension"]) for item in data]
                reported = payload.get("usage", {}).get("prompt_tokens")
                if not isinstance(reported, int) or reported < 0:
                    raise ServiceError("Provider did not supply valid input-token usage.")
                for text, vector in zip(batch, vectors):
                    self.cache.put(text, vector)
                    results[text] = vector
                self.cache.connection.execute("UPDATE calls SET status='success',reported_tokens=? WHERE id=?", (reported, request_id))
                self.cache.connection.commit()
            except Exception as error:
                self.cache.connection.rollback()
                self.cache.connection.execute("UPDATE calls SET status='uncertain_or_failed' WHERE id=?", (request_id,))
                self.cache.connection.commit()
                if isinstance(error, ServiceError):
                    raise
                raise ServiceError(f"Embedding request failed ({type(error).__name__}); sensitive details omitted; cached earlier batches are retained.") from None
            print(f"{purpose}: embedded batch {index + 1}/{len(batches)} ({len(batch)} inputs); earlier results cached.", flush=True)
        return {text: results[text] for text in texts}

    def usage(self):
        usage = self.cache.usage()
        usage["known_rate_estimate_usd"] = usage["reported_input_tokens"] * self.config["embedding_input_usd_per_million"] / 1_000_000
        usage["estimate_note"] = "Recorded successful embedding calls only; provider billing is authoritative. Failed/uncertain attempts are listed separately."
        return usage

    def close(self):
        self.client.close()
        self.cache.close()


def query_input(question: str, config: dict):
    return f"Instruct: {config['query_instruction']}\nQuery: {question}"
