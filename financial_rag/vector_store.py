"""Versioned Pinecone storage; no index creation, namespace clearing or deletes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import ssl
import time
from urllib.parse import urlsplit

import httpx
import numpy as np

from financial_rag.chunking import TokenCounter, config_and_hash, load_corpus, prepare_units, validate_chunks
from financial_rag.embeddings import ServiceError, require_credit_policy
from financial_rag.sources import ROOT, sha256, write_json


def load_chunk_build(root: Path = ROOT):
    config, config_hash = config_and_hash(root)
    blocks, corpus = load_corpus(root)
    report = json.loads((root / "data/processed/chunking_summary.json").read_text())
    if report["config_sha256"] != config_hash or report["corpus_sha256"] != corpus["corpus_sha256"]:
        raise ServiceError("Chunk build differs from the current corpus/configuration.")
    counter = TokenCounter(root)
    units = prepare_units(blocks, counter, config)
    strategies, hashes = {}, []
    for strategy in ["fixed", "semantic"]:
        path = root / f"data/processed/{strategy}_chunks.jsonl"
        data = path.read_bytes()
        digest = sha256(data)
        if digest != report["strategies"][strategy]["file_sha256"]:
            raise ServiceError("Chunk-file fingerprint mismatch.")
        chunks = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        if any(c["config_sha256"] != config_hash or c["corpus_sha256"] != corpus["corpus_sha256"] or c["strategy"] != strategy for c in chunks):
            raise ServiceError("Mixed chunk configurations or corpus versions.")
        validate_chunks(chunks, units, counter, config)
        strategies[strategy] = chunks
        hashes.append(digest)
    build_id = sha256(json.dumps([config_hash, corpus["corpus_sha256"], hashes]).encode())[:12]
    return strategies, config, report, counter, build_id


def encoded_payload(namespace, vectors):
    return json.dumps({"namespace": namespace, "vectors": vectors}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def upsert_batches(namespace, records, max_bytes=1_800_000):
    batch = []
    for record in records:
        if len(encoded_payload(namespace, [record])) > max_bytes:
            raise ServiceError("A vector record exceeds the bounded upsert payload size.")
        if batch and (len(batch) >= 32 or len(encoded_payload(namespace, batch + [record])) > max_bytes):
            yield batch
            batch = []
        batch.append(record)
    if batch:
        yield batch


def vector_record(chunk, vector):
    metadata = {key: chunk[key] for key in ["strategy", "text", "evidence_ids", "document_id", "source_title", "source_url", "source_sha256", "section", "token_count", "corpus_sha256", "config_sha256", "period_labels"]}
    metadata["segments_json"] = json.dumps(chunk["segments"], ensure_ascii=False, separators=(",", ":"))
    if len(json.dumps(metadata, ensure_ascii=False).encode()) > 38_000:
        raise ServiceError("Chunk metadata exceeds the conservative per-record limit.")
    return {"id": chunk["chunk_id"], "values": vector.tolist(), "metadata": metadata}


class PineconeStore:
    def __init__(self, root: Path = ROOT, transport=None):
        require_credit_policy(root)
        self.root = root
        self.name = os.getenv("PINECONE_INDEX_NAME")
        self.prefix = os.getenv("PINECONE_NAMESPACE_PREFIX", "")
        if self.name != "msft-financial-rag" or not re.fullmatch(r"msft-fy2025-v1", self.prefix):
            raise ServiceError("This build may only write the agreed project index/namespace prefix.")
        if not os.getenv("PINECONE_API_KEY"):
            raise ServiceError("Configure the Pinecone key locally.")
        self.headers = {"Api-Key": os.environ["PINECONE_API_KEY"], "X-Pinecone-Api-Version": "2025-10"}
        self.client = httpx.Client(verify=ssl.create_default_context(), timeout=45, follow_redirects=False, transport=transport)
        self.usage = []
        try:
            description = self.request("GET", f"https://api.pinecone.io/indexes/{self.name}")
            if description.get("dimension") != 4096 or description.get("metric") != "cosine" or description.get("vector_type", "dense") != "dense" or not description.get("status", {}).get("ready"):
                raise ServiceError("Pinecone index is not ready or dimension/metric/type is incompatible; no writes performed.")
            host = description.get("host", "")
            parsed = urlsplit("https://" + host)
            if not parsed.hostname or not parsed.hostname.endswith(".pinecone.io") or parsed.port or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
                raise ServiceError("Refusing to send a key to an unexpected Pinecone index host.")
            self.endpoint = "https://" + host
        except Exception:
            self.client.close()
            raise

    def request(self, method, url, *, payload=None, params=None, content=None):
        try:
            response = self.client.request(method, url, headers={**self.headers, "Content-Type": "application/json"}, json=payload, params=params, content=content)
            if not 200 <= response.status_code < 300:
                raise ServiceError(f"Pinecone {urlsplit(url).path} returned HTTP {response.status_code}; sensitive response omitted.")
            result = response.json()
            if result.get("usage"):
                self.usage.append({"operation": urlsplit(url).path, **result["usage"]})
            return result
        except ServiceError:
            raise
        except Exception as error:
            raise ServiceError(f"Pinecone request failed ({type(error).__name__}); sensitive details omitted.") from None

    def stats(self):
        return self.request("POST", self.endpoint + "/describe_index_stats", payload={})

    def list_ids(self, namespace):
        identifiers, token, seen_tokens = set(), None, set()
        for _ in range(100):
            params = {"namespace": namespace, "limit": 100}
            if token:
                params["paginationToken"] = token
            result = self.request("GET", self.endpoint + "/vectors/list", params=params)
            identifiers.update(record["id"] for record in result.get("vectors", []))
            token = result.get("pagination", {}).get("next")
            if not token:
                return identifiers
            if token in seen_tokens:
                raise ServiceError("Repeated Pinecone pagination token.")
            seen_tokens.add(token)
        raise ServiceError("Namespace listing exceeded the bounded page count.")

    def query(self, namespace, vector, chunks, top_k=10):
        result = self.request("POST", self.endpoint + "/query", payload={"namespace": namespace, "vector": vector.tolist(), "topK": top_k, "includeValues": False, "includeMetadata": True})
        lookup = {c["chunk_id"]: c for c in chunks}
        matches = result.get("matches", [])
        if len(matches) != min(top_k, len(chunks)) or len({m["id"] for m in matches}) != len(matches):
            raise ServiceError("Retrieval returned an unexpected number of unique project candidates.")
        for match in matches:
            expected = lookup.get(match["id"])
            metadata = match.get("metadata", {})
            if expected is None or any(metadata.get(key) != expected[key] for key in ["evidence_ids", "corpus_sha256", "config_sha256", "text"]):
                raise ServiceError("Retrieved candidate metadata does not match the frozen local chunks.")
        return matches

    def index_build(self, strategies, embedding_cache, build_id):
        entries = {}
        for strategy, chunks in strategies.items():
            entries[strategy] = {
                "namespace": f"{self.prefix}-{strategy}-{build_id}",
                "expected_count": len(chunks),
                "expected_ids_sha256": sha256(json.dumps(sorted(c["chunk_id"] for c in chunks)).encode()),
            }
        path = self.root / f"artifacts/index_journal_{build_id}.json"
        old = json.loads(path.read_text()) if path.exists() else None
        if old and (old.get("index") != self.name or old.get("namespaces") != entries):
            raise ServiceError("Index journal differs from the requested build; no writes performed.")
        stats = self.stats()
        for strategy, entry in entries.items():
            count = stats.get("namespaces", {}).get(entry["namespace"], {}).get("vectorCount", 0)
            if count and old is None:
                raise ServiceError("Target namespace already contains records without this build's ownership journal; left untouched.")
        if old is None:
            write_json(path, {"index": self.name, "build_id": build_id, "namespaces": entries})
        report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "index": self.name, "build_id": build_id, "dimension": 4096, "metric": "cosine", "namespaces": {}, "success": False}
        for strategy, chunks in strategies.items():
            namespace = entries[strategy]["namespace"]
            expected = {chunk["chunk_id"] for chunk in chunks}
            existing = self.list_ids(namespace)
            if not existing.issubset(expected):
                raise ServiceError("Target namespace contains unexpected IDs; no clearing or overwrite attempted.")
            pending = []
            for chunk in chunks:
                vector = embedding_cache.get(chunk["text"])
                if vector is None:
                    raise ServiceError("Chunk embedding is missing; generate it before indexing.")
                if chunk["chunk_id"] not in existing:
                    pending.append(vector_record(chunk, vector))
            payload_bytes, upserted = 0, 0
            for batch in upsert_batches(namespace, pending):
                content = encoded_payload(namespace, batch)
                result = self.request("POST", self.endpoint + "/vectors/upsert", content=content)
                if result.get("upsertedCount") != len(batch):
                    raise ServiceError("Pinecone did not acknowledge every submitted vector.")
                payload_bytes += len(content)
                upserted += len(batch)
                print(f"{strategy}: upserted {upserted}/{len(pending)} missing vectors.", flush=True)
            deadline = time.monotonic() + 60
            while True:
                identifiers = self.list_ids(namespace)
                count = self.stats().get("namespaces", {}).get(namespace, {}).get("vectorCount", 0)
                if identifiers == expected and count == len(expected):
                    break
                if time.monotonic() > deadline:
                    raise ServiceError("Vector IDs/counts are not yet consistent; build remains resumable, not marked complete.")
                time.sleep(2)
            samples = list(dict.fromkeys([c["chunk_id"] for c in chunks[:2] + chunks[-2:]]))
            result = self.request("GET", self.endpoint + "/vectors/fetch", params=[("namespace", namespace)] + [("ids", identifier) for identifier in samples])
            lookup = {c["chunk_id"]: c for c in chunks}
            for identifier in samples:
                actual = result.get("vectors", {}).get(identifier)
                expected_chunk = lookup[identifier]
                if not actual or actual.get("metadata", {}).get("text") != expected_chunk["text"] or not np.allclose(actual.get("values", []), embedding_cache.get(expected_chunk["text"]), atol=1e-6):
                    raise ServiceError("Fetched vector or metadata failed sample verification.")
            report["namespaces"][strategy] = {**entries[strategy], "verified_count": count, "all_ids_verified": True, "sample_fetch_verified": len(samples), "new_vectors_written": upserted, "existing_vectors_reused": len(existing), "upsert_payload_bytes": payload_bytes}
            print(f"{strategy}: verified all {count} IDs/counts and {len(samples)} fetched vector/metadata samples.", flush=True)
        report["success"] = True
        report["provider_usage"] = self.usage
        report["write_units_note"] = "Upsert responses did not return write-unit usage; no zero-cost or quota-remaining claim inferred."
        write_json(self.root / "artifacts/index_report.json", report)
        return report

    def close(self):
        self.client.close()
