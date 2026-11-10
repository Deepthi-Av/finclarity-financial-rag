"""Common evidence preparation plus token-bounded fixed/semantic chunkers."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re

import numpy as np
from tokenizers import Tokenizer

from financial_rag.sources import ROOT, read_manifest, sha256, verify_snapshot, write_json


def config_and_hash(root: Path = ROOT) -> tuple[dict, str]:
    config = json.loads((root / "config/chunking.json").read_text(encoding="utf-8"))
    return config, sha256(json.dumps(config, sort_keys=True).encode())


class TokenCounter:
    def __init__(self, root: Path = ROOT):
        config, _ = config_and_hash(root)
        path = root / "data/processed/tokenizer.json"
        metadata = json.loads(path.with_name("tokenizer_metadata.json").read_text())
        if metadata["revision"] != config["tokenizer_revision"] or sha256(path.read_bytes()) != metadata["sha256"]:
            raise ValueError("Pinned tokenizer provenance mismatch.")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def split_text(self, text: str, limit: int):
        start = 0
        while start < len(text):
            remaining = text[start:]
            encoded = self.tokenizer.encode(remaining, add_special_tokens=False)
            end = len(text) if len(encoded.ids) <= limit else start + encoded.offsets[limit - 1][1]
            while end > start and self.count(text[start:end]) > limit:
                end -= 1
            if end <= start:
                raise ValueError("Unable to split text within the token limit.")
            yield start, end
            start = end


def load_corpus(root: Path = ROOT) -> tuple[list[dict], dict]:
    manifest = read_manifest(root)
    for source in manifest["sources"]:
        verify_snapshot(source, root)
    data = (root / "data/processed/corpus.jsonl").read_bytes()
    summary = json.loads((root / "data/processed/summary.json").read_text())
    if sha256(data) != summary["corpus_sha256"]:
        raise ValueError("Corpus checksum mismatch; rerun ingestion before chunking.")
    blocks = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    sources = {source["document_id"]: source for source in manifest["sources"]}
    for block in blocks:
        source = sources.get(block["document_id"])
        if not source or source["sha256"] != block["source_sha256"] or source["url"] != block["source_url"]:
            raise ValueError("Evidence does not match the approved source snapshot.")
    if len(blocks) != summary["block_count"] or len({b["evidence_id"] for b in blocks}) != len(blocks):
        raise ValueError("Evidence count or IDs are inconsistent.")
    return blocks, summary


def prefix(unit: dict) -> str:
    return f"Source: {unit['source_title']}\nSection: {unit['section']}\n"


def render(units: list[dict]) -> str:
    return prefix(units[0]) + "\n".join(unit["text"] for unit in units)


def prepare_units(blocks: list[dict], counter, config: dict) -> list[dict]:
    units = []
    for block in blocks:
        available = config["max_chunk_tokens"] - counter.count(prefix(block)) - 8
        if available < 64:
            raise ValueError("Section heading consumes too much of the chunk budget.")
        if block["kind"] == "table_row":
            if counter.count(block["text"]) > available:
                raise ValueError("Table row exceeds the budget; change the documented policy instead of silently splitting it.")
            spans = [(0, len(block["text"]))]
        else:
            spans, cursor = [], 0
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z“\"])', block["text"])
            for sentence in sentences:
                start = block["text"].find(sentence, cursor)
                if start < 0:
                    raise ValueError("Sentence could not be mapped to its source evidence.")
                for local_start, local_end in counter.split_text(sentence, available):
                    spans.append((start + local_start, start + local_end))
                cursor = start + len(sentence)
        for start, end in spans:
            unit = {key: block[key] for key in ["document_id", "source_title", "source_url", "source_sha256", "section", "section_id", "evidence_id", "kind", "period_labels"]}
            unit.update({"start": start, "end": end, "text": block["text"][start:end], "unit_id": f"{block['evidence_id']}:{start}-{end}"})
            units.append(unit)
    return units


def grouped_units(units: list[dict]) -> list[list[dict]]:
    groups = OrderedDict()
    for unit in units:
        key = (unit["document_id"], unit["section_id"], unit["section"])
        groups.setdefault(key, []).append(unit)
    return list(groups.values())


def make_chunk(units, strategy, reason, counter, config_hash, corpus_hash, overlap=0):
    text = render(units)
    spans = [{key: u[key] for key in ["evidence_id", "unit_id", "start", "end", "text"]} for u in units]
    digest = sha256(json.dumps([strategy, config_hash, corpus_hash, spans], sort_keys=True).encode())[:20]
    return {
        "chunk_id": f"{units[0]['document_id']}:{strategy}:{digest}", "strategy": strategy,
        "text": text, "token_count": counter.count(text), "boundary_reason": reason,
        "overlap_tokens": overlap, "evidence_ids": list(dict.fromkeys(u["evidence_id"] for u in units)),
        "segments": spans, "document_id": units[0]["document_id"],
        "source_title": units[0]["source_title"], "source_url": units[0]["source_url"],
        "source_sha256": units[0]["source_sha256"], "section": units[0]["section"],
        "period_labels": sorted({year for u in units for year in u["period_labels"]}),
        "config_sha256": config_hash, "corpus_sha256": corpus_hash,
    }


def fixed_chunks(units, counter, config, config_hash, corpus_hash):
    chunks = []
    for group in grouped_units(units):
        current, carried_overlap = [], 0
        for unit in group:
            if current and counter.count(render(current + [unit])) > config["max_chunk_tokens"]:
                chunks.append(make_chunk(current, "fixed", "token_limit", counter, config_hash, corpus_hash, carried_overlap))
                tail = []
                for previous in reversed(current):
                    candidate = [previous] + tail
                    if counter.count("\n".join(item["text"] for item in candidate)) > config["fixed_overlap_tokens"]:
                        break
                    tail = candidate
                while tail and counter.count(render(tail + [unit])) > config["max_chunk_tokens"]:
                    tail.pop(0)
                carried_overlap = counter.count("\n".join(item["text"] for item in tail)) if tail else 0
                current = tail
            current.append(unit)
        if current:
            chunks.append(make_chunk(current, "fixed", "section_end", counter, config_hash, corpus_hash, carried_overlap))
    return chunks


def semantic_window_texts(units, config):
    texts = []
    for group in grouped_units(units):
        if len(group) < 2:
            continue
        buffer = config["semantic_buffer_sentences"]
        texts.extend(render(group[max(0, i - buffer):i + buffer + 1]) for i in range(len(group)))
    return list(dict.fromkeys(texts))


def semantic_chunks(units, vectors, counter, config, config_hash, corpus_hash):
    chunks, diagnostics = [], []
    for group in grouped_units(units):
        distances = []
        if len(group) > 1:
            buffer = config["semantic_buffer_sentences"]
            embeddings = [np.asarray(vectors[render(group[max(0, i - buffer):i + buffer + 1])], dtype=np.float32) for i in range(len(group))]
            normalized = [vector / np.linalg.norm(vector) for vector in embeddings]
            distances = [float(np.clip(1 - np.dot(a, b), 0, 2)) for a, b in zip(normalized, normalized[1:])]
        threshold = float(np.percentile(distances, config["semantic_breakpoint_percentile"])) if distances else None
        diagnostics.append({"document_id": group[0]["document_id"], "section": group[0]["section"], "unit_count": len(group), "distance_threshold": threshold, "distances": distances})
        current = []
        for i, unit in enumerate(group):
            if current and counter.count(render(current + [unit])) > config["max_chunk_tokens"]:
                chunks.append(make_chunk(current, "semantic", "token_limit", counter, config_hash, corpus_hash))
                current = []
            current.append(unit)
            if i < len(distances) and distances[i] > threshold and counter.count(render(current)) >= config["semantic_min_tokens"]:
                chunks.append(make_chunk(current, "semantic", "semantic_distance", counter, config_hash, corpus_hash))
                current = []
        if current:
            chunks.append(make_chunk(current, "semantic", "section_end", counter, config_hash, corpus_hash))
    return chunks, diagnostics


def validate_chunks(chunks, units, counter, config):
    expected = {u["unit_id"] for u in units}
    covered = {segment["unit_id"] for c in chunks for segment in c["segments"]}
    if expected != covered or len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Chunk coverage or uniqueness failed.")
    lookup = {u["unit_id"]: u for u in units}
    for chunk in chunks:
        if counter.count(chunk["text"]) != chunk["token_count"] or chunk["token_count"] > config["max_chunk_tokens"]:
            raise ValueError("Chunk token accounting failed.")
        for segment in chunk["segments"]:
            unit = lookup[segment["unit_id"]]
            if unit["document_id"] != chunk["document_id"] or unit["section"] != chunk["section"] or any(unit[key] != segment[key] for key in ["text", "evidence_id", "start", "end"]):
                raise ValueError("Source boundary or evidence mapping failed.")
        if chunk["evidence_ids"] != list(dict.fromkeys(segment["evidence_id"] for segment in chunk["segments"])):
            raise ValueError("Chunk citation IDs do not match its actual evidence segments.")


def save_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8", newline="\n")
    temporary.replace(path)


def chunk_stats(chunks):
    tokens = [c["token_count"] for c in chunks]
    return {"count": len(chunks), "tokens_min": min(tokens), "tokens_mean": round(float(np.mean(tokens)), 2), "tokens_max": max(tokens), "total_tokens": sum(tokens), "actual_overlap_tokens": sum(c["overlap_tokens"] for c in chunks), "semantic_distance_boundaries": sum(c["boundary_reason"] == "semantic_distance" for c in chunks)}
