"""Download only the two approved public sources; never load API credentials."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, HTTPSHandler


ROOT = Path(__file__).resolve().parents[1]
APPROVED_URLS = {
    "AR25": "https://www.microsoft.com/investor/reports/ar25/",
    "ER25Q4": "https://www.microsoft.com/en-us/Investor/earnings/FY-2025-Q4/press-release-webcast",
}
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "data/sources.json").read_text(encoding="utf-8"))
    entries = manifest.get("sources", [])
    if len(entries) != 2 or {s.get("document_id") for s in entries} != set(APPROVED_URLS):
        raise ValueError("The corpus must contain exactly the two approved Microsoft sources.")
    for source in entries:
        if source.get("url") != APPROVED_URLS[source["document_id"]]:
            raise ValueError("Source URL differs from the approved corpus.")
    return manifest


def raw_path(source: dict, root: Path = ROOT) -> Path:
    return root / "data" / "raw" / (source["document_id"] + ".html")


def verify_snapshot(source: dict, root: Path = ROOT) -> bytes:
    data = raw_path(source, root).read_bytes()
    if not source.get("downloaded_at_utc") or sha256(data) != source.get("sha256"):
        raise ValueError(f"{source['document_id']}: missing provenance or snapshot checksum mismatch.")
    return data


def check_official_url(url: str) -> None:
    parts = urlsplit(url)
    if (
        parts.scheme != "https" or parts.hostname != "www.microsoft.com"
        or parts.port not in (None, 443) or parts.username or parts.password
    ):
        raise ValueError("Downloads and redirects must remain on Microsoft's official HTTPS host.")


class OfficialRedirectHandler(HTTPRedirectHandler):
    max_redirections = 3

    def redirect_request(self, request, fp, code, message, headers, new_url):
        check_official_url(new_url)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def download_sources(root: Path = ROOT, *, refresh: bool = False) -> dict:
    manifest = read_manifest(root)
    opener = build_opener(OfficialRedirectHandler(), HTTPSHandler(context=ssl.create_default_context()))
    for source in manifest["sources"]:
        path = raw_path(source, root)
        if path.exists() and not refresh:
            verify_snapshot(source, root)
            print(f"{source['document_id']}: verified cached snapshot; no download.", flush=True)
            continue
        check_official_url(source["url"])
        request = Request(source["url"], headers={
            "User-Agent": "MicrosoftFinancialResultsExplorer/0.1 (educational source snapshot)",
            "Accept": "text/html,application/xhtml+xml",
        })
        with opener.open(request, timeout=30) as response:
            check_official_url(response.url)
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"{source['document_id']}: expected HTML, received {content_type}.")
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(data) > MAX_DOWNLOAD_BYTES or b"<html" not in data[:8192].lower():
                raise ValueError(f"{source['document_id']}: oversized or invalid HTML snapshot.")
            metadata = {
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": sha256(data),
                "local_path": path.relative_to(root).as_posix(),
                "byte_count": len(data),
                "content_type": content_type,
                "final_url": response.url,
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".html.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        source.update(metadata)
        write_json(root / "data/sources.json", manifest)
        print(f"{source['document_id']}: saved {len(data):,} bytes, SHA-256 {metadata['sha256'][:12]}...", flush=True)
    return manifest
