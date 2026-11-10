"""Extract a bounded, traceable corpus without embedding or provider calls.

The adapters deliberately target the two approved Microsoft publications.
Tables retain their merged column headings rather than becoming loose numbers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

from bs4 import BeautifulSoup, Tag

from financial_rag.sources import ROOT, read_manifest, sha256, verify_snapshot, write_json


PARSER_VERSION = "microsoft-html-v1"
ANNUAL_SECTIONS = {
    "discussion-analysis": "Management Discussion and Analysis",
    "income-statements": "Income Statements",
    "cash-flows-statements": "Cash Flows Statements",
    "segment-information-geo-data": "Note 18 Segment Information and Geographic Data",
}
EARNINGS_HEADINGS = {
    "business highlights": "Business Highlights",
    "fiscal year 2025 results": "Fiscal Year 2025 Results",
    "constant currency": "Constant Currency",
    "financial performance constant currency reconciliation": "Financial Performance Constant Currency Reconciliation",
    "segment revenue constant currency reconciliation": "Segment Revenue Constant Currency Reconciliation",
    "selected product and service revenue constant currency reconciliation": "Selected Product and Service Revenue Constant Currency Reconciliation",
}
EARNINGS_EXCLUSIONS = {
    "business outlook", "quarterly highlights, product releases, and enhancements",
    "webcast details", "about microsoft", "forward-looking statements",
}
NUMBER = re.compile(r"^\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?(?:%|ppt)?$", re.I)
YEARS = re.compile(r"\b20\d{2}\b")
UNITS = re.compile(r"\([^)]*(?:millions|billions|per share|percentages)[^)]*\)", re.I)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\$\s+(?=[\d(])", "$", text)


def text_of(tag: Tag) -> str:
    return clean_text(tag.get_text(" ", strip=True))


def expand_table(table: Tag) -> tuple[list[list[dict | None]], list[Tag]]:
    """Expand rowspan/colspan into a grid while retaining each cell's origin."""
    rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
    occupied: dict[tuple[int, int], dict] = {}
    for row_index, row in enumerate(rows):
        column_index = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (row_index, column_index) in occupied:
                column_index += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            if not 1 <= rowspan <= 100 or not 1 <= colspan <= 100:
                raise ValueError("Unsupported table span; inspect the source before ingestion.")
            item = {
                "text": text_of(cell), "origin": [row_index, column_index],
                "rowspan": rowspan, "colspan": colspan,
                "header_ids": list(cell.get("headers", [])),
            }
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    key = (row_index + row_offset, column_index + column_offset)
                    if key in occupied:
                        raise ValueError("Overlapping HTML table cells.")
                    occupied[key] = item
            column_index += colspan
    width = max((col for _, col in occupied), default=-1) + 1
    if width > 100 or len(rows) > 2000:
        raise ValueError("Table exceeds the bounded parser limits.")
    return [[occupied.get((row, col)) for col in range(width)] for row in range(len(rows))], rows


def unique_cells(row: list[dict | None]) -> list[dict]:
    seen = set()
    result = []
    for item in row:
        if item is not None and tuple(item["origin"]) not in seen:
            seen.add(tuple(item["origin"]))
            result.append(item)
    return result


def has_measures(row: list[dict | None]) -> bool:
    values = [cell["text"] for cell in unique_cells(row) if cell["origin"][1] > 0 and NUMBER.fullmatch(cell["text"])]
    # A row of bare years is a header; a monetary value such as $2,025 is not.
    return bool(values) and not all(re.fullmatch(r"20\d{2}", value) for value in values)


def header_count(grid: list[list[dict | None]], rows: list[Tag]) -> int:
    explicit = [i for i, row in enumerate(rows) if row.find_parent("thead") is not None]
    if explicit:
        return max(explicit) + 1
    for i, row in enumerate(grid):
        if has_measures(row):
            if i == 0:
                raise ValueError("A numeric table has no identifiable header rows.")
            # A section label can sit between the true column headings and the
            # first numeric row. Keep it in the body so its group is preserved.
            while i > 0:
                nonempty = [cell for cell in unique_cells(grid[i - 1]) if cell["text"]]
                if len(nonempty) != 1:
                    break
                label = nonempty[0]["text"]
                if UNITS.fullmatch(label) or re.match(r"(?:Year Ended|June 30|Three Months Ending)", label, re.I):
                    break
                i -= 1
            return i
    # MD&A also includes two-column metric definitions, with no numeric values.
    return 0


def serialize_table(table: Tag, section: str) -> dict:
    grid, rows = expand_table(table)
    count = header_count(grid, rows)
    caption_tag = table.find("caption")
    caption = text_of(caption_tag) if caption_tag else section
    header_texts = [cell["text"] for row in grid[:count] for cell in unique_cells(row)]
    units = list(dict.fromkeys(UNITS.findall(" ".join([caption] + header_texts))))
    unit_text = "; ".join(units) or "Units not explicitly stated in this table."
    by_id = {tag["id"]: text_of(tag) for tag in table.find_all(id=True)}
    columns = []
    for col in range(len(grid[0]) if grid else 0):
        labels = []
        for row in grid[:count]:
            item = row[col]
            if item and item["text"] and not UNITS.fullmatch(item["text"]):
                if item["text"] not in labels:
                    labels.append(item["text"])
        # In annual statements the shared period label occupies the left cell.
        if col > 0:
            for row in grid[:count]:
                first = row[0]
                if first and re.match(r"(?:Year Ended|June 30|Three Months Ending)", first["text"], re.I):
                    if first["text"] not in labels:
                        labels.insert(0, first["text"])
        columns.append(" / ".join(labels))
    records = []
    group = ""
    for row_index in range(count, len(grid)):
        cells = unique_cells(grid[row_index])
        if not cells:
            continue
        left = grid[row_index][0]
        label = left["text"] if left else ""
        values = []
        for cell in cells:
            col = cell["origin"][1]
            if col == 0 or not cell["text"]:
                continue
            explicit_labels = [by_id[key] for key in cell["header_ids"] if by_id.get(key)]
            heading = " / ".join(dict.fromkeys(explicit_labels)) or columns[col]
            if not heading:
                heading = "Description" if count == 0 else f"Unlabelled source column {col + 1}"
            values.append({"column": heading, "value": cell["text"], "source_column": col})
        if not values:
            if label:
                # Cash-flow subheadings qualify adjustments within Operations;
                # they must not replace that activity for its eventual total.
                if not label.startswith(("Adjustments to reconcile", "Changes in operating assets")):
                    group = label
            continue
        if not label:
            raise ValueError("A data row has no row label; refusing to detach financial values.")
        row_group = group
        restricted_groups = {
            "Revenue:": {"Product", "Service and other", "Total revenue"},
            "Cost of revenue:": {"Product", "Service and other", "Total cost of revenue"},
            "Earnings per share:": {"Basic", "Diluted"},
            "Weighted average shares outstanding:": {"Basic", "Diluted"},
        }
        if group in restricted_groups and label not in restricted_groups[group]:
            row_group = ""
        row_label = " / ".join(part for part in [row_group, label] if part)
        text = f"{caption}. Units: {unit_text}. Row: {row_label}. "
        text += "; ".join(f"{value['column']}: {value['value']}" for value in values)
        records.append({
            "row_index": row_index, "row_label": label, "row_group": row_group,
            "values": values, "text": text,
            "period_labels": sorted(set(YEARS.findall(" ".join([row_label] + [v["column"] for v in values])))),
        })
        if label.casefold().startswith(("net cash from operations", "net cash used in financing", "net cash used in investing")):
            group = ""
    return {
        "caption": caption, "units": unit_text, "header_row_count": count,
        "columns": columns,
        "grid": [[cell["text"] if cell else "" for cell in row] for row in grid],
        "rows": records,
    }


def evidence_block(source: dict, section_id: str, section: str, locator: str, text: str, kind: str, **details) -> dict:
    text = clean_text(text)
    digest = sha256(f"{source['document_id']}|{section_id}|{locator}|{text}".encode("utf-8"))[:16]
    return {
        "evidence_id": f"{source['document_id']}:{section_id}:{digest}",
        "document_id": source["document_id"], "source_title": source["title"],
        "source_url": source["url"], "source_sha256": source["sha256"],
        "source_period_note": source["period_note"],
        "section_id": section_id, "section": section, "locator": locator,
        "kind": kind, "text": text,
        "period_labels": sorted(set(YEARS.findall(text))),
        **details,
    }


def extract_source(source: dict, data: bytes) -> tuple[list[dict], list[dict], dict]:
    soup = BeautifulSoup(data, "html.parser", from_encoding="utf-8")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "form", "svg"]):
        tag.decompose()
    if "\ufffd" in soup.get_text():
        raise ValueError(f"{source['document_id']}: source decoding produced replacement characters.")
    if source["document_id"] == "AR25":
        containers = [(key, title, soup.find(id=key)) for key, title in ANNUAL_SECTIONS.items()]
    else:
        containers = [("pressreleasecontent", "Financial Summary", soup.find(id="pressreleasecontent"))]
    if any(container is None for _, _, container in containers):
        raise ValueError(f"{source['document_id']}: required financial section is absent.")
    blocks = []
    tables = []
    excluded_tables = []
    excluded_narrative = 0
    for section_id, base_title, container in containers:
        current_title = base_title
        active = True
        table_index = 0
        for ordinal, tag in enumerate(container.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"])):
            if tag.find_parent("table"):
                continue
            text = text_of(tag)
            if not text:
                continue
            if source["document_id"] == "AR25" and section_id == "income-statements":
                if tag.name == "h2" and "COMPREHENSIVE" in text.upper():
                    break
            if tag.name == "table":
                table_index += 1
                caption_tag = tag.find("caption")
                caption = text_of(caption_tag) if caption_tag else ""
                title = current_title
                if source["document_id"] == "ER25Q4":
                    if caption:
                        upper = caption.upper()
                        if "COMPREHENSIVE" in upper or "BALANCE SHEETS" in upper:
                            excluded_tables.append(caption)
                            continue
                        if not any(label in upper for label in ["INCOME STATEMENTS", "CASH FLOWS", "SEGMENT REVENUE"]):
                            raise ValueError("Unrecognized earnings-release table caption.")
                        title = caption
                    elif not active:
                        excluded_tables.append(f"Table {table_index}: outside selected sections")
                        continue
                parsed = serialize_table(tag, title)
                if not parsed["rows"]:
                    raise ValueError("A selected table produced no evidence rows.")
                table_id = f"{source['document_id']}:{section_id}:table-{table_index}"
                table_record = {"table_id": table_id, "document_id": source["document_id"], "source_url": source["url"], "section": title, **parsed}
                tables.append(table_record)
                for row in parsed["rows"]:
                    blocks.append(evidence_block(
                        source, section_id, title, f"table-{table_index}/row-{row['row_index']}", row["text"], "table_row",
                        table_id=table_id, row_index=row["row_index"], row_label=row["row_label"], row_group=row["row_group"],
                        column_values=row["values"], units=parsed["units"], period_labels=row["period_labels"],
                    ))
                continue
            if tag.name.startswith("h"):
                current_title = base_title if text.casefold() == base_title.casefold() else f"{base_title} / {text}"
                continue
            if tag.name == "p" and tag.find_parent("li"):
                continue
            if tag.name == "li" and tag.find_parent("li"):
                continue
            if source["document_id"] == "ER25Q4":
                heading = text.casefold()
                if heading in EARNINGS_HEADINGS:
                    current_title = EARNINGS_HEADINGS[heading]
                    active = True
                    continue
                if heading in EARNINGS_EXCLUSIONS or heading.startswith("for more information") or heading.startswith("note to editors:"):
                    active = False
                    continue
            if not active:
                excluded_narrative += 1
                continue
            if len(text.split()) < 3:
                continue
            blocks.append(evidence_block(source, section_id, current_title, f"text-{ordinal}", text, "paragraph"))
    if not blocks or not tables:
        raise ValueError(f"{source['document_id']}: no usable financial corpus extracted.")
    return blocks, tables, {"excluded_tables": excluded_tables, "excluded_narrative_blocks": excluded_narrative}


def ingest(root: Path = ROOT) -> dict:
    manifest = read_manifest(root)
    all_blocks, all_tables, source_reports = [], [], []
    for source in manifest["sources"]:
        data = verify_snapshot(source, root)
        blocks, tables, exclusions = extract_source(source, data)
        words = sum(len(block["text"].split()) for block in blocks)
        report = {
            "document_id": source["document_id"], "source_sha256": source["sha256"],
            "block_count": len(blocks), "table_count": len(tables), "word_count": words,
            "kind_counts": dict(Counter(block["kind"] for block in blocks)),
            "sections": sorted(set(block["section"] for block in blocks)), **exclusions,
        }
        source_reports.append(report)
        all_blocks.extend(blocks)
        all_tables.extend(tables)
    identifiers = [block["evidence_id"] for block in all_blocks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate evidence IDs.")
    total_words = sum(report["word_count"] for report in source_reports)
    if total_words > manifest["max_selected_words"]:
        raise ValueError(f"Selected corpus is {total_words:,} words, over the configured cap. Narrow sections explicitly; do not truncate silently.")
    serialized = "".join(json.dumps(block, ensure_ascii=False, sort_keys=True) + "\n" for block in all_blocks)
    summary = {
        "schema_version": 1, "parser_version": PARSER_VERSION,
        "corpus_id": manifest["corpus_id"], "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": sha256(serialized.encode("utf-8")), "block_count": len(all_blocks),
        "table_count": len(all_tables), "word_count": total_words,
        "sources": source_reports, "embeddings_created": False, "vectors_written": False,
    }
    output = root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "corpus.jsonl.tmp"
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(output / "corpus.jsonl")
    write_json(output / "tables.json", all_tables)
    write_json(output / "summary.json", summary)
    preview = ["# Extracted financial evidence", "", "Generated from verified source snapshots. No model calls.", ""]
    for block in all_blocks:
        preview.extend([f"## {block['evidence_id']}", f"Source: {block['source_title']} | Section: {block['section']} | Locator: {block['locator']}", "", block["text"], ""])
    (output / "preview.md").write_text("\n".join(preview), encoding="utf-8")
    for source, report in zip(manifest["sources"], source_reports):
        source["ingestion"] = {"parser_version": PARSER_VERSION, "block_count": report["block_count"], "table_count": report["table_count"], "word_count": report["word_count"]}
    write_json(root / "data/sources.json", manifest)
    return summary
