"""Turning uploaded files into something a workflow can use.

Parsing is deliberately deterministic and bounded:

* **Deterministic**, because counting rows, finding blanks and spotting
  duplicates is arithmetic. Asking a model to do it would be slower, costlier
  and less reliable -- the same reason the policy engine holds the thresholds.
* **Bounded**, because a 50,000-row spreadsheet handed to an agent would
  exhaust its context and its budget. Parsing keeps a sample for the agent and
  computes statistics over the whole file in code.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from typing import Any

from cortexflow.modules.document.domain.models import (
    ColumnProfile,
    DocumentKind,
    NumericSummary,
    ParsedDocument,
)
from cortexflow.shared.errors import ValidationError
from cortexflow.shared.observability.logging import get_logger

logger = get_logger(__name__)

MAX_SAMPLE_ROWS = 50

MAX_TRIAGE_ROWS = 5_000
"""The most rows one step will judge individually.

Well above any file a person assembles by hand, and far below the size at
which holding the rows in memory becomes the problem.
"""
"""Rows retained for agents to read. Statistics still cover every row."""

MAX_TEXT_CHARS = 20_000
MAX_COLUMNS = 80
MAX_PDF_PAGES = 40
EXAMPLES_PER_COLUMN = 3

_EXTENSIONS: dict[str, DocumentKind] = {
    ".xlsx": DocumentKind.SPREADSHEET,
    ".xlsm": DocumentKind.SPREADSHEET,
    ".csv": DocumentKind.CSV,
    ".tsv": DocumentKind.CSV,
    ".json": DocumentKind.JSON,
    ".pdf": DocumentKind.PDF,
    ".txt": DocumentKind.TEXT,
    ".md": DocumentKind.TEXT,
}


def detect_kind(filename: str, content_type: str = "") -> DocumentKind:
    """Decide how to parse, from the extension first and the media type second."""
    lowered = filename.lower()
    for extension, kind in _EXTENSIONS.items():
        if lowered.endswith(extension):
            return kind

    if "spreadsheet" in content_type or "excel" in content_type:
        return DocumentKind.SPREADSHEET
    if "csv" in content_type:
        return DocumentKind.CSV
    if "json" in content_type:
        return DocumentKind.JSON
    if "pdf" in content_type:
        return DocumentKind.PDF
    return DocumentKind.TEXT


def parse(data: bytes, *, filename: str, content_type: str = "") -> ParsedDocument:
    """Parse an uploaded file into its bounded projection."""
    kind = detect_kind(filename, content_type)
    try:
        match kind:
            case DocumentKind.SPREADSHEET:
                return _parse_spreadsheet(data)
            case DocumentKind.CSV:
                return _parse_csv(data)
            case DocumentKind.JSON:
                return _parse_json(data)
            case DocumentKind.PDF:
                return _parse_pdf(data)
            case _:
                return _parse_text(data)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            "Could not read this file",
            filename=filename,
            kind=str(kind),
            error=f"{type(exc).__name__}: {exc}"[:200],
        ) from exc


_TRUE = {"true", "yes", "y", "1", "t"}
_FALSE = {"false", "no", "n", "0", "f"}


def coerce(value: Any) -> Any:
    """Turn a spreadsheet cell into something a rule can compare.

    Every cell arrives as text, and ``facts.amount > facts.limit`` against the
    string "18500" is either a type error or, worse, a silent lexicographic
    comparison. Blanks become None rather than 0, because a missing amount is
    not a free one.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False

    # Thousands separators and currency symbols are how people write numbers
    # in a spreadsheet; they should not make a row unjudgeable.
    cleaned = text.replace(",", "").replace("₹", "").replace("$", "").replace("£", "")
    cleaned = cleaned.replace("€", "").strip()
    try:
        number = float(cleaned)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def read_rows(
    data: bytes,
    *,
    filename: str,
    content_type: str = "",
    limit: int = MAX_TRIAGE_ROWS,
) -> tuple[list[dict[str, Any]], int]:
    """Read every row of a tabular file, up to ``limit``.

    ``parse`` keeps only a sample, because a document is stored as a whole and
    must stay small. Judging a file row by row needs all of it, so this reads
    from the original bytes instead -- bounded, because an unbounded read of an
    uploaded file is how one upload takes the process down.

    Returns the rows and the true total, which differ when the file is longer
    than ``limit``. A caller that does not report that difference is lying
    about what it examined.
    """
    kind = detect_kind(filename, content_type)
    try:
        match kind:
            case DocumentKind.SPREADSHEET:
                _, records = _spreadsheet_records(data)
            case DocumentKind.CSV:
                _, records = _csv_records(data)
            case DocumentKind.JSON:
                payload = json.loads(_decode(data))
                if not (
                    isinstance(payload, list)
                    and payload
                    and isinstance(payload[0], dict)
                ):
                    return [], 0
                _, records = _json_records(payload)
            case _:
                return [], 0
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            "Could not read this file",
            filename=filename,
            kind=str(kind),
            error=f"{type(exc).__name__}: {exc}"[:200],
        ) from exc

    return records[:limit], len(records)


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------
def _spreadsheet_records(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    """Every row of the first sheet, with no sample cap applied."""
    from openpyxl import load_workbook

    # read_only + data_only: stream the sheet and take computed values rather
    # than formula text, which is what a reviewer would actually see.
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            return [], []
        columns = _clean_header(header)
        records = [
            dict(zip(columns, _stringify_row(row, len(columns)), strict=False))
            for row in rows
            if any(cell is not None and str(cell).strip() for cell in row)
        ]
    finally:
        workbook.close()
    return columns, records


def _parse_spreadsheet(data: bytes) -> ParsedDocument:
    columns, records = _spreadsheet_records(data)
    if not columns:
        return ParsedDocument(
            kind=DocumentKind.SPREADSHEET,
            warnings=("The spreadsheet is empty.",),
        )
    return _build_tabular(DocumentKind.SPREADSHEET, columns, records)


def _csv_records(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    """Every row of a delimited file, with no sample cap applied."""
    text = _decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect)
    header = next(reader, None)
    if header is None:
        return [], []

    columns = _clean_header(header)
    records = [
        dict(zip(columns, _stringify_row(row, len(columns)), strict=False))
        for row in reader
        if any(cell.strip() for cell in row)
    ]
    return columns, records


def _parse_csv(data: bytes) -> ParsedDocument:
    columns, records = _csv_records(data)
    if not columns:
        return ParsedDocument(kind=DocumentKind.CSV, warnings=("The file is empty.",))
    return _build_tabular(DocumentKind.CSV, columns, records)


def _parse_json(data: bytes) -> ParsedDocument:
    payload = json.loads(_decode(data))

    # A list of objects is tabular; anything else is treated as a document.
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        columns, records = _json_records(payload)
        return _build_tabular(DocumentKind.JSON, columns, records)

    return ParsedDocument(
        kind=DocumentKind.JSON,
        text=json.dumps(payload, indent=2, default=str)[:MAX_TEXT_CHARS],
        row_count=1,
    )


def _json_records(payload: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Every object in a JSON array, with no sample cap applied."""
    columns: list[str] = []
    # The header is taken from a sample: a million-record file should not cost
    # a million dictionary scans just to learn its shape.
    for record in payload[:MAX_SAMPLE_ROWS]:
        for key in record:
            if key not in columns:
                columns.append(str(key))
    columns = columns[:MAX_COLUMNS]
    records = [
        {column: _stringify(record.get(column)) for column in columns}
        for record in payload
    ]
    return columns, records


def _build_tabular(
    kind: DocumentKind, columns: list[str], records: list[dict[str, Any]]
) -> ParsedDocument:
    profiles = tuple(_profile_column(name, records) for name in columns)
    sample = tuple(records[:MAX_SAMPLE_ROWS])

    warnings: list[str] = []
    if not records:
        warnings.append("The file has a header but no data rows.")
    for profile in profiles:
        if not profile.empty:
            continue
        if profile.non_empty:
            warnings.append(
                f"Column '{profile.name}' is missing {profile.empty} of "
                f"{profile.empty + profile.non_empty} values."
            )
        else:
            # A wholly empty column is more noteworthy than a patchy one, and
            # easy to miss if it is only reported as "missing N of N".
            warnings.append(f"Column '{profile.name}' is empty in every row.")

    return ParsedDocument(
        kind=kind,
        text=_tabular_text(columns, sample, len(records)),
        columns=profiles,
        rows=sample,
        row_count=len(records),
        truncated=len(records) > MAX_SAMPLE_ROWS,
        warnings=tuple(warnings[:20]),
    )


MAX_VALUE_COUNT_CARDINALITY = 12
"""Above this, listing every value is noise rather than a useful fact."""


def _profile_column(name: str, records: list[dict[str, Any]]) -> ColumnProfile:
    values = [str(record.get(name, "") or "").strip() for record in records]
    present = [v for v in values if v]
    inferred = _infer_type(present)

    # Aggregates are computed here, over *every* row, because the sample kept
    # for agents is deliberately partial. A policy engine comparing against a
    # threshold needs the real maximum, not the largest of the first fifty.
    numeric: NumericSummary | None = None
    if inferred == "number" and present:
        numbers = [float(v.replace(",", "")) for v in present]
        numeric = NumericSummary(
            minimum=min(numbers),
            maximum=max(numbers),
            total=round(sum(numbers), 4),
            mean=round(sum(numbers) / len(numbers), 4),
        )

    # Only where values actually repeat. An identifier column has few
    # distinct values relative to nothing at all -- every value occurs once --
    # and listing them is noise, not a fact worth acting on.
    counts: dict[str, int] = {}
    distinct = set(present)
    if present and len(distinct) <= MAX_VALUE_COUNT_CARDINALITY and len(distinct) < len(present):
        counts = dict(Counter(present).most_common())

    return ColumnProfile(
        name=name,
        non_empty=len(present),
        empty=len(values) - len(present),
        distinct=len(set(present)),
        inferred_type=inferred,
        examples=tuple(dict.fromkeys(present))[:EXAMPLES_PER_COLUMN],
        numeric=numeric,
        value_counts=counts,
    )


def _infer_type(values: list[str]) -> str:
    if not values:
        return "empty"
    sample = values[:200]
    if all(_looks_numeric(v) for v in sample):
        return "number"
    if all(_looks_date(v) for v in sample):
        return "date"
    if all("@" in v and "." in v for v in sample):
        return "email"
    return "string"


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ""))
    except ValueError:
        return False
    return True


def _looks_date(value: str) -> bool:
    from datetime import datetime

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            datetime.strptime(value[:10], fmt)
        except ValueError:
            continue
        return True
    return False


def _tabular_text(
    columns: list[str], rows: tuple[dict[str, Any], ...], total: int
) -> str:
    """Render a sample as readable text for agents."""
    lines = [f"Columns: {', '.join(columns)}", f"Total rows: {total}", ""]
    for index, row in enumerate(rows, start=1):
        rendered = "; ".join(f"{c}: {row.get(c, '')}" for c in columns if row.get(c))
        lines.append(f"Row {index}: {rendered}")
    if total > len(rows):
        lines.append(f"... and {total - len(rows)} further rows not shown.")
    return "\n".join(lines)[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def _parse_pdf(data: bytes) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages[:MAX_PDF_PAGES]
    extracted = [(page.extract_text() or "").strip() for page in pages]
    text = "\n\n".join(part for part in extracted if part)

    warnings: list[str] = []
    if len(reader.pages) > MAX_PDF_PAGES:
        warnings.append(
            f"Only the first {MAX_PDF_PAGES} of {len(reader.pages)} pages were read."
        )
    if not text.strip():
        # A scanned PDF needs OCR. Saying so beats handing an agent an empty
        # string and letting it invent content.
        warnings.append(
            "No text layer was found. This looks like a scanned document and "
            "needs OCR before it can be read."
        )

    return ParsedDocument(
        kind=DocumentKind.PDF,
        text=text[:MAX_TEXT_CHARS],
        row_count=len(reader.pages),
        truncated=len(reader.pages) > MAX_PDF_PAGES,
        warnings=tuple(warnings),
    )


def _parse_text(data: bytes) -> ParsedDocument:
    text = _decode(data)
    return ParsedDocument(
        kind=DocumentKind.TEXT,
        text=text[:MAX_TEXT_CHARS],
        row_count=text.count("\n") + 1 if text else 0,
        truncated=len(text) > MAX_TEXT_CHARS,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean_header(header: Any) -> list[str]:
    """Normalise header cells, keeping every column addressable."""
    columns: list[str] = []
    seen: Counter[str] = Counter()
    for index, cell in enumerate(list(header)[:MAX_COLUMNS]):
        name = str(cell).strip() if cell is not None else ""
        if not name:
            name = f"column_{index + 1}"
        seen[name] += 1
        # Duplicate headers are common in exported spreadsheets; suffix rather
        # than silently dropping a column.
        columns.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return columns


def _stringify_row(row: Any, width: int) -> list[str]:
    values = [_stringify(cell) for cell in list(row)[:width]]
    values.extend([""] * (width - len(values)))
    return values


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
