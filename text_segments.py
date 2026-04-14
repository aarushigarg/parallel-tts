from __future__ import annotations

from pathlib import Path

SEGMENT_BOUNDARY_PUNCTUATION = set("。！？!?、，,；;：:」』）)]】〉》\n")


def split_text_segments(text: str, *, min_chars: int = 12, max_chars: int = 180) -> list[str]:
    cleaned = str(text).strip()
    if cleaned == "":
        return []

    raw_segments: list[str] = []
    current: list[str] = []
    for ch in cleaned:
        current.append(ch)
        if ch in SEGMENT_BOUNDARY_PUNCTUATION:
            segment = "".join(current).strip()
            if segment:
                raw_segments.append(segment)
            current = []
    tail = "".join(current).strip()
    if tail:
        raw_segments.append(tail)

    if not raw_segments:
        raw_segments = [cleaned]

    sized_segments: list[str] = []
    for segment in raw_segments:
        if max_chars <= 0 or len(segment) <= max_chars:
            sized_segments.append(segment)
            continue
        for start in range(0, len(segment), max_chars):
            piece = segment[start : start + max_chars].strip()
            if piece:
                sized_segments.append(piece)

    merged_segments: list[str] = []
    pending = ""
    for segment in sized_segments:
        pending = f"{pending}{segment}" if pending else segment
        if len(pending) >= min_chars:
            merged_segments.append(pending)
            pending = ""
    if pending:
        if merged_segments:
            merged_segments[-1] = f"{merged_segments[-1]}{pending}"
        else:
            merged_segments.append(pending)

    return merged_segments


def load_text_for_segmentation(
    *,
    text: str | None = None,
    text_file: str | Path | None = None,
    text_file_lines: int | None = None,
) -> str:
    if text is not None and str(text).strip() != "":
        return str(text)
    if text_file is None:
        raise ValueError("Either text or text_file is required.")

    path = Path(text_file).expanduser()
    max_lines = None if text_file_lines is None else int(text_file_lines)
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("|")
            if len(parts) == 4:
                row_text = parts[1].strip()
            else:
                row_text = stripped
            if row_text:
                texts.append(row_text)
            if max_lines is not None and len(texts) >= max_lines:
                break

    if not texts:
        raise ValueError(f"No text found in {path}.")
    return "".join(texts)
