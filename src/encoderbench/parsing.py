"""Pre-registered native zero-shot response parser."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedResponse:
    prediction: str
    matched: bool
    mode: str


def parse_response(text: str | None, disease: str, inference_error: bool = False) -> ParsedResponse:
    if inference_error:
        return ParsedResponse("ERR", False, "error")
    if disease not in ("ad", "scz"):
        raise ValueError("disease must be 'ad' or 'scz'")
    positive = "AD" if disease == "ad" else "SCZ"
    if not text or not text.strip():
        return ParsedResponse("UNK", False, "unparsed")
    labels = f"{positive}|CN"
    exact = re.search(rf"final\s*answer\s*[:\-]?\s*({labels})\b", text, re.IGNORECASE)
    if exact:
        return ParsedResponse(exact.group(1).upper(), True, "exact")
    abbreviations = re.findall(rf"\b({labels})\b", text, re.IGNORECASE)
    if abbreviations:
        return ParsedResponse(abbreviations[-1].upper(), False, "abbreviation")
    lowered = text.lower()
    if disease == "ad" and "alzheimer" in lowered and "normal" not in lowered:
        return ParsedResponse("AD", False, "semantic")
    if disease == "scz" and ("schizophrenia" in lowered or "schizophrenic" in lowered) \
            and "normal" not in lowered:
        return ParsedResponse("SCZ", False, "semantic")
    if "normal" in lowered or "no evidence" in lowered:
        return ParsedResponse("CN", False, "semantic")
    return ParsedResponse("UNK", False, "unparsed")


def format_rates(rows: list[dict[str, object]]) -> dict[str, float]:
    count = len(rows)
    if count == 0:
        return {key: 0.0 for key in ("exact_format_rate", "valid_fallback_rate",
                                     "nonexact_format_rate", "unk_err_rate")}
    exact = sum(bool(row.get("matched")) for row in rows)
    fallback = sum(row.get("mode") in ("abbreviation", "semantic") for row in rows)
    invalid = sum(row.get("prediction") in ("UNK", "ERR") for row in rows)
    return {"exact_format_rate": exact / count, "valid_fallback_rate": fallback / count,
            "nonexact_format_rate": (count - exact) / count, "unk_err_rate": invalid / count}

