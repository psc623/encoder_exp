import pytest

from encoderbench.parsing import format_rates, parse_response


@pytest.mark.parametrize(
    ("text", "disease", "prediction", "matched", "mode"),
    [
        ("Reasoning\nFinal Answer: AD", "ad", "AD", True, "exact"),
        ("final answer - cn", "ad", "CN", True, "exact"),
        ("AD was considered, but CN", "ad", "CN", False, "abbreviation"),
        ("Findings suggest Alzheimer disease", "ad", "AD", False, "semantic"),
        ("No evidence of a disorder", "scz", "CN", False, "semantic"),
        ("The schizophrenia cohort is favored", "scz", "SCZ", False, "semantic"),
        ("cannot decide", "scz", "UNK", False, "unparsed"),
    ],
)
def test_hierarchical_parser(text, disease, prediction, matched, mode):
    parsed = parse_response(text, disease)
    assert (parsed.prediction, parsed.matched, parsed.mode) == (prediction, matched, mode)


def test_format_rate_definitions():
    rows = [{"matched": True, "mode": "exact", "prediction": "AD"},
            {"matched": False, "mode": "semantic", "prediction": "CN"},
            {"matched": False, "mode": "unparsed", "prediction": "UNK"}]
    rates = format_rates(rows)
    assert rates == {"exact_format_rate": 1 / 3, "valid_fallback_rate": 1 / 3,
                     "nonexact_format_rate": 2 / 3, "unk_err_rate": 1 / 3}
