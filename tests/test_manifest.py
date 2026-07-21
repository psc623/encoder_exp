import csv
from collections import Counter

from encoderbench.manifest import carve_validation, stratified_subject_split, validate_manifest


def _rows(cn: int = 125, scz: int = 50):
    rows = []
    for label, count in (("CN", cn), ("SCZ", scz)):
        for index in range(count):
            subject = f"{label}-{index:03d}"
            rows.append({"path": f"/{subject}.nii.gz", "file_id": subject,
                         "subject_id": subject, "group": label, "is_repeat": 0, "split": ""})
    return rows


def test_ucla_expected_two_level_counts():
    rows = stratified_subject_split(_rows(), 0.5, 0)
    rows = carve_validation(rows, 0.15, 0)
    counts = Counter((row["split"], row["group"]) for row in rows)
    assert counts == Counter({("test", "CN"): 62, ("test", "SCZ"): 25,
                              ("validation", "CN"): 9, ("validation", "SCZ"): 4,
                              ("train", "CN"): 54, ("train", "SCZ"): 21})
    assert validate_manifest(rows, "scz")["subjects"] == 175


def test_repeat_scans_never_cross_splits():
    rows = _rows(cn=8, scz=8)
    repeat = dict(rows[0])
    repeat["file_id"] += "_2"
    repeat["is_repeat"] = 1
    rows.append(repeat)
    rows = carve_validation(stratified_subject_split(rows), 0.15, 0)
    subject_splits = {row["split"] for row in rows if row["subject_id"] == repeat["subject_id"]}
    assert len(subject_splits) == 1

