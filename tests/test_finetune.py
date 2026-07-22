import torch
from torch import nn

from encoderbench.finetune import configure_parameter_budget


class DummyExtractor:
    name = "dummy"

    def __init__(self):
        self.model = nn.ModuleDict({
            "early": nn.Linear(10, 10, bias=False),
            "middle": nn.Linear(10, 10, bias=False),
            "late": nn.Linear(10, 10, bias=False),
            "norm": nn.LayerNorm(10),
        })

    def finetune_groups(self):
        return [
            ("late", self.model["late"]),
            ("middle", self.model["middle"]),
            ("early", self.model["early"]),
        ], [("norm", self.model["norm"])]


def test_parameter_budget_selects_complete_output_groups():
    extractor = DummyExtractor()
    audit = configure_parameter_budget(extractor, 220)
    assert [item["name"] for item in audit["selected_groups"]] == ["norm", "late", "middle"]
    assert audit["trainable_encoder_parameters"] == 220
    assert extractor.model["late"].weight.requires_grad
    assert extractor.model["middle"].weight.requires_grad
    assert not extractor.model["early"].weight.requires_grad


def test_budget_never_splits_the_first_primary_group():
    extractor = DummyExtractor()
    audit = configure_parameter_budget(extractor, 50)
    assert [item["name"] for item in audit["selected_groups"]] == ["norm", "late"]
    assert audit["trainable_encoder_parameters"] == 120
