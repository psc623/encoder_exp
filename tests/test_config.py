from pathlib import Path

from encoderbench.config import load_config


def test_default_config_enforces_frozen_protocol():
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    assert config.raw["data"]["axis"] == 2
    assert config.raw["features"]["pooled_grid"] == [4, 4, 4]
    assert config.raw["bridge"]["micro_batch_size"] * config.raw["bridge"]["gradient_accumulation"] == 16

