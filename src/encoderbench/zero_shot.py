"""Native MedGemma and BrainGemma3D zero-shot free-generation evaluation."""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from encoderbench.manifest import read_manifest
from encoderbench.metrics import binary_metrics, cluster_bootstrap, force_invalid_wrong
from encoderbench.parsing import format_rates, parse_response
from encoderbench.preprocessing import volume_to_slices, volume_to_tensor
from encoderbench.prompts import native_multimodal_content, native_text
from encoderbench.utils import checkpoint_identifier, ensure_parent, write_json


def _load_architecture(repository: Path):
    source = repository / "braingemma3d_architecture.py"
    spec = importlib.util.spec_from_file_location("encoderbench_native_braingemma", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_zero_shot(manifest_path: str | Path, disease: str, model_name: str,
                  config: dict[str, Any], output_dir: str | Path,
                  device: str = "cuda") -> dict[str, Any]:
    import torch

    rows = read_manifest(manifest_path, split="test")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompt = native_text(disease)
    generation = {"temperature": 0.0, "do_sample": False,
                  "max_new_tokens": int(config["evaluation"]["zero_shot_max_new_tokens"])}
    if model_name == "medgemma":
        generator, checkpoint = _medgemma_generator(config, device, disease, generation)
    elif model_name == "braingemma3d":
        generator, checkpoint = _braingemma_generator(config, device, prompt, generation)
    else:
        raise ValueError("model_name must be 'medgemma' or 'braingemma3d'")
    predictions = []
    for row in rows:
        started = time.monotonic()
        try:
            raw = generator(row["path"])
            parsed = parse_response(raw, disease)
        except Exception as exc:  # each inference failure remains auditable and is forced wrong
            raw = f"{type(exc).__name__}: {exc}"
            parsed = parse_response(None, disease, inference_error=True)
        predictions.append({"file_id": row["file_id"], "subject_id": row["subject_id"],
                            "true": row["group"], "prediction": parsed.prediction,
                            "matched": parsed.matched, "mode": parsed.mode,
                            "latency_seconds": round(time.monotonic() - started, 3), "raw": raw})
    prediction_path = ensure_parent(output / f"zero_shot_{model_name}_predictions.csv")
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    positive = "AD" if disease == "ad" else "SCZ"
    truth = np.asarray([row["true"] for row in predictions])
    raw_prediction = np.asarray([row["prediction"] for row in predictions])
    scored, invalid = force_invalid_wrong(truth, raw_prediction, positive)
    subjects = np.asarray([row["subject_id"] for row in predictions])
    metrics = binary_metrics(truth, scored, None, positive)
    confidence = cluster_bootstrap(truth, scored, None, subjects, positive,
                                   int(config["evaluation"]["bootstrap_samples"]),
                                   int(config["evaluation"]["bootstrap_seed"]))
    result = {"kind": "zero_shot", "disease": disease, "model": model_name,
              "checkpoint": checkpoint, "prompt": prompt, "generation": generation,
              "format_rates": format_rates(predictions), "forced_wrong_count": invalid,
              "metrics": metrics, "subject_cluster_ci": confidence,
              "predictions": str(prediction_path)}
    write_json(output / f"zero_shot_{model_name}_summary.json", result)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _medgemma_generator(config: dict[str, Any], device: str, disease: str,
                        generation: dict[str, Any]):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint = Path(config["checkpoints"]["medgemma"]).resolve()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to(device).eval().requires_grad_(False)
    data = config["data"]

    def generate(path: str) -> str:
        images = volume_to_slices(path, axis=data["axis"], count=data["num_slices"],
                                  bounds=data["slice_range"], roi_fraction=data["roi_fraction"])
        messages = [{"role": "user", "content": native_multimodal_content(disease, images)}]
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[rendered], images=[images], return_tensors="pt", padding=True)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False,
                                       max_new_tokens=generation["max_new_tokens"])
        continuation = generated[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(continuation, skip_special_tokens=True)[0]

    return generate, checkpoint_identifier(checkpoint)


def _braingemma_generator(config: dict[str, Any], device: str, prompt: str,
                          generation: dict[str, Any]):
    import torch

    root = Path(config["checkpoints"]["braingemma3d"]).resolve()
    architecture = _load_architecture(Path(config["source_repositories"]["braingemma3d"]).resolve())
    model = architecture.BrainGemma3D(
        vision_model_dir=str(root / "vision_model"),
        language_model_dir=str(root / "language_model"), depth=2, num_vision_tokens=32,
        freeze_vision=True, freeze_language=True, device_map={"": torch.device(device).index or 0},
    )
    projector = torch.load(root / "projector_vis_scale.pt", map_location=model.lm_device,
                           weights_only=False)
    missing, unexpected = model.vision_projector.load_state_dict(projector["vision_projector"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"BrainGemma3D projector mismatch: missing={missing}, unexpected={unexpected}")
    if projector.get("vis_scale") is not None:
        model.vis_scale.data.copy_(torch.as_tensor(projector["vis_scale"], device=model.lm_device))
    model.eval().requires_grad_(False)
    shape = config["data"]["volume_shapes"]["braingemma3d"]

    def generate(path: str) -> str:
        volume = volume_to_tensor(path, shape).unsqueeze(0)
        return model.generate_report(volume, prompt=prompt,
                                     max_new_tokens=generation["max_new_tokens"],
                                     min_new_tokens=1, temperature=0.0, top_p=1.0,
                                     repetition_penalty=1.0, no_repeat_ngram_size=0)

    audit = checkpoint_identifier(root)
    audit["projector_missing_keys"] = list(missing)
    audit["projector_unexpected_keys"] = list(unexpected)
    return generate, audit
