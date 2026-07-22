"""Native MedGemma and BrainGemma3D zero-shot free-generation evaluation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from encoderbench.manifest import read_manifest
from encoderbench.metrics import binary_metrics, cluster_bootstrap, force_invalid_wrong
from encoderbench.parsing import format_rates, parse_response
from encoderbench.preprocessing import volume_to_slices, volume_to_tensor
from encoderbench.prompts import native_multimodal_content, native_text
from encoderbench.utils import checkpoint_identifier, ensure_parent, sha256_file


PREDICTION_FIELDS = (
    "file_id", "subject_id", "true", "prediction", "matched", "mode",
    "latency_seconds", "raw",
)
_FINAL_ANSWER = {
    "ad": re.compile(r"final\s+answer\s*[:\-]\s*(?:AD|CN)\b", re.IGNORECASE),
    "scz": re.compile(r"final\s+answer\s*[:\-]\s*(?:SCZ|CN)\b", re.IGNORECASE),
}


def _load_architecture(repository: Path):
    source = repository / "braingemma3d_architecture.py"
    spec = importlib.util.spec_from_file_location("encoderbench_native_braingemma", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contains_final_answer(text: str, disease: str) -> bool:
    return bool(_FINAL_ANSWER[disease].search(text))


def _stopping_criteria(tokenizer: Any, disease: str, prefix_length: int = 0):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class FinalAnswerStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001, ANN003
            stopped = []
            for sequence in input_ids:
                generated = sequence[prefix_length:]
                # The answer marker is short; limiting the decode keeps this check inexpensive.
                text = tokenizer.decode(generated[-64:], skip_special_tokens=True)
                stopped.append(_contains_final_answer(text, disease))
            return torch.tensor(stopped, device=input_ids.device, dtype=torch.bool)

    return StoppingCriteriaList([FinalAnswerStoppingCriteria()])


def _shard_rows(rows: list[dict[str, str]], num_shards: int,
                shard_index: int) -> list[dict[str, str]]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return rows[shard_index::num_shards]


def _artifact_paths(output: Path, model_name: str, num_shards: int,
                    shard_index: int) -> tuple[Path, Path]:
    base = f"zero_shot_{model_name}"
    if num_shards > 1:
        base += f".shard_{shard_index:03d}_of_{num_shards:03d}"
    return output / f"{base}_predictions.csv", output / f"{base}_state.json"


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    output = ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                     dir=output.parent, prefix=f".{output.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def _atomic_json(path: Path, value: Any) -> Path:
    output = ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise ValueError(f"Prediction file has unexpected columns: {path}")
        rows = []
        for row in reader:
            item: dict[str, Any] = dict(row)
            item["matched"] = str(item["matched"]).lower() == "true"
            item["latency_seconds"] = float(item["latency_seconds"])
            rows.append(item)
    return rows


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["file_id"]), str(row["subject_id"])


def _protocol(manifest_path: str | Path, disease: str, model_name: str,
              config: dict[str, Any], num_shards: int) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    generation = {"temperature": 0.0, "do_sample": False,
                  "max_new_tokens": int(config["evaluation"]["zero_shot_max_new_tokens"]),
                  "stop_on_final_answer": True}
    checkpoint_path = config["checkpoints"]["medgemma" if model_name == "medgemma"
                                                    else "braingemma3d"]
    precision = "bf16"
    return {
        "schema_version": 2,
        "disease": disease,
        "model": model_name,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "prompt": native_text(disease),
        "generation": generation,
        "precision": precision,
        "attention_implementation": "sdpa",
        "checkpoint": checkpoint_identifier(checkpoint_path),
        "num_shards": num_shards,
    }


def _protocol_digest(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_resume(prediction_path: Path, state_path: Path, protocol: dict[str, Any],
                    shard_rows: list[dict[str, str]], shard_index: int,
                    restart: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if restart:
        prediction_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    if prediction_path.exists() and not state_path.exists():
        raise ValueError(
            f"Legacy prediction file has no resumable state: {prediction_path}. "
            "Use --restart to replace it under the current protocol."
        )
    digest = _protocol_digest(protocol)
    expected_keys = [_row_key(row) for row in shard_rows]
    if state_path.exists():
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("protocol_sha256") != digest:
            raise ValueError(
                f"Existing shard state uses a different protocol: {state_path}. "
                "Use --restart to replace that shard."
            )
        if state.get("shard_index") != shard_index:
            raise ValueError(f"Shard state index mismatch in {state_path}")
    else:
        state = {
            "protocol": protocol,
            "protocol_sha256": digest,
            "shard_index": shard_index,
            "expected_rows": len(shard_rows),
            "complete": False,
        }
        _atomic_json(state_path, state)

    predictions = _read_predictions(prediction_path)
    expected = set(expected_keys)
    seen: set[tuple[str, str]] = set()
    truth_by_key = {_row_key(row): row["group"] for row in shard_rows}
    for prediction in predictions:
        key = _row_key(prediction)
        if key not in expected:
            raise ValueError(f"Prediction {key} does not belong to shard {shard_index}")
        if key in seen:
            raise ValueError(f"Duplicate prediction {key} in {prediction_path}")
        if prediction["true"] != truth_by_key[key]:
            raise ValueError(f"Prediction label mismatch for {key} in {prediction_path}")
        seen.add(key)
    if state.get("complete") and len(predictions) != len(shard_rows):
        raise ValueError(f"State marks an incomplete prediction file as complete: {state_path}")
    return predictions, state


def _summarize(predictions: list[dict[str, Any]], disease: str, model_name: str,
               protocol: dict[str, Any], evaluation: dict[str, Any],
               prediction_path: Path) -> dict[str, Any]:
    positive = "AD" if disease == "ad" else "SCZ"
    truth = np.asarray([row["true"] for row in predictions])
    raw_prediction = np.asarray([row["prediction"] for row in predictions])
    scored, invalid = force_invalid_wrong(truth, raw_prediction, positive)
    subjects = np.asarray([row["subject_id"] for row in predictions])
    metrics = binary_metrics(truth, scored, None, positive)
    confidence = cluster_bootstrap(
        truth, scored, None, subjects, positive,
        int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"]),
    )
    return {
        "kind": "zero_shot", "disease": disease, "model": model_name,
        "checkpoint": protocol["checkpoint"], "prompt": protocol["prompt"],
        "generation": protocol["generation"], "precision": protocol["precision"],
        "attention_implementation": protocol["attention_implementation"],
        "sharding": {"num_shards": protocol["num_shards"], "merged": True},
        "format_rates": format_rates(predictions), "forced_wrong_count": invalid,
        "metrics": metrics, "subject_cluster_ci": confidence,
        "predictions": str(prediction_path),
    }


def _merge_shards(output: Path, rows: list[dict[str, str]], disease: str, model_name: str,
                  protocol: dict[str, Any], evaluation: dict[str, Any],
                  strict: bool) -> dict[str, Any] | None:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    digest = _protocol_digest(protocol)
    for shard_index in range(protocol["num_shards"]):
        prediction_path, state_path = _artifact_paths(
            output, model_name, protocol["num_shards"], shard_index
        )
        if not prediction_path.is_file() or not state_path.is_file():
            if strict:
                raise ValueError(f"Shard {shard_index} is not available for merging")
            return None
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        shard_expected = _shard_rows(rows, protocol["num_shards"], shard_index)
        predictions = _read_predictions(prediction_path)
        if (state.get("protocol_sha256") != digest or not state.get("complete")
                or len(predictions) != len(shard_expected)):
            if strict:
                raise ValueError(f"Shard {shard_index} is incomplete or uses another protocol")
            return None
        expected = {_row_key(row): row["group"] for row in shard_expected}
        observed = {_row_key(row): row["true"] for row in predictions}
        if observed != expected:
            raise ValueError(f"Shard {shard_index} rows do not match the manifest")
        for prediction in predictions:
            key = _row_key(prediction)
            if key in merged:
                raise ValueError(f"Duplicate prediction across shards: {key}")
            merged[key] = prediction

    expected_keys = [_row_key(row) for row in rows]
    missing = [key for key in expected_keys if key not in merged]
    if missing:
        raise ValueError(f"Merged predictions are missing {len(missing)} rows")
    ordered = [merged[key] for key in expected_keys]
    canonical_prediction, _ = _artifact_paths(output, model_name, 1, 0)
    _atomic_csv(canonical_prediction, ordered)
    result = _summarize(ordered, disease, model_name, protocol,
                        evaluation, canonical_prediction)
    _atomic_json(output / f"zero_shot_{model_name}_summary.json", result)
    return result


def run_zero_shot(manifest_path: str | Path, disease: str, model_name: str,
                  config: dict[str, Any], output_dir: str | Path,
                  device: str = "cuda", num_shards: int = 1, shard_index: int = 0,
                  restart: bool = False, merge_only: bool = False) -> dict[str, Any]:
    import torch

    rows = read_manifest(manifest_path, split="test")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(manifest_path, disease, model_name, config, num_shards)
    if merge_only:
        if num_shards == 1:
            raise ValueError("--merge-only requires --num-shards greater than one")
        result = _merge_shards(output, rows, disease, model_name, protocol,
                               config["evaluation"], strict=True)
        assert result is not None
        return result

    assigned_rows = _shard_rows(rows, num_shards, shard_index)
    prediction_path, state_path = _artifact_paths(output, model_name, num_shards, shard_index)
    predictions, state = _prepare_resume(
        prediction_path, state_path, protocol, assigned_rows, shard_index, restart
    )
    completed = {_row_key(row) for row in predictions}
    pending = [row for row in assigned_rows if _row_key(row) not in completed]

    generator: Callable[[str], str] | None = None
    if pending:
        if model_name == "medgemma":
            generator = _medgemma_generator(config, device, disease, protocol["generation"])
        elif model_name == "braingemma3d":
            generator = _braingemma_generator(config, device, disease, protocol["prompt"],
                                               protocol["generation"])
        else:
            raise ValueError("model_name must be 'medgemma' or 'braingemma3d'")

        from tqdm.auto import tqdm

        progress = tqdm(pending, initial=len(predictions), total=len(assigned_rows),
                        desc=f"{model_name} shard {shard_index + 1}/{num_shards}", unit="scan")
        for row in progress:
            started = time.monotonic()
            try:
                raw = generator(row["path"])
                parsed = parse_response(raw, disease)
            except Exception as exc:  # each inference failure remains auditable and is forced wrong
                raw = f"{type(exc).__name__}: {exc}"
                parsed = parse_response(None, disease, inference_error=True)
            latency = round(time.monotonic() - started, 3)
            predictions.append({
                "file_id": row["file_id"], "subject_id": row["subject_id"],
                "true": row["group"], "prediction": parsed.prediction,
                "matched": parsed.matched, "mode": parsed.mode,
                "latency_seconds": latency, "raw": raw,
            })
            _atomic_csv(prediction_path, predictions)
            progress.set_postfix(latency=f"{latency:.1f}s", mode=parsed.mode)

    state["complete"] = True
    state["completed_rows"] = len(predictions)
    _atomic_json(state_path, state)
    if generator is not None:
        del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if num_shards > 1:
        merged = _merge_shards(output, rows, disease, model_name, protocol,
                               config["evaluation"], strict=False)
        if merged is not None:
            return merged
        return {
            "kind": "zero_shot_shard", "disease": disease, "model": model_name,
            "shard_index": shard_index, "num_shards": num_shards,
            "completed_rows": len(predictions), "predictions": str(prediction_path),
            "status": "complete; waiting for remaining shards",
        }

    by_key = {_row_key(row): row for row in predictions}
    predictions = [by_key[_row_key(row)] for row in assigned_rows]
    _atomic_csv(prediction_path, predictions)
    result = _summarize(predictions, disease, model_name, protocol,
                        config["evaluation"], prediction_path)
    _atomic_json(output / f"zero_shot_{model_name}_summary.json", result)
    return result


def _medgemma_generator(config: dict[str, Any], device: str, disease: str,
                        generation: dict[str, Any]) -> Callable[[str], str]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint = Path(config["checkpoints"]["medgemma"]).resolve()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    data = config["data"]

    def generate(path: str) -> str:
        images = volume_to_slices(path, axis=data["axis"], count=data["num_slices"],
                                  bounds=data["slice_range"], roi_fraction=data["roi_fraction"])
        messages = [{"role": "user", "content": native_multimodal_content(disease, images)}]
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[rendered], images=[images], return_tensors="pt", padding=True)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        stopping = _stopping_criteria(processor.tokenizer, disease, inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=generation["max_new_tokens"],
                stopping_criteria=stopping, use_cache=True,
            )
        continuation = generated[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(continuation, skip_special_tokens=True)[0]

    return generate


def _braingemma_generator(config: dict[str, Any], device: str, disease: str, prompt: str,
                          generation: dict[str, Any]) -> Callable[[str], str]:
    import torch
    from transformers import AutoModelForCausalLM

    root = Path(config["checkpoints"]["braingemma3d"]).resolve()
    architecture = _load_architecture(Path(config["source_repositories"]["braingemma3d"]).resolve())

    # The released loader always enables 4-bit bitsandbytes quantization. A100-class
    # GPUs have ample memory for this 4B model, and native BF16 avoids decode-time
    # dequantization while retaining the same frozen checkpoint.
    def load_bf16_language_model(model_dir: str, device_map=None):
        return AutoModelForCausalLM.from_pretrained(
            model_dir, device_map=device_map, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True,
        )

    architecture.load_medgemma_lm_local = load_bf16_language_model
    model = architecture.BrainGemma3D(
        vision_model_dir=str(root / "vision_model"),
        language_model_dir=str(root / "language_model"), depth=2, num_vision_tokens=32,
        freeze_vision=True, freeze_language=True, device_map={"": torch.device(device).index or 0},
    )
    projector = torch.load(root / "projector_vis_scale.pt", map_location=model.lm_device,
                           weights_only=False)
    missing, unexpected = model.vision_projector.load_state_dict(
        projector["vision_projector"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"BrainGemma3D projector mismatch: missing={missing}, unexpected={unexpected}")
    if projector.get("vis_scale") is not None:
        model.vis_scale.data.copy_(torch.as_tensor(projector["vis_scale"], device=model.lm_device))
    model.eval().requires_grad_(False)
    shape = config["data"]["volume_shapes"]["braingemma3d"]

    def generate(path: str) -> str:
        volume = volume_to_tensor(path, shape).unsqueeze(0)
        vis = model.encode_volume(volume)
        tokens = model.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=256,
        ).to(model.lm_device)
        text_embeddings = model.language_model.get_input_embeddings()(tokens.input_ids)
        inputs_embeds = torch.cat([vis, text_embeddings], dim=1)
        attention = torch.ones(inputs_embeds.shape[:2], device=model.lm_device, dtype=torch.long)
        stopping = _stopping_criteria(model.tokenizer, disease)
        with torch.inference_mode():
            output_ids = model.language_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention,
                max_new_tokens=generation["max_new_tokens"], min_new_tokens=1,
                do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                stopping_criteria=stopping, use_cache=True,
            )
        return model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return generate
