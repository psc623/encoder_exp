"""Command-line entrypoint for the phase-gated experiment workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from encoderbench.config import DISEASES, ENCODERS, ExperimentConfig, load_config


def _positive(disease: str) -> str:
    return "AD" if disease == "ad" else "SCZ"


def _seeds(value: str, configured: list[int]) -> list[int]:
    result = configured if value == "all" else [int(item.strip()) for item in value.split(",")]
    if not result or not set(result) <= {0, 1, 2}:
        raise ValueError("Training seeds must be a non-empty subset of 0,1,2")
    return result


def _cache_path(config: ExperimentConfig, disease: str, encoder: str) -> Path:
    return config.output_root / "cache" / disease / f"{encoder}.npz"


def _gate(config: ExperimentConfig, disease: str) -> None:
    if disease == "scz":
        from encoderbench.phase import require_ad_lock

        require_ad_lock(config.output_root, config.source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="encoderbench", description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config (default config/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("build-manifest", help="Build exact ADNI or UCLA split manifests")
    manifest.add_argument("disease", choices=DISEASES)
    manifest.add_argument("--source-manifest", default=os.environ.get("ADNI_SOURCE_MANIFEST"))
    manifest.add_argument("--bids-root", default=os.environ.get("UCLA_BIDS_ROOT"))
    manifest.add_argument("--participants", default=os.environ.get("UCLA_PARTICIPANTS_TSV"))
    manifest.add_argument("--out", default=None)

    audit = sub.add_parser("audit-data", help="Audit every volume and create representative montages")
    audit.add_argument("disease", choices=DISEASES)
    audit.add_argument("--manifest", default=None)
    audit.add_argument("--out-dir", default=None)

    smoke = sub.add_parser("smoke", help="Phase 0 two-case checkpoint/grid smoke test")
    smoke.add_argument("disease", choices=DISEASES)
    smoke.add_argument("encoder", choices=ENCODERS)
    smoke.add_argument("--manifest", default=None)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--out", default=None)

    token = sub.add_parser("token-audit", help="Audit A/B in the actual chat generation context")
    token.add_argument("disease", choices=DISEASES)
    token.add_argument("--out", default=None)

    cache = sub.add_parser("cache", help="Extract and cache one encoder's 64-token grid")
    cache.add_argument("disease", choices=DISEASES)
    cache.add_argument("encoder", choices=ENCODERS)
    cache.add_argument("--manifest", default=None)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--out", default=None)

    probe = sub.add_parser("probe", help="Train formal attention-probe seeds")
    probe.add_argument("disease", choices=DISEASES)
    probe.add_argument("encoder", choices=ENCODERS)
    probe.add_argument("--cache", default=None)
    probe.add_argument("--seeds", default="all", help="all or comma-separated subset of 0,1,2")
    probe.add_argument("--shuffled-labels", action="store_true")
    probe.add_argument("--device", default="auto")
    probe.add_argument("--out-dir", default=None)

    zero = sub.add_parser("zero-shot", help="Run a frozen native VLM with free generation")
    zero.add_argument("disease", choices=DISEASES)
    zero.add_argument("model", choices=("medgemma", "braingemma3d"))
    zero.add_argument("--manifest", default=None)
    zero.add_argument("--device", default="cuda")
    zero.add_argument("--out-dir", default=None)

    bridge = sub.add_parser("bridge", help="Train formal frozen-LLM bridge seeds")
    bridge.add_argument("disease", choices=DISEASES)
    bridge.add_argument("encoder", choices=ENCODERS)
    bridge.add_argument("kind", choices=("linear", "resampler"))
    bridge.add_argument("--cache", default=None)
    bridge.add_argument("--seeds", default="all", help="all or comma-separated subset of 0,1,2")
    bridge.add_argument("--device", default="cuda")
    bridge.add_argument("--out-dir", default=None)

    sub.add_parser("lock-ad", help="Verify all 50 AD runs and freeze configuration for SCZ")

    report = sub.add_parser("report", help="Aggregate seeds and paired comparisons")
    report.add_argument("--input-root", default=None)
    report.add_argument("--out-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = _dispatch(args, config)
        if result is not None:
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


def _dispatch(args: argparse.Namespace, config: ExperimentConfig) -> Any:
    raw = config.raw
    if args.command == "build-manifest":
        from encoderbench.manifest import build_adni_from_source, build_ucla, validate_manifest
        from encoderbench.utils import write_json

        output = Path(args.out or config.manifest(args.disease))
        if args.disease == "ad":
            if not args.source_manifest:
                raise ValueError("ADNI requires --source-manifest or ADNI_SOURCE_MANIFEST")
            rows = build_adni_from_source(args.source_manifest, output)
            skipped = []
        else:
            if not args.bids_root or not args.participants:
                raise ValueError("SCZ requires --bids-root and --participants (or matching env vars)")
            rows, skipped = build_ucla(args.bids_root, args.participants, output)
            write_json(output.with_suffix(".skipped.json"), skipped)
        return {"manifest": str(output.resolve()), "audit": validate_manifest(rows, args.disease),
                "skipped_count": len(skipped)}
    if args.command == "audit-data":
        from encoderbench.workflows import audit_dataset

        manifest = args.manifest or config.manifest(args.disease)
        output = args.out_dir or config.output_root / "audits" / args.disease / "dataset"
        return audit_dataset(manifest, args.disease, output, raw["data"])
    if args.command == "smoke":
        from encoderbench.workflows import smoke_extractor

        _gate(config, args.disease)
        manifest = args.manifest or config.manifest(args.disease)
        output = args.out or config.output_root / "audits" / args.disease / f"{args.encoder}_smoke.json"
        return smoke_extractor(manifest, args.disease, args.encoder, raw, output, args.device)
    if args.command == "token-audit":
        from transformers import AutoTokenizer

        from encoderbench.llm import audit_answer_tokens
        from encoderbench.utils import write_json

        _gate(config, args.disease)
        tokenizer = AutoTokenizer.from_pretrained(raw["checkpoints"]["medgemma"], local_files_only=True)
        audit = audit_answer_tokens(tokenizer, args.disease).to_dict()
        audit["injection_order"] = "BOS + 64 visual tokens + prompt tokens + complete answer sequence"
        audit["label_mask"] = "BOS, visual, and prompt positions=-100; every answer position=scored"
        audit["masked_prefix_positions"] = 64 + len(audit["context_ids"])
        audit["answer_a_scored_positions"] = len(audit["answer_a_ids"])
        audit["answer_b_scored_positions"] = len(audit["answer_b_ids"])
        output = args.out or config.output_root / "audits" / args.disease / "tokenizer.json"
        write_json(output, audit)
        return audit
    if args.command == "cache":
        from encoderbench.workflows import cache_features

        _gate(config, args.disease)
        output = Path(args.out or _cache_path(config, args.disease, args.encoder))
        result = cache_features(args.manifest or config.manifest(args.disease), args.disease,
                                args.encoder, raw, output, args.device)
        return {"cache": str(result)}
    if args.command == "probe":
        from encoderbench.training import run_probe

        _gate(config, args.disease)
        cache = args.cache or _cache_path(config, args.disease, args.encoder)
        output = args.out_dir or config.output_root / "attention" / args.disease / args.encoder
        results = [run_probe(cache, output, _positive(args.disease), seed, raw["probe"],
                             raw["evaluation"], args.shuffled_labels, args.device)
                   for seed in _seeds(args.seeds, raw["probe"]["seeds"])]
        return {"runs": results}
    if args.command == "zero-shot":
        from encoderbench.zero_shot import run_zero_shot

        _gate(config, args.disease)
        output = args.out_dir or config.output_root / "zero_shot" / args.disease
        return run_zero_shot(args.manifest or config.manifest(args.disease), args.disease,
                             args.model, raw, output, args.device)
    if args.command == "bridge":
        from encoderbench.training import run_bridge

        _gate(config, args.disease)
        cache = args.cache or _cache_path(config, args.disease, args.encoder)
        output = args.out_dir or config.output_root / "bridge" / args.disease / args.kind / args.encoder
        results = [run_bridge(cache, output, _positive(args.disease), args.disease, args.kind,
                              seed, raw["bridge"], raw["evaluation"],
                              raw["checkpoints"]["medgemma"], args.device)
                   for seed in _seeds(args.seeds, raw["bridge"]["seeds"])]
        return {"runs": results}
    if args.command == "lock-ad":
        from encoderbench.phase import create_ad_lock

        return {"lock": str(create_ad_lock(config.output_root, config.source))}
    if args.command == "report":
        from encoderbench.report import generate_report

        return generate_report(args.input_root or config.output_root,
                               args.out_dir or config.output_root / "reports",
                               int(raw["evaluation"]["bootstrap_samples"]),
                               int(raw["evaluation"]["bootstrap_seed"]))
    raise AssertionError(f"Unhandled command: {args.command}")
