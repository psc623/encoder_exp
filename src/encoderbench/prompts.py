"""Frozen prompts for native zero-shot and supervised bridge experiments."""

from __future__ import annotations


BRIDGE_PROMPTS = {
    "ad": """Classify this brain MRI as:
A. Alzheimer's disease
B. Cognitively normal
Output only A or B.""",
    "scz": """Classify this brain MRI as:
A. Schizophrenia
B. Cognitively normal
Output only A or B.""",
}

NATIVE = {
    "ad": {
        "instruction": (
            "You are an expert neuroradiologist analyzing a T1-weighted structural brain MRI. "
            "Below is a contiguous, inferior-to-superior stack of axial slices from a single "
            "patient's skull-stripped, spatially normalized scan. Review all slices carefully "
            "as one 3D volume."
        ),
        "query": (
            "Based on the imaging above, assess whether this patient's brain shows evidence of "
            "Alzheimer's disease. Focus on hallmark findings: medial temporal lobe and hippocampal "
            "atrophy, widening of the temporal horns of the lateral ventricles, general ventricular "
            "enlargement, and cortical sulcal widening. Provide brief reasoning, then conclude with "
            "exactly one line: 'Final Answer: AD' if findings are consistent with Alzheimer's "
            "disease, or 'Final Answer: CN' if the brain appears cognitively normal."
        ),
    },
    "scz": {
        "instruction": (
            "You are an expert neuroradiologist reviewing a T1-weighted structural brain MRI for a "
            "research cohort study. Below is a contiguous, inferior-to-superior stack of axial "
            "slices from one participant. Review all slices carefully as one 3D volume. Structural "
            "MRI cannot diagnose schizophrenia clinically; this task tests cohort-level signal only."
        ),
        "query": (
            "For this research classification task, classify the participant as schizophrenia "
            "cohort (SCZ) or control cohort (CN) from the structural MRI. Provide brief reasoning, "
            "then conclude with exactly one line: 'Final Answer: SCZ' or 'Final Answer: CN'."
        ),
    },
}


def native_text(disease: str) -> str:
    if disease not in NATIVE:
        raise ValueError("disease must be 'ad' or 'scz'")
    return f"{NATIVE[disease]['instruction']}\n\n{NATIVE[disease]['query']}"


def native_multimodal_content(disease: str, images: list[object]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = [{"type": "text", "text": NATIVE[disease]["instruction"]}]
    for number, image in enumerate(images, 1):
        blocks.extend(({"type": "image", "image": image},
                       {"type": "text", "text": f"SLICE {number}"}))
    blocks.append({"type": "text", "text": NATIVE[disease]["query"]})
    return blocks

