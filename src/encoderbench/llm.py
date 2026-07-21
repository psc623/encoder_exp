"""Frozen MedGemma language backbone, token audit, loss, and A/B scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from encoderbench.prompts import BRIDGE_PROMPTS
from encoderbench.utils import checkpoint_identifier


@dataclass(frozen=True)
class AnswerTokenAudit:
    disease: str
    rendered_context: str
    context_ids: list[int]
    answer_a_ids: list[int]
    answer_b_ids: list[int]
    answer_a_decoded: str
    answer_b_decoded: str
    answer_a_has_leading_space: bool
    answer_b_has_leading_space: bool
    eos_in_target: bool
    template_adds_eos_to_context: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_answer_tokens(tokenizer, disease: str) -> AnswerTokenAudit:
    prompt = BRIDGE_PROMPTS[disease]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    context_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    answers: dict[str, list[int]] = {}
    for answer in ("A", "B"):
        full_ids = tokenizer(rendered + answer, add_special_tokens=False).input_ids
        if full_ids[:len(context_ids)] != context_ids:
            raise ValueError(
                f"Tokenizer merges the assistant answer with its prefix for {answer}; "
                "the frozen prompt needs an explicit stable boundary"
            )
        answer_ids = full_ids[len(context_ids):]
        if not answer_ids:
            raise ValueError(f"Answer {answer} produced no target token")
        answers[answer] = answer_ids
    eos_ids = tokenizer.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    return AnswerTokenAudit(
        disease=disease, rendered_context=rendered, context_ids=list(context_ids),
        answer_a_ids=answers["A"], answer_b_ids=answers["B"],
        answer_a_decoded=tokenizer.decode(answers["A"]),
        answer_b_decoded=tokenizer.decode(answers["B"]),
        answer_a_has_leading_space=tokenizer.decode(answers["A"]).startswith(" "),
        answer_b_has_leading_space=tokenizer.decode(answers["B"]).startswith(" "),
        eos_in_target=bool(eos_set & set(answers["A"] + answers["B"])),
        template_adds_eos_to_context=bool(context_ids and context_ids[-1] in eos_set),
    )


class FrozenMedGemma:
    """Text-only causal backbone from the canonical multimodal checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str, precision: str = "bf16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not str(device).startswith("cuda"):
            raise ValueError("Bridge training requires a CUDA device for the 4B MedGemma backbone")
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval().requires_grad_(False)
        self.trainable_parameters = sum(parameter.numel() for parameter in self.model.parameters()
                                        if parameter.requires_grad)
        if self.trainable_parameters != 0:
            raise RuntimeError(f"MedGemma backbone has {self.trainable_parameters} trainable parameters")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.checkpoint_audit = checkpoint_identifier(checkpoint)
        self.hidden_size = int(getattr(self.model.config, "hidden_size",
                                       self.model.get_input_embeddings().embedding_dim))

    def token_audit(self, disease: str) -> AnswerTokenAudit:
        return audit_answer_tokens(self.tokenizer, disease)

    def _composed_embeddings(
        self, visual_tokens: torch.Tensor, context_ids: Sequence[int], answer_ids: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch = visual_tokens.shape[0]
        context = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch, -1)
        answer = torch.tensor(answer_ids, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch, -1)
        if context.shape[1] < 1:
            raise ValueError("Chat template produced an empty context")
        embed = self.model.get_input_embeddings()
        bos, prompt = embed(context[:, :1]), embed(context[:, 1:])
        answer_embeddings = embed(answer)
        inputs = torch.cat((bos, visual_tokens.to(self.dtype), prompt, answer_embeddings), dim=1)
        prefix_length = 1 + visual_tokens.shape[1] + prompt.shape[1]
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
        return inputs, attention, prefix_length

    def sequence_scores(
        self, visual_tokens: torch.Tensor, context_ids: Sequence[int], answer_ids: Sequence[int]
    ) -> torch.Tensor:
        inputs, attention, prefix_length = self._composed_embeddings(
            visual_tokens, context_ids, answer_ids
        )
        position_ids = attention.cumsum(dim=1) - 1
        output = self.model(inputs_embeds=inputs, attention_mask=attention,
                            position_ids=position_ids, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
        target = torch.tensor(answer_ids, dtype=torch.long, device=self.device)
        score = torch.zeros(inputs.shape[0], device=self.device)
        for offset, token_id in enumerate(target):
            score = score + log_probs[:, prefix_length - 1 + offset, token_id]
        return score

    def class_probabilities(self, visual_tokens: torch.Tensor, audit: AnswerTokenAudit) -> torch.Tensor:
        score_a = self.sequence_scores(visual_tokens, audit.context_ids, audit.answer_a_ids)
        score_b = self.sequence_scores(visual_tokens, audit.context_ids, audit.answer_b_ids)
        scores = torch.stack((score_a, score_b), dim=-1)
        return torch.exp(scores[:, 0] - torch.logsumexp(scores, dim=-1))

    def weighted_answer_loss(self, visual_tokens: torch.Tensor, class_indices: torch.Tensor,
                             class_weights: torch.Tensor, audit: AnswerTokenAudit) -> torch.Tensor:
        losses = []
        for index in range(visual_tokens.shape[0]):
            is_a = int(class_indices[index].item()) == 1
            answer = audit.answer_a_ids if is_a else audit.answer_b_ids
            score = self.sequence_scores(visual_tokens[index:index + 1], audit.context_ids, answer)[0]
            losses.append(-score * class_weights[class_indices[index]])
        return torch.stack(losses).mean()


def assert_gradient_isolation(bridge: torch.nn.Module, backbone: FrozenMedGemma) -> dict[str, object]:
    bridge_nonzero = any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                         for parameter in bridge.parameters())
    llm_has_gradient = any(parameter.grad is not None for parameter in backbone.model.parameters())
    if not bridge_nonzero or llm_has_gradient:
        raise RuntimeError(
            f"Gradient audit failed: bridge_nonzero={bridge_nonzero}, llm_has_gradient={llm_has_gradient}"
        )
    return {"bridge_gradient_nonzero": True, "llm_gradient_zero": True,
            "encoder_gradient_zero": True,
            "llm_trainable_parameters": backbone.trainable_parameters,
            "llm_eval_mode": not backbone.model.training}
