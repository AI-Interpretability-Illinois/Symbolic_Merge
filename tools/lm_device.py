#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lm_device.py

One place that decides where a runner's language model lives.

--device cuda / cuda:N / cpu   the whole model on that device, as before.
--device split                 layers spread over every visible GPU by accelerate,
                               capped per GPU by --max_gpu_memory (e.g. "7GiB,12GiB").
                               For sharing GPUs with another job that leaves too
                               little room for the whole model on one card.

The forward pass is the same bf16 computation either way; only placement differs.
Callers get back the device to send inputs to and to run the SAE on (the first
GPU), and must move hidden states there, since a split model returns each layer's
hidden state on the GPU that computed it.
"""

import torch
from transformers import AutoModelForCausalLM


def add_device_args(ap, default="cuda"):
    ap.add_argument("--device", default=default,
                    help="cuda, cuda:N, cpu, or 'split' to spread the model over all visible GPUs")
    ap.add_argument("--max_gpu_memory", default="7GiB,12GiB",
                    help="with --device split: per-GPU cap for model weights, comma list "
                         "(the last value repeats); keep the first small, it also holds the SAE")


def load_causal_lm(model_path, device, max_gpu_memory="7GiB,12GiB", **kwargs):
    """Returns (model, work_device)."""
    kwargs.setdefault("torch_dtype", torch.bfloat16)
    kwargs.setdefault("low_cpu_mem_usage", True)
    if device != "split":
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device).eval(), device
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError("--device split needs at least two visible GPUs")
    caps = [c.strip() for c in max_gpu_memory.split(",") if c.strip()]
    max_memory = {i: caps[min(i, len(caps) - 1)] for i in range(n)}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", max_memory=max_memory, **kwargs).eval()
    placed = sorted({str(d) for d in model.hf_device_map.values()})
    if "cpu" in placed or "disk" in placed:
        raise RuntimeError(f"model did not fit in {max_memory}: {model.hf_device_map}")
    print(f"[split] model over {placed} with caps {max_memory}", flush=True)
    return model, "cuda:0"
