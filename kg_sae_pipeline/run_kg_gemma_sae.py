#!/usr/bin/env python3
"""Extract dense and tokenwise-SAE entity representations for KG contexts."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kg_sae_utils import SAE, aggregate_context_sae, load_contexts, safe_name, token_ids

# Gemma Scope layer_20/width_131k/average_l0_114: the canonical release, and the
# only layer-20 residual 131k variant with full Neuronpedia auto-interp coverage
# (source id gemma-2-9b/20-gemmascope-res-131k). Keep runs on this SAE so feature
# indices stay comparable across benchmarks and remain interpretable.
DEFAULT_SAE_PATH = ("/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz")
DEFAULT_MODEL_PATH = "/projects/biro/shared/models/gemma-2-9b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae_path", default=DEFAULT_SAE_PATH,
                        help="defaults to average_l0_114 (Neuronpedia-interpretable)")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stable_frequencies", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug_first", action="store_true")
    args = parser.parse_args()

    grouped, labels, errors = load_contexts(args.input)
    entity_ids = sorted(grouped)[args.start:]
    if args.limit:
        entity_ids = entity_ids[:args.limit]

    output_dir = Path(args.output_dir).expanduser()
    representation_dir = output_dir / "representations"
    representation_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(args.model_path).expanduser())

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    sae = SAE(Path(args.sae_path).expanduser(), args.device)
    print(f"entities={len(entity_ids)} SAE width={sae.width} d_in={sae.d_in}")
    thresholds = [float(x) for x in args.stable_frequencies.split(",") if x.strip()]
    manifest = []

    for number, entity_id in enumerate(entity_ids, 1):
        file_path = representation_dir / (safe_name(entity_id) + ".pt")
        if file_path.exists() and not args.overwrite:
            print(f"[{number}/{len(entity_ids)}] skip {entity_id}")
            continue

        contexts = grouped[entity_id][:args.contexts] if args.contexts else grouped[entity_id]
        saved = []
        dense_vectors = []
        print(f"[{number}/{len(entity_ids)}] {entity_id} :: {labels[entity_id]} :: contexts={len(contexts)}")

        for context_number, context in enumerate(contexts, 1):
            encoded = tokenizer(
                context["text"],
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            token_indices = token_ids(offsets, context["char_span"])
            start, end = context["char_span"]
            if min(offsets[i][0] for i in token_indices) > start or max(offsets[i][1] for i in token_indices) < end:
                raise RuntimeError(f"symbol truncated: {entity_id} context {context_number}")

            inputs = {key: value.to(args.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.hidden_states[args.layer + 1][0]
                token_hidden = hidden[token_indices]

                # Dense baseline: pool exact symbol-token hidden states.
                dense_vector = token_hidden.mean(0)
                if dense_vector.numel() != sae.d_in:
                    raise RuntimeError(f"hidden {dense_vector.numel()} != SAE d_in {sae.d_in}")

                # Main SAE representation: encode each token first, then pool
                # the sparse features. This is not SAE(mean(hidden_states)).
                indices, values, token_nnz = sae.encode(token_hidden)

            record = {key: value for key, value in context.items() if key not in ("dense", "sae")}
            record.update(
                token_indices=[int(index) for index in token_indices],
                symbol_tokens=tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][0, token_indices].tolist()
                ),
                dense=dense_vector.detach().float().cpu(),
                sae={
                    "idx": indices.detach().to(torch.int32).cpu(),
                    "val": values.detach().float().cpu(),
                    "size": sae.width,
                    "pooling": "mean_after_tokenwise_sae",
                    "token_nnz": token_nnz.cpu().tolist(),
                },
            )
            saved.append(record)
            dense_vectors.append(record["dense"])

            if args.debug_first and number == 1:
                print(
                    " context", context_number,
                    "span", context["char_span"],
                    "tokens", record["symbol_tokens"],
                    "token_nnz", record["sae"]["token_nnz"],
                    "pooled_nnz", len(record["sae"]["idx"]),
                )
            del output, hidden, token_hidden, dense_vector, inputs

        sae_mean, sae_frequency = aggregate_context_sae(saved, sae.width)
        representation = {
            "entity_iri": entity_id,
            "label": labels[entity_id],
            "contexts": saved,
            "dense_mean": torch.stack(dense_vectors).mean(0),
            "sae_mean": sae_mean,
            "sae_frequency": sae_frequency,
            "sae_stable": {
                f"{threshold:.2f}": sae_mean * (sae_frequency >= threshold).float()
                for threshold in thresholds
            },
            "meta": {
                "layer": args.layer,
                "hidden_states_index": args.layer + 1,
                "model_path": model_path,
                "sae_path": str(Path(args.sae_path).expanduser()),
                "sae_width": sae.width,
                "symbol_span_only": True,
                "sae_span_pooling": "mean_after_tokenwise_sae",
                "num_contexts": len(saved),
            },
        }
        temporary = Path(str(file_path) + ".tmp")
        torch.save(representation, temporary)
        temporary.replace(file_path)
        manifest.append({
            "entity_iri": entity_id,
            "label": labels[entity_id],
            "file": str(file_path),
            "contexts": len(saved),
        })

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"entities": manifest, "parse_errors": errors}, f, indent=2, ensure_ascii=False)
    print("DONE:", representation_dir)


if __name__ == "__main__":
    main()
