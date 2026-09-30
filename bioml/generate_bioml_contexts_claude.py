#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_bioml_contexts_claude.py

Bio-ML context generation with Claude (Azure AI Foundry) in place of Bio-8B.

The prompt is the max-diverse protocol of generate_bioml_contexts_bio8b_maxdiverse.py,
imported rather than copied: same system prompt, same per-entity diversity plans
(semantic focus, syntax style, entity position), same LEFT/RIGHT fragment format,
same parser that inserts the exact preferred label. The only protocol difference is
that Claude Opus 5 takes no temperature/top_p, so diversity comes from the plans.

Every entity of every requested split is generated into one store, so an entity
shared by several splits (most DOID candidates) is generated once:

    <output_dir>/contexts/{src,tgt}/<tail>_<sha10>.json   same fields as the Bio-8B runs
    <output_dir>/views/<split>/selected_queries.json      one per split
    <output_dir>/views/<split>/contexts -> ../../contexts

A view directory is a drop-in context_run_dir for the SAE runner, the hidden-state
cache and analyze_bioml_idf.py. Reruns resume: complete entity files are skipped.

The API key is read from a file (default ~/.config/symbolic_merge/azure_anthropic_key),
never from the command line.

Example:
    python bioml/generate_bioml_contexts_claude.py --repo_dir <OAEI-Bio-ML> \
        --splits valid,train --output_dir <store> --workers 16
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_bioml_contexts_bio8b_maxdiverse import (  # noqa: E402
    SYSTEM_PROMPT, atomic_json_dump, build_prompt, complete_cache, deterministic_plans,
    entity_output_path, iri_tail, join_context, load_cache, load_queries, metadata_for,
    recover_pairs)

DEFAULT_ENDPOINT = "https://xiaocong-resource.services.ai.azure.com/anthropic"
DEFAULT_KEY_FILE = "~/.config/symbolic_merge/azure_anthropic_key"


class LocalClient:
    """A local Hugging Face chat model behind the one call this module makes
    (client.messages.create), so every task script can generate one side of its
    pairs with a different model than the other. Concurrent calls from the worker
    threads are gathered into batches for model.generate."""

    def __init__(self, model_path, device="cuda", batch_size=16, max_new_tokens=1500):
        import queue
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.queue = torch, queue
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
        self.device, self.batch_size, self.max_new_tokens = device, batch_size, max_new_tokens
        self.requests = queue.Queue()
        self.messages = self
        threading.Thread(target=self._serve, daemon=True).start()

    def create(self, model, max_tokens, system, messages, **_):
        from concurrent.futures import Future
        done = Future()
        self.requests.put((system, messages[0]["content"], done))
        return done.result()

    def _serve(self):
        from types import SimpleNamespace as NS
        while True:
            batch = [self.requests.get()]
            try:
                while len(batch) < self.batch_size:
                    batch.append(self.requests.get(timeout=0.5))
            except self.queue.Empty:
                pass
            try:
                texts = [self.tok.apply_chat_template(
                    [{"role": "system", "content": s}, {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True) for s, u, _ in batch]
                enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
                with self.torch.inference_mode():
                    out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                              do_sample=True, temperature=0.8, top_p=0.95,
                                              pad_token_id=self.tok.pad_token_id)
                width = enc["input_ids"].shape[1]
                for i, (_, _, done) in enumerate(batch):
                    gen = out[i, width:]
                    n_out = int((gen != self.tok.pad_token_id).sum())
                    stop = "max_tokens" if n_out >= self.max_new_tokens else "end_turn"
                    done.set_result(NS(
                        content=[NS(type="text", text=self.tok.decode(gen, skip_special_tokens=True))],
                        stop_reason=stop,
                        usage=NS(input_tokens=int(enc["attention_mask"][i].sum()), output_tokens=n_out)))
            except Exception as e:  # hand the failure to every waiting caller
                for _, _, done in batch:
                    if not done.done():
                        done.set_exception(RuntimeError(f"local generation failed: {e}"))


class OpenAIClient:
    """An OpenAI Responses-API model (e.g. gpt-5.6-sol) behind the one call this module
    makes, so a pair's two sides can come from different model families."""

    def __init__(self, key_file, base_url=None, effort="low"):
        from openai import OpenAI
        key = Path(os.path.expanduser(key_file)).read_text().strip()
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=8, timeout=300)
        self.effort = effort
        self.messages = self

    def create(self, model, max_tokens, system, messages, **_):
        from types import SimpleNamespace as NS
        r = self.client.responses.create(
            model=model, instructions=system, input=messages[0]["content"],
            max_output_tokens=max_tokens, reasoning={"effort": self.effort}, store=False)
        stop = "end_turn" if r.status == "completed" else "max_tokens"
        return NS(content=[NS(type="text", text=r.output_text or "")], stop_reason=stop,
                  usage=NS(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens))


def call(client, args, prompt, system=SYSTEM_PROMPT):
    extra = {"output_config": {"effort": args.effort}} if args.backend == "claude" else {}
    msg = client.messages.create(
        model=args.model, max_tokens=args.max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}], **extra)
    if msg.stop_reason == "refusal":
        raise RuntimeError("refusal")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.stop_reason, msg.usage


def generate_entity(client, args, job, system=SYSTEM_PROMPT, prompt_fn=build_prompt,
                    plans_fn=deterministic_plans, validate_fn=None):
    """Returns the entity record; retries until the reply parses into n contexts.

    system / prompt_fn / plans_fn default to the Bio-ML protocol; other
    context-generation benchmarks pass their own domain prompt. validate_fn(job,
    contexts) -> None or a reason string rejects a reply (e.g. one that names the
    entity's counterpart), which is then retried like an unparseable one."""
    prompt = prompt_fn(job["meta"], args.contexts)
    usage = {"input_tokens": 0, "output_tokens": 0}
    last = None
    for attempt in range(1, args.parse_retries + 2):
        raw, stop, u = call(client, args, prompt, system)
        usage["input_tokens"] += u.input_tokens
        usage["output_tokens"] += u.output_tokens
        pairs = recover_pairs(raw)
        if len(pairs) >= args.contexts:
            label = job["meta"]["preferred_label"]
            contexts = [join_context(p.get("left", ""), label, p.get("right", ""))
                        for p in pairs[:args.contexts]]
            bad = validate_fn(job, contexts) if validate_fn else None
            if bad:
                last = f"rejected: {bad}"
                continue
            return {
                "entity_iri": job["iri"], "side": job["side"],
                "preferred_label": label,
                "entity_metadata_given_to_model": job["meta"],
                "diversity_plans": plans_fn(job["meta"], args.contexts),
                "system_prompt": system, "user_prompt": prompt,
                "raw_model_output": raw, "contexts": contexts,
                "parsed_output": {"contexts": [{"text": c["text"]} for c in contexts]},
                "generator": {"model": args.model, "effort": args.effort,
                              "attempts": attempt, "stop_reason": stop, "usage": usage},
            }
        last = f"parsed {len(pairs)}/{args.contexts} contexts (stop_reason={stop})"
    # No template fallback: a silently templated entity would contaminate the benchmark.
    raise RuntimeError(last)


def add_api_args(ap):
    ap.add_argument("--contexts", type=int, default=5)
    ap.add_argument("--backend", choices=("claude", "openai", "local"), default="claude",
                    help="claude: Azure Foundry; openai: OpenAI Responses API (--openai_model); "
                         "local: a Hugging Face chat model on this machine")
    ap.add_argument("--openai_model", default="gpt-5.6-sol")
    ap.add_argument("--openai_key_file", default="~/.config/symbolic_merge/openai_key")
    ap.add_argument("--openai_base_url", default="https://xiaocong-resource.services.ai.azure.com/openai/v1",
                    help="OpenAI-compatible endpoint; the Azure Foundry resource by default")
    ap.add_argument("--local_model", default=None, help="path or HF id for --backend local")
    ap.add_argument("--local_device", default="cuda")
    ap.add_argument("--local_batch", type=int, default=16)
    ap.add_argument("--sides", default="src,tgt",
                    help="which sides to generate here; the two sides of a pair can come "
                         "from different backends, run separately into the same store")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--key_file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--parse_retries", type=int, default=2)


def make_client(args):
    if args.backend == "openai":
        args.model = args.openai_model
        client = OpenAIClient(args.openai_key_file, args.openai_base_url, args.effort)
        return client
    if args.backend == "local":
        args.model = args.local_model  # recorded in each entity's generator field
        args.effort = None
        return LocalClient(args.local_model, args.local_device, args.local_batch)
    from anthropic import AnthropicFoundry
    key = Path(os.path.expanduser(args.key_file)).read_text().strip()
    return AnthropicFoundry(api_key=key, base_url=args.endpoint, max_retries=8, timeout=300)


def generate_all(client, args, pending, **gen_kwargs):
    """Generate and write every pending job concurrently; returns (totals, failures)."""
    lock = threading.Lock()
    tot = {"done": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "retried": 0}
    failures = []
    t0 = time.time()

    sides = set(args.sides.split(","))
    pending = [j for j in pending if j["side"] in sides]

    def work(job):
        rec = generate_entity(client, args, job, **gen_kwargs)
        atomic_json_dump(rec, job["path"])
        return rec["generator"]

    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(work, j): j for j in pending}
        for f in as_completed(futs):
            j = futs[f]
            with lock:
                try:
                    g = f.result()
                    tot["done"] += 1
                    tot["input_tokens"] += g["usage"]["input_tokens"]
                    tot["output_tokens"] += g["usage"]["output_tokens"]
                    tot["retried"] += g["attempts"] > 1
                except Exception as e:  # API errors, refusals, unparseable replies
                    tot["failed"] += 1
                    failures.append({"iri": j["iri"], "side": j["side"], "error": str(e)[:300]})
                n = tot["done"] + tot["failed"]
                if n % 100 == 0 or n == len(pending):
                    rate = n / max(1e-9, time.time() - t0)
                    print(f"[{n}/{len(pending)}] done={tot['done']} failed={tot['failed']} "
                          f"retried={tot['retried']} in={tot['input_tokens']:,} "
                          f"out={tot['output_tokens']:,} {rate:.2f} ent/s "
                          f"eta={(len(pending) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    return tot, failures


def write_view(out, name, queries):
    """views/<name>/{selected_queries.json, contexts -> ../../contexts}: a context_run_dir."""
    view = out / "views" / name
    view.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(queries, view / "selected_queries.json")
    link = view / "contexts"
    if not link.exists():
        link.symlink_to(Path("..") / ".." / "contexts")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_dir", type=Path, required=True, help="OAEI-Bio-ML checkout")
    ap.add_argument("--pair", default="NCIT-DOID")
    ap.add_argument("--splits", default="valid", help="comma list, e.g. valid,train")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--limit_queries", type=int, default=0, help="per split; 0 = all")
    add_api_args(ap)
    args = ap.parse_args()

    if args.pair != "NCIT-DOID":
        raise SystemExit("only NCIT-DOID has public metadata caches (SNOMED is licence-gated)")
    repo = args.repo_dir.expanduser().resolve() / "bio-ml"
    out = args.output_dir.expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    queries = {}
    for sp in splits:
        queries[sp] = load_queries(repo / args.pair / f"local.{sp}.cands.tsv", 0,
                                   args.limit_queries or 10 ** 9)
        write_view(out, sp, queries[sp])

    ncit = load_cache(repo / "ontology_metadata_cache" / "NCIT_metadata_cache.pt", "NCIT")
    doid = load_cache(repo / "ontology_metadata_cache" / "DOID_metadata_cache.pt", "DOID")
    src = list(dict.fromkeys(q["src"] for qs in queries.values() for q in qs))
    tgt = list(dict.fromkeys(c for qs in queries.values() for q in qs for c in q["candidates"]))
    jobs = ([{"side": "src", "iri": i, "meta": metadata_for(i, ncit[i], "NCIT")} for i in src]
            + [{"side": "tgt", "iri": i, "meta": metadata_for(i, doid[i], "DOID")} for i in tgt])
    for j in jobs:
        j["path"] = entity_output_path(out, j["side"], j["iri"])
    pending = [j for j in jobs if not complete_cache(j["path"], j["iri"], args.contexts)]
    print(f"splits={ {s: len(q) for s, q in queries.items()} } entities={len(jobs)} "
          f"(src {len(src)}, tgt {len(tgt)}) pending={len(pending)} "
          f"model={args.model} effort={args.effort} workers={args.workers}", flush=True)

    tot, failures = generate_all(make_client(args), args, pending)

    summary = {"splits": {s: len(q) for s, q in queries.items()}, "total_entities": len(jobs),
               "cached": len(jobs) - len(pending), **tot, "model": args.model,
               "effort": args.effort, "contexts_per_entity": args.contexts,
               "protocol": "bio8b max-diverse prompt, generated by Claude",
               "failures": failures}
    atomic_json_dump(summary, out / "generation_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} entities failed; rerun to retry them")


if __name__ == "__main__":
    main()
