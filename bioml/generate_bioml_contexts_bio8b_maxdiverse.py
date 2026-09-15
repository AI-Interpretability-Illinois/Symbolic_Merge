#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, ast, csv, hashlib, json, os, random, re, tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def iri_tail(iri):
    s = str(iri).rstrip("/")
    return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]


def sha10(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def atomic_json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_candidates(raw):
    try:
        obj = ast.literal_eval(raw)
    except Exception:
        obj = json.loads(raw)
    return list(dict.fromkeys(str(x).strip() for x in obj if str(x).strip()))


def load_queries(path, start, limit):
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for qid, row in enumerate(reader):
            if qid < start:
                continue
            if len(out) >= limit:
                break
            out.append({
                "query_id": qid,
                "src": row["SrcEntity"].strip(),
                "gold": row["TgtEntity"].strip(),
                "candidates": parse_candidates(row["TgtCandidates"]),
            })
    if not out:
        raise RuntimeError(f"No queries loaded from {path}")
    return out


def load_cache(path, name):
    print(f"[cache] loading {name}: {path}", flush=True)
    try:
        wrapper = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        wrapper = torch.load(path, map_location="cpu")
    info = wrapper["info"]
    print(f"[cache] {name}: {len(info):,} entities", flush=True)
    return info


def metadata_for(iri, raw, ontology):
    return {
        "entity_iri": iri,
        "ontology": ontology,
        "preferred_label": raw.get("label") or iri_tail(iri),
        "synonyms": list(raw.get("synonyms") or []),
        "definitions": list(raw.get("definitions") or []),
        "parent_labels": list(raw.get("parent_labels") or []),
    }


SEMANTIC_AXES = [
    "core definition / concept identity",
    "ontology classification / broader biomedical category",
    "anatomy / tissue / organ system / anatomical localization",
    "pathology / pathophysiology / mechanism / biological process",
    "clinical manifestation / phenotype / symptom / functional consequence",
    "etiology / genetic basis / environmental cause / risk factor",
    "diagnosis / imaging / laboratory finding / biomarker / clinical recognition",
    "treatment / management / prognosis / clinical course",
    "epidemiology / population / developmental stage / age-related presentation",
    "terminology / synonymy / relationship to a nearby biomedical concept",
]

SYNTAX_STYLES = [
    "direct declarative clause",
    "subordinate-clause-first sentence",
    "relative-clause construction",
    "passive construction",
    "clinical observation framing",
    "classification statement",
    "cause-to-consequence construction",
    "contrastive or distinguishing construction",
    "diagnostic reasoning construction",
    "definition-by-description construction",
    "conditional construction",
    "appositive / parenthetical construction",
]

POSITION_PLANS = [
    ("early", 0.20, "place the entity around the first quarter of the sentence, but not as the first token"),
    ("early_middle", 0.35, "place the entity around one-third of the way through the sentence"),
    ("middle", 0.50, "place the entity near the middle of the sentence"),
    ("late_middle", 0.68, "place the entity around two-thirds of the way through the sentence"),
    ("late", 0.85, "place the entity near the end, with only a short trailing phrase after it"),
]


def deterministic_plans(meta, n):
    seed = int(hashlib.sha1(meta["entity_iri"].encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    axes = SEMANTIC_AXES[:]
    styles = SYNTAX_STYLES[:]
    rng.shuffle(axes)
    rng.shuffle(styles)
    positions = POSITION_PLANS[:] if n == 5 else [POSITION_PLANS[i % 5] for i in range(n)]
    out = []
    for i in range(n):
        name, frac, instr = positions[i]
        out.append({
            "slot": i + 1,
            "semantic_focus": axes[i % len(axes)],
            "syntax_style": styles[i % len(styles)],
            "position_name": name,
            "target_symbol_fraction": frac,
            "position_instruction": instr,
        })
    return out


SYSTEM_PROMPT = """You are a biomedical language model.

Create highly diverse biomedical sentence contexts for one entity.

The SAME entity will appear in all contexts, so maximize everything else:
semantic content, surrounding vocabulary, grammar, clause structure, sentence rhythm,
biomedical perspective, and where the entity appears in the sentence.

Do not merely paraphrase the same definition. Use the supplied ontology metadata
and your own biomedical knowledge when helpful. Keep statements medically plausible.

You will NOT write the entity name itself. Generate LEFT and RIGHT text fragments;
Python will insert the exact preferred label between them.

Return only valid JSON with no commentary."""


def build_prompt(meta, n):
    plans = deterministic_plans(meta, n)
    return f"""ENTITY METADATA:
{json.dumps(meta, ensure_ascii=False, indent=2)}

Generate exactly {n} MAXIMALLY DIVERSE biomedical contexts for this SAME entity.

For each context, write:
- "left": text BEFORE the entity
- "right": text AFTER the entity

Python will construct:
LEFT + [EXACT ENTITY LABEL] + RIGHT

Do NOT include the preferred label itself in either fragment.

CRITICAL DIVERSITY GOAL:
The non-entity tokens across the {n} contexts should overlap as little as reasonably possible while remaining natural and medically meaningful.

Vary ALL of the following:
- biomedical content / perspective
- main verbs and nouns
- modifiers and descriptive vocabulary
- grammar and clause order
- active vs passive voice
- sentence length and rhythm
- discourse framing
- relation expressed around the entity
- entity position within the sentence

Do NOT:
- paraphrase the same definition repeatedly
- reuse the same opening phrase
- reuse the same linking phrase
- reuse the same main verb across multiple contexts unless unavoidable
- use the same sentence template repeatedly
- place the entity in the same position every time

Follow these DIFFERENT plans independently:
{json.dumps(plans, ensure_ascii=False, indent=2)}

POSITION RULE:
Approximate the requested entity position by controlling LEFT versus RIGHT length.
- early: short LEFT, much longer RIGHT
- early_middle: moderate LEFT, longer RIGHT
- middle: similar LEFT and RIGHT lengths
- late_middle: longer LEFT, shorter RIGHT
- late: much longer LEFT, very short RIGHT

Even for an early position, put some meaningful biomedical material before the entity.

Return ONLY JSON in this exact form:
{{
  "contexts": [
    {{"left": "text before entity", "right": "text after entity"}},
    {{"left": "text before entity", "right": "text after entity"}}
  ]
}}

The contexts array must contain exactly {n} objects.
"""


def render_chat(tok, prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{prompt}\n\nASSISTANT:\n"


def strip_fence(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def recover_pairs(raw):
    cleaned = strip_fence(raw)

    def pull(obj):
        if not isinstance(obj, dict):
            return None
        xs = obj.get("contexts")
        if not isinstance(xs, list):
            return None
        out = []
        for x in xs:
            if isinstance(x, dict):
                out.append({
                    "left": str(x.get("left", "")).strip(),
                    "right": str(x.get("right", "")).strip(),
                })
        return out if out else None

    try:
        got = pull(json.loads(cleaned))
        if got:
            return got
    except Exception:
        pass

    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a >= 0 and b > a:
        try:
            got = pull(json.loads(cleaned[a:b+1]))
            if got:
                return got
        except Exception:
            pass

    return []


def fallback_pairs(meta, n):
    defs = meta.get("definitions") or []
    parents = meta.get("parent_labels") or []
    syns = meta.get("synonyms") or []
    d = str(defs[0]).strip().rstrip(".") if defs else "a biomedical concept represented in this ontology"
    p = str(parents[0]).strip() if parents else "a broader biomedical category"
    s = str(syns[0]).strip() if syns else "an alternative biomedical term"

    base = [
        {"left": "In biomedical terminology,", "right": f" can be described as {d}."},
        {"left": f"Within {p}, the concept", "right": " occupies a more specific pathological category."},
        {"left": "When clinicians consider anatomy, phenotype, and disease pattern,", "right": " may enter the differential description."},
        {"left": f"Although related terminology such as {s} may appear in the literature, the ontology concept", "right": " retains its preferred designation."},
        {"left": f"A biomedical record characterized by {d} corresponds to", "right": "."},
    ]
    return [base[i % len(base)] for i in range(n)]


def clean_fragment(text, label):
    s = str(text).strip()
    s = re.sub(re.escape(label), "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def join_context(left, label, right):
    left = clean_fragment(left, label)
    right = clean_fragment(right, label)

    if not left:
        left = "In this biomedical setting,"
    if not right:
        right = "."

    text = f"{left} {label}"
    if right[0] in ".,;:!?)]":
        text += right
    else:
        text += " " + right

    text = re.sub(r"\s+", " ", text).strip()
    if text[-1] not in ".!?":
        text += "."

    start = text.index(label)
    end = start + len(label)
    before_words = len(text[:start].split())
    total_words = max(1, len(text.split()))

    return {
        "text": text,
        "symbol": label,
        "char_span": [start, end],
        "symbol_word_fraction": before_words / total_words,
    }


def make_contexts(raw, meta, n):
    pairs = recover_pairs(raw)
    if len(pairs) < n:
        fills = fallback_pairs(meta, n)
        pairs = pairs + fills[len(pairs):n]
    else:
        pairs = pairs[:n]

    label = meta["preferred_label"]
    return [join_context(p.get("left", ""), label, p.get("right", "")) for p in pairs]


def resolve_dtype(name):
    return {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }[name.lower()]


class Generator:
    def __init__(self, model_path, device, dtype, max_input_tokens):
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens

        print(f"[model] loading tokenizer: {model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        print(f"[model] loading Bio-8B-it: {model_path}", flush=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
            ).to(self.device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
            ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def generate_batch(self, prompts, max_new_tokens, temperature, top_p):
        rendered = [render_chat(self.tok, p) for p in prompts]
        enc = self.tok(rendered, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_input_tokens)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        width = enc["input_ids"].shape[1]

        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        out = self.model.generate(**enc, **kwargs)
        return [self.tok.decode(row[width:], skip_special_tokens=True).strip() for row in out]


def entity_output_path(run_dir, side, iri):
    return run_dir / "contexts" / side / f"{iri_tail(iri)}_{sha10(iri)}.json"


def complete_cache(path, iri, n):
    if not path.exists():
        return False
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
        return x.get("entity_iri") == iri and isinstance(x.get("contexts"), list) and len(x["contexts"]) == n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", type=Path, required=True)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--pair", default="NCIT-DOID")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--contexts", type=int, default=5)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_input_tokens", type=int, default=3072)
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo = args.repo_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    pool = repo / "bio-ml" / args.pair / f"local.{args.split}.cands.tsv"
    queries = load_queries(pool, args.start, args.limit)

    ncit = load_cache(repo / "bio-ml" / "ontology_metadata_cache" / "NCIT_metadata_cache.pt", "NCIT")
    doid = load_cache(repo / "bio-ml" / "ontology_metadata_cache" / "DOID_metadata_cache.pt", "DOID")

    src_iris = list(dict.fromkeys(q["src"] for q in queries))
    tgt_iris = list(dict.fromkeys(c for q in queries for c in q["candidates"]))

    jobs = []
    for iri in src_iris:
        jobs.append({"side": "src", "iri": iri, "meta": metadata_for(iri, ncit[iri], "NCIT")})
    for iri in tgt_iris:
        jobs.append({"side": "tgt", "iri": iri, "meta": metadata_for(iri, doid[iri], "DOID")})

    atomic_json_dump(queries, run_dir / "selected_queries.json")
    atomic_json_dump({
        "queries": len(queries),
        "source_entities": len(src_iris),
        "target_entities": len(tgt_iris),
        "total_entities": len(jobs),
        "contexts_per_entity": args.contexts,
        "protocol": "max-diverse semantic+lexical+syntactic+symbol-position context generation",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }, run_dir / "run_config.json")

    print("\n" + "=" * 110)
    print("BIO-8B MAX-DIVERSE CONTEXT GENERATION")
    print("=" * 110)
    print(f"queries          : {len(queries)}")
    print(f"source entities  : {len(src_iris)}")
    print(f"target entities  : {len(tgt_iris)}")
    print(f"total entities   : {len(jobs)}")
    print(f"contexts/entity  : {args.contexts}")
    print(f"batch_size       : {args.batch_size}")

    pending, cached = [], 0
    for idx, job in enumerate(jobs, 1):
        path = entity_output_path(run_dir, job["side"], job["iri"])
        item = dict(job)
        item["index"] = idx
        item["path"] = path
        if not args.overwrite and complete_cache(path, job["iri"], args.contexts):
            cached += 1
        else:
            pending.append(item)

    print(f"\npending={len(pending)} cached={cached}")
    if not pending:
        print("Everything already complete.")
        return

    gen = Generator(str(Path(args.model_path).expanduser().resolve()),
                    args.device, resolve_dtype(args.dtype), args.max_input_tokens)

    completed = 0
    for b in range(0, len(pending), args.batch_size):
        batch = pending[b:b+args.batch_size]
        prompts = [build_prompt(j["meta"], args.contexts) for j in batch]
        raws = gen.generate_batch(prompts, args.max_new_tokens, args.temperature, args.top_p)

        for job, prompt, raw in zip(batch, prompts, raws):
            contexts = make_contexts(raw, job["meta"], args.contexts)
            result = {
                "entity_iri": job["iri"],
                "side": job["side"],
                "preferred_label": job["meta"]["preferred_label"],
                "entity_metadata_given_to_model": job["meta"],
                "diversity_plans": deterministic_plans(job["meta"], args.contexts),
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompt,
                "raw_model_output": raw,
                "contexts": contexts,
                "parsed_output": {"contexts": [{"text": c["text"]} for c in contexts]},
            }
            atomic_json_dump(result, job["path"])
            completed += 1

            print(f"\n[completed {completed}/{len(pending)}] "
                  f"entity {job['index']}/{len(jobs)} {job['side']} "
                  f"{iri_tail(job['iri'])} :: {job['meta']['preferred_label']}")
            for i, c in enumerate(contexts, 1):
                print(f"  {i}. {c['text']}")
                print(f"     char_span={c['char_span']} symbol_fraction≈{c['symbol_word_fraction']:.2f}")

    summary = {
        "total_entities": len(jobs),
        "cached": cached,
        "newly_completed": completed,
        "failed": 0,
        "contexts_per_entity": args.contexts,
        "validation": False,
        "retry": False,
    }
    atomic_json_dump(summary, run_dir / "generation_summary.json")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
