import ast
import csv
import math
import torch
import numpy as np

CACHE = (
    "neurosymbolic_outputs_v3/"
    "NCIT-DOID_valid_n50_ctx5_L20_stable0.6/"
    "representations_v3.pt"
)

POOL = "bio-ml/NCIT-DOID/local.valid.cands.tsv"
LIMIT = 50

METHODS = [
    "dense_single",
    "dense_mean",
    "sae_single",
    "sae_mean",
    "sae_stable",
]


def cosine(a, b):
    a = a.float()
    b = b.float()

    na = a.norm()
    nb = b.norm()

    if float(na) == 0.0 or float(nb) == 0.0:
        return -1.0

    return float(torch.dot(a, b) / (na * nb))


print("Loading cache:")
print(CACHE)

try:
    cache = torch.load(
        CACHE,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    cache = torch.load(
        CACHE,
        map_location="cpu",
    )

src_cached = {
    k[len("src|"):]
    for k in cache
    if k.startswith("src|")
}

tgt_cached = {
    k[len("tgt|"):]
    for k in cache
    if k.startswith("tgt|")
}

print()
print("Cached representations")
print("----------------------")
print("Sources:", len(src_cached))
print("Targets:", len(tgt_cached))
print("Total:", len(src_cached) + len(tgt_cached))


# --------------------------------------------------
# Load first 50 Bio-ML ranking rows
# --------------------------------------------------

queries = []

with open(POOL, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for qid, row in enumerate(reader):
        if qid >= LIMIT:
            break

        candidates = ast.literal_eval(
            row["TgtCandidates"]
        )

        candidates = list(
            dict.fromkeys(
                str(x).strip()
                for x in candidates
            )
        )

        queries.append({
            "query_id": qid,
            "src": row["SrcEntity"].strip(),
            "gold": row["TgtEntity"].strip(),
            "candidates": candidates,
        })


# --------------------------------------------------
# Determine which queries are fully computable
# --------------------------------------------------

complete = []
incomplete = []

for q in queries:
    src_ok = q["src"] in src_cached

    missing_targets = [
        tgt
        for tgt in q["candidates"]
        if tgt not in tgt_cached
    ]

    if src_ok and not missing_targets:
        complete.append(q)
    else:
        incomplete.append({
            "query_id": q["query_id"],
            "src_ok": src_ok,
            "missing_targets": len(missing_targets),
        })


print()
print("Query coverage")
print("--------------")
print(
    f"Complete queries: {len(complete)} / {len(queries)}"
)
print(
    f"Incomplete queries: {len(incomplete)} / {len(queries)}"
)

if incomplete:
    print()
    print("First incomplete queries:")
    for x in incomplete[:10]:
        print(
            f"  query {x['query_id']}: "
            f"source_cached={x['src_ok']}, "
            f"missing_targets={x['missing_targets']}"
        )

if not complete:
    raise SystemExit(
        "\nNo query currently has all 100 candidate "
        "representations cached."
    )


# --------------------------------------------------
# Score completed queries
# --------------------------------------------------

print()
print("Results on COMPLETE 100-candidate queries only")
print("===============================================")

all_results = {}

for method in METHODS:

    reciprocal_ranks = []
    hits1 = 0
    hits5 = 0
    hits10 = 0

    ranks = []

    for q in complete:

        svec = cache[
            f"src|{q['src']}"
        ][method]

        scored = []

        for tgt in q["candidates"]:

            tvec = cache[
                f"tgt|{tgt}"
            ][method]

            score = cosine(
                svec,
                tvec,
            )

            if not math.isfinite(score):
                score = -1.0

            scored.append(
                (tgt, score)
            )

        scored.sort(
            key=lambda x: (-x[1], x[0])
        )

        rank = next(
            i + 1
            for i, (tgt, _)
            in enumerate(scored)
            if tgt == q["gold"]
        )

        ranks.append(rank)

        reciprocal_ranks.append(
            1.0 / rank
        )

        hits1 += int(rank <= 1)
        hits5 += int(rank <= 5)
        hits10 += int(rank <= 10)

    n = len(complete)

    mrr = float(
        np.mean(reciprocal_ranks)
    )

    h1 = hits1 / n
    h5 = hits5 / n
    h10 = hits10 / n

    all_results[method] = {
        "MRR": mrr,
        "H@1": h1,
        "H@5": h5,
        "H@10": h10,
        "ranks": ranks,
    }

    print(
        f"[{method:12s}] "
        f"MRR={mrr:.4f} "
        f"H@1={h1:.4f} "
        f"H@5={h5:.4f} "
        f"H@10={h10:.4f}"
    )


# --------------------------------------------------
# Show per-query gold ranks
# --------------------------------------------------

print()
print("Gold rank per completed query")
print("-----------------------------")

header = (
    "qid  "
    + "  ".join(f"{m:>12s}" for m in METHODS)
)

print(header)

for i, q in enumerate(complete):

    values = [
        all_results[m]["ranks"][i]
        for m in METHODS
    ]

    print(
        f"{q['query_id']:>3d}  "
        + "  ".join(
            f"{r:>12d}"
            for r in values
        )
    )

print()
print(
    "IMPORTANT: These metrics are for the "
    f"{len(complete)} fully completed queries only, "
    "not the full requested 50-query experiment."
)
