#!/usr/bin/env python3
"""Generate the paper's LaTeX tables from the result files in data/, so every
number in the paper is copied by code rather than by hand."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(HERE / "data" / "all_results.csv")))
val = lambda r, k: float(r[k]) if r.get(k) else None
METHODS = ["strings", "dense", "dense_pc1", "sae_mean", "sae_idf"]


def fmt(v, best=False, nd=3):
    if v is None:
        return "--"
    s = f"{v:.{nd}f}"
    return r"\textbf{" + s + "}" if best else s


def cells(d, methods=METHODS, nd=3):
    present = {m: d[m] for m in methods if d.get(m) is not None}
    top = max(present, key=present.get)
    return [fmt(d.get(m), m == top, nd) for m in methods]


def write(name, text):
    (OUT / name).write_text(text)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- main table
def pick(task, metric="MRR", group="main"):
    return next(r for r in rows if r["task"] == task and r["metric"] == metric and r["group"] == group)


xl = [r for r in rows if r["task"].startswith("XLCoST") and r["metric"] == "MRR"]
xl_mean = {m: sum(val(r, m) for r in xl) / len(xl) for m in METHODS}
xl6 = [r for r in rows if r["task"].startswith("XLCoST") and r["metric"] == "Precision@6"]
xl6_mean = {m: sum(val(r, m) for r in xl6) / len(xl6) for m in METHODS}
spec = [
    ("Formal mathematics (miniF2F)", [
        ("Lean 4 $\\to$ Isabelle", "MRR", {m: val(pick("miniF2F Lean4->Isabelle"), m) for m in METHODS}),
        ("Lean 4 $\\to$ Metamath", "MRR", {m: val(pick("miniF2F Lean4->Metamath"), m) for m in METHODS}),
        ("Lean 4 $\\to$ HOL Light", "MRR", {m: val(pick("miniF2F Lean4->HOL Light"), m) for m in METHODS})]),
    ("Code (XLCoST, program level)", [
        ("7 languages, mean", "MRR$^\\dagger$", xl_mean),
        ("7 languages, mean", "P@6$^\\dagger$", xl6_mean)]),
    ("Table schemas (Valentine, 551 pairs)", [
        ("source $\\to$ target columns", "F1", {m: val(pick("Valentine (551 pairs)", "F1Score"), m) for m in METHODS}),
        ("source $\\to$ target columns", "MRR", {m: val(pick("Valentine (551 pairs)", "MeanReciprocalRank"), m) for m in METHODS})]),
    ("Knowledge graphs (OAEI Common-KG)", [
        ("NELL $\\to$ DBpedia", "MRR", {m: val(pick("Common-KG NELL->DBpedia"), m) for m in METHODS}),
        ("YAGO $\\to$ Wikidata", "MRR", {m: val(pick("Common-KG YAGO->Wikidata"), m) for m in METHODS})]),
    ("Cross-lingual ontologies (MultiFarm)", [
        ("Chinese $\\to$ English", "MRR", {m: val(pick("MultiFarm Chinese->English"), m) for m in METHODS}),
        ("Russian $\\to$ English", "MRR", {m: val(pick("MultiFarm Russian->English"), m) for m in METHODS}),
        ("Arabic $\\to$ English", "MRR", {m: val(pick("MultiFarm Arabic->English"), m) for m in METHODS})]),
]
E = HERE.parent / "paper" / "data" / "embed_baseline.csv"
EMB = {}
if E.exists():
    for r in csv.DictReader(open(E)):
        EMB[(r["task"], r["metric"])] = float(r["emb"])
emb_key = {"Lean 4 $\\to$ Isabelle": "Lean 4 $\\to$ Isabelle", "Lean 4 $\\to$ Metamath": "Lean 4 $\\to$ Metamath", "Lean 4 $\\to$ HOL Light": "Lean 4 $\\to$ HOL Light",
           "7 languages, mean": "XLCoST, 7 languages", "source $\\to$ target columns": "Valentine columns", "NELL $\\to$ DBpedia": "NELL $\\to$ DBpedia",
           "YAGO $\\to$ Wikidata": "YAGO $\\to$ Wikidata", "Chinese $\\to$ English": "MultiFarm zh $\\to$ en", "Russian $\\to$ English": "MultiFarm ru $\\to$ en", "Arabic $\\to$ English": "MultiFarm ar $\\to$ en"}
METHODS_E = ["strings", "dense", "dense_pc1", "sae_mean", "sae_idf", "bge"]
lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Task & Metric & Strings & Dense & \makecell{Dense,\\PC1 rm.} & \makecell{SAE\\unweighted} & \textbf{SAE-IDF} & \makecell{BGE-M3\\retriever} & $\Delta$ dense \\",
         r"\midrule"]
for group, items in spec:
    lines.append(r"\multicolumn{9}{@{}l}{\textit{" + group + r"}} \\")
    for name, metric, d in items:
        delta = d["sae_idf"] - d["dense"]
        mkey = {"MRR": "MRR", "MRR$^\\dagger$": "MRR", "P@6$^\\dagger$": "P@6", "F1": "F1"}[metric]
        d = dict(d); d["bge"] = EMB.get((emb_key[name], mkey))
        lines.append(f"\\quad {name} & {metric} & " + " & ".join(cells(d, METHODS_E)) + f" & {delta:+.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_main.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- appendix: XLCoST per language
lang = {"Java": "Java", "Cpp": "C++", "Python": "Python", "Csharp": "C\\#", "Javascript": "JavaScript", "PHP": "PHP", "C": "C"}
lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Query language & Programs & Metric & Strings & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
for L, name in lang.items():
    for metric, label in (("MRR", "MRR"), ("Precision@6", "P@6")):
        r = pick(f"XLCoST {L}", metric)
        d = {m: val(r, m) for m in METHODS}
        first = f"{name} & {int(r['n']):,}" if metric == "MRR" else " & "
        lines.append(f"{first} & {label} & " + " & ".join(cells(d)) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_xlcost.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- appendix: other tasks
lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Task & Queries & Strings & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
for task, name in (("Bio-ML NCIT->DOID valid", "Bio-ML NCIT $\\to$ DOID, valid split"), ("Bio-ML NCIT->DOID train", "Bio-ML NCIT $\\to$ DOID, train split"),
                   ("Anatomy mouse->human", "OAEI Anatomy, mouse $\\to$ human")):
    r = pick(task, "MRR", "appendix")
    lines.append(f"{name} & {int(r['n']):,} & " + " & ".join(cells({m: val(r, m) for m in METHODS})) + r" \\")
lines.append(r"\midrule")
for task, name in (("PubChem SMILES->IUPAC", "PubChem SMILES $\\to$ IUPAC"), ("PubChem SMILES->InChI", "PubChem SMILES $\\to$ InChI"), ("PubChem IUPAC->InChI", "PubChem IUPAC $\\to$ InChI")):
    r = pick(task, "MRR", "appendix")
    lines.append(f"{name} & {int(r['n']):,} & " + " & ".join(fmt(val(r, m)) for m in METHODS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_appendix_tasks.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- appendix: generated contexts
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Task & Contexts & Strings & Dense & $-$PC1 & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
gen_tasks = [("Common-KG NELL->DBpedia", "NELL $\\to$ DBpedia"), ("Common-KG YAGO->Wikidata", "YAGO $\\to$ Wikidata"),
             ("MultiFarm Chinese->English", "MultiFarm zh $\\to$ en"), ("MultiFarm Russian->English", "MultiFarm ru $\\to$ en"),
             ("MultiFarm Arabic->English", "MultiFarm ar $\\to$ en"), ("Drugs brand->generic", "Drug brand $\\to$ generic")]
note_label = {"Claude, both sides": "one LLM", "Claude src + GPT-5.6 tgt": "two LLMs",
              "Claude, both sides, shared class metadata": "one LLM, shared metadata", "native context": "native"}
for task, name in gen_tasks:
    rs = [r for r in rows if r["task"] == task and r["metric"] == "MRR"]
    rs = sorted(rs, key=lambda r: (r["group"] != "generated", r["note"]))
    for i, r in enumerate(rs):
        d = {m: val(r, m) for m in METHODS}
        lines.append(f"{name if i == 0 else ''} & {note_label.get(r['note'], r['note'])} & " + " & ".join(cells(d)) + r" \\")
    lines.append(r"\addlinespace[2pt]")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_generated.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- ablation table (compact)
A = {(r["task"], r["config"]): r for r in csv.DictReader(open(HERE / "data" / "ablation_minif2f.csv"))}
def ab(task, cfg, m="sae_idf"):
    return float(A[(task, cfg)][m])
widths_canon = [("16k", "L20_w16k", 68), ("32k", "L20_w32k", 57), ("65k", "L20_w65k", 55), ("131k", "L20_w131k", 114),
                ("262k", "L20_w262k_l0_50", 50), ("524k", "L20_w524k", 48), ("1M", "L20_w1m", 101)]
l0s = [("11", "L20_w131k_l0_11"), ("19", "L20_w131k_l0_19"), ("34", "L20_w131k_l0_34"), ("53", "L20_w131k_l0_53"),
       ("62", "L20_w131k_l0_62"), ("114", "L20_w131k"), ("221", "L20_w131k_l0_221"), ("276", "L20_w131k_l0_276")]
g2b = [("16k", "g2b_L12_w16k"), ("65k", "g2b_L12_w65k"), ("262k", "g2b_L12_w262k")]
lines = [r"\begin{tabular}{@{}p{0.34\linewidth}p{0.3\linewidth}p{0.3\linewidth}@{}}", r"\toprule",
         r"Setting (Gemma-2-9B, layer 20, unless noted) & HOL Light & Metamath \\", r"\midrule",
         f"Dense baseline & {ab('HOLLight','L20_w131k','dense'):.3f} & {ab('Metamath','L20_w131k','dense'):.3f} \\\\",
         r"\addlinespace[2pt]",
         r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textit{SAE width} 16k / 32k / 65k / 131k / 262k / 524k / 1M, each at its canonical $L_0$ (" + " / ".join(str(l) for _, _, l in widths_canon) + r")} \\",
         "\\quad SAE-IDF & " + " / ".join(f"{ab('HOLLight', c):.2f}" for _, c, _ in widths_canon) + " & " + " / ".join(f"{ab('Metamath', c):.2f}" for _, c, _ in widths_canon) + r" \\",
         r"\addlinespace[2pt]",
         r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textit{Sparsity}, average $L_0$ " + " / ".join(l for l, _ in l0s) + r" (131k features)} \\",
         "\\quad SAE-IDF & " + " / ".join(f"{ab('HOLLight', c):.2f}" for _, c in l0s) + " & " + " / ".join(f"{ab('Metamath', c):.2f}" for _, c in l0s) + r" \\",
         r"\addlinespace[2pt]",
         r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textit{Smaller model}: Gemma-2-2B, layer 12, width 16k / 65k / 262k} \\",
         f"\\quad Dense & {ab('HOLLight','g2b_L12_w16k','dense'):.3f} & {ab('Metamath','g2b_L12_w16k','dense'):.3f} \\\\",
         "\\quad SAE-IDF & " + " / ".join(f"{ab('HOLLight', c):.2f}" for _, c in g2b) + " & " + " / ".join(f"{ab('Metamath', c):.2f}" for _, c in g2b) + r" \\",
         r"\addlinespace[2pt]",
         r"\multicolumn{3}{@{}p{\linewidth}@{}}{\textit{Larger model}: Gemma-2-27B, layer 10 / 22 / 34 (131k, canonical $L_0$ 106 / 82 / 72)} \\",
         f"\\quad Dense & " + " / ".join(f"{ab('HOLLight', c, 'dense'):.2f}" for c in ("g27b_L10_w131k","g27b_L22_w131k","g27b_L34_w131k")) + " & " + " / ".join(f"{ab('Metamath', c, 'dense'):.2f}" for c in ("g27b_L10_w131k","g27b_L22_w131k","g27b_L34_w131k")) + r" \\",
         "\\quad SAE-IDF & " + " / ".join(f"{ab('HOLLight', c):.2f}" for c in ("g27b_L10_w131k","g27b_L22_w131k","g27b_L34_w131k")) + " & " + " / ".join(f"{ab('Metamath', c):.2f}" for c in ("g27b_L10_w131k","g27b_L22_w131k","g27b_L34_w131k")) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
write("tab_ablation.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Llama table
cfgs = [("llama_L8_w131k", "Layer 8, 131k"), ("llama_L12_w131k", "Layer 12, 131k"), ("llama_L15_w131k", "Layer 15, 131k"),
        ("llama_L20_w131k", "Layer 20, 131k"), ("llama_L24_w131k", "Layer 24, 131k"), ("llama_L15_w32k", "Layer 15, 32k")]
lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Lean 4 $\to$ HOL Light} & \multicolumn{4}{c}{Lean 4 $\to$ Metamath} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Llama Scope SAE & Dense & PC1 rm. & SAE unw. & \textbf{SAE-IDF} & Dense & PC1 rm. & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
for c, name in cfgs:
    parts = []
    for t in ("HOLLight", "Metamath"):
        d = {m: ab(t, c, m) for m in ("dense", "dense_pc1", "sae_mean", "sae_idf")}
        parts += cells(d, ["dense", "dense_pc1", "sae_mean", "sae_idf"])
    lines.append(f"{name} & " + " & ".join(parts) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_llama.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- full ablation (appendix)
order = [("L5_w131k", "Gemma-2-9B, layer 5, 131k"), ("L9_w131k", "layer 9"), ("L10_w131k", "layer 10"), ("L15_w131k", "layer 15"),
         ("L20_w131k", "layer 20 ($L_0$ 114, main table)"), ("L25_w131k", "layer 25"), ("L30_w131k", "layer 30"), ("L31_w131k", "layer 31"),
         ("L35_w131k", "layer 35"), ("L40_w131k", "layer 40"),
         ("L20_w16k", "layer 20, 16k ($L_0$ 68)"), ("L20_w16k_l0_58", "layer 20, 16k ($L_0$ 58)"), ("L20_w32k", "layer 20, 32k ($L_0$ 57)"),
         ("L20_w65k", "layer 20, 65k ($L_0$ 55)"), ("L20_w262k_l0_50", "layer 20, 262k ($L_0$ 50)"), ("L20_w524k", "layer 20, 524k ($L_0$ 48)"),
         ("L20_w1m", "layer 20, 1M ($L_0$ 101)"), ("L20_w1m_l0_57", "layer 20, 1M ($L_0$ 57)"),
         ("L20_w131k_l0_11", "layer 20, 131k, $L_0$ 11"), ("L20_w131k_l0_19", "$L_0$ 19"), ("L20_w131k_l0_34", "$L_0$ 34"), ("L20_w131k_l0_53", "$L_0$ 53"),
         ("L20_w131k_l0_62", "$L_0$ 62"), ("L20_w131k_l0_221", "$L_0$ 221"), ("L20_w131k_l0_276", "$L_0$ 276"),
         ("g2b_L12_w16k", "Gemma-2-2B, layer 12, 16k"), ("g2b_L12_w65k", "Gemma-2-2B, layer 12, 65k"), ("g2b_L12_w262k", "Gemma-2-2B, layer 12, 262k"),
         ("llama_L8_w131k", "Llama-3.1-8B, layer 8, 131k"), ("llama_L12_w131k", "Llama-3.1-8B, layer 12, 131k"), ("llama_L15_w131k", "Llama-3.1-8B, layer 15, 131k"),
         ("llama_L15_w32k", "Llama-3.1-8B, layer 15, 32k"), ("llama_L20_w131k", "Llama-3.1-8B, layer 20, 131k"), ("llama_L24_w131k", "Llama-3.1-8B, layer 24, 131k"),
         ("g27b_L10_w131k", "Gemma-2-27B, layer 10, 131k ($L_0$ 106)"), ("g27b_L22_w131k", "Gemma-2-27B, layer 22, 131k ($L_0$ 82)"),
         ("g27b_L22_w131k_l0_48", "Gemma-2-27B, layer 22, 131k ($L_0$ 48)"), ("g27b_L22_w131k_l0_150", "Gemma-2-27B, layer 22, 131k ($L_0$ 150)"),
         ("g27b_L34_w131k", "Gemma-2-27B, layer 34, 131k ($L_0$ 72)")]
assert len(order) == 39, len(order)
lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Lean 4 $\to$ HOL Light} & \multicolumn{4}{c}{Lean 4 $\to$ Metamath} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Configuration & Dense & PC1 rm. & SAE unw. & \textbf{SAE-IDF} & Dense & PC1 rm. & SAE unw. & \textbf{SAE-IDF} \\", r"\midrule"]
for c, name in order:
    parts = []
    for t in ("HOLLight", "Metamath"):
        d = {m: ab(t, c, m) for m in ("dense", "dense_pc1", "sae_mean", "sae_idf")}
        parts += cells(d, ["dense", "dense_pc1", "sae_mean", "sae_idf"])
    lines.append(f"{name} & " + " & ".join(parts) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_ablation_full.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- interpretability per task
I = list(csv.DictReader(open(HERE / "data" / "interpretability_summary.csv")))
D = {(r["task"], r["method"]): r for r in I}
tasks = [("miniF2F Lean 4 → Isabelle", "Lean 4 $\\to$ Isabelle"), ("miniF2F Lean 4 → Metamath", "Lean 4 $\\to$ Metamath"),
         ("miniF2F Lean 4 → HOL Light", "Lean 4 $\\to$ HOL Light"), ("XLCoST, Java to 6 languages", "XLCoST, Java $\\to$ 6 languages"),
         ("Valentine, whole column", "Valentine columns"), ("Common-KG NELL → DBpedia", "NELL $\\to$ DBpedia"),
         ("Common-KG YAGO → Wikidata", "YAGO $\\to$ Wikidata"), ("MultiFarm zh → en", "MultiFarm zh $\\to$ en"),
         ("MultiFarm ru → en", "MultiFarm ru $\\to$ en"), ("MultiFarm ar → en", "MultiFarm ar $\\to$ en")]
lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Task & Matches & \makecell[r]{Common ($>$1\%):\\unw. $\to$ SAE-IDF} & \makecell[r]{Rare ($<$0.1\%)\\SAE-IDF} & \makecell[r]{Features for\\50\% / 80\%} & \makecell[r]{Concept / surface / notation\\(SAE-IDF top-5)} \\", r"\midrule"]
tot = 0
for t, name in tasks:
    a, b = D[(t, "sae_mean")], D[(t, "sae_idf")]
    tot += int(b["pairs"])
    lines.append(f"{name} & {b['pairs']} & {float(a['common']):.0%} $\\to$ {float(b['common']):.0%} & {float(b['rare']):.0%} & "
                 f"{float(b['median_n50']):.0f} / {float(b['median_n80']):.0f} & {float(b['concept']):.0%} / {float(b['surface']):.0%} / {float(b['notation']):.0%} \\\\".replace("%", r"\%"))
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_interp.tex", "\n".join(lines) + "\n")
print("total matches decomposed:", tot)

# ---------------------------------------------------------------- Valentine column types
B = json.load(open(HERE / "data" / "boundary_valentine.json"))
print("boundary keys:", list(B.keys())[:10])
