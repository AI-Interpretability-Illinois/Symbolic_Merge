#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_native_views.py

Contexts for ontology/KG symbols taken from the datasets themselves, with no text
generation: each view pairs ONE piece of the symbol's own system-internal
evidence with the symbol, evidence first and symbol last, e.g.

    Instance: Edward Thomas (poet). Class: Writer            (Common-KG)
    domain 作者. Term: 撰写论文                                  (MultiFarm)
    Parent: Intestinal Atresia. Class: Duodenal Atresia        (Bio-ML, Anatomy)

The symbol comes last because the model is causal: its tokens see only what
precedes them, so a view that opened with the symbol would give every view the
same representation. This mirrors Valentine's "name" preset (value first, then
the column name). The symbol's own tokens are pooled exactly as in the
generated-context pipeline (cache with --contexts 0 --label_last).

Evidence per task, all from the benchmark's own files:
  commonkg   instances of the class in its own KG (NELL vs DBpedia, YAGO vs Wikidata)
  multifarm  the term's own axioms in its own ontology and language: parents and
             children, domain/range, properties whose domain/range it is,
             sub-/super-properties, inverses, disjoint classes
  bioml      NCIT / DOID definitions and parent classes
  anatomy    MA / NCI definitions, parents and part-of
Synonyms are never used: across these pairs a synonym is often the other system's
label, which would hand over the answer.

Views per entity are capped at --max_views (a seeded sample when there are more);
an entity with no evidence gets the single view "Class: <label>".

Output per task: a context store (selected_queries.json or views/<name>/, and
contexts/{src,tgt}/*.json) that cache_bioml_hidden.py, encode_bioml_cached.py and
analyze_bioml_idf.py read unchanged.

Example:
    python native/build_native_views.py commonkg --data_dir <commonkg> --output_dir <out>
"""

import argparse
import hashlib
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for sub in ("bioml", "kg", "multilingual", "ontology"):
    sys.path.insert(0, str(REPO / sub))
from generate_bioml_contexts_bio8b_maxdiverse import (  # noqa: E402
    atomic_json_dump, entity_output_path, load_cache, load_queries)
import generate_bioml_contexts_claude as G  # noqa: E402

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"


SAMPLE_SALT = ""   # set from --seed; salts every per-entity sampling key


def seeded_sample(items, k, key):
    items = sorted(dict.fromkeys(i for i in items if i))
    if len(items) <= k:
        return items
    key = key + SAMPLE_SALT
    return sorted(random.Random(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)).sample(items, k))


def write_entity(out, side, iri, label, views, kind):
    texts = views or [f"{kind}: {label}"]
    atomic_json_dump({"entity_iri": iri, "side": side, "preferred_label": label,
                      "native": True, "contexts": [{"text": t} for t in texts],
                      "parsed_output": {"contexts": [{"text": t} for t in texts]}},
                     entity_output_path(out, side, iri))
    return len(views)


def report(name, counts):
    n = len(counts)
    print(f"{name}: {n} entities, views/entity mean {sum(counts) / max(1, n):.1f}, "
          f"without evidence {sum(c == 0 for c in counts)}", flush=True)


# --- Common-KG ---------------------------------------------------------------------

def commonkg(args):
    import prepare_commonkg as CK
    for case in CK.CASES:
        out = args.output_dir / case
        refs, src_kg, tgt_kg, queries = CK.load_case(case, args.data_dir)
        atomic_json_dump(queries, out / "selected_queries.json")
        counts = []
        for side, kg, iris in (("src", src_kg, sorted({a for a, _ in refs})),
                               ("tgt", tgt_kg, queries[0]["candidates"])):
            for iri in iris:
                label = CK.local_name(iri)
                views = [f"Instance: {i}. Class: {label}"
                         for i in seeded_sample(kg[iri], args.max_views, iri)]
                counts.append(write_entity(out, side, iri, label, views, "Class"))
        report(f"commonkg/{case} ({len(queries)} queries)", counts)


# --- MultiFarm ---------------------------------------------------------------------

def ontology_axioms(path):
    """{iri: [(relation, other iri)]} for every labelled entity, both directions."""
    root = ET.parse(path).getroot()
    rel = defaultdict(list)
    out_tags = {RDFS + "subClassOf": ("subClassOf", "superClassOf"),
                RDFS + "subPropertyOf": ("subPropertyOf", "superPropertyOf"),
                RDFS + "domain": ("domain", "domainOf"),
                RDFS + "range": ("range", "rangeOf"),
                OWL + "inverseOf": ("inverseOf", "inverseOf"),
                OWL + "disjointWith": ("disjointWith", "disjointWith")}
    for el in root:
        about = el.attrib.get(RDF + "about")
        if not about:
            continue
        for child in el:
            if child.tag not in out_tags:
                continue
            other = child.attrib.get(RDF + "resource")
            if not other:
                continue
            fwd, back = out_tags[child.tag]
            rel[about].append((fwd, other))
            rel[other].append((back, about))
    return rel


def multifarm(args):
    import prepare_multifarm as MF
    out = args.output_dir
    by_pair, src_ents, tgt_ents = MF.collect_queries(args.data_dir, args.pairs)
    for pair, queries in by_pair.items():
        G.write_view(out, pair, queries)
        print(f"multifarm {pair}: {len(queries)} queries", flush=True)
    axioms, labels = {}, {}

    def onto(o, lang):
        if (o, lang) not in axioms:
            path = args.data_dir / "ont" / lang / f"{o}-{lang}.owl"
            axioms[o, lang] = ontology_axioms(path)
            labels[o, lang] = {i: e["preferred_label"]
                               for i, e in MF.parse_ontology(path, o, lang).items()}
        return axioms[o, lang], labels[o, lang]

    counts = []
    for side, ents in (("src", src_ents), ("tgt", tgt_ents)):
        for iri, e in sorted(ents.items()):
            rel, lab = onto(e["onto"], e["lang"])
            items = [f"{r} {lab[o]}. Term: {e['preferred_label']}"
                     for r, o in rel.get(iri, []) if o in lab]
            views = seeded_sample(items, args.max_views, iri)
            counts.append(write_entity(out, side, iri, e["preferred_label"], views, "Term"))
    report("multifarm", counts)


# --- Bio-ML and Anatomy ------------------------------------------------------------

def definition_parent_views(label, definitions, parents, key, k):
    items = [f"Definition: {d.strip().rstrip('.')}. Class: {label}" for d in definitions]
    for p in parents:
        items.append(f"Part of: {p[len('part of '):]}. Class: {label}" if p.startswith("part of ")
                     else f"Parent: {p}. Class: {label}")
    return seeded_sample(items, k, key)


def bioml(args):
    repo = args.data_dir / "bio-ml"
    out = args.output_dir
    ncit = load_cache(repo / "ontology_metadata_cache" / "NCIT_metadata_cache.pt", "NCIT")
    doid = load_cache(repo / "ontology_metadata_cache" / "DOID_metadata_cache.pt", "DOID")
    src, tgt = set(), set()
    for sp in args.splits.split(","):
        qs = load_queries(repo / "NCIT-DOID" / f"local.{sp}.cands.tsv", 0, 10 ** 9)
        G.write_view(out, sp, qs)
        src |= {q["src"] for q in qs}
        tgt |= {c for q in qs for c in q["candidates"]}
        print(f"bioml {sp}: {len(qs)} queries", flush=True)
    counts = []
    for side, cache, iris in (("src", ncit, src), ("tgt", doid, tgt)):
        for iri in sorted(iris):
            raw = cache[iri]
            label = raw.get("label") or iri
            views = definition_parent_views(label, raw.get("definitions") or [],
                                            raw.get("parent_labels") or [], iri, args.max_views)
            counts.append(write_entity(out, side, iri, label, views, "Class"))
    report("bioml", counts)


def anatomy(args):
    import prepare_anatomy as PA
    d, out = args.data_dir, args.output_dir
    mouse = PA.parse_ontology(d / "source.rdf", "MA")
    human = PA.parse_ontology(d / "target.rdf", "NCI")
    refs = []
    for cell in ET.parse(d / "reference.rdf").getroot().iter(PA.ALIGN + "Cell"):
        a = cell.find(PA.ALIGN + "entity1").attrib[RDF + "resource"]
        b = cell.find(PA.ALIGN + "entity2").attrib[RDF + "resource"]
        if (cell.findtext(PA.ALIGN + "relation") or "=").strip() == "=" and a in mouse and b in human:
            refs.append((a, b))
    targets = sorted(human)
    atomic_json_dump([{"query_id": i, "src": a, "gold": b, "candidates": targets}
                      for i, (a, b) in enumerate(sorted(refs))], out / "selected_queries.json")
    counts = []
    for side, ents, iris in (("src", mouse, sorted({a for a, _ in refs})), ("tgt", human, targets)):
        for iri in iris:
            m = ents[iri]
            views = definition_parent_views(m["preferred_label"], m["definitions"],
                                            m["parent_labels"], iri, args.max_views)
            counts.append(write_entity(out, side, iri, m["preferred_label"], views, "Class"))
    report(f"anatomy ({len(refs)} queries)", counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=("commonkg", "multifarm", "bioml", "anatomy"))
    ap.add_argument("--data_dir", type=Path, required=True,
                    help="commonkg: dir with the two checkouts; multifarm: dataset-2015-testing; "
                         "bioml: OAEI-Bio-ML checkout; anatomy: dir with source/target/reference.rdf")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_views", type=int, default=16)
    ap.add_argument("--pairs", default="cn-en,en-ru,ar-en", help="multifarm")
    ap.add_argument("--splits", default="valid,train", help="bioml")
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = the default sample; any other value draws a different sample of "
                         "views per entity (for a seed study)")
    args = ap.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    global SAMPLE_SALT
    SAMPLE_SALT = f"#seed{args.seed}" if args.seed else ""
    {"commonkg": commonkg, "multifarm": multifarm, "bioml": bioml, "anatomy": anatomy}[args.task](args)


if __name__ == "__main__":
    main()
