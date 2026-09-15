#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_oaei_starwars_cached.py

Shared-cache Star Wars OAEI KG preprocessing + natural/diverse triple-grounded contexts.

Shared KG cache:
  /u/ramon2004/KG/data/starwars_swtor/
    raw/
    parsed/

Experiment output:
  /u/ramon2004/KG/data/starwars_swtor_q5/
    selected_queries.json
    contexts/
    entity_metadata/
    preparation_summary.json
"""

import argparse, hashlib, json, os, random, re, tempfile, time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, SKOS

SCHEMA_TYPES = {
    str(OWL.Class), str(RDF.Property), str(OWL.ObjectProperty),
    str(OWL.DatatypeProperty), str(OWL.AnnotationProperty), str(RDFS.Class)
}
IMAGE = str(URIRef("http://dbkwik.webdatacommons.org/ontology/Image"))

SKIP_TAILS = {
    "label", "comment", "abstract", "wikiPageExternalLink",
    "wikiPageWikiLinkText", "thumbnail", "depiction", "rights"
}

STYLE_CARDS = [
    ("relation_first",
     "Begin from related nodes or relations and introduce <ENTITY> later. Prefer relational evidence."),
    ("type_attribute",
     "Use type/category/attribute evidence in compact encyclopedic prose. Put <ENTITY> around the middle."),
    ("neighborhood_narrative",
     "Connect several supplied graph facts in a fluent relational narrative with substantial left context before <ENTITY>."),
    ("reverse_relation",
     "Prefer reverse/passive phrasing for incoming edges and identify <ENTITY> later."),
    ("mixed_summary",
     "Blend different supplied KG facts into a natural summary and place <ENTITY> relatively late.")
]

def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def tail(x):
    s = unquote(str(x)).rstrip("/")
    s = s.rsplit("#",1)[-1] if "#" in s else s.rsplit("/",1)[-1]
    return s.replace("_"," ").strip()

def safe_id(iri):
    b = re.sub(r"[^A-Za-z0-9._-]+","_",tail(iri))[:70] or "entity"
    return b + "_" + hashlib.sha1(str(iri).encode()).hexdigest()[:10]

def label_from_graph(g, x):
    u = URIRef(str(x))
    for p in (RDFS.label, SKOS.prefLabel):
        for o in g.objects(u,p):
            if isinstance(o, Literal) and str(o).strip():
                return re.sub(r"\s+"," ",str(o)).strip()
    return tail(x)

def useful_pred(p):
    return tail(p) not in SKIP_TAILS

def rdf_complete(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0: return False
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        end = f.read().lower()
    return b"</rdf:rdf>" in end

def download_first(urls, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if rdf_complete(path):
        print(f"[download] valid cached {path}")
        return
    if path.exists():
        print(f"[download] removing incomplete cache {path}")
        path.unlink()
    last = None
    for url in urls:
        tmp = path.with_suffix(path.suffix + ".part")
        if tmp.exists(): tmp.unlink()
        try:
            print("[download]", url, flush=True)
            with requests.get(url, stream=True, timeout=(30,300), allow_redirects=True) as r:
                if r.status_code != 200:
                    last = RuntimeError(f"HTTP {r.status_code}: {url}")
                    continue
                expected = r.headers.get("Content-Length")
                expected = int(expected) if expected else None
                got = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(8*1024*1024):
                        if chunk:
                            f.write(chunk); got += len(chunk)
            if expected is not None and got != expected:
                raise RuntimeError(f"incomplete download: got {got}, expected {expected}")
            if not rdf_complete(tmp):
                raise RuntimeError("missing </rdf:RDF>; file appears truncated")
            os.replace(tmp, path)
            print(f"[download] saved {path} ({path.stat().st_size/1024**2:.1f} MB)")
            return
        except Exception as e:
            last = e
            print("[download] failed:", e, flush=True)
            if tmp.exists(): tmp.unlink()
    raise RuntimeError(f"All download URLs failed for {path}: {last}")

def download_case(pair, raw):
    base="https://oaei.webdatacommons.org/tdrs/testdata/persistent/knowledgegraph"
    versions=("v4","v3")
    src,tgt,ref=raw/"source.rdf",raw/"target.rdf",raw/"reference.xml"
    download_first([f"{base}/{v}/suite/{pair}/component/source/" for v in versions],src)
    download_first([f"{base}/{v}/suite/{pair}/component/target/" for v in versions],tgt)
    download_first([f"{base}/{v}/suite/{pair}/component/reference.xml" for v in versions],ref)
    return src,tgt,ref

def load_graph(path):
    g=Graph()
    print("[rdf] parsing",path,flush=True)
    g.parse(path,format="xml")
    print(f"[rdf] {len(g):,} triples")
    return g

def parse_alignment(path):
    root=ET.parse(path).getroot()
    rr="{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
    out=[]
    for c in root.iter():
        if not c.tag.endswith("Cell"): continue
        a=b=None
        for z in c:
            if z.tag.endswith("entity1"): a=z.attrib.get(rr)
            elif z.tag.endswith("entity2"): b=z.attrib.get(rr)
        if a and b: out.append((a,b))
    if not out: raise RuntimeError("No alignment correspondences found")
    return out

def edge_kind(predicate_iri, relation):
    p=predicate_iri.lower(); r=relation.lower()
    if predicate_iri == str(RDF.type): return "type"
    if "/property/" in p: return "attribute_or_relation"
    if "subject" in r or "category" in r: return "category"
    if "wikipagewikilink" in p: return "wiki_link"
    return "relation"

def graph_to_cache(g):
    labels={}
    nodes=set()
    for s,p,o in g:
        if isinstance(s,URIRef): nodes.add(str(s))
        if isinstance(o,URIRef): nodes.add(str(o))
    print(f"[cache] labeling {len(nodes):,} URI nodes",flush=True)
    for iri in nodes:
        labels[iri]=label_from_graph(g,iri)

    types={}
    for s,o in g.subject_objects(RDF.type):
        if isinstance(s,URIRef) and isinstance(o,URIRef):
            types.setdefault(str(s),[]).append(str(o))

    incident={}
    print("[cache] building incident edge index",flush=True)
    for s,p,o in g:
        if not isinstance(s,URIRef) or not useful_pred(p): continue
        s_iri=str(s)
        rel=labels.get(str(p),tail(p))
        if isinstance(o,Literal):
            oval=re.sub(r"\s+"," ",str(o)).strip()[:180]
            o_iri=None
        elif isinstance(o,URIRef):
            oval=labels.get(str(o),tail(o)); o_iri=str(o)
        else:
            continue

        ro={"direction":"out","subject":labels.get(s_iri,tail(s_iri)),
            "relation":rel,"object":oval,"predicate_iri":str(p),
            "neighbor_iri":o_iri}
        ro["kind"]=edge_kind(ro["predicate_iri"],rel)
        incident.setdefault(s_iri,[]).append(ro)

        if o_iri:
            ri={"direction":"in","subject":labels.get(s_iri,tail(s_iri)),
                "relation":rel,"object":labels.get(o_iri,tail(o_iri)),
                "predicate_iri":str(p),"neighbor_iri":s_iri}
            ri["kind"]=edge_kind(ri["predicate_iri"],rel)
            incident.setdefault(o_iri,[]).append(ri)

    clean={}
    for iri,rows in incident.items():
        seen=set(); nonwiki=[]; wiki=[]
        for r in rows:
            k=(r["direction"],r["subject"],r["relation"],r["object"])
            if k in seen: continue
            seen.add(k)
            (wiki if r["kind"]=="wiki_link" else nonwiki).append(r)
        clean[iri]=nonwiki[:200]+wiki[:80]
    return {"labels":labels,"types":types,"incident":clean}

def save_pt(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=str(path)+".tmp"
    torch.save(obj,tmp)
    os.replace(tmp,path)

def is_schema(types, iri):
    if set(types)&SCHEMA_TYPES: return True
    s=iri.lower()
    return "/property/" in s or "/class/" in s or "/ontology/" in s

def excluded(types):
    t=set(types)
    return IMAGE in t or str(SKOS.Concept) in t

def prepare_shared_cache(pair, kg_cache_dir, force_reparse=False):
    raw=kg_cache_dir/"raw"; parsed=kg_cache_dir/"parsed"
    parsed.mkdir(parents=True,exist_ok=True)
    src_rdf,tgt_rdf,ref_xml=download_case(pair,raw)
    sp=parsed/"source_graph_cache.pt"
    tp=parsed/"target_graph_cache.pt"
    rp=parsed/"instance_reference.json"

    if force_reparse or not sp.exists():
        g=load_graph(src_rdf); c=graph_to_cache(g); save_pt(c,sp); del g,c
        print("[cache] saved",sp)
    else: print("[cache] using",sp)

    if force_reparse or not tp.exists():
        g=load_graph(tgt_rdf); c=graph_to_cache(g); save_pt(c,tp); del g,c
        print("[cache] saved",tp)
    else: print("[cache] using",tp)

    sc=torch.load(sp,map_location="cpu")
    tc=torch.load(tp,map_location="cpu")

    if force_reparse or not rp.exists():
        refs=parse_alignment(ref_xml)
        inst=[]
        for s,t in refs:
            st=sc["types"].get(s,[]); tt=tc["types"].get(t,[])
            if is_schema(st,s) or is_schema(tt,t): continue
            if excluded(st) or excluded(tt): continue
            inst.append({"src":s,"tgt":t})
        atomic_json(inst,rp)
        print(f"[cache] saved {rp} ({len(inst):,} instance pairs)")
    else: print("[cache] using",rp)

    with open(rp,encoding="utf-8") as f:
        refs=json.load(f)
    return sc,tc,refs

def select_fact_sets(rows,n_contexts=5,facts_per_context=5):
    pr={"type":0,"attribute_or_relation":1,"relation":2,"category":3,"wiki_link":4}
    rows=sorted(rows,key=lambda r:(pr.get(r["kind"],9),r["relation"],r["subject"],r["object"]))
    if not rows: return [[] for _ in range(n_contexts)]
    by={}
    for r in rows: by.setdefault(r["kind"],[]).append(r)
    pref=[
        ["relation","attribute_or_relation","type"],
        ["type","category","attribute_or_relation"],
        ["relation","wiki_link","attribute_or_relation"],
        ["relation","attribute_or_relation","category"],
        ["attribute_or_relation","relation","type","wiki_link"],
    ]
    used=set(); out=[]
    for i in range(n_contexts):
        cur=[]
        for k in pref[i%len(pref)]:
            for r in by.get(k,[]):
                key=(r["direction"],r["subject"],r["relation"],r["object"])
                if key not in used:
                    cur.append(r); used.add(key); break
            if len(cur)>=facts_per_context: break
        for r in rows:
            if len(cur)>=facts_per_context: break
            key=(r["direction"],r["subject"],r["relation"],r["object"])
            if key not in used:
                cur.append(r); used.add(key)
        if len(cur)<facts_per_context:
            for r in rows:
                if len(cur)>=facts_per_context: break
                if r not in cur: cur.append(r)
        out.append(cur)
    return out

def facts_prompt(rows):
    out=[]
    for i,r in enumerate(rows,1):
        if r["direction"]=="out":
            out.append(f"{i}. <ENTITY> --[{r['relation']}]--> {r['object']}")
        else:
            out.append(f"{i}. {r['subject']} --[{r['relation']}]--> <ENTITY>")
    return "\n".join(out)

def build_chat_prompt(tokenizer, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def call_llm_local(model, tokenizer, system, user, temp, seed, max_tokens=220):
    prompt = build_chat_prompt(tokenizer, system, user)
    inputs = tokenizer(prompt, return_tensors="pt")

    # With device_map="auto", input embeddings live on one CUDA device.
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    generator = torch.Generator(device=input_device)
    generator.manual_seed(seed)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.95,
            generator=generator,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def fallback(rows):
    if not rows:
        return "Within this knowledge graph, the selected node is <ENTITY>."
    clauses=[]
    for r in rows:
        if r["direction"]=="out":
            clauses.append(f"the node is connected by {r['relation']} to {r['object']}")
        else:
            clauses.append(f"{r['subject']} is connected to the node by {r['relation']}")
    return "Considering that " + "; ".join(clauses) + ", the corresponding entity is <ENTITY>."


def verbalize(rows, style_instruction, model, tokenizer, seed):
    sys=("You are a strict knowledge-graph verbalizer. Convert only the supplied triples into fluent natural English. "
         "Do not use outside knowledge or infer/add facts. Preserve <ENTITY> exactly once. Do not output a triple list.")
    usr=f"""STYLE:
{style_instruction}

TRIPLES:
{facts_prompt(rows)}

Requirements:
- Use only supplied facts.
- Mention <ENTITY> exactly once.
- Write 1-2 fluent sentences.
- Keep meaningful grounded context before <ENTITY> when possible.
- Return only the context."""

    for a in range(3):
        try:
            x=call_llm_local(
                model, tokenizer, sys, usr,
                temp=0.85 + 0.05*a,
                seed=seed + a,
            )
            x=re.sub(r"^```(?:text)?\s*|\s*```$","",x.strip())
            x=re.sub(r"\s+"," ",x).strip().strip('"')
            if x.count("<ENTITY>")==1 and 40<=len(x)<=1200:
                return x,False
            print(f"[LLM retry {a+1}/3] invalid output: "
                  f"<ENTITY> count={x.count('<ENTITY>')}, len={len(x)}",
                  flush=True)
        except Exception as e:
            print(f"[LLM retry {a+1}/3] {type(e).__name__}: {e}",flush=True)
            time.sleep(1)

    return fallback(rows),True


def build_contexts(iri, cache, n_contexts, facts_per_context, model, tokenizer, seed):
    entity=cache["labels"].get(iri,tail(iri))
    rows=cache["incident"].get(iri,[])
    factsets=select_fact_sets(rows,n_contexts,facts_per_context)
    out=[]
    for i,fs in enumerate(factsets):
        stylename,styleinst=STYLE_CARDS[i%len(STYLE_CARDS)]
        text,fallback_used=verbalize(
            fs, styleinst, model, tokenizer,
            seed=seed + i * 1009,
        )
        start=text.index("<ENTITY>")
        final=text.replace("<ENTITY>",entity,1)
        out.append({
            "text":final,
            "symbol":entity,
            "char_span":[start,start+len(entity)],
            "style":stylename,
            "grounding_triples":fs,
            "used_fallback":fallback_used
        })
    return rows,out


def tokset(x): return set(re.findall(r"[a-z0-9]+",x.lower()))
def jaccard(a,b):
    A,B=tokset(a),tokset(b)
    return len(A&B)/max(1,len(A|B))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kg_cache_dir",type=Path,required=True)
    ap.add_argument("--output_dir",type=Path,required=True)
    ap.add_argument("--pair",default="starwars-swtor",choices=["starwars-swtor","starwars-swg"])
    ap.add_argument("--limit",type=int,default=5)
    ap.add_argument("--candidate_count",type=int,default=100)
    ap.add_argument("--contexts",type=int,default=5)
    ap.add_argument("--facts_per_context",type=int,default=5)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--model_path",type=str,required=True)
    ap.add_argument("--max_memory_per_gpu",type=str,default=None,
                    help="Optional per-GPU cap, e.g. 76GiB for A100-80GB or 37GiB for A100-40GB")
    ap.add_argument("--force_reparse",action="store_true")
    args=ap.parse_args()

    kgdir=args.kg_cache_dir.expanduser().resolve()
    out=args.output_dir.expanduser().resolve()
    sc,tc,refs=prepare_shared_cache(args.pair,kgdir,args.force_reparse)

    print("[llm] loading tokenizer:", args.model_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    max_memory = None
    if args.max_memory_per_gpu:
        n_gpu = torch.cuda.device_count()
        max_memory = {i: args.max_memory_per_gpu for i in range(n_gpu)}
        max_memory["cpu"] = "64GiB"

    print(
        f"[llm] loading Qwen directly with Transformers on "
        f"{torch.cuda.device_count()} visible GPUs (bf16, device_map=auto)",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[llm] local model ready", flush=True)

    bysrc={}
    for x in refs: bysrc.setdefault(x["src"],x["tgt"])
    pairs=list(sorted(bysrc.items()))[:args.limit]
    universe=sorted(set(x["tgt"] for x in refs))
    if len(universe)<args.candidate_count:
        raise RuntimeError(f"Only {len(universe)} target instances")

    queries=[]; needed=set()
    for qi,(s,gold) in enumerate(pairs):
        sl=sc["labels"].get(s,tail(s))
        ranked=sorted((t for t in universe if t!=gold),
                      key=lambda t:(-jaccard(sl,tc["labels"].get(t,tail(t))),tc["labels"].get(t,tail(t)),t))
        cand=[gold]+ranked[:args.candidate_count-1]
        random.Random(args.seed+qi).shuffle(cand)
        needed.update(cand)
        queries.append({"query_id":qi,"src":s,"gold":gold,"candidates":cand})

    for side,cache,iris in [("src",sc,[s for s,_ in pairs]),("tgt",tc,sorted(needed))]:
        print(f"[{side}] entities: {len(iris)}")
        for i,iri in enumerate(iris,1):
            sid=safe_id(iri)
            cp=out/"contexts"/side/f"{sid}.json"
            if cp.exists(): continue
            rows,contexts=build_contexts(iri,cache,args.contexts,args.facts_per_context,model,tokenizer,args.seed + i*100003)
            meta={"entity_iri":iri,"label":cache["labels"].get(iri,tail(iri)),
                  "n_incident_triples":len(rows),"incident_triples":rows}
            atomic_json(meta,out/"entity_metadata"/side/f"{sid}.json")
            atomic_json({"entity_iri":iri,"side":side,"label":meta["label"],"preferred_label":meta["label"],
                         "contexts":contexts,"parsed_output":{"contexts":[{"text":x["text"]} for x in contexts]}},cp)
            if i%25==0 or i==len(iris): print(f"[{side}] {i}/{len(iris)}",flush=True)

    atomic_json(queries,out/"selected_queries.json")
    atomic_json({"pair":args.pair,"kg_cache_dir":str(kgdir),"queries":len(queries),
                 "candidate_count":args.candidate_count,"contexts_per_entity":args.contexts,
                 "facts_per_context":args.facts_per_context,
                 "context_policy":"natural diverse contexts grounded only in cached incident KG triples",
                 "verbalizer_model":args.model_path},out/"preparation_summary.json")
    print("DONE:",out)

if __name__=="__main__":
    main()
