#!/usr/bin/env python3
import argparse, json, os, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Gemma Scope layer_20/width_131k/average_l0_114: the canonical release, and the
# only layer-20 residual 131k variant with full Neuronpedia auto-interp coverage
# (its source id is gemma-2-9b/20-gemmascope-res-131k). Keep runs on this SAE so
# feature indices stay comparable across benchmarks and remain interpretable.
DEFAULT_SAE_PATH = ("/projects/biro/shared/sae/gemma-scope-9b-pt-res-131k/layer_20/width_131k/average_l0_114/params.npz")
DEFAULT_MODEL_PATH = "/projects/biro/shared/models/gemma-2-9b"

ID_KEYS=("entity_iri","iri","entity_id","id","uri","focus_iri")
LABEL_KEYS=("label","symbol","entity_label","name","focus_label")
TEXT_KEYS=("text","context","sentence")
SPAN_KEYS=("char_span","symbol_char_span","span")

def first(d, keys):
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return None

def span2(x):
    if isinstance(x,dict) and "start" in x and "end" in x: return [int(x["start"]),int(x["end"])]
    if isinstance(x,(list,tuple)) and len(x)==2: return [int(x[0]),int(x[1])]
    return None

def norm_context(c,eid=None,label=None):
    text=first(c,TEXT_KEYS)
    if not isinstance(text,str): return None
    eid=first(c,ID_KEYS) or eid
    lab=first(c,LABEL_KEYS) or label
    if isinstance(lab,dict): lab=first(lab,LABEL_KEYS)
    if lab is None: raise ValueError("context has no label/symbol")
    lab=str(lab)
    sp=span2(first(c,SPAN_KEYS))
    if sp is None:
        pos=[m.start() for m in re.finditer(re.escape(lab),text)]
        if len(pos)!=1: raise ValueError(f"missing char_span; exact label occurs {len(pos)} times")
        sp=[pos[0],pos[0]+len(lab)]
    s,e=sp
    if text[s:e]!=lab: raise ValueError(f"span mismatch: {text[s:e]!r} != {lab!r}")
    z=dict(c); z.update(text=text,char_span=sp,symbol=lab)
    if eid is not None: z["entity_iri"]=str(eid)
    return z

def harvest(obj,source):
    out=[]
    def add(r,eid=None,label=None):
        if not isinstance(r,dict): return
        eid=first(r,ID_KEYS) or eid
        lab=first(r,LABEL_KEYS) or label
        if isinstance(lab,dict): lab=first(lab,LABEL_KEYS)
        if isinstance(r.get("contexts"),list):
            for c in r["contexts"]:
                nc=norm_context(c,eid,lab)
                if nc: out.append((str(nc.get("entity_iri",eid or source)),nc["symbol"],nc))
        elif first(r,TEXT_KEYS) is not None:
            nc=norm_context(r,eid,lab)
            out.append((str(nc.get("entity_iri",eid or source)),nc["symbol"],nc))
    if isinstance(obj,list):
        for x in obj: add(x)
    elif isinstance(obj,dict):
        if "contexts" in obj or first(obj,TEXT_KEYS) is not None: add(obj)
        else:
            used=False
            for k in ("entities","records","items","data"):
                if isinstance(obj.get(k),list):
                    used=True
                    for x in obj[k]: add(x)
            if not used:
                for k,v in obj.items():
                    if isinstance(v,dict): add(v,k)
                    elif isinstance(v,list):
                        for x in v:
                            if isinstance(x,dict): add(x,k)
    return out

def load_contexts(path):
    p=Path(path).expanduser()
    fs=[p] if p.is_file() else sorted(p.rglob("*.json"))
    G=defaultdict(list); labels={}; errs=[]
    for fp in fs:
        try:
            obj=json.load(open(fp,encoding="utf-8"))
            for eid,lab,c in harvest(obj,fp.stem):
                G[eid].append(c); labels[eid]=lab
        except Exception as e: errs.append((str(fp),str(e)))
    if not G:
        raise RuntimeError("No usable contexts. Need text + label/symbol + char_span.\n"+str(errs[:10]))
    for eid in G:
        seen=set(); u=[]
        for c in G[eid]:
            k=(c["text"],tuple(c["char_span"]),c["symbol"])
            if k not in seen: seen.add(k); u.append(c)
        G[eid]=u
    return G,labels,errs

class SAE(torch.nn.Module):
    def __init__(self,path,device):
        super().__init__(); d=np.load(str(path))
        def get(*ks):
            for k in ks:
                if k in d:return d[k]
            raise KeyError(f"missing {ks}; keys={list(d.keys())}")
        W=get("W_enc","w_enc"); b=get("b_enc"); th=get("threshold","thresholds")
        self.register_buffer("W",torch.from_numpy(W).to(torch.bfloat16))
        self.register_buffer("b",torch.from_numpy(b).to(torch.bfloat16))
        self.register_buffer("th",torch.from_numpy(th).to(torch.bfloat16))
        self.to(device); self.d_in=W.shape[0]; self.width=W.shape[1]
    @torch.inference_mode()
    def encode(self,x):
        pre=x.to(self.W.dtype)@self.W+self.b
        z=torch.where(pre>self.th,pre,torch.zeros_like(pre))
        idx=torch.nonzero(z,as_tuple=False).flatten()
        return idx,z[idx]

def token_ids(offsets,sp):
    s,e=sp
    ids=[i for i,(a,b) in enumerate(offsets) if b>a and a<e and b>s]
    if not ids: raise ValueError(f"no tokens overlap {sp}")
    return ids

def aggregate(cs,width):
    sums=torch.zeros(width); counts=torch.zeros(width,dtype=torch.int32)
    for c in cs:
        idx=c["sae"]["idx"].long(); val=c["sae"]["val"].float()
        sums[idx]+=val; counts[idx]+=1
    n=len(cs)
    return sums/n,counts.float()/n

def safe(s): return (re.sub(r"[^A-Za-z0-9._-]+","_",str(s)).strip("_") or "entity")[:180]

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--input",required=True)
    a.add_argument("--output_dir",required=True)
    a.add_argument("--model_path",default=DEFAULT_MODEL_PATH)
    a.add_argument("--sae_path",default=DEFAULT_SAE_PATH,
                   help="defaults to average_l0_114 (Neuronpedia-interpretable)")
    a.add_argument("--layer",type=int,default=20)
    a.add_argument("--contexts",type=int,default=5)
    a.add_argument("--device",default="cuda")
    a.add_argument("--max_length",type=int,default=2048)
    a.add_argument("--stable_frequencies",default="0.2,0.4,0.6,0.8,1.0")
    a.add_argument("--start",type=int,default=0)
    a.add_argument("--limit",type=int,default=0)
    a.add_argument("--overwrite",action="store_true")
    a.add_argument("--debug_first",action="store_true")
    args=a.parse_args()

    G,labels,errs=load_contexts(args.input)
    ids=sorted(G)[args.start:]
    if args.limit: ids=ids[:args.limit]
    out=Path(args.output_dir).expanduser(); repdir=out/"representations"; repdir.mkdir(parents=True,exist_ok=True)
    model_path=str(Path(args.model_path).expanduser())
    tok=AutoTokenizer.from_pretrained(model_path,use_fast=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
        local_files_only=True,low_cpu_mem_usage=True).to(args.device).eval()
    sae=SAE(Path(args.sae_path).expanduser(),args.device)
    print(f"entities={len(ids)} SAE width={sae.width} d_in={sae.d_in}")
    thresholds=[float(x) for x in args.stable_frequencies.split(",") if x.strip()]
    manifest=[]

    for n,eid in enumerate(ids,1):
        fp=repdir/(safe(eid)+".pt")
        if fp.exists() and not args.overwrite:
            print(f"[{n}/{len(ids)}] skip {eid}"); continue
        cs=G[eid][:args.contexts] if args.contexts else G[eid]
        saved=[]; dense=[]
        print(f"[{n}/{len(ids)}] {eid} :: {labels[eid]} :: contexts={len(cs)}")
        for j,c in enumerate(cs,1):
            enc=tok(c["text"],return_tensors="pt",return_offsets_mapping=True,
                    truncation=True,max_length=args.max_length,add_special_tokens=True)
            offsets=enc.pop("offset_mapping")[0].tolist()
            tids=token_ids(offsets,c["char_span"])
            s,e=c["char_span"]
            if min(offsets[i][0] for i in tids)>s or max(offsets[i][1] for i in tids)<e:
                raise RuntimeError(f"symbol truncated: {eid} context {j}")
            inputs={k:v.to(args.device) for k,v in enc.items()}
            with torch.inference_mode():
                o=model(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
                h=o.hidden_states[args.layer+1][0]
                # IMPORTANT: mean ONLY exact symbol-token hidden states.
                v=h[tids].mean(0)
                if v.numel()!=sae.d_in: raise RuntimeError(f"hidden {v.numel()} != SAE d_in {sae.d_in}")
                idx,val=sae.encode(v)
            r={k:v0 for k,v0 in c.items() if k not in ("dense","sae")}
            r.update(token_indices=[int(x) for x in tids],
                     symbol_tokens=tok.convert_ids_to_tokens(enc["input_ids"][0,tids].tolist()),
                     dense=v.detach().float().cpu(),
                     sae={"idx":idx.detach().to(torch.int32).cpu(),"val":val.detach().float().cpu()})
            saved.append(r); dense.append(r["dense"])
            if args.debug_first and n==1:
                print(" context",j,"span",c["char_span"],"tokens",r["symbol_tokens"],"nnz",len(r["sae"]["idx"]))
            del o,h,v,inputs
        sm,sf=aggregate(saved,sae.width)
        rep={"entity_iri":eid,"label":labels[eid],"contexts":saved,
             "dense_mean":torch.stack(dense).mean(0),"sae_mean":sm,"sae_frequency":sf,
             "sae_stable":{f"{t:.2f}":sm*(sf>=t).float() for t in thresholds},
             "meta":{"layer":args.layer,"hidden_states_index":args.layer+1,
                     "model_path":model_path,"sae_path":str(Path(args.sae_path).expanduser()),
                     "sae_width":sae.width,"symbol_span_only":True,"num_contexts":len(saved)}}
        tmp=Path(str(fp)+".tmp"); torch.save(rep,tmp); os.replace(tmp,fp)
        manifest.append({"entity_iri":eid,"label":labels[eid],"file":str(fp),"contexts":len(saved)})
    json.dump({"entities":manifest,"parse_errors":errs},open(out/"manifest.json","w"),indent=2,ensure_ascii=False)
    print("DONE:",repdir)

if __name__=="__main__": main()
