#!/usr/bin/env python3
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import torch


def load_pt(p):
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(p, map_location='cpu')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def load_rep_dirs(run_dirs):
    src, tgt = {}, {}
    for run in run_dirs:
        for side, store in [('src', src), ('tgt', tgt)]:
            for p in (run/'representations'/side).glob('*.pt'):
                x = load_pt(p)
                store.setdefault(x['entity_iri'], x)
    return src, tgt


def load_queries(context_dirs):
    out, seen = [], set()
    for d in context_dirs:
        rows = json.loads((d/'selected_queries.json').read_text())
        for q in rows:
            qid = int(q.get('query_id', q.get('qid')))
            if qid in seen: continue
            seen.add(qid)
            out.append({'qid':qid,'src':q['src'],'gold':q['gold'],'candidates':q['candidates']})
    return sorted(out, key=lambda x:x['qid'])


def dense_cos(a,b):
    a=a.detach().cpu().float(); b=b.detach().cpu().float()
    na=float(torch.linalg.vector_norm(a)); nb=float(torch.linalg.vector_norm(b))
    return float(torch.dot(a,b))/(na*nb) if na and nb else 0.0


def norm_sparse(x):
    idx=x['idx'].detach().cpu().long(); val=x['val'].detach().cpu().float()
    if idx.numel()>1:
        o=torch.argsort(idx); idx=idx[o]; val=val[o]
    return {'idx':idx,'val':val,'size':int(x['size'])}


def sparse_cos(a,b):
    ai,av=a['idx'],a['val']; bi,bv=b['idx'],b['val']
    if ai.numel()==0 or bi.numel()==0: return 0.0
    if ai.numel()<=bi.numel():
        d=dict(zip(ai.tolist(),av.tolist())); dot=sum(d.get(i,0.0)*v for i,v in zip(bi.tolist(),bv.tolist()))
    else:
        d=dict(zip(bi.tolist(),bv.tolist())); dot=sum(d.get(i,0.0)*v for i,v in zip(ai.tolist(),av.tolist()))
    na=float(torch.linalg.vector_norm(av)); nb=float(torch.linalg.vector_norm(bv))
    return dot/(na*nb) if na and nb else 0.0


def existing(rep, method):
    agg=rep['aggregate']
    if method=='sae_mean': return norm_sparse(agg['sae_mean'])
    obj=agg['sae_stable']['0.80']; return norm_sparse(obj['vector'] if 'vector' in obj else obj)


def build_background(reps):
    counts=Counter(); total=0
    for rep in reps:
        for c in rep['contexts']:
            counts.update(set(int(i) for i in c['sae']['idx'].detach().cpu().tolist()))
            total += 1
    return counts,total


def build_signature(rep,bg,bg_total,eps=1e-6):
    n=len(rep['contexts']); size=int(rep['contexts'][0]['sae']['size'])
    vals=defaultdict(lambda:[0.0]*n)
    for ci,c in enumerate(rep['contexts']):
        for i,v in zip(c['sae']['idx'].detach().cpu().tolist(), c['sae']['val'].detach().cpu().float().tolist()):
            vals[int(i)][ci]=float(v)
    buckets={k:([],[]) for k in ['sae_info','sae_info_reliability','sae_logodds']}
    for j,xs in vals.items():
        x=torch.tensor(xs,dtype=torch.float32)
        pe=float((x>0).sum())/n; pb=bg.get(j,0)/bg_total
        mean=float(x.mean()); std=float(x.std(unbiased=False)); cv=std/(abs(mean)+eps)
        info=max(math.log((pe+eps)/(pb+eps)),0.0)
        pe2=min(max(pe,eps),1-eps); pb2=min(max(pb,eps),1-eps)
        lod=max(math.log(pe2/(1-pe2))-math.log(pb2/(1-pb2)),0.0)
        weights={
            'sae_info': mean*pe*info,
            'sae_info_reliability': mean*pe*info/(1+cv),
            'sae_logodds': mean*lod,
        }
        for k,w in weights.items():
            if w>0:
                buckets[k][0].append(j); buckets[k][1].append(w)
    return {k:{'idx':torch.tensor(ii,dtype=torch.long),'val':torch.tensor(vv,dtype=torch.float32),'size':size} for k,(ii,vv) in buckets.items()}


def metrics(ranks):
    n=len(ranks)
    return {'n':n,'MRR':sum(1/r for r in ranks)/n,'H@1':sum(r<=1 for r in ranks)/n,'H@5':sum(r<=5 for r in ranks)/n,'H@10':sum(r<=10 for r in ranks)/n,'MeanRank':sum(ranks)/n}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sae_run_dirs',required=True)
    ap.add_argument('--context_run_dirs',required=True)
    ap.add_argument('--output_dir',type=Path,required=True)
    args=ap.parse_args()
    sae_dirs=[Path(x).expanduser().resolve() for x in args.sae_run_dirs.split(',')]
    ctx_dirs=[Path(x).expanduser().resolve() for x in args.context_run_dirs.split(',')]
    out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)

    print('='*100); print('INFORMATION-WEIGHTED SAE ANALYSIS -- CPU ONLY'); print('='*100)
    src,tgt=load_rep_dirs(sae_dirs); qs=load_queries(ctx_dirs)
    print('queries:',len(qs),'source reps:',len(src),'target reps:',len(tgt))
    bg,bg_total=build_background(list(src.values())+list(tgt.values()))
    print('background entity-context pairs:',bg_total,'features observed:',len(bg))

    print('Building information-weighted signatures...')
    srcsig={i:build_signature(r,bg,bg_total) for i,r in src.items()}
    tgtsig={i:build_signature(r,bg,bg_total) for i,r in tgt.items()}

    methods=['dense_mean','sae_mean','sae_stable_0.80','sae_info','sae_info_reliability','sae_logodds']
    ranks=defaultdict(list); rows=[]
    for q in qs:
        s,g=q['src'],q['gold']; cands=[c for c in q['candidates'] if c in tgt]
        if s not in src or g not in cands: raise RuntimeError(f'missing representation qid={q["qid"]}')
        row={'qid':q['qid'],'source_label':src[s]['label'],'gold_label':tgt[g]['label']}
        for m in methods:
            scored=[]
            for c in cands:
                if m=='dense_mean': score=dense_cos(src[s]['aggregate']['dense_mean'],tgt[c]['aggregate']['dense_mean'])
                elif m in ('sae_mean','sae_stable_0.80'): score=sparse_cos(existing(src[s],m),existing(tgt[c],m))
                else: score=sparse_cos(srcsig[s][m],tgtsig[c][m])
                scored.append((c,score))
            scored.sort(key=lambda z:(-z[1],z[0]))
            r=next(i+1 for i,(iri,_) in enumerate(scored) if iri==g)
            row[m+'_rank']=r; ranks[m].append(r)
        rows.append(row)
    write_csv(out/'per_query_ranks.csv',rows)
    summary=[{'method':m,**metrics(ranks[m])} for m in methods]
    write_csv(out/'ranking_summary.csv',summary)
    print('\nFINAL SUMMARY')
    for s in summary:
        print(f"[{s['method']:<24}] MRR={s['MRR']:.4f} H@1={s['H@1']:.4f} H@5={s['H@5']:.4f} H@10={s['H@10']:.4f} MeanRank={s['MeanRank']:.2f}")
    (out/'analysis.json').write_text(json.dumps({'n_queries':len(qs),'background_contexts':bg_total,'summary':summary},indent=2))
    print('\nSaved to:',out)

if __name__=='__main__': main()
