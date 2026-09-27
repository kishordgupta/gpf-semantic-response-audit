"""Balanced complete-cell sensitivity; preserves canonical responses unchanged."""
import itertools, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lexical_features import fit_locked_tfidf
from analyze import ROOT, MODELS, COMPANY, SEED, clean

def ci(x, groups):
    """Cluster resampling ratio-of-sums: valid for unequal cluster sizes."""
    levels=sorted(set(groups))
    sums=np.array([x[groups==g].sum() for g in levels])
    counts=np.array([(groups==g).sum() for g in levels])
    draws=np.random.default_rng(SEED).integers(0,len(levels),(2000,len(levels)))
    v=sums[draws].sum(1)/counts[draws].sum(1)
    return np.quantile(v,[.025,.975]).tolist()

def run():
    df=pd.read_csv(ROOT/'data/responses.csv',keep_default_na=False)
    audit=json.loads((ROOT/'results/data_quality.json').read_text())
    excluded=set(audit['sensitivity']['alignment_sensitivity_excluded_cells'])
    cells=sorted(df.cell_id.unique())
    meta=df.drop_duplicates('cell_id').set_index('cell_id').loc[cells]
    keep=~meta.index.isin(excluded)
    smeta=meta[keep]; groups=smeta.y_group_id.to_numpy()
    pairs=list(itertools.combinations(MODELS,2))
    stored=np.load(ROOT/'results/matched_distances.npz')
    tables={n:[] for n in ['pairwise_summary','pairwise_by_y','pairwise_by_family','pairwise_by_article','company_distances','identity_sensitivity','y_importance']}
    same=np.array([COMPANY[a]==COMPANY[b] for a,b in pairs])
    contrasts=[]; stability=[]
    for enc in stored.files:
        full=stored[enc]; arr=full[keep]
        for j,(a,b) in enumerate(pairs):
            d=arr[:,j]; lo,hi=ci(d,groups)
            common=dict(encoder=enc,model_a=a,model_b=b)
            tables['pairwise_summary'].append(dict(**common,company_a=COMPANY[a],company_b=COMPANY[b],mean_distance=float(d.mean()),ci_low=lo,ci_high=hi,n_prompts=len(d)))
            for dim,y in sorted(set(zip(smeta.y_group_dimension,smeta.y_group_name))):
                mask=(smeta.y_group_dimension==dim)&(smeta.y_group_name==y)
                tables['pairwise_by_y'].append(dict(**common,y_group_dimension=dim,y_group_name=y,mean_distance=float(d[mask].mean()),n_prompts=int(mask.sum())))
            for field,name in [('prompt_family','pairwise_by_family'),('x_group_id','pairwise_by_article')]:
                for val in sorted(smeta[field].unique()):
                    mask=smeta[field]==val
                    tables[name].append(dict(**common,**{field:val},mean_distance=float(d[mask].mean()),n_prompts=int(mask.sum())))
        cp={}
        for j,(a,b) in enumerate(pairs): cp.setdefault(tuple(sorted([COMPANY[a],COMPANY[b]])),[]).append(j)
        for (ca,cb),pis in cp.items():
            d=arr[:,pis].mean(1);lo,hi=ci(d,groups)
            tables['company_distances'].append(dict(encoder=enc,company_a=ca,company_b=cb,model_pairs=len(pis),n_prompts=len(arr),mean_distance=float(d.mean()),ci_low=lo,ci_high=hi))
        a=arr[:,same].mean(1);b=arr[:,~same].mean(1);lo,hi=ci(b-a,groups)
        contrasts.append(dict(encoder=enc,within_company_mean=float(a.mean()),between_company_mean=float(b.mean()),between_minus_within=float((b-a).mean()),delta_ci_low=lo,delta_ci_high=hi,within_model_pairs=int(same.sum()),between_model_pairs=int((~same).sum()),n_prompts=len(arr)))
        stability.append(dict(encoder=enc,pair_rank_spearman=float(spearmanr(full.mean(0),arr.mean(0)).statistic),max_absolute_pair_mean_change=float(np.max(np.abs(full.mean(0)-arr.mean(0)))),mean_absolute_pair_mean_change=float(np.mean(np.abs(full.mean(0)-arr.mean(0))))))
        if enc=='tfidf':
            # Refit IDF on the same complete corpus with the original locked features.
            z=fit_locked_tfidf(df.response.map(clean))
        else: z=np.load(ROOT/f'embeddings/{enc}.npz')['embeddings']
        sub=df[~df.cell_id.isin(excluded)]
        for (m,dim),g in sub.groupby(['model_name','y_group_dimension']):
            blockmeans=[]; n_pairs=0; perlabel={}
            for _,block in g.groupby(['x_group_id','prompt_family']):
                rows=block.index.to_numpy()
                D=1-((z[rows]@z[rows].T).toarray() if enc=='tfidf' else z[rows]@z[rows].T)
                D=np.clip(D,0,2);tri=D[np.triu_indices(len(rows),1)]
                blockmeans.append(float(tri.mean()));n_pairs+=len(tri)
                for j,y in enumerate(block.y_group_name):perlabel.setdefault(y,[]).append(float(np.delete(D[j],j).mean()))
            lo,hi=ci(np.array(blockmeans),np.arange(len(blockmeans)))
            tables['identity_sensitivity'].append(dict(encoder=enc,model_name=m,company=COMPANY[m],y_group_dimension=dim,mean_distance=float(np.mean(blockmeans)),ci_low=lo,ci_high=hi,n_pairs=n_pairs,n_blocks=len(blockmeans)))
            for y,vals in perlabel.items():
                tables['y_importance'].append(dict(encoder=enc,model_name=m,company=COMPANY[m],y_group_dimension=dim,y_group_name=y,mean_distance_to_other_identities=float(np.mean(vals)),n_blocks=len(vals)))
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(ROOT/f'results/{name}_screened.csv',index=False)
    (ROOT/'results/company_contrast_screened.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    notes={'full_cells':len(cells),'screened_cells':int(keep.sum()),'screened_rows':int(keep.sum())*12,'excluded_cells':sorted(excluded),'reason':'Conservative sensitivity excludes every model response at five cells with suspected Gemini prompt-response inconsistencies; original data never removed or reassigned.','bootstrap':'2000 identity-cluster draws using sum/count ratio; preserves row-weighted estimand despite unequal cluster sizes. Conditional on the two fixed articles.','rank_stability':stability}
    (ROOT/'results/robustness_summary.json').write_text(json.dumps(notes,indent=2)+'\n')
    # Nonflagged inspection candidates covering agreement, disagreement, and company pairs.
    arr=stored['mpnet']; eligible=np.where(keep)[0]
    selections=[]
    for selection,k in [('screened_high',-1),('screened_low',0)]:
        local=np.argsort(arr[keep].ravel())[k]
        ci0,pi=np.unravel_index(local,arr[keep].shape)
        selections.append((eligible[ci0],pi,selection))
    for pi in np.where(same)[0]:
        values=arr[keep,pi]
        for label,ordidx in [('same_company_median',len(values)//2),('same_company_high',-1)]:
            selections.append((eligible[np.argsort(values)[ordidx]],pi,label))
    lookup=df.set_index(['cell_id','model_name']); out=[]
    for i,j,label in selections:
        c=cells[i];a,b=pairs[j];ra,rb=lookup.loc[(c,a)],lookup.loc[(c,b)]
        out.append(dict(selection=label,cell_id=c,model_a=a,model_b=b,distance=float(arr[i,j]),prompt_family=ra.prompt_family,prompt_question=ra.prompt_question,y_group_dimension=ra.y_group_dimension,y_group_name=ra.y_group_name,news_source=ra.news_source,response_a=ra.response,response_b=rb.response))
    (ROOT/'results/cases_screened.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(notes,indent=2))

if __name__=='__main__':run()
