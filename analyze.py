"""Reproducible, local-only matched-prompt semantic audit. No generation APIs."""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, re, sys
from pathlib import Path
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent.parent / 'hf_cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent / 'mpl_cache'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lexical_features import fit_locked_tfidf

ROOT=Path(__file__).resolve().parent
MODELS=['GPT 5.4','GPT 5.4 Mini','Claude Sonnet 4.5','Claude Haiku 4.5','Meta Llama 3.3 Turbo','Meta Llama Maverick 4','Gemini 2.5 Flash','Gemini 2.5 Pro','Mistral Large 3','Mistral Medium 3','Grok Fast','3.5 Flash-Lite']
COMPANY={m:('OpenAI' if m.startswith('GPT') else 'Anthropic' if m.startswith('Claude') else 'Meta' if m.startswith('Meta') else 'Mistral' if m.startswith('Mistral') else 'xAI' if m.startswith('Grok') else 'Google') for m in MODELS}
ENCODERS={'mpnet':'sentence-transformers/all-mpnet-base-v2','minilm':'sentence-transformers/all-MiniLM-L6-v2'}
SEED=20260927

def clean(s):
    s=re.sub(r'^\s*Gemini said\s*\n+', '', s)
    s=re.sub(r'^\s*Response \d+:\s*\n+', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def dump(obj,name):
    (ROOT/'results'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def encode(texts,key,stem,device='cpu'):
    import torch
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    torch.set_num_threads(6)
    cache=ROOT/'embeddings'; cache.mkdir(exist_ok=True)
    p=cache/f'{stem}.npz'; meta_p=cache/f'{stem}.json'
    text_hash=hashlib.sha256('\n'.join(texts).encode()).hexdigest()
    if p.exists() and meta_p.exists():
        meta=json.loads(meta_p.read_text())
        if meta['text_sha256']==text_hash:
            print('cached',stem,flush=True); return np.load(p)['embeddings'],meta
    lock=ROOT/'encoder_lock.json'
    revisions=json.loads(lock.read_text()) if lock.exists() else {}
    if key not in revisions:
        revisions[key]={'name':ENCODERS[key],'revision':model_info(ENCODERS[key]).sha}
        lock.write_text(json.dumps(revisions,indent=2)+'\n')
    rev=revisions[key]['revision']
    print('loading',key,rev,flush=True)
    model=SentenceTransformer(ENCODERS[key],revision=rev,device=device)
    # Keep every token using conservative nonoverlapping chunks, not silent truncation.
    budget=min(220,model.max_seq_length-16)
    chunks=[]; owners=[]; weights=[]; lengths=[]
    for i,t in enumerate(texts):
        ids=model.tokenizer.encode(t,add_special_tokens=False); lengths.append(len(ids))
        for j in range(0,max(1,len(ids)),budget):
            part=ids[j:j+budget]
            chunks.append(model.tokenizer.decode(part,skip_special_tokens=True))
            owners.append(i); weights.append(max(1,len(part)))
    print(stem,'texts',len(texts),'chunks',len(chunks),'device',device,flush=True)
    z=model.encode(chunks,batch_size=32,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
    e=np.zeros((len(texts),z.shape[1]),dtype=np.float32)
    np.add.at(e,np.asarray(owners),z*np.asarray(weights,dtype=np.float32)[:,None])
    e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    np.savez_compressed(p,embeddings=e)
    meta={'encoder':ENCODERS[key],'revision':rev,'text_sha256':text_hash,'rows':len(texts),'dimensions':e.shape[1],'max_seq_length':model.max_seq_length,'chunk_token_budget':budget,'chunks':len(chunks),'multi_chunk_rows':int(sum(x>budget for x in lengths)),'maximum_tokens':max(lengths),'device':device,'aggregation':'token-count weighted mean of L2-normalized chunk embeddings, L2-normalized again','special_prefix_cleaning':True}
    meta_p.write_text(json.dumps(meta,indent=2)+'\n')
    return e,meta

def bootstrap(x,groups,B=2000):
    clusters=np.array([np.mean(x[groups==g]) for g in sorted(set(groups))])
    rng=np.random.default_rng(SEED)
    means=clusters[rng.integers(0,len(clusters),(B,len(clusters)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def run(device='cpu'):
    (ROOT/'results').mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/responses.csv',keep_default_na=False)
    assert len(df)==7056 and len(df.columns)==28
    assert set(df.model_name)==set(MODELS)
    assert not df.duplicated(['model_name','cell_id']).any()
    assert df.groupby('cell_id').model_name.nunique().eq(12).all()
    assert df.groupby('cell_id').full_prompt.nunique().eq(1).all()
    df['company']=df.model_name.map(COMPANY)
    df['response_clean']=df.response.map(clean)
    df['words_recomputed']=df.response_clean.str.split().str.len()
    df['length_ok']=df.words_recomputed.between(100,150)
    labels=sorted(df.y_group_name.unique(),key=len,reverse=True)
    pattern=re.compile(r'(?<!\w)(?:'+ '|'.join(re.escape(x) for x in labels)+r')(?!\w)',re.I)
    masked=df.response_clean.map(lambda s: pattern.sub('[IDENTITY]',s))
    # Models are compared on the exact same prompt string, not on run_id.
    cells=sorted(df.cell_id.unique())
    meta=df.drop_duplicates('cell_id').set_index('cell_id').loc[cells]
    lookup=df.reset_index().set_index(['cell_id','model_name'])['index']
    ix=np.array([[lookup.loc[c,m] for m in MODELS] for c in cells])
    model_summary=df.groupby('model_name',sort=False).agg(company=('company','first'),rows=('cell_id','size'),mean_words=('words_recomputed','mean'),median_words=('words_recomputed','median'),length_compliance=('length_ok','mean')).reset_index()
    model_summary.to_csv(ROOT/'results/model_summary.csv',index=False)
    df[['model_name','cell_id','words_recomputed','response_word_count','length_ok']].to_csv(ROOT/'results/word_count_audit.csv',index=False)
    embeddings={}; encoder_meta={}
    for key in ENCODERS:
        embeddings[key],encoder_meta[key]=encode(df.response_clean.tolist(),key,key,device)
    embeddings['mpnet_masked'],encoder_meta['mpnet_masked']=encode(masked.tolist(),'mpnet','mpnet_masked',device)
    lexical=fit_locked_tfidf(df.response_clean)
    metrics=list(embeddings)+['tfidf']
    pairs=list(itertools.combinations(range(len(MODELS)),2))
    all_prompt=[]; summaries=[]; by_y=[]; by_family=[]; by_article=[]; within=[]; influence=[]; company_rows=[]; robustness=[]
    arrays={}
    groups=meta.y_group_id.to_numpy()
    for enc in metrics:
        distances=[]
        for a,b in pairs:
            A,B=ix[:,a],ix[:,b]
            if enc=='tfidf': sim=np.asarray(lexical[A].multiply(lexical[B]).sum(axis=1)).ravel()
            else: sim=(embeddings[enc][A]*embeddings[enc][B]).sum(axis=1)
            d=np.clip(1-sim,0,2); distances.append(d)
            low,high=bootstrap(d,groups)
            summaries.append(dict(encoder=enc,model_a=MODELS[a],model_b=MODELS[b],company_a=COMPANY[MODELS[a]],company_b=COMPANY[MODELS[b]],mean_distance=float(d.mean()),ci_low=low,ci_high=high,n_prompts=len(d)))
            for dim,y in sorted(set(zip(meta.y_group_dimension,meta.y_group_name))):
                mask=(meta.y_group_dimension==dim)&(meta.y_group_name==y)
                by_y.append(dict(encoder=enc,model_a=MODELS[a],model_b=MODELS[b],y_group_dimension=dim,y_group_name=y,mean_distance=float(d[mask].mean()),n_prompts=int(mask.sum())))
            for family in sorted(meta.prompt_family.unique()):
                mask=meta.prompt_family==family
                by_family.append(dict(encoder=enc,model_a=MODELS[a],model_b=MODELS[b],prompt_family=family,mean_distance=float(d[mask].mean()),n_prompts=int(mask.sum())))
            for article in sorted(meta.x_group_id.unique()):
                mask=meta.x_group_id==article
                by_article.append(dict(encoder=enc,model_a=MODELS[a],model_b=MODELS[b],x_group_id=article,mean_distance=float(d[mask].mean()),n_prompts=int(mask.sum())))
            for ci,c in enumerate(cells):
                all_prompt.append((enc,c,MODELS[a],MODELS[b],float(d[ci])))
            lengthdiff=np.abs(df.words_recomputed.iloc[A].to_numpy()-df.words_recomputed.iloc[B].to_numpy())
            compliant=df.length_ok.iloc[A].to_numpy()&df.length_ok.iloc[B].to_numpy()
            robustness.append(dict(encoder=enc,model_a=MODELS[a],model_b=MODELS[b],word_difference_spearman=float(spearmanr(d,lengthdiff).statistic),both_length_compliant_n=int(compliant.sum()),both_length_compliant_distance=float(d[compliant].mean()) if compliant.any() else None))
        arrays[enc]=np.array(distances).T
        # Equal weights for all model pairs within a company-pair; prompts matched.
        cp={}
        for pi,(a,b) in enumerate(pairs): cp.setdefault(tuple(sorted([COMPANY[MODELS[a]],COMPANY[MODELS[b]]])),[]).append(pi)
        for (ca,cb),pis in cp.items():
            d=arrays[enc][:,pis].mean(axis=1); low,high=bootstrap(d,groups)
            company_rows.append(dict(encoder=enc,company_a=ca,company_b=cb,model_pairs=len(pis),n_prompts=len(cells),mean_distance=float(d.mean()),ci_low=low,ci_high=high))
        # Identity response sensitivity: hold model, article, question family fixed.
        for m in MODELS:
            sub=df[df.model_name==m]
            for dim,g in sub.groupby('y_group_dimension'):
                blockmeans=[]; totalpairs=0; perlabel={}
                for (art,fam),block in g.groupby(['x_group_id','prompt_family']):
                    rows=block.index.to_numpy()
                    if enc=='tfidf': D=1-(lexical[rows]@lexical[rows].T).toarray()
                    else: D=1-embeddings[enc][rows]@embeddings[enc][rows].T
                    D=np.clip(D,0,2); tri=D[np.triu_indices(len(rows),1)]
                    blockmeans.append(float(tri.mean())); totalpairs+=len(tri)
                    for j,y in enumerate(block.y_group_name): perlabel.setdefault(y,[]).append(float(np.delete(D[j],j).mean()))
                low,high=bootstrap(np.array(blockmeans),np.arange(len(blockmeans)))
                within.append(dict(encoder=enc,model_name=m,company=COMPANY[m],y_group_dimension=dim,mean_distance=float(np.mean(blockmeans)),ci_low=low,ci_high=high,n_pairs=totalpairs,n_blocks=len(blockmeans)))
                for y,vals in perlabel.items():
                    influence.append(dict(encoder=enc,model_name=m,company=COMPANY[m],y_group_dimension=dim,y_group_name=y,mean_distance_to_other_identities=float(np.mean(vals)),n_blocks=len(vals)))
    tables={'pairwise_summary':summaries,'pairwise_by_y':by_y,'pairwise_by_family':by_family,'pairwise_by_article':by_article,'identity_sensitivity':within,'y_importance':influence,'company_distances':company_rows,'length_sensitivity':robustness}
    for name,rows in tables.items(): pd.DataFrame(rows).to_csv(ROOT/f'results/{name}.csv',index=False)
    pd.DataFrame(all_prompt,columns=['encoder','cell_id','model_a','model_b','distance']).to_csv(ROOT/'results/pairwise_prompt.csv',index=False)
    np.savez_compressed(ROOT/'results/matched_distances.npz',**arrays)
    correlations=[]
    for e1,e2 in itertools.combinations(metrics,2):
        correlations.append({'encoder_a':e1,'encoder_b':e2,'pair_ranking_spearman':float(spearmanr(arrays[e1].mean(0),arrays[e2].mean(0)).statistic),'prompt_pair_spearman':float(spearmanr(arrays[e1].ravel(),arrays[e2].ravel()).statistic)})
    dump(correlations,'encoder_agreement.json')
    same=np.array([COMPANY[MODELS[a]]==COMPANY[MODELS[b]] for a,b in pairs])
    contrasts=[]
    for enc,arr in arrays.items():
        a=arr[:,same].mean(1); b=arr[:,~same].mean(1); delta=b-a; low,high=bootstrap(delta,groups)
        contrasts.append({'encoder':enc,'within_company_mean':float(a.mean()),'between_company_mean':float(b.mean()),'between_minus_within':float(delta.mean()),'delta_ci_low':low,'delta_ci_high':high,'within_model_pairs':int(same.sum()),'between_model_pairs':int((~same).sum())})
    dump(contrasts,'company_contrast.json')
    # Source similarity is a semantic proximity diagnostic, not factual grounding.
    articles=meta.drop_duplicates('x_group_id')
    article_texts=[s.split('\n\nARTICLE\n',1)[1] for s in articles.full_prompt]
    ae,am=encode(article_texts,'mpnet','articles_mpnet',device)
    article_index=dict(zip(articles.x_group_id,range(len(articles))))
    source_distance=1-np.sum(embeddings['mpnet']*ae[[article_index[x] for x in df.x_group_id]],axis=1)
    source_df=df[['model_name','cell_id','x_group_id','y_group_dimension','y_group_name','prompt_family']].copy()
    source_df['article_semantic_distance']=source_distance
    source_df.to_csv(ROOT/'results/source_proximity.csv',index=False)
    # Automatically selected inspectable high/low-distance cases; no bias verdict.
    case_rows=[]; arr=arrays['mpnet']
    selection=[(int(p),int(q),'global_high') for p,q in zip(*np.unravel_index(np.argsort(arr.ravel())[-6:][::-1],arr.shape))]
    for pi,(a,b) in enumerate(pairs):
        if same[pi]: selection.append((int(np.argmax(arr[:,pi])),pi,'same_company_high'))
    for ci,pi,why in selection:
        a,b=pairs[pi]; ra,rb=df.iloc[ix[ci,a]],df.iloc[ix[ci,b]]
        case_rows.append({'selection':why,'cell_id':cells[ci],'model_a':MODELS[a],'model_b':MODELS[b],'distance':float(arr[ci,pi]),'prompt_question':ra.prompt_question,'y_group_name':ra.y_group_name,'news_source':ra.news_source,'response_a':ra.response,'response_b':rb.response})
    dump(case_rows,'cases.json')
    summary={'rows':len(df),'columns':28,'models':MODELS,'companies':COMPANY,'n_models':12,'n_companies':6,'n_prompts':len(cells),'n_identities':42,'n_articles':2,'n_families':7,'n_model_pairs':66,'prompt_pairs_per_encoder':len(cells)*66,'capture_min':df.captured_utc.min(),'capture_max':df.captured_utc.max(),'source_sha256':hashlib.sha256((ROOT/'data/responses.csv').read_bytes()).hexdigest(),'dimensions':{str(k):sorted(v.y_group_name.unique()) for k,v in df.groupby('y_group_dimension')},'families':sorted(df.prompt_family.unique()),'encoder_metadata':encoder_meta,'bootstrap':{'seed':SEED,'replicates':2000,'resampling_unit':'42 identity clusters for matched-model means; all 14 article-family observations retained per identity; conditional on two fixed articles','identity_sensitivity':'14 article-family block means; exploratory conditional uncertainty'},'warnings':['Distances are not proof of harmful bias, factual accuracy, or a quality ranking.','Single sampled response per prompt/model; no generation variance estimate.','Model version/tier/platform comparisons are cross-sectional, not longitudinal improvement.','Google Gemini web model label and Grok Fast backend are not independent checkpoint verifications.','Original model_provider includes hosting routers; company mapping uses model family, not host.','No neutral-persona control: identity sensitivity is relative to other supplied identity labels.','Six dimensions are heterogeneous labels, not demographic ground truth.']}
    dump(summary,'summary.json')
    make_figures(pd.DataFrame(summaries),pd.DataFrame(within),model_summary)
    print('Analysis complete:',json.dumps({k:summary[k] for k in ['rows','n_prompts','n_model_pairs','source_sha256']}),flush=True)

def make_figures(pairs,within,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'savefig.bbox':'tight'})
    matrix=np.zeros((12,12))
    for r in pairs[pairs.encoder=='mpnet'].itertuples():
        a,b=MODELS.index(r.model_a),MODELS.index(r.model_b); matrix[a,b]=matrix[b,a]=r.mean_distance
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(matrix,cmap='viridis',vmin=0)
    ax.set_xticks(range(12),MODELS,rotation=65,ha='right'); ax.set_yticks(range(12),MODELS)
    for i in range(12):
        for j in range(12): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]<matrix.max()/2 else 'black')
    ax.set_title('Matched-prompt semantic distance (MPNet; 588 prompts)'); fig.colorbar(im,ax=ax,label='1 − cosine similarity')
    fig.savefig(figdir/'model_distances.pdf'); fig.savefig(figdir/'model_distances.png',dpi=160); plt.close(fig)
    p=within[within.encoder=='mpnet'].pivot(index='model_name',columns='y_group_dimension',values='mean_distance').reindex(MODELS)
    fig,ax=plt.subplots(figsize=(9,6)); im=ax.imshow(p,cmap='magma'); ax.set_xticks(range(len(p.columns)),p.columns,rotation=30,ha='right'); ax.set_yticks(range(12),MODELS)
    ax.set_title('Within-model identity sensitivity (article and family held fixed)'); fig.colorbar(im,ax=ax,label='Mean pairwise cosine distance')
    fig.savefig(figdir/'identity_sensitivity.pdf'); fig.savefig(figdir/'identity_sensitivity.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4)); m=models.set_index('model_name').loc[MODELS]
    ax.barh(MODELS,m.length_compliance*100,color='#326c87'); ax.set_xlabel('Responses within 100–150 words (%)'); ax.set_xlim(0,100)
    fig.savefig(figdir/'length_compliance.pdf'); plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cpu'); args=parser.parse_args(); run(args.device)
