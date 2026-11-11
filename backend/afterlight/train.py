import argparse, json, random, time
from datetime import datetime, timezone
import joblib, numpy as np, torch
from sklearn.ensemble import RandomForestClassifier
from .core import ARTIFACTS, catalog, load_pair, features, score_matrix, dataset_hash
from .model import PairUNet

def train(epochs=20, size=128):
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    samples=catalog()['samples'];groups={k:{s['event'] for s in samples if s['split']==k} for k in ['train','validation','test']}
    if any(not x for x in groups.values()):raise ValueError('Need non-empty train, validation and test event groups.')
    if any(groups[a]&groups[b] for a,b in [('train','test'),('train','validation'),('validation','test')]):raise ValueError('Event leakage detected')
    parts={k:[(s,*load_pair(s,size)) for s in samples if s['split']==k] for k in groups}
    ARTIFACTS.mkdir(exist_ok=True,parents=True)
    counts=np.zeros(5)
    for _,_,y in parts['train']:counts+=np.bincount(y[y!=255].ravel(),minlength=5)
    weights=(counts.sum()/np.maximum(counts,1))**0.8;weights=np.minimum(weights,150);weights/=weights.mean()
    model=PairUNet();optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),ignore_index=255)
    def evaluate(predict,split):
        cm=np.zeros((5,5),dtype=np.int64)
        for _,x,y in parts[split]:
            p=predict(x);v=y!=255;cm+=np.bincount((y[v].astype(int)*5+p[v]).ravel(),minlength=25).reshape(5,5)
        return score_matrix(cm)
    def cnn(x):
        model.eval()
        with torch.no_grad():return model(torch.tensor(x.transpose(2,0,1)[None])).argmax(1)[0].numpy().astype('uint8')
    history=[];best=-1;started=time.time()
    for ep in range(epochs):
        model.train();losses=[]
        order=list(parts['train']);random.shuffle(order)
        for _,x,y in order:
            if not np.any(y!=255):continue
            if random.random()<.5:x=x[:,::-1].copy();y=y[:,::-1].copy()
            optimizer.zero_grad();target=torch.tensor(y[None].copy(),dtype=torch.long);logits=model(torch.tensor(x.transpose(2,0,1)[None].copy()));loss=loss_fn(logits,target);valid=target!=255;binary=(target>0)&valid;prob=1-torch.softmax(logits,1)[:,0];dice=1-(2*(prob*binary).sum()+1)/((prob*valid).sum()+binary.sum()+1);loss=loss+0.5*dice;loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=evaluate(cnn,'validation');row={'epoch':ep+1,'loss':float(np.mean(losses)),'validation_macro_f1':val['damage_macro_f1']};history.append(row)
        if val['damage_macro_f1']>best:
            best=val['damage_macro_f1'];torch.save({'state_dict':model.state_dict(),'size':size},ARTIFACTS/'pair-unet.pt')
        print(json.dumps(row),flush=True)
    model.load_state_dict(torch.load(ARTIFACTS/'pair-unet.pt',map_location='cpu',weights_only=True)['state_dict'])
    rng=np.random.default_rng(42);xx=[];yy=[]
    for _,x,y in parts['train']:
        f=features(x).reshape(-1,10);flat=y.ravel()
        for cls in range(5):
            ids=np.flatnonzero(flat==cls)
            if len(ids):ids=rng.choice(ids,min(len(ids),300),replace=False);xx.append(f[ids]);yy.append(flat[ids])
    forest=RandomForestClassifier(n_estimators=80,max_depth=14,class_weight='balanced',n_jobs=4,random_state=42)
    forest.fit(np.concatenate(xx),np.concatenate(yy));joblib.dump(forest,ARTIFACTS/'random-forest.joblib',compress=3)
    models=[{'id':'pair-unet','name':'Pair U-Net','kind':'deep-learning','metrics':evaluate(cnn,'test')},
            {'id':'random-forest','name':'Random Forest','kind':'machine-learning','metrics':evaluate(lambda x:forest.predict(features(x).reshape(-1,10)).reshape(x.shape[:2]),'test')}]
    report={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Small-sample research pilot, not a validated disaster-response system','seed':42,'size':size,'epochs':epochs,'seconds':round(time.time()-started,2),'dataset_hash':dataset_hash(samples),'splits':{k:len(v) for k,v in parts.items()},'events':{k:sorted(v) for k,v in groups.items()},'history':history,'models':models,'evaluation':'Pooled pixel confusion matrix on held-out events; four foreground classes, absent classes reported; not the official xView2 challenge score. Validation selects CNN checkpoint; test used only for final evaluation.'}
    (ARTIFACTS/'report.json').write_text(json.dumps(report,indent=2));print('Saved evaluated pilot models and report.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--size',type=int,default=128);a=p.parse_args();train(a.epochs,a.size)
