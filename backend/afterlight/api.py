from functools import lru_cache
import io, json, threading
from typing import Literal
import joblib, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from .core import DATA, ARTIFACTS, CLASSES, COLORS, catalog, load_pair, features, findings, png64, scores, dataset_hash

app=FastAPI(title='AFTERLIGHT',version='1.0.0',docs_url='/api/docs',openapi_url='/api/openapi.json')
lock=threading.Lock()
Image.MAX_IMAGE_PIXELS=16_000_000

@lru_cache(maxsize=2)
def dataset_valid(report_mtime, catalog_mtime):
    r=json.loads((ARTIFACTS/'report.json').read_text())
    return r.get('dataset_hash')==dataset_hash(catalog()['samples'])

def report():
    p=ARTIFACTS/'report.json'
    if not p.exists():return None
    value=json.loads(p.read_text())
    value['data_matches']=dataset_valid(p.stat().st_mtime_ns,(DATA/'catalog.json').stat().st_mtime_ns)
    return value

@lru_cache(maxsize=1)
def stats():
    c=catalog();counts=np.zeros(5,dtype=np.int64)
    for s in c['samples']:
        a=np.asarray(Image.open(DATA/s['mask']));counts+=np.bincount(a[a<5].ravel(),minlength=5)
    return {'sample_count':len(c['samples']),'event_count':len({s['event'] for s in c['samples']}),
      'split_counts':{k:sum(s['split']==k for s in c['samples']) for k in ['train','validation','test']},
      'class_pixels':counts.tolist(),'classes':CLASSES,'provenance':c['provenance']}

def scene(sid):
    s=next((s for s in catalog()['samples'] if s['id']==sid),None)
    if s is None:raise HTTPException(404,'Scene not found')
    return s

@lru_cache(maxsize=2)
def get_model(name):
    if not report():raise HTTPException(503,'No trained model. Run the training service first.')
    if name=='random-forest':return joblib.load(ARTIFACTS/'random-forest.joblib')
    if name=='pair-unet':
        import torch
        from .model import PairUNet
        torch.set_num_threads(2)
        m=PairUNet();m.load_state_dict(torch.load(ARTIFACTS/'pair-unet.pt',map_location='cpu',weights_only=True)['state_dict']);m.eval();return m
    raise HTTPException(400,'Unknown model')

def infer(x,name,truth=None):
    if not (report() or {}).get('data_matches'):raise HTTPException(409,'Dataset changed. Retrain models and restart the API before inference.')
    with lock:
        model=get_model(name)
        if name=='random-forest':
            p=model.predict_proba(features(x).reshape(-1,10));pred=model.classes_[p.argmax(1)].reshape(x.shape[:2]);conf=p.max(1).reshape(x.shape[:2])
        else:
            import torch
            with torch.no_grad():
                p=torch.softmax(model(torch.tensor(x.transpose(2,0,1)[None])),1)[0].numpy();pred=p.argmax(0).astype('uint8');conf=p.max(0)
    return {'model':name,'size':x.shape[0],'overlay':png64(COLORS[pred]),'mask':png64(pred.astype('uint8')),
      'regions':findings(pred,conf),'class_pixels':np.bincount(pred.ravel(),minlength=5).tolist(),
      'metrics':scores(truth,pred) if truth is not None else None,
      'score_note':'Uncalibrated model scores, not certified probabilities. Connected regions are not verified individual buildings.'}

@app.get('/api/health')
def health():return {'status':'ok','model_ready':bool((report() or {}).get('data_matches'))}
@app.get('/api/catalog')
def get_catalog():return catalog()
@app.get('/api/stats')
def get_stats():return stats()
@app.get('/api/models')
def get_report():return report() or {'models':[],'history':[],'scope':'No trained model'}
@app.get('/api/images/{sid}/{kind}')
def image(sid:str,kind:Literal['pre','post','mask','labels']):
    s=scene(sid)
    if kind=='labels':
        from fastapi.responses import Response
        a=np.asarray(Image.open(DATA/s['mask']));a=np.where(a<5,a,0);b=io.BytesIO();Image.fromarray(COLORS[a]).save(b,format='PNG');return Response(b.getvalue(),media_type='image/png')
    return FileResponse(DATA/s[kind],media_type='image/png')
@app.post('/api/predict/{sid}')
def predict(sid:str,model:Literal['pair-unet','random-forest']='pair-unet'):
    s=scene(sid);size=(report() or {}).get('size',128);x,y=load_pair(s,size);return infer(x,model,y)

async def read_image(upload):
    raw=await upload.read(8*1024*1024+1)
    if len(raw)>8*1024*1024:raise HTTPException(413,'Each image must be at most 8 MiB')
    try:
        im=Image.open(io.BytesIO(raw))
        if im.width<32 or im.height<32 or im.width*im.height>16_000_000:raise ValueError()
        if im.format not in ['PNG','JPEG']:raise ValueError()
        im.load()
        return im.convert('RGB')
    except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError):raise HTTPException(400,'Use valid PNG/JPEG images, at least 32×32 and at most 16 megapixels')

@app.post('/api/predict-upload')
async def predict_upload(before:UploadFile=File(...),after:UploadFile=File(...),model:Literal['pair-unet','random-forest']='pair-unet'):
    a,b=await read_image(before),await read_image(after)
    if a.size!=b.size:raise HTTPException(400,'Before and after dimensions must match. Images must already be spatially aligned.')
    size=(report() or {}).get('size',128);x=np.concatenate([np.asarray(im.resize((size,size)),dtype=np.float32)/255 for im in [a,b]],axis=-1)
    return infer(x,model)
