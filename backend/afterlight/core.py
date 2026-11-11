from pathlib import Path
import base64, hashlib, io, json, os
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(os.environ.get('AFTERLIGHT_ROOT', Path(__file__).resolve().parents[2]))
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
CLASSES = ['background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed']
COLORS = np.array([[0,0,0,0],[75,214,169,190],[244,198,93,200],[247,142,77,205],[239,94,111,210]], dtype=np.uint8)

def catalog():
    p = DATA / 'catalog.json'
    return json.loads(p.read_text()) if p.exists() else {'samples': [], 'provenance': {}}

def load_pair(sample, size=256):
    images = [np.asarray(Image.open(DATA / sample[k]).convert('RGB').resize((size,size)), dtype=np.float32)/255 for k in ['pre','post']]
    mask = np.asarray(Image.open(DATA / sample['mask']).resize((size,size), Image.Resampling.NEAREST), dtype=np.uint8) if sample.get('mask') else None
    return np.concatenate(images, axis=2), mask

def features(x):
    before,after=x[...,:3],x[...,3:]
    return np.concatenate([before,after,np.abs(before-after),after.mean(-1,keepdims=True)-before.mean(-1,keepdims=True)],axis=-1)

def scores(truth, prediction):
    valid=truth!=255
    cm=np.bincount((truth[valid].astype(int)*5+prediction[valid].astype(int)).ravel(),minlength=25).reshape(5,5)
    return score_matrix(cm)

def score_matrix(cm):
    cm=np.asarray(cm);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(5,dtype=float),where=den>0)
    iou=np.divide(tp,den-tp,out=np.zeros(5,dtype=float),where=(den-tp)>0)
    present=den[1:]>0
    btp=cm[1:,1:].sum(); bfp=cm[0,1:].sum();bfn=cm[1:,0].sum()
    return {'confusion_matrix':cm.tolist(),'per_class_f1':f1.tolist(),'per_class_iou':iou.tolist(),
      'damage_macro_f1':float(f1[1:][present].mean()) if present.any() else 0.,
      'building_iou':float(btp/max(btp+bfp+bfn,1)), 'building_dice':float(2*btp/max(2*btp+bfp+bfn,1)),
      'evaluated_pixels':int(cm.sum()), 'absent_classes':[CLASSES[i] for i in range(1,5) if den[i]==0]}

def png64(arr):
    b=io.BytesIO();Image.fromarray(arr).save(b,format='PNG');return base64.b64encode(b.getvalue()).decode()

def findings(mask, confidence=None):
    regions,n=ndimage.label(mask>0)
    result=[]
    for i,sl in enumerate(ndimage.find_objects(regions),1):
        if sl is None:continue
        select=regions[sl]==i;area=int(select.sum())
        if area<4:continue
        values=mask[sl][select];cls=int(np.bincount(values,minlength=5)[1:].argmax()+1)
        ys,xs=sl
        result.append({'id':i,'class_id':cls,'area_pixels':area,'bbox':[xs.start,ys.start,xs.stop,ys.stop],
                       'score':float(confidence[sl][select].mean()) if confidence is not None else None})
    return sorted(result,key=lambda r:-r['area_pixels'])[:100]

def dataset_hash(samples):
    h=hashlib.sha256()
    for s in sorted(samples,key=lambda x:x['id']):
        for k in ['pre','post','mask']:
            if s.get(k):h.update((DATA/s[k]).read_bytes())
        h.update(json.dumps(s,sort_keys=True).encode())
    return h.hexdigest()
