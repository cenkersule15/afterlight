import argparse, collections, hashlib, io, json, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .core import DATA

TEST_EVENTS={'santa-rosa-wildfire','hurricane-michael'}
VALIDATION_EVENTS={'mexico-earthquake','midwest-flooding'}

def split_for(event):
    return 'test' if event in TEST_EVENTS else 'validation' if event in VALIDATION_EVENTS else 'train'

def prepare_parquet(path, per_event=8):
    import pyarrow.parquet as pq
    DATA.mkdir(parents=True,exist_ok=True);(DATA/'scenes').mkdir(exist_ok=True)
    counts=collections.Counter();samples=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            sid=Path(row['image_name']).name;event=sid.rsplit('_',1)[0]
            if counts[event]>=per_event:continue
            sample={'id':sid,'event':event,'split':split_for(event),'source':'xBD / xView2','resolution':512,'original_resolution':1024}
            for kind,column in [('pre','t1_image'),('post','t2_image'),('mask','t2_mask')]:
                im=Image.open(io.BytesIO(row[column]['bytes']))
                if kind=='mask':
                    im=im.convert('L');a=np.asarray(im).copy();a[a>4]=255
                    im=Image.fromarray(a).resize((512,512),Image.Resampling.NEAREST)
                else:im=im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
                name=f'scenes/{sid}_{kind}.png';im.save(DATA/name);sample[kind]=name
            a=np.asarray(Image.open(DATA/sample['mask']));sample['class_pixels']=np.bincount(a[a<5].ravel(),minlength=5).tolist()
            counts[event]+=1;samples.append(sample)
    provenance={'name':'xBD pilot subset','authors':'Gupta et al. / DIU / CMU SEI; imagery: Maxar Open Data Program',
      'source':'https://huggingface.co/datasets/kshitijrajsharma/xview2-xbd',
      'mirror_revision':'079adbab22aa09633dfd50ece0dd0137431a9780',
      'source_shard':'data/train-00000-of-00017.parquet','source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      'license':'CC BY-NC-SA 4.0','license_url':'https://creativecommons.org/licenses/by-nc-sa/4.0/',
      'transforms':'Selected at most 8 tiles per event from first training shard; RGB downsampled 1024 to 512, masks nearest-neighbour; IDs >4 ignored as 255. Training/evaluation resized again to model resolution.',
      'scope':'Small convenience sample. Event-disjoint custom pilot splits; NOT official xView2 train/test or challenge score. No accuracy claims outside this subset.'}
    (DATA/'catalog.json').write_text(json.dumps({'provenance':provenance,'samples':samples},indent=2))
    print('Prepared',len(samples),'pairs across',len(counts),'events:',dict(counts))

def import_xbd(root):
    from shapely import wkt
    DATA.mkdir(parents=True,exist_ok=True);(DATA/'scenes').mkdir(exist_ok=True)
    samples=[]
    for label in sorted(Path(root).rglob('*_post_disaster.json')):
        labels=json.loads(label.read_text());sid=label.name.replace('_post_disaster.json','');event=sid.rsplit('_',1)[0]
        post=next(iter(Path(root).rglob(sid+'_post_disaster.png')),None);pre=next(iter(Path(root).rglob(sid+'_pre_disaster.png')),None)
        if not post or not pre:continue
        with Image.open(post) as im:mask=Image.new('L',im.size)
        draw=ImageDraw.Draw(mask)
        for feature in labels['features']['xy']:
            poly=wkt.loads(feature['wkt']);cls={'no-damage':1,'minor-damage':2,'major-damage':3,'destroyed':4}.get(feature['properties'].get('subtype'),255)
            for shape in getattr(poly,'geoms',[poly]):
                draw.polygon(list(shape.exterior.coords),fill=cls)
                for hole in shape.interiors:draw.polygon(list(hole.coords),fill=0)
        s={'id':sid,'event':event,'split':split_for(event),'source':'xBD local import','resolution':512}
        for kind,p in [('pre',pre),('post',post)]:
            name=f'scenes/{sid}_{kind}.png';Image.open(p).convert('RGB').resize((512,512)).save(DATA/name);s[kind]=name
        name=f'scenes/{sid}_mask.png';mask.resize((512,512),Image.Resampling.NEAREST).save(DATA/name);s['mask']=name;a=np.asarray(Image.open(DATA/name));s['class_pixels']=np.bincount(a[a<5].ravel(),minlength=5).tolist();samples.append(s)
    if not samples:raise ValueError('No matching xBD PNG and JSON pairs found')
    (DATA/'catalog.json').write_text(json.dumps({'provenance':{'name':'Local xBD import','license':'CC BY-NC-SA 4.0','source':'https://xview2.org/','scope':'Custom event-disjoint split; review event coverage before training'},'samples':samples},indent=2))
    print('Imported',len(samples),'pairs')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--parquet');g.add_argument('--xbd');p.add_argument('--per-event',type=int,default=8);a=p.parse_args()
    prepare_parquet(a.parquet,a.per_event) if a.parquet else import_xbd(a.xbd)
