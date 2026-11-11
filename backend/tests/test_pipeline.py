import io, json, unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from afterlight.api import app
from afterlight.core import DATA, ARTIFACTS, catalog, dataset_hash, scores, load_pair

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app);cls.data=catalog();cls.sid=next(s['id'] for s in cls.data['samples'] if s['split']=='test')
    def test_event_disjoint(self):
        groups={k:{s['event'] for s in self.data['samples'] if s['split']==k} for k in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:self.assertFalse(groups[a]&groups[b])
        self.assertTrue(all(groups.values()))
    def test_paired_paths_and_masks(self):
        for s in self.data['samples']:
            with Image.open(DATA/s['pre']) as a,Image.open(DATA/s['post']) as b:self.assertEqual(a.size,b.size)
            with Image.open(DATA/s['mask']) as im:mask=np.asarray(im)
            self.assertTrue(set(np.unique(mask))<={0,1,2,3,4,255})
    def test_reproducible_report(self):
        report=json.loads((ARTIFACTS/'report.json').read_text());self.assertEqual(report['dataset_hash'],dataset_hash(self.data['samples']))
        self.assertEqual(report['splits'],{k:sum(s['split']==k for s in self.data['samples']) for k in ['train','validation','test']})
    def test_perfect_metrics(self):
        a=np.array([[0,1,2],[3,4,255]],dtype='uint8');m=scores(a,a);self.assertEqual(m['damage_macro_f1'],1);self.assertEqual(m['building_iou'],1);self.assertEqual(m['evaluated_pixels'],5)
    def test_health_and_metadata(self):
        self.assertEqual(self.client.get('/api/health').json()['status'],'ok');self.assertEqual(self.client.get('/api/stats').json()['sample_count'],len(self.data['samples']))
    def test_unknown_scene(self):self.assertEqual(self.client.post('/api/predict/not-a-scene').status_code,404)
    def test_invalid_model(self):self.assertEqual(self.client.post('/api/predict/'+self.sid+'?model=unknown').status_code,422)
    def test_real_model_outputs(self):
        for model in ['pair-unet','random-forest']:
            r=self.client.post('/api/predict/'+self.sid+'?model='+model);self.assertEqual(r.status_code,200)
            d=r.json();self.assertEqual(sum(d['class_pixels']),128**2);self.assertTrue(d['overlay']);self.assertIsNotNone(d['metrics'])
    def test_overlay(self):self.assertEqual(self.client.get('/api/images/'+self.sid+'/labels').headers['content-type'],'image/png')
    def test_invalid_upload(self):
        r=self.client.post('/api/predict-upload',files={'before':('x.png',b'not an image','image/png'),'after':('y.png',b'bad','image/png')});self.assertEqual(r.status_code,400)
    def test_mismatched_upload(self):
        files={}
        for name,size in [('before',(40,40)),('after',(50,40))]:
            out=io.BytesIO();Image.new('RGB',size).save(out,format='PNG');files[name]=(name+'.png',out.getvalue(),'image/png')
        self.assertEqual(self.client.post('/api/predict-upload',files=files).status_code,400)
    def test_valid_upload_has_no_metrics(self):
        s=self.data['samples'][0];files={name:(name+'.png',(DATA/s[k]).read_bytes(),'image/png') for name,k in [('before','pre'),('after','post')]}
        d=self.client.post('/api/predict-upload',files=files).json();self.assertIsNone(d['metrics']);self.assertTrue(d['overlay'])
    def test_stale_data_rejected(self):
        with patch('afterlight.api.dataset_valid',return_value=False):
            self.assertEqual(self.client.post('/api/predict/'+self.sid).status_code,409)
    def test_channel_pair(self):
        x,y=load_pair(self.data['samples'][0],128);self.assertEqual(x.shape,(128,128,6));self.assertEqual(y.shape,(128,128))

if __name__=='__main__':unittest.main()
