"""Restore the non-commercial xBD pilot subset after cloning a source-only repository."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
from urllib.request import urlopen
root=Path(__file__).resolve().parents[1]
source=root/'data/raw/source.parquet';source.parent.mkdir(parents=True,exist_ok=True)
revision='079adbab22aa09633dfd50ece0dd0137431a9780'
url=f'https://huggingface.co/datasets/kshitijrajsharma/xview2-xbd/resolve/{revision}/data/train-00000-of-00017.parquet'
print('xBD / xView2; Maxar Open Data; CC BY-NC-SA 4.0. Non-commercial use only.')
print('Downloading approximately 509 MB. The packaged ZIP already includes the prepared subset.')
if not source.exists():
    temp=source.with_suffix('.part')
    with urlopen(url,timeout=120) as response,temp.open('wb') as out:
        while chunk:=response.read(4*1024*1024):out.write(chunk)
    temp.rename(source)
expected=json.loads((root/'data/catalog.json').read_text())['provenance']['source_sha256']
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise SystemExit('Checksum mismatch. Delete data/raw/source.parquet and retry from the pinned source.')
env={**os.environ,'PYTHONPATH':str(root/'backend')}
subprocess.run([sys.executable,'-m','afterlight.prepare','--parquet',str(source)],env=env,cwd=root,check=True)
