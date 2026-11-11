import json, shutil, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
print('AFTERLIGHT environment check')
print('Docker:',shutil.which('docker') or 'not found: install and open Docker Desktop')
if shutil.which('docker'):
    subprocess.run(['docker','compose','version'],check=False)
    subprocess.run(['docker','info','--format','{{.ServerVersion}}'],check=False)
for name in ['data/catalog.json','artifacts/report.json','artifacts/pair-unet.pt','artifacts/random-forest.joblib']:
    p=root/name;print(name,'OK' if p.exists() else 'MISSING')
