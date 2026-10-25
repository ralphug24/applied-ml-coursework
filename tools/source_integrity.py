"""Compare the published Git source bytes with the supplied-source checksums."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'provenance.json').read_text())
different=[]
for item in records:
    content=subprocess.run(['git','show','HEAD:'+item['public_path']],cwd=ROOT,capture_output=True,check=True).stdout
    if hashlib.sha256(content).hexdigest()!=item['sha256']:different.append(item['public_path'])
print(json.dumps({'sources':len(records),'git_byte_differences':different}))
