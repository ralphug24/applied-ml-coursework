"""Record checksums of the published code, data and fitted model artifacts."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
records=[]
for directory in ['data','models']:
    for file in sorted((ROOT/directory).rglob('*')):
        if not file.is_file() or 'downloads' in file.parts or any(p.startswith('transformer-') for p in file.parts):continue
        records.append({'path':file.relative_to(ROOT).as_posix(),'bytes':file.stat().st_size,'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
(ROOT/'reports/artifact-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Recorded',len(records),'data/model artifacts.')
