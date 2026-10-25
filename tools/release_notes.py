from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'reports/runs/full/hw10/metrics.json').read_text())
lines=['# Full transformer coursework checkpoints','',
'BART-base and RoBERTa-base sequence classifiers from the completed CPSC 6300 syllogism-validity workflow. Each ZIP includes the fitted model, configuration and tokenizer.','',
f'The run uses {r["train_rows"]} training, {r["validation_rows"]} validation and {r["test_rows"]} held-out examples, with three epochs, batch size 16, learning rate 2e-5 and seed 42.','',
'See the [saved-model guide](https://github.com/ralphug24/applied-ml-coursework/blob/main/models/README.md) for loading examples and the [run record](https://github.com/ralphug24/applied-ml-coursework/blob/main/reports/runs/full/hw10/metrics.json) for metrics, class-level errors and pretrained model revisions. The benchmark includes a majority baseline; these results do not establish general reasoning capability.','',
'## ZIP checksums','']
for file in sorted((ROOT/'releases').glob('*-checkpoint.zip')):
    with file.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    lines.append(f'- `{file.name}` — {file.stat().st_size:,} bytes; SHA-256 `{sha}`.')
(ROOT/'releases/release-notes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote model-release notes and checksums.')
