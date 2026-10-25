"""Add configuration evidence read from the completed runs' own saved arguments."""
from pathlib import Path
import json,os
os.environ['USE_TF']='0'
import torch
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'reports/runs/full/hw10/metrics.json'
r=json.loads(p.read_text())
for name,key in [('bart-base','facebook/bart-base'),('roberta-base','FacebookAI/roberta-base')]:
    path=ROOT/'models/full/hw10'/('transformer-'+name)/'training_args.bin'
    args=torch.load(path,weights_only=False,map_location='cpu')
    fields=['num_train_epochs','learning_rate','weight_decay','per_device_train_batch_size','per_device_eval_batch_size','seed','eval_strategy','save_strategy']
    r['metrics'][key]['training_configuration']={field:getattr(args,field) for field in fields}
    r['metrics'][key]['training_configuration_source']='Completed checkpoint training_args.bin'
p.write_text(json.dumps(r,indent=2),encoding='utf-8')
print('Recorded configuration from both completed checkpoint argument files.')
