"""Build a readable summary directly from completed run records."""
from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parents[1]
REPO='https://github.com/ralphug24/applied-ml-coursework'

def summary(r):
    task=r['task']
    if task=='hw03':return f"{r['input_rows']:,} input records; {r['retained_rows']:,} retained. Data preparation."
    if task=='hw04':return f"{r['titanic_rows']:,} Titanic records; {r['cleveland_rows']:,} public Cleveland records. Descriptive analysis."
    if task=='hw05':return f"{r['shapes_rows']:,} shapes; {r['digit_rows']:,} MNIST rows in the cluster search."
    if task=='hw06':return f"{r['selected_model']}: MSE {r['test']['mse']:.3f}; RMSE {r['test']['rmse']:.3f}; R² {r['test']['r2']:.3f}."
    if task in ['hw07','hw12']:return f"{r['selected_model']}: accuracy {r['test']['accuracy']:.3f}; macro F1 {r['test']['macro_f1']:.3f}."
    if task=='hw08':return f"{r['epochs']} epochs; ordinary test parameter MAE {r['evaluation']['in_distribution']['parameter_mae']:.4f}; extended-case MAE {r['evaluation']['extended']['parameter_mae']:.4f}."
    if task=='hw09':return f"{r['epochs']} epochs; CIFAR-100 accuracy {r['cnn']['metrics']['accuracy']:.3f}; Reuters accuracy {r['lstm']['metrics']['accuracy']:.3f}. Separate tasks."
    if task=='hw10':return '; '.join(f"{n}: accuracy {m['accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}" for n,m in r['metrics'].items())
    if task=='hw11':return f"{r['transactions']:,} baskets; {r['frequent_itemsets']:,} frequent itemsets; {r['rules_confidence_at_least_08']} rules at confidence ≥ 0.8."
    return ''

def render():
    rows=[];preview_rows=[]
    records=[]
    for p in sorted((ROOT/'reports/runs').glob('*/*/metrics.json')):
        r=json.loads(p.read_text(encoding='utf-8'));records.append(r)
        url=REPO+'/blob/main/'+p.relative_to(ROOT).as_posix()
        row=f'<tr><th>{html.escape(r["task"])}</th><td>{html.escape(r["profile"])}</td><td>{html.escape(summary(r))}</td><td>{r["elapsed_seconds"]:.1f} s</td><td><a href="{url}">Full record</a></td></tr>'
        (preview_rows if r['profile']=='preview' else rows).append(row)
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Applied ML — recorded experiment results</title><style>body{font:16px/1.8 system-ui,sans-serif;background:#101718;color:#edf1ed;max-width:1100px;margin:auto;padding:36px 22px}a{color:#e7ba5c}h1{line-height:1.2}p{color:#bcc9c1}table{width:100%;border-collapse:collapse;font-size:.92rem}th,td{padding:13px;border-bottom:1px solid #35423b;text-align:left;vertical-align:top}thead{color:#e7ba5c}.scroll{overflow:auto}code{font-family:monospace}</style><a href="https://ralphugboko.dev/projects?discipline=school-projects&amp;project=applied-ml-coursework">← Portfolio case study</a><h1>Applied ML: recorded experiment results</h1><p>Raphael Ugboko · CPSC 6300 coursework and portable experiments</p><p>This table is generated from completed metrics files. Full and preview profiles have different scopes. Preview runs demonstrate execution and are not competitive performance benchmarks. Metrics from different datasets are not a model ranking.</p><div class="scroll"><table><thead><tr><th>Task</th><th>Profile</th><th>Recorded result</th><th>Elapsed</th><th>Evidence</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><h2>How to read the records</h2><p>Each full record includes task configuration, sample scope, seed, actual execution timestamp, environment versions and code checksum. Training occurred on a shared CPU environment with two threads per process; elapsed time is not a hardware benchmark.</p><p>Original Fall 2025 saved submission outputs are retained separately. Portable versions update paths and document changes to splitting, preprocessing and evaluation. Read the <a href="https://github.com/ralphug24/applied-ml-coursework/blob/main/docs/methodology.md">methodology guide</a> before comparing results.</p></html>'''
    if preview_rows:
        content=content.replace('<h2>How to read the records</h2>','<details><summary>Preview execution records · reduced training scope</summary><p>These runs check that the workflows execute. Read their sample sizes and optimization steps before interpreting a metric.</p><div class="scroll"><table><tbody>'+''.join(preview_rows)+'</tbody></table></div></details><h2>How to read the records</h2>')
    (ROOT/'reports/run-results.html').write_text(content,encoding='utf-8')
    (ROOT/'reports/run-index.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Summarized',len(records),'completed runs.')

if __name__=='__main__':render()
