"""Export archived notebooks and package actual run evidence for the portfolio."""
from pathlib import Path
import json, shutil, sys, re
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nbformat
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
TITLES={'hw03':'Data preparation','hw04':'Exploratory analysis','hw05':'Clustering','hw06':'Regression and cross-validation','hw07':'Feature and model selection','hw08':'Dense neural regression','hw09':'CNN and LSTM classifiers','hw10':'Transformer fine-tuning','hw11':'Association mining','hw12':'PCA and SVD'}

def package():
    exporter=HTMLExporter(template_name='classic',sanitize_html=True)
    exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    for task,title in TITLES.items():
        nb=nbformat.read(ROOT/'original/submissions'/f'{task}.ipynb',as_version=4)
        # Archived content is rendered, never executed here.
        html,_=exporter.from_notebook_node(nb)
        html=html.replace('<title>Notebook</title>',f'<title>{task} · {title} · Original coursework</title>')
        html=html.replace('<body>','<body><div style="padding:18px 24px;background:#101718;color:white;font-family:system-ui">Original Fall 2025 coursework submission · Saved outputs. <a style="color:#eec16d" href="https://ralphugboko.dev/projects?discipline=school-projects&amp;project=applied-ml-coursework">Portfolio case study</a></div>',1)
        (ROOT/'reports/original'/f'{task}.html').write_text(html,encoding='utf-8')
        profile='full'
        notebook=nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell(f'# {title}\n\nCPSC 6300 · Raphael Ugboko\n\nThis portable version runs the documented task implementation. The original coursework is in `original/submissions/{task}.ipynb`. Read `docs/methodology.md` for changes and run-profile definitions. The selected **{profile}** profile records its actual scope; switch to `full` for course-sized training where supported. Install the matching requirements file from the repository README before starting.'),
            nbformat.v4.new_code_cell('from pathlib import Path\nimport sys\nROOT = Path.cwd() if (Path.cwd() / "coursework.py").exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\nfrom coursework import run_task'),
            nbformat.v4.new_code_cell(f'result = run_task("{task}", profile="{profile}")'),
            nbformat.v4.new_markdown_cell('## Read the results\n\nInspect the profile, row counts, split plan and metrics together. Preview runs demonstrate execution; they do not establish competitive model quality.'),
            nbformat.v4.new_code_cell('import json\nprint(json.dumps(result, indent=2))'),
            nbformat.v4.new_code_cell(f'from IPython.display import display, Image\nfor figure in sorted((ROOT / "reports/runs/{profile}/{task}").glob("*.png")):\n    display(Image(filename=str(figure)))')
        ],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
        notebook_path=ROOT/'notebooks'/f'{task}.ipynb'
        if not notebook_path.exists() or not any(c.get('execution_count') for c in nbformat.read(notebook_path,as_version=4).cells):
            nbformat.write(notebook,notebook_path)
    source=ROOT/'original/submissions/hw08.ipynb'
    original=json.loads(source.read_text(encoding='utf-8'))
    # Export one actual saved prediction plot as historical coursework evidence.
    import base64
    for output in original['cells'][26].get('outputs',[]):
        data=output.get('data',{}).get('image/png')
        if data:
            (ROOT/'reports/original/hw08-extended-example.png').write_bytes(base64.b64decode(''.join(data)));break
    entries=[]
    for file in sorted((ROOT/'reports/runs').glob('*/*/metrics.json')):
        content=json.loads(file.read_text());entries.append(content)
    (ROOT/'reports/run-index.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    print('Exported',len(TITLES),'archived reports and runnable task notebooks;',len(entries),'recorded runs.')

if __name__=='__main__':package()
