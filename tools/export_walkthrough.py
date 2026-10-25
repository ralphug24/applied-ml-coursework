"""Export the executed beginner notebook as a standalone, readable HTML page."""
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / 'notebooks/START_HERE.ipynb', as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
if not all(cell.execution_count is not None for cell in code_cells):
    raise RuntimeError('Execute every notebook cell before exporting the walkthrough.')
if any(output.output_type == 'error' for cell in code_cells for output in cell.outputs):
    raise RuntimeError('Resolve notebook errors before exporting.')
exporter = HTMLExporter(template_name='basic')
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
body, _ = exporter.from_notebook_node(notebook)
style = '''
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b0e10;color:#e8ebee;font:17px/1.7 system-ui,sans-serif}main{max-width:980px;margin:auto;padding:36px 24px 72px}nav{display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #30363a;padding-bottom:22px;margin-bottom:32px}a{color:#e6b85e;text-underline-offset:3px}h1{font-size:clamp(28px,5vw,42px);line-height:1.2}h2{font-size:26px;line-height:1.3;margin-top:50px}h3{line-height:1.3}p,li{max-width:82ch}pre{overflow:auto;background:#151b1f;border:1px solid #30383e;border-radius:10px;padding:18px;font:14px/1.6 ui-monospace,monospace}code{font-family:ui-monospace,monospace;font-size:.88em}p code,li code{background:#1b2226;padding:2px 5px;border-radius:4px}table{border-collapse:collapse;display:block;max-width:100%;overflow:auto;font-size:15px;margin:20px 0}td,th{padding:9px 16px;text-align:right;border-bottom:1px solid #333c42}th{color:#e6b85e}img{max-width:100%;height:auto;background:white;border-radius:8px}.output_wrapper{margin-bottom:30px}.output_area{overflow:auto}.input_area .highlight{background:transparent}.anchor-link{display:none}.input{margin:20px 0}.output_stderr{color:#e6b85e}footer{border-top:1px solid #30363a;margin-top:40px;padding-top:20px;color:#a4abb0;font-size:14px}@media(max-width:600px){main{padding:24px 16px}pre{font-size:12px}td,th{padding:7px 10px}}
'''
html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Start here · Applied Machine Learning · Raphael Ugboko</title><style>' + style + '</style></head><body><main><nav aria-label="Walkthrough links"><a href="https://ralphugboko.dev/projects?discipline=school-projects&amp;project=applied-ml-coursework">← Case study</a><a href="https://colab.research.google.com/github/ralphug24/applied-ml-coursework/blob/main/notebooks/START_HERE.ipynb">Run in Google Colab ↗</a><a href="https://github.com/ralphug24/applied-ml-coursework/blob/main/docs/START_HERE.md">Setup &amp; glossary ↗</a></nav>' + body + '<footer>Executed notebook output · The notebook records execution timestamps and library versions. Code, data and the full assignment collection are available in the linked repository.</footer></main></body></html>'
(ROOT / 'reports/start-here.html').write_text(html, encoding='utf-8')
print('Exported all executed cells and chart to reports/start-here.html')
