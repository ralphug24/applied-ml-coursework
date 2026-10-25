"""Execute selected task notebooks with this environment's registered kernel."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('tasks',nargs='+');args=parser.parse_args()
for task in args.tasks:
    path=ROOT/'notebooks'/f'{task}.ipynb'
    notebook=nbformat.read(path,as_version=4)
    client=NotebookClient(notebook,timeout=1800,kernel_name='applied-ml-coursework',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.write(notebook,path)
    print('Executed',task,flush=True)
