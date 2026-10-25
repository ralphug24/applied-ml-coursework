"""Package completed transformer checkpoints for optional GitHub Release downloads."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+');args=parser.parse_args()
out=ROOT/'releases';out.mkdir(exist_ok=True)
for name in args.names:
    folder=ROOT/'models/full/hw10'/('transformer-'+name)
    if not (folder/'model.safetensors').exists():raise FileNotFoundError(folder)
    path=out/(name+'-coursework-checkpoint.zip')
    with ZipFile(path,'w',compression=ZIP_DEFLATED,compresslevel=1) as archive:
        for file in sorted(folder.rglob('*')):
            if file.is_file():archive.write(file,file.relative_to(folder.parent))
    print(path.name,path.stat().st_size,flush=True)
