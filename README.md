# Applied Machine Learning: Training and Evaluation

**Raphael Ugboko · Clemson University · CPSC 4300/6300 Applied Data Science · Fall 2025**

A connected collection of ten course assignments, from data preparation and clustering to regression, neural networks and transformer fine-tuning. The case study explains the inputs, targets, training objectives and evaluation choices behind the work.

**[Portfolio case study](https://ralphugboko.dev/projects?discipline=school-projects&project=applied-ml-coursework)** · **[Learning and methodology guide](docs/methodology.md)** · **[Assignment map](docs/assignment-map.md)**

## What I learned

I practiced preparing data, fitting and comparing models, choosing an objective for a task, and interpreting results beyond a single score. The coursework included guided exercises and supplied starter code. My submissions document my completed exercises and interpretations; the portable code in this repository makes the workflows easier to run and inspect.

## Run the code

Use Python 3.12 and a virtual environment. Commands run from the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python coursework.py hw06
python coursework.py hw07
python coursework.py hw12
python predict.py hw06 --row 0
```

For neural networks:

```bash
python -m pip install -r requirements-neural.txt
python coursework.py hw08
python predict.py hw08 --row 0
python coursework.py hw09 --profile preview
# Course-sized training (more compute):
python coursework.py hw09 --profile full
```

For pretrained transformer fine-tuning:

```bash
python -m pip install -r requirements-transformers.txt
python coursework.py hw10 --profile preview
python coursework.py hw10 --profile full
```

The transformer run downloads public pretrained weights and the exercise CSV. A GPU is useful for full fine-tuning, but the code also supports CPU. Downloads are cached; no account token is embedded in the code. The two-step preview only demonstrates execution and must not be interpreted as a model-quality benchmark. Transformer checkpoints are saved locally by the run; they are excluded from Git because of their size. The code, original saved output and run metrics remain inspectable in the repository.

The same tasks can be opened in `notebooks/`. Each notebook imports the documented task implementation and shows its saved report and figures after execution. `python coursework.py classical` runs the non-neural collection, including public dataset downloads. `all` includes every task and can require substantial time and disk space.

## Assignments

| ID | Topic | Main learning | Command |
| --- | --- | --- | --- |
| hw03 | Goodreads data preparation | Missing values, types, parsing and grouping | `python coursework.py hw03` |
| hw04 | Exploratory analysis | Distributions, sample composition and relationships | `python coursework.py hw04` |
| hw05 | Clustering | K-means, Ward, DBSCAN, inertia and silhouette | `python coursework.py hw05 --profile preview` |
| hw06 | Regression | Baselines, model comparison, 10-fold CV and final testing | `python coursework.py hw06` |
| hw07 | Feature and model selection | Training-only selection and classification metrics | `python coursework.py hw07` |
| hw08 | Dense neural regression | A 1,000-sample waveform mapped to four parameters | `python coursework.py hw08` |
| hw09 | CNN and LSTM | Image classification and sequence classification | `python coursework.py hw09 --profile preview` |
| hw10 | BART and RoBERTa | Supervised fine-tuning, imbalance and macro F1 | `python coursework.py hw10 --profile preview` |
| hw11 | Association mining | Frequent itemsets, support, confidence and lift | `python coursework.py hw11` |
| hw12 | PCA and SVD | Train-fitted transformations and predictive evaluation | `python coursework.py hw12` |

## Results and files

- `original/submissions/`: the ten named coursework submissions, with their original saved output. `provenance.json` records source filenames and checksums.
- `coursework.py`: the portable training and analysis implementation.
- `notebooks/`: task notebooks with explanations and run commands.
- `reports/original/`: original notebook exports and saved text output.
- `reports/runs/`: actual run metrics, predictions and figures, separated by `full` or `preview` profile.
- `models/`: saved fitted pipelines and neural models from recorded runs.
- `data/`: the supplied exercise data needed by the portable tasks, plus a cache directory for public downloads.
- `docs/`: objectives, evaluation rationale, changes and limitations.

### Read the run profile before comparing numbers

`full` uses the documented full configuration for a task. `preview` limits training rows, epochs or optimization steps for selected expensive tasks. Both profiles record their actual scope. The clustering search bounds the MNIST analysis to 5,000 rows in `full` and 1,000 in `preview`, with sampled silhouette calculations; it does not repeat the original unbounded full-MNIST search.

Coursework outputs and new run records are different evidence sources. Portable versions update paths and APIs, make validation boundaries explicit, and correct documented methodological issues. They are not expected to reproduce every archived number exactly. See [methodology](docs/methodology.md) and the actual metrics JSON files for configurations and execution dates.

The optional `mystery.csv` visualization in Homework 4 is retained in the original submission only because its input file was absent from the shared folder. The ML workflows do not depend on it.

## Attribution and scope

The work was completed in Carlos Toxtli's Applied Data Science course. Supplied notebooks contain teaching material credited to their original course authors, including Brandon Walker and Chris Kalahiki. Some starter notebooks retain earlier-year headers; the collection here documents Raphael's Fall 2025 coursework. I do not claim authorship of the starter curriculum, datasets or pretrained models.

The housing regression is a historical classroom exercise. The original dataset includes a race-derived feature; the exercise is not offered as a deployable valuation system.

The studies are coursework demonstrations. Fire classification is not an operational forecast; association rules do not establish causation; and benchmark accuracy does not by itself establish a useful deployed system. No blanket license is asserted over third-party teaching material or datasets.
