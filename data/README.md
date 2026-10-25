# Data sources and scope

Supplied exercise files are recorded in `provenance.json`; data and model hashes are in `reports/artifact-manifest.json`.

- `hw03/goodreads.csv`: supplied book-record cleaning exercise.
- `hw05/multishapes.csv`: supplied clustering exercise.
- `hw06/`: supplied historical Boston housing development and test files. The course target is MEDV. This dataset includes a race-derived predictor and is retained as historical coursework material, not as a deployed valuation dataset.
- `hw07/`: supplied Online News Popularity and Algerian Forest Fires exercise files. The unrelated 558 MB `housing.csv` from the lab folder is not required by the submitted tasks and is not redistributed here.
- `hw08/`: the six supplied sine-wave `.npy` arrays. Inputs are 1,000 curve samples; targets are four generating parameters. NumPy reads use `allow_pickle=False`.
- `hw10/alt_syllogisms.csv`: the public exercise dataset downloaded using the file ID in the supplied transformer assignment. It contains 768 text/label records in the downloaded version.
- `hw11/groceries-groceries.csv`: supplied transaction-basket data.
- `hw12/wines.csv`: supplied chemical attributes, quality values and red/white indicator.

The following public datasets are downloaded automatically into the ignored `data/downloads/` cache or Keras's dataset cache:

- Titanic: [seaborn example data](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).
- Cleveland heart exercise: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), `processed.cleveland.data` from the public archive. This restores the standard public exercise data; the Drive copy did not contain the original heart CSV.
- MNIST: official TensorFlow/Keras NPZ hosting.
- CIFAR-100 and Reuters: TensorFlow/Keras dataset loaders.
- BART-base and RoBERTa-base weights: their public Hugging Face model repositories. Actual run records retain the pretrained model commit when available.

The optional `mystery.csv` exercise input was not available. Its saved visualization remains in the archived Homework 4 submission and is not invented or replaced in the runnable implementation.

The data support course exercises. Sampling, label definitions and split choices are documented with each run; no population-level health, commercial or safety conclusion is asserted from these exercises.
