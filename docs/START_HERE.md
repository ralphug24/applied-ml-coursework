# Start here: a first model, explained

No course knowledge is required. Start with one small regression example before the larger neural-network and transformer assignments.

1. **Read the worked example:** [view the walkthrough with saved outputs](https://ralphugboko.dev/assets/coursework/applied-ml/start-here.html). No setup or account is needed to read it.
2. **Run the same example:** [open in Google Colab](https://colab.research.google.com/github/ralphug24/applied-ml-coursework/blob/main/notebooks/START_HERE.ipynb). Sign in if requested, connect a runtime, then choose **Runtime → Run all**. Review the notebook before confirming execution. It uses a CPU, standard Python packages and two small public data files; no API key is needed.
3. **Explore further:** return to the [case study](https://ralphugboko.dev/projects?discipline=school-projects&project=applied-ml-coursework), then open the full regression or neural-network notebook.

## What happens in the walkthrough?

- Read 450 labeled neighborhood observations. Thirteen measurements are inputs; a historical median housing value is the known answer.
- Compare a mean baseline, linear regression and a 100-tree Random Forest using the same ten validation folds.
- Select the lowest mean validation squared error, then train that model on all 450 examples.
- Open a separate 56-row test file only after model selection. Predict its answers, inspect individual errors, and plot predictions against known values.
- Interpret MSE, RMSE, MAE and R². The saved run selected Random Forest: test MSE approximately **12.55**, RMSE **3.54**, MAE **2.59**, R² **0.313**. These are error measures, not accuracy percentages.

The example uses the historical dataset supplied for the assignment. Its target is in historical thousands of dollars. It includes problematic demographic features and is not suitable as a current valuation system. The three-model walkthrough is a smaller teaching path through Homework 6; the full implementation compares six candidates.

## Run locally instead

Install **Python 3.12** and download this repository from GitHub using **Code → Download ZIP**, then extract it. Open a terminal in the extracted folder: it should contain `coursework.py` and `requirements.txt`.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install notebook
.venv\Scripts\python.exe -m notebook notebooks/START_HERE.ipynb
```

On macOS/Linux with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install notebook
.venv/bin/python -m notebook notebooks/START_HERE.ipynb
```

In the notebook, select the virtual environment's Python kernel if prompted, then choose **Run → Run All Cells**. Wait for the final chart. Package installation can take longer than the small training example. The notebook uses local data when available; Colab downloads the same files. The saved outputs show what a successful run looks like.

To run the **full six-model regression task** without a notebook, from the same folder:

```powershell
.venv\Scripts\python.exe coursework.py hw06
```

It saves metrics and predictions under `reports/runs/full/hw06/` and the fitted model under `models/full/hw06/`. To load that saved model and make one prediction:

```powershell
.venv\Scripts\python.exe predict.py hw06 --row 0
```

The base setup is enough for this example. TensorFlow and transformer packages are optional later steps in the main README. You do not need to install everything to begin.

## A short glossary

| Term | Meaning in this collection |
| --- | --- |
| Feature/input | A measurement supplied to the model, such as average rooms. |
| Target/label | The known answer used for learning and checking predictions. |
| Model | A learned rule that maps inputs to predictions. |
| Training | Adjusting internal settings using examples and an objective. |
| Loss/objective | A quantity training tries to reduce, such as squared error. |
| Metric | A measure used to judge predictions. It may be the same quantity as the loss, or a different one. |
| Baseline | A simple reference, such as always predicting the training average or the most common class. |
| Validation | Held-out examples used to choose a model or settings. |
| Cross-validation | Repeating training and validation across several divisions of the development data. |
| Test set | Examples reserved for a final evaluation after choices are made. Repeatedly tuning against them weakens that final check. |
| MSE / RMSE / MAE | Squared error, its square root, and absolute error, averaged across examples. Lower is better. |
| R² | Squared-error performance relative to predicting the evaluation set's mean. It can be negative and is not percent accuracy. |
| Classification | Predicting a category rather than a continuous number. |
| Precision / recall | Of the positive predictions, how many were right / of the actual positives, how many were found. |
| Macro F1 | Calculate F1 (a balance of precision and recall) for each class, then average equally across classes. |
| Epoch / batch | One pass through training examples / a group processed together during a training step. |
| Neural-network weight | An internal learned setting. In the waveform exercise, these are distinct from the **four curve parameters** the network predicts. |
| ReLU / Adam | A neural layer's rule that keeps positive values and replaces negative values with zero / an algorithm that updates network weights to reduce loss. |
| Fine-tuning | Continuing training from an existing pretrained model on a task's examples. |

## If something goes wrong

- **Cannot download the data:** check internet access and retry. In Colab, reconnect the runtime. Locally, keep the extracted `data/hw06/` folder in the repository.
- **Module not found:** use the virtual environment's Python in the commands above. If a notebook still reports this, select that environment's kernel and restart it.
- **File not found:** open the terminal in the repository folder, not inside `notebooks/` or the ZIP archive.
- **PowerShell blocks activation:** the commands above call the environment directly; activation is unnecessary.
- **Scores differ slightly:** check the printed library versions and fixed seed. The saved notebook records a successful run; the full repository pins the versions used for the documented runs.
- **A large neural run is slow:** return to this small example first. Later use the documented `--profile preview` command for an execution check; preview scores are not full model-quality benchmarks.

The [methods guide](methodology.md) explains the full collection's choices and limitations. The [assignment map](assignment-map.md) connects each submission to its topic and runnable task.
