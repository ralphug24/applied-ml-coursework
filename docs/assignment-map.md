# Assignment map and evidence

This collection follows the folder and submission numbering in the shared CPSC 6300 Drive. The submission notebooks are retained under `original/submissions/hwXX.ipynb`. The portable workflow is in `coursework.py`; corresponding runnable notebooks are in `notebooks/`.

| Assignment | Inputs | Desired output or analysis | Training objective / criterion | Evaluation |
| --- | --- | --- | --- | --- |
| 3 | Goodreads records | Cleaned and grouped records | No supervised training | Types, missingness, row counts and descriptive plots |
| 4 | Titanic and Cleveland exercise records | Descriptive distributions and relationships | No supervised training | Sample counts, plots and interpretation of sample scope |
| 5 | Scaled 2D shapes and MNIST pixels | Cluster assignments | K-means within-cluster squared distance; Ward merge criterion; DBSCAN density connectivity | Inertia, silhouette, noise count and visual structure |
| 6 | Supplied historical housing attributes | Continuous MEDV target | Squared error; Ridge/Lasso add L2/L1 penalties | Shared 10-fold CV, MSE, RMSE, MAE and R-squared |
| 7 | News features and recorded forest-fire conditions | Feature ranking; fire/not-fire label | Regression-appropriate F score; log loss / squared hinge / penalized squared loss | Training-only CV macro F1; held-out precision, recall and confusion matrix |
| 8 | 1,000 sampled waveform values | Four parameters a, b, c, d | Mean absolute error, Adam | Validation curve, parameter MAE, per-parameter error and nine extended cases |
| 9 | CIFAR-100 RGB images; Reuters word-index sequences | 100 image classes; 46 news categories | Categorical cross-entropy, Adam | Held-out accuracy, macro F1 and class-level results |
| 10 | Syllogism text | Binary validity label | Cross-entropy, AdamW fine-tuning of BART-base and RoBERTa-base | Majority baseline, accuracy, macro F1 and confusion matrices |
| 11 | Grocery transaction baskets | Frequent itemsets and association rules | Support counting and Apriori pruning | Support, confidence, lift and low-support stability considerations |
| 12 | Wine chemical attributes and red/white indicator | Quality categories: <=5, 6–7, >=8 | PCA variance retention / SVD decomposition; logistic classification loss | Training-only CV, held-out accuracy, macro F1 and per-class errors |

## Portable versions

The runnable code preserves the tasks while documenting changes needed to make execution and interpretation clearer. It uses local data paths, current APIs, seeded randomness, saved artifacts and explicit run profiles. It does not silently overwrite the original notebook's method or result.

## Archived numerical examples

- Homework 6: Random Forest's original held-out MSE is 12.54685 and R-squared is 0.31326. The archived shuffled-fold CV values are about 11.67 MSE and 0.868 mean R-squared. MSE and R-squared describe different aspects of performance and can change differently across distributions.
- Homework 8: the original output reports 0.0272 parameter MAE on 2,000 ordinary test examples and 0.5103 on nine extended examples. Those nine examples are a diagnostic stress case, not a population performance estimate.
- Homework 9: the original output reports 0.3715 CIFAR-100 accuracy and 0.6060 Reuters accuracy. The CIFAR test set was also supplied as validation during fitting, so the portable version introduces a separate validation split.
- Homework 10: both archived model outputs show accuracy 0.90909 and macro F1 0.47619. Those values alone do not establish superiority over a majority baseline. RoBERTa's checkpoint selection used the same split later reported as test, which is corrected in the portable split plan.
- Homework 12: the original 10-component PCA and SVD examples each report about 0.71154 test accuracy. This is a descriptive archived comparison, not evidence that reduction must improve a classifier.

The repository's actual run JSON files are the authoritative source for the portable versions' results, dates, profiles and software environment.
