# Learning, training and evaluation

## My role and what I learned

These guided Applied Data Science assignments gave me practice preparing data, fitting models and interpreting their output. The collection demonstrates the progression of those skills; the portable code makes the process inspectable and executable.

The common workflow is: clarify the input and target; inspect data; create suitable partitions; fit preprocessing on training data; establish a baseline; train candidates; compare them using development data; evaluate on a separate test split; inspect errors; and document the next step.

## 1. Distinguish the objective from the evaluation metric

A loss is what a learning algorithm optimizes. An evaluation metric tells us how well its predictions serve a specified purpose. They may differ.

- **Regression:** squared error penalizes large residuals. MSE is in squared target units; RMSE returns to the target's units; MAE summarizes absolute error; R-squared relates residual error to the variance of the evaluated target.
- **Logistic classification:** log loss rewards probability assigned to the observed class. Accuracy counts correct labels. Macro F1 gives each class equal weight and helps reveal poor minority-class performance.
- **Linear SVC:** squared hinge loss encourages a separating margin. A margin score is not automatically a calibrated probability.
- **Neural waveform regression:** MAE averages absolute errors across the four generating parameters. It is appropriate for this continuous-output exercise, but does not mean the model has learned a general Fourier transform.
- **CNN/LSTM and transformer classification:** cross-entropy aligns the output distribution with the class labels. Sparse categorical cross-entropy uses integer labels; ordinary categorical cross-entropy uses equivalent one-hot labels.
- **Clustering and association mining:** these have their own structure criteria. Silhouette, support, confidence and lift are not substitutes for supervised predictive accuracy.

## 2. Keep model selection separate from final evaluation

The regression workflow compares all candidates on the same shuffled ten-fold partition of the 450 development observations. It selects the lowest mean MSE, refits on those development data, then loads and evaluates the supplied 56-row external test file. A preliminary 80/20 development split illustrates subset versus all-feature regression.

The archived notebook's earlier linear-regression cross-validation used unshuffled folds; the later comparison used shuffled folds. That change affects the estimate, so differences between those stages cannot be attributed entirely to a better model. The portable comparison keeps the split policy fixed across candidates.

R-squared also depends on the variation in the evaluation target: `R² = 1 - SSE/SST`. A relatively narrow test-target distribution can produce a lower R-squared even when MSE is close to the CV MSE. Report the test set and both metrics together instead of calling R-squared a percentage of predictions that were correct.

The fire and wine classifiers use a held-out final split and training-only five-fold selection. Scalers and dimensionality reducers are inside the cross-validation pipeline. The fire split is a random classification exercise on recorded conditions; temporal and geographic generalization would need different split designs.

The portable CIFAR workflow holds validation data out of the training set instead of repeatedly viewing the final test set during fitting. The transformer workflow uses development/validation/test partitions, removes exact repeated wording before splitting and avoids selecting checkpoints using test labels. It does not claim independence across logical forms: related syllogism templates can remain in different partitions.

## 3. Model training I can explain: waveform parameter regression

**Input:** 1,000 equally spaced samples of a curve over 0 to 10π.

**Desired output:** four parameters in `a*sin(b*x) + c*cos(d*x)`.

**Architecture in the submission:** dense layers of 128, 64 and 32 ReLU units, followed by four linear outputs. The first dense layer has `1000*128 + 128 = 128128` parameters. The archived written answer's different count was an error; the model definition and portable code establish the correct calculation.

**Loss and optimizer:** mean absolute error and Adam, as specified by the exercise. A linear output supports continuous parameter values. In the full portable run, 8,000 rows train the network and 2,000 rows provide validation for 50 epochs at batch size 32. The ordinary test set has 2,000 rows; the extended diagnostic set has nine.

**Evaluation:** parameter MAE, error for each output, training/validation curves and waveforms implied by predicted parameters. Reconstructing a curve from predicted parameters helps inspect the prediction, but is not equivalent to parameter accuracy. Some parameter combinations can represent similar curves, and low amplitudes can make frequencies difficult to identify.

**Learning:** acceptable error within the training range does not establish extrapolation outside it. The original saved losses differ sharply between ordinary test and extended examples. This is a useful example of examining the data distribution in addition to a headline metric.

## 4. Why an impressive accuracy number may be insufficient

The archived transformer outputs both report about 90.9% accuracy and 0.476 macro F1. This pair is consistent with a majority-only classifier on a heavily imbalanced binary test split. Without saved predictions, it does not prove that is exactly what happened; it does show why accuracy alone cannot establish learned logical reasoning.

The portable benchmark explicitly includes a majority baseline, macro F1 and a confusion matrix. It records which class is missed and the validity prevalence. A stronger future study would group splits by logical form and assess sensitivity across seeds and prompts. Fine-tuning a classifier is not RLHF, reward modeling or a demonstration of general reasoning capability.

## 5. Preprocessing and representation learning

Homework 7's original chi-square exercise treated raw share counts as class labels. The portable news selector uses the regression-appropriate F statistic and only training rows. This remains a simple univariate association ranking; it does not establish causality or automatically improve a predictive model.

PCA retains variance in the standardized predictors. It does not seek directions that best separate quality categories. In the wine task, a visually strong separation of red versus white wines is not necessarily helpful for the actual quality target. The classifier and its held-out class-level errors determine predictive usefulness.

For clustering, scaling affects distance-based algorithms. K-means favors compact groups; Ward merges clusters by its variance criterion; DBSCAN identifies dense connected regions and marks noise. Compare criteria suited to the question instead of treating the numeric cluster label as a meaningful category by itself.

## 6. Association rules need the correct denominator

For a rule A → B:

- support = P(A and B);
- confidence = P(B given A);
- lift = P(B given A) / P(B).

The denominator of lift is the consequent's marginal support. The portable implementation uses this definition and applies Apriori subset pruning. It retains the exercise's selected 28 products, support threshold 0.0001 and maximum itemset size of three. Very rare baskets can produce high confidence or lift from few observations, so inspect support before making a recommendation. The recorded rules do not measure business impact.

## 7. What changed in the portable package

| Area | Portable change | Reason |
| --- | --- | --- |
| Colab setup | Repository-relative paths and explicit public-data cache | Run without mounting a personal Drive |
| Formatting cells | External classroom CSS is not fetched during training | Keep execution focused on the task |
| Deprecated imports/readers | Supported APIs and whitespace CSV parsing | Make the environment explicit |
| Randomness | Recorded seeds and bounded CPU threads | Make configurations inspectable |
| Selection | Training-only CV and validation partitions | Keep final test data out of fitting and selection |
| News selector | F regression for share counts | Match the criterion to a regression target |
| CNN labels | Integer labels with sparse cross-entropy | Equivalent loss without unnecessary one-hot arrays |
| MNIST helper | Official NPZ reader, bounded seeded subset | Avoid cloning a third-party executable helper and unbounded pairwise silhouette work |
| Mining | Small transparent Apriori implementation | Keep support/confidence/lift calculations inspectable |
| Transformer input | Correct DataFrame variable, validated labels and exact-text deduplication | Remove stale notebook state and document split assumptions |
| Artifacts | Metrics, predictions, plots, fitted models and hashes | Connect a claim to an actual run |

The original submissions are kept separately. Their written explanations include unfinished answers and some incorrect interpretations; this guide explains the corrected reasoning without changing the archived records.

## 8. Running and interpreting profiles

The `full` waveform configuration uses the full supplied training set and 50 epochs. Full CNN/LSTM runs use five epochs. Full transformer runs use three epochs. Preview CNN/LSTM runs use smaller training and evaluation subsets and one epoch; transformer preview runs use two optimization steps per model. These preview results are execution demonstrations, not competitive performance claims.

Run files record actual execution date, input scope, configuration, dependency versions and code checksum. The data were supplied for coursework or downloaded from the documented public source. The package does not establish production readiness, causal user outcomes or performance on new domains.

## References for the implementation

- [TensorFlow installation](https://www.tensorflow.org/install/pip)
- [Keras dense layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Course notebooks and named submissions recorded in `provenance.json`.
