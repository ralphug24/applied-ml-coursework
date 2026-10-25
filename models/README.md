# Saved models

The folders separate `full` training configurations from `preview` execution demonstrations. The corresponding configuration and evaluation are in `reports/runs/<profile>/<task>/metrics.json`.

| Artifact | Load with | Purpose |
| --- | --- | --- |
| `full/hw05/shapes-kmeans.joblib` | `joblib.load` | K-means centroids for the standardized shapes exercise |
| `full/hw06/regression.joblib` | `joblib.load` | Selected housing regression model |
| `full/hw07/fire-classifier.joblib` | `joblib.load` | Selected classification pipeline, including scaling |
| `full/hw08/sinewave.keras` | `tf.keras.models.load_model` | Four-parameter waveform regression |
| `full/hw09/cifar100-cnn.keras` | `tf.keras.models.load_model` | Five-epoch, 100-class image classifier |
| `full/hw09/reuters-lstm.keras` | `tf.keras.models.load_model` | Five-epoch, 46-category sequence classifier |
| `full/hw12/wine-pipeline.joblib` | `joblib.load` | Selected wine-quality pipeline |

Use `python predict.py hw06 --row 0` or `python predict.py hw08 --row 0` from the repository root to inspect a saved example without retraining. See the root README for the dependency environment.

## Transformer checkpoints

Full BART-base and RoBERTa-base checkpoints are distributed as optional ZIP assets in the repository's **[model release](https://github.com/ralphug24/applied-ml-coursework/releases/tag/v1.0.0)** because each checkpoint exceeds GitHub's individual Git-file limit. Their code and metrics are in the normal repository.

After downloading and extracting one ZIP, pass its extracted directory to the Hugging Face loaders:

```python
import os
os.environ["USE_TF"] = "0"  # These transformer checkpoints use PyTorch.
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "transformer-bart-base"  # or transformer-roberta-base
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
batch = tokenizer("All A are B. All B are C. Therefore all A are C.",
                  return_tensors="pt", truncation=True, max_length=256)
model.eval()
import torch
with torch.no_grad():
    predicted_id = model(**batch).logits.argmax(-1).item()
print({0: "invalid", 1: "valid"}[predicted_id])
```

An example prediction demonstrates loading and inference. It does not establish reasoning capability. Compare the full run's class-level performance with the majority baseline before interpreting a checkpoint's usefulness.

The CNN expects 32×32 RGB inputs normalized to [0, 1]. The Reuters LSTM expects the dataset loader's word indices, a 10,000-token vocabulary and sequences padded to length 200. The waveform model expects the original 1,000 sample values. Use the same preprocessing as `coursework.py` when loading a saved artifact.

Model hashes and byte sizes for the Git-hosted artifacts are recorded in `reports/artifact-manifest.json`. Release notes record hashes for the larger transformer ZIP files.
