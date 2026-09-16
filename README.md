# Pediatric Pneumonia Dual-Modal Classification

This repository contains the core training and evaluation code accompanying the
manuscript **Dual-Modal Respiratory Signal-Based Auxiliary Diagnostic Method for
Pediatric Pneumonia**. It implements the proposed hierarchical CA-LMF multimodal
fusion framework, which combines regularized canonical correlation analysis
(rCCA)-based cross-modal attention (CA) with low-rank multimodal fusion (LMF),
together with the weighted SMOTE-Voting-Stacking (SVS) ensemble classifier.

## Data

No participant-level pediatric research data are distributed with this public
repository. This includes feature values, labels, sample identifiers, original
filenames, and fold assignments. The expected four-file schema is documented in
[`data/README.md`](data/README.md).

The informed-consent documents specify that participants' personal information
must be kept confidential in a non-public database. The approved consent and
institutional ethical review do not provide for unrestricted public dissemination
of individual-level pediatric research data. These materials may be made
available from the corresponding author upon reasonable request and subject to
institutional and ethics committee approval.

Authorized users with approved data access must provide local data files that
follow the documented schema before running the experiment. The configured
frequency-domain column ranges used by the CA module are declared in
`configs/experiment_config.yaml`.

## Installation

Python 3.9 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows Command Prompt, activate the environment with:

```text
.venv\Scripts\activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproduce the main experiment

After supplying authorized local data, run the fixed-fold out-of-fold evaluation
from the repository root:

```bash
python src/evaluation.py --config configs/experiment_config.yaml
```

The command fits every preprocessing and fusion component using the current
training fold only. SMOTE is also restricted to that training fold. It writes
OOF predictions, per-fold metrics, and a five-fold summary to the local
`outputs/` directory. Outputs may contain participant-level identifiers, labels,
predictions, or derived values, so that directory is ignored by Git and must not
be published.

To reproduce the modality-level held-out SHAP analysis:

```bash
python src/oof_shap.py --config configs/experiment_config.yaml
```

Each sample is explained only by a model trained without that sample. SHAP files
and figures are also written to `outputs/`.

## Repository layout

```text
pediatric-pneumonia-dual-modal/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- CITATION.cff
|-- .gitignore
|-- data/
|   `-- README.md
|-- src/
|   |-- feature_fusion.py
|   |-- svs_classifier.py
|   |-- evaluation.py
|   `-- oof_shap.py
`-- configs/
    `-- experiment_config.yaml
```

## Reproducibility notes

- Feature matrices, labels, and outer-fold assignments must be supplied locally
  by authorized users; they are not included in the public repository.
- The outer folds are read from the local `data/fold_indices.csv`; they are not
  regenerated.
- PCA, scaling, regularized CCA, and SMOTE are fitted independently inside each
  fold.
- Random seeds and model hyperparameters are fixed in the code and configuration.
- `outputs/` is created automatically and is not part of the source release.

## License and citation

The code is released under the MIT License. Citation metadata and the complete
author list are provided in `CITATION.cff`.