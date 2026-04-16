# MFGCN-Pep

## Integrating Multi-modal Language Models with Graph Structural Learning for Bioactive Peptide Prediction

![Graphical Abstract](graphical_abstract.png)

Code and data resources for the paper **Integrating Multi-modal Language Models with Graph Structural Learning for Bioactive Peptide Prediction**.

## Overview

MFGCN-Pep integrates multi-modal protein language model features with graph structural learning for bioactive peptide prediction. The repository currently contains:

- the main feature extraction, training, and inference code
- raw data files used to build benchmark datasets
- a graphical abstract provided with the project

## Repository Structure

```text
MFGCN-Pep/
|-- code/
|   |-- MLLMFeature.py
|   |-- Modeling.py
|   `-- Model_pred.py
|-- Raw Data/
`-- graphical_abstract.png
```

## Main Workflow

The intended main workflow is:

1. Run `code/MLLMFeature.py` to generate sequence-level features, including ProtT5, ESM, AAindex, and attention-derived graph inputs.
2. Run `code/Modeling.py` to train the main cross-validation model.
3. Run `code/Model_pred.py` to load the trained model and perform evaluation or prediction.

## Data Notes

The `Raw Data/` directory contains the collected source datasets. The training scripts expect processed dataset folders under a `data/<DATASET_NAME>/` layout, for example:

```text
data/
`-- AHT/
    |-- labels.csv
    `-- feats/
        |-- aaindex/
        |-- attn/
        |-- esm/
        `-- t5/
```

`Raw Data/AAindex.txt` is included as the AAindex source table used during feature construction.

## Environment

Recommended environment:

- Python 3.10+
- PyTorch
- transformers
- numpy
- pandas
- scikit-learn
- biopython
- tqdm

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Depending on your hardware, you may want to install a CUDA-enabled PyTorch build manually from the official PyTorch installation instructions.

## Reproducibility

The scripts expose several options through environment variables, including:

- `DATASET_NAME`
- `SEED`
- `EPOCHS`
- `LEARNING_RATE`
- `HIDDEN_DIM`
- `N_SPLITS`

Please refer to the script headers in `code/Modeling.py`, `code/MLLMFeature.py`, and `code/Model_pred.py` for the current defaults.

## License

The code in this repository is released under the MIT License. Please review the original sources of the raw datasets before redistribution or reuse of the data files in `Raw Data/`.


## Contact

For questions about the repository, please open an issue on GitHub.
