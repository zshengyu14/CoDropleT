# CoDropleT: AlphaFold2-based Prediction of Protein Co-condensation Propensity

This repository contains the official implementation of **CoDropleT (co-condensation into droplet transformer)**, a deep learning model for predicting the propensity of protein pairs to co-condense. The method leverages structural and evolutionary representations from AlphaFold2 to predict the composition of protein condensates and membraneless organelles (MLOs).

### Citation

If you use CoDropleT in your research, please cite the original publication:

> Zhang, S., Lim, C. M., Occhetta, M., & Vendruscolo, M. (2024). AlphaFold2-based prediction of the co-condensation propensity of proteins. *PNAS*, 121(34), e2315005121. https://doi.org/10.1073/pnas.2315005121

---

## 🚀 Getting Started with Google Colab

The easiest way to use CoDropleT is through our interactive Google Colab notebooks, which provide a user-friendly interface and access to the necessary computational resources (including GPUs) for free.

The workflow is divided into two main steps, each with its own notebook:

### Step 1: Generate Protein Representations

Before you can predict co-condensation, you must first generate the necessary structural and sequence representations for each of your proteins.

* **Notebook:** [**AlphaFold2_representations.ipynb**](https://github.com/zshengyu14/ColabFold_distmats/blob/main/AlphaFold2_representations.ipynb)
* **Input:** Protein sequences.
* **Output:** A `.zip` file for each protein containing its `.npy` (protein representations) files.

You must run this notebook for all the proteins you wish to analyze.

### Step 2: Predict Co-condensation with CoDropleT

Once you have the representation `.zip` files from Step 1, you can use the main CoDropleT notebook to perform the analysis.

* **Notebook:** **`CoDropleT_Colab.ipynb`** (in this repository)
* **Input:** The `.zip` files generated in Step 1.
* **Output:** Co-condensation scores for all protein pairs (including self-pairs) and interactive visualizations of the per-residue profiles.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zshengyu14/CoDropleT/blob/main/CoDropleT_Colab.ipynb)

This interactive notebook allows you to:
- Upload your protein `.zip` files or select them directly from Google Drive.
- Automatically generate all unique protein pairs for analysis.
- Visualize the 3D structures and 2D co-condensation profiles for any pair.
- Download all results in a single package.

---

## 🖥️ Local Installation and Usage (Advanced)

For users who wish to run the pipeline locally, this repository contains all the necessary Python scripts.

### Requirements
- A Python environment (e.g., Conda).
- JAX, Haiku, and TensorFlow.
- A local installation of AlphaFold2 (specifically, a version that saves representations, such as [this fork](https://github.com/zshengyu14/alphafold)) and its associated databases.

### Run Inference

Execute the `run_model.py` script from your terminal:
```bash
python CoDropleT/run_model.py \
  --test_csv path/to/your/input.csv \
  --model_ckpt CoDropleT/params/params.pkl \
  --results_dir ./results
