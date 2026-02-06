# Analysis of Deep Image Prior and Exploiting Self-Guidance for Image Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2402.04097-b31b1b.svg)](https://arxiv.org/abs/2402.04097)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

> **Official PyTorch Implementation** for the paper "Analysis of Deep Image Prior and Exploiting Self-Guidance for Image Reconstruction" (IEEE TCI 2025).

## 📖 Overview

This repository implements the **Self-Guided Deep Image Prior (DIP)** technique. While standard DIP leverages the inductive bias of Convolutional Neural Networks (CNNs) for unsupervised image restoration, it often requires careful early stopping to avoid overfitting to noise.

Our approach introduces a **Self-Guidance** mechanism that utilizes the model's own predictions during training to regularize the optimization process. This enables robust image restoration (including denoising, inpainting, and MRI reconstruction) without the need for ground-truth supervision.

## ✨ Features

- **Self-Guided DIP**: A novel regularization strategy to prevent overfitting in DIP.
- **Versatile Architectures**: Implementations of various backbones including standard UNet and custom Deep/Shallow variants.
- **Unsupervised Learning**: Perform high-quality image restoration without external training datasets.

## 📂 Repository Structure

```text
📦 Self-Guided-DIP
 ┣ 📂 models             # Core model architectures (Baseline & Proposed)
 ┣ 📂 unet               # UNet implementation details
 ┣ 📂 utils              # Utility functions (Dataloaders, Metrics, Visualization)
 ┣ 📂 data               # (Place your dataset files here)
 ┣ 📜 self_guided_DIP_demo.ipynb  # Main entry point: Jupyter Notebook for training/testing
 ┣ 📜 two_channel_dataset_DIP_github_demo.ipynb  # Handles MRI k-space data loading, VDRS mask generation, and data preprocessing.
 ┗ 📜 requirements.txt   # Python dependencies
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sjames40/Self-Guided-DIP.git
cd Self-Guided-DIP
```

### 2. Environment Setup
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

```bash
# Create a new conda environment
conda create --name self-guided-dip python=3.9
conda activate self-guided-dip

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation
To reproduce the results, please download the specific k-space datasets used in our experiments.

1.  **Dataset**:  
    - [**fastMRI Dataset website**](https://fastmri.med.nyu.edu/) (Download will take some time)
    - [**Stanford 2D FSE website**](http://mridata.org/list?project=Stanford%202D%20FSE) (or download our copy via [**Google Drive**](https://drive.google.com/drive/folders/1CEI3SH2Amw1wRlQJygeWj1r1TI4jAOjm?usp=sharing))

3.  **Setup**:  
    We **recommend** downloading the **fastMRI dataset** first, as it is the primary dataset used to generate the results in `self_guided_DIP_demo.ipynb`. 
    
    **Download Instructions:**
    * **For fastMRI**: Please visit the official website to obtain the license/agreement and then download the data.
    * **For Stanford 2D FSE**: The **full dataset** is available on the official website. We also provide a **partial dataset** (subset) via [**Google Drive**](https://drive.google.com/drive/folders/1CEI3SH2Amw1wRlQJygeWj1r1TI4jAOjm?usp=sharing) for quick testing.
    
    Once downloaded, unzip the files and place them into the project directory (e.g., inside a folder named `data` or as specified in the notebook).

## 🏃 Usage

The core logic and experiments are contained in the Jupyter Notebook.

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open **`self_guided_DIP_demo.ipynb`**.
3.  Run the cells sequentially to initialize the model, load data, and perform restoration.

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{DBLP:journals/tci/LiangBQWR25,
  author    = {Shijun Liang and
               Evan Bell and
               Qing Qu and
               Rongrong Wang and
               Saiprasad Ravishankar},
  title     = {Analysis of Deep Image Prior and Exploiting Self-Guidance for Image Reconstruction},
  journal   = {{IEEE} Trans. Computational Imaging},
  volume    = {11},
  pages     = {435--451},
  year      = {2025},
  url       = {https://doi.org/10.1109/TCI.2025.3540706},
  doi       = {10.1109/TCI.2025.3540706}
}
```

## 📬 Correspondence

For questions regarding the paper or code, please contact:

* **Shijun Liang**: `liangs16@msu.edu`
* **Haijie Yuan**: `yuanhai1@msu.edu`
* **Prof. Saiprasad Ravishankar**: `ravisha3@msu.edu`
