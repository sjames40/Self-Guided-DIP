# Self-Guided-DIP

## Overview
This repository implements the **Self-Guided Deep Image Prior (DIP)** technique, which enables effective unsupervised learning for image restoration tasks. The approach leverages the inherent structure of convolutional neural networks (CNNs) to solve inverse problems in imaging.

## Repo structure
```
📦
├─ models
│  ├─ baseline_pai.py
│  ├─ baseline_pat.py
│  ├─ sgld.py
│  ├─ vanilla_decoder.py
│  └─ vanilla_dip.py
├─ unet
├─ util
│  ├─ baboon
│  ├─ barbara
│  ├─ lena
│  └─ pepper
├─ utils
│  ├─ baboon
│  ├─ barbara
│  ├─ lena
│  └─ pepper
├─ two_channel_data....
└─ self_guided_DIP.ipynb
```
## Features
- **Self-Guided DIP**: Apply self-guidance during DIP to improve restoration quality.
- **Model Implementations**: Includes model architectures like UNet.
- **Image Restoration**: Enhance image quality without supervision.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/sjames40/Self-Guided-DIP.git
```

2 Install the required dependencies:
```bash
conda create --name self-guided-dip
conda activate self-guided-dip
pip install -r requirements.txt
```

3 Usage Download the dataset from Dropbox: Data avaliable on **https://www.dropbox.com/scl/fi/801dxovhbkp2bkl2krz5x/NEW_KSPACE.zip?rlkey=4u3b32f6c4pfujsv3kp7z5bdk&st=hwe9thrv&dl=0**
Open and run the self_guided_DIP.ipynb Jupyter notebook to train and evaluate the model on image restoration tasks.
Directory Structure
models/: Contains model architecture code.
unet/: Implementation of the UNet model.
utils/: Utility functions for the project.
