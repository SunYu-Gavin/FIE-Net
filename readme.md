# FIE-Net

## Project Profile
This item is used for academic papers: FIE-Net: Multimodal Driving Vigilance Estimation Using Enhanced Intra- and Cross-Modal Feature Interaction.

## Functionality
FIE-Net is a network that focuses on enhancing intra- and cross-modal feature interactions by introducing a Modal Feature Interaction Enhancement Module, a Cross-Modal Feature Interaction Enhancement Module for interaction enhancement, and additionally, a Polynomial Dimension Expansion Module to extend one-dimensional features to two dimensions to preserve and extract spatial features in modal data. For more information, please feel free to read our paper for more details.


## Requirements
- Python >= 3.8
- Dependency packages: see `requirements.txt`.

## Install
```bash
pip install -r requirements.txt
```

## Dataset
To enable cross-sectional comparison of models, the SEED-VIG dataset was used. For code completeness, some of the data is provided in the project to be able to realize the training and testing of the code, the complete SEED-VIG dataset is available on their website https://bcmi.sjtu.edu.cn/home/seed/ .

## Run
Run the SEED-FIENet.py file under this project to start training and testing.