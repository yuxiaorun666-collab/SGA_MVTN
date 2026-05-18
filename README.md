# SGA_MVTN

This repository provides the code implementation for the paper "A Neuroscientific Prior-Guided Multi-View Network With Structured Channel Selection for EEG Emotion Recognition".

## Brief Introduction

This project proposes SGA-MVTN, a neuroscientific prior-guided two-stage framework for electroencephalography (EEG)-based emotion recognition. The framework integrates structured neurophysiological priors into representation learning and channel selection. It consists of two core modules:

* **Multi-View Tower Network (MVTN)**: Explicitly decouples EEG differential entropy features into anatomical spatial views and frequency-band-specific spectral views, enabling physiologically grounded learning of region-wise and band-wise emotional representations.


* **Structured Genetic Algorithm (SGA)**: Embeds functional brain region grouping and contralateral hemispheric pairing into evolutionary operators, guiding the search toward compact and topologically coherent electrode subsets for topology-aware channel selection.




<img width="1934" height="958" alt="system" src="https://github.com/user-attachments/assets/a8809cfb-0b30-42ee-83f0-bdbb74861648" />



## Code Structure Overview

The main files and directories included in this repository are:

* **`seed_MVTN_model.py`**: Contains the model definition of the Multi-View Tower Network (MVTN) for spatial and spectral feature decoupling and emotion classification.
* **`SGA_model.py`**: Contains the logic implementation of the Structured Genetic Algorithm (SGA) for topology-constrained EEG channel search and optimization.
* **`seed_MVTN_modelparameter/`**: A directory for storing weight parameters or pre-trained configurations related to the MVTN model.

## Data
* **`seed`**: https://bcmi.sjtu.edu.cn/home/seed/index.html
* **`seed-iv`**: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
