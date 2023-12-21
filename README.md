# FlarePrediction
Due to the large size of the model weight files in the weight folder, we have provided a Baidu Netdisk link for access. The link is as follows:https://pan.baidu.com/s/1k6sK1njopeD7rP7zdJpRLw?pwd=fn36 
Additionally, we have shared a more comprehensive project link on Baidu Netdisk, which includes all the code and datasets mentioned in this paper. The link is as follows:https://pan.baidu.com/s/1X7lYKW0bQvg9byAWFpuEqg?pwd=uxis

This repository contains the source code for predicting solar flares. The codebase is organized into several folders, each serving a specific purpose:

common/: This directory contains reusable code components that are utilized across the project. It includes utility functions and common definitions that are central to the operation of the code.

config/: The configuration files for the project are located here. These files contain essential settings and parameters that control various aspects of the code execution, ensuring consistency and ease of configuration management.

layer/: This folder is dedicated to the implementation of attention mechanisms. It contains code that outlines how the attention layers within the neural network models are structured and function.

main/: The main Python scripts of the project reside in this directory. These are the primary entry points for running the software and contain the core logic for solar flare prediction.

model/: In this directory, you will find Python files that define various neural network models. Each file represents a different model architecture used for predicting solar flares.

pic_draw/: This folder contains the code for drawing ROC (Receiver Operating Characteristic) curves. These scripts are used for visualizing and evaluating the performance of the predictive models.

util/: The util directory houses various utility functions. These Python files contain supplementary code that supports the main functionality, such as data processing, logging, and other helper functions.

train_feature_importent folder/: Contains the training code for the related models.

train folder/: Includes the training code for models used in the feature importance analysis.

outputs folder/: Stores the results of various models and related experiments.

detect folder/: Contains code related to model prediction.

detect_feture_importence/ and feature_importance_cut/ folders: Hold code related to experiments on feature importance.

data/: Includes data used in our paper
