# B3-classifier

This repository contains tools for building and evaluating a deep learning model for BBB (Blood-Brain Barrier) permeability classification using a graph neural network. The model is implemented using `PyTorch` and `PyTorch Geometric` libraries.

## Usage

1) Load the Dataset: The dataset classification_extended_test.tsv is loaded and displayed. 
2) Data Preprocessing: The dataset is preprocessed, converting SMILES strings into molecular graphs.
3) Model Definition: A custom graph neural network is defined using PyTorch and PyTorch Geometric.
4) Training and Evaluation: The model is trained on the preprocessed data and evaluated using metrics like ROC AUC and F1 score.

## Key Components
1) B3DBDataset: A custom dataset class for processing SMILES strings and creating graph representations.
2) GNN Model: A graph neural network model implemented with multiple convolutional layers, attention mechanisms, and global pooling.
3) Training Loop: The training loop includes forward propagation, loss computation, backpropagation, and evaluation.

# Results

The notebook includes visualizations of training progress and evaluation metrics to help you understand the model's performance.


