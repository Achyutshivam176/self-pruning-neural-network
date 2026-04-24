# Self-Pruning Neural Network

This project implements a neural network that learns to prune itself during training.

## Key Idea
Each weight is paired with a learnable gate parameter (0–1). During training:
- Gates are computed using sigmoid
- Effective weight = weight × gate
- L1 regularization pushes gates toward zero

## Features
- Custom PrunableLinear layer
- Sparsity loss (L1 on gates)
- Lambda-based sparsity comparison
- Gate distribution visualization

## How to Run
pip install -r requirements.txt  
python main.py

## Note
This is a conceptual implementation focusing on architecture and sparsity mechanism.
