#!/usr/bin/env python3

import torch
import torch.nn as nn


class DemographicMLP(nn.Module):
    """Multi-layer perceptron for demographic prediction from audio embeddings."""
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        """
        Args:
            input_dim: Input feature dimension (e.g., WavLM embedding size)
            hidden_dims: List of hidden layer dimensions [512, 256, 128]
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.network(x)
