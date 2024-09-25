import torch
from torch import nn

def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    """
    Creates a projection layer with specified configurations.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )

def extract_elements(sequence, indices):
    B, L, D = sequence.shape
    K = indices.shape[1]

    # Expand indices to [B, K, D]
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)

    # Gather the elements
    extracted_elements = torch.gather(sequence, 1, expanded_indices)

    return extracted_elements