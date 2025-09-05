from typing import List
import torch
import torch.nn as nn

# --------------------- Projection Layer ---------------------
# Implements an MLP to project the aggregated feature vector into the target embedding space.


class Projection(nn.Module):
    """
    Projection module: projects input features into a target embedding space.

    This module applies a configurable stack of linear layers, each followed by
    LayerNorm and ReLU activation, and ends with a final linear projection to
    the desired output dimension.

    """

    def __init__(self, d_in: int = 1024, d_out: int = 768, hidden: List[int] | None = [3072, 2048, 2048, 1536]):
        """
        Initialize the projection network.

        Args:
            d_in (int, default=1024): Input feature dimension.
            d_out (int, default=768): Output feature dimension.
            hidden (List[int] | None, default=[3072, 2048, 2048, 1536]): List of hidden layer sizes.
                Each entry adds a block: Linear → LayerNorm → ReLU.
                If None or empty, the module reduces to a single Linear(d_in, d_out).
        """
        super().__init__()
        hidden = hidden or []
        layers, dim = [], d_in

        # Build hidden layers: Linear → LayerNorm → ReLU
        for h in hidden:
            layers += [
                nn.Linear(in_features=dim, out_features=h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ]
            dim = h

        # Final projection to the output dimension
        layers.append(nn.Linear(in_features=dim, out_features=d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, d_in).

        Returns:
            torch.Tensor: Projected tensor of shape (B, d_out).
        """
        return self.net(x)
