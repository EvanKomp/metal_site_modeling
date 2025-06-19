# metalsitenn/nn/mlp.py
'''
* Author: Evan Komp
* Created: 6/19/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import torch
import torch.nn as nn
from typing import Optional, Union


def get_torch_activation(activation: Union[str, nn.Module]) -> nn.Module:
    """
    Convert activation string to torch activation module.
    
    Args:
        activation: Activation function name (string) or torch module
        
    Returns:
        PyTorch activation module
        
    Raises:
        ValueError: If activation string is not recognized
    """
    if isinstance(activation, nn.Module):
        return activation
    
    if not isinstance(activation, str):
        raise ValueError(f"Activation must be string or nn.Module, got {type(activation)}")
    
    activation = activation.lower()
    
    activation_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),  # SiLU is Swish
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=-1),
        'log_softmax': nn.LogSoftmax(dim=-1),
        'identity': nn.Identity(),
        'none': nn.Identity(),
        'linear': nn.Identity(),
    }
    
    if activation not in activation_map:
        available = ', '.join(activation_map.keys())
        raise ValueError(f"Unknown activation '{activation}'. Available: {available}")
    
    return activation_map[activation]


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        n_hidden_layers: Number of hidden layers (not including output layer)
        hidden_activation: Activation function for hidden layers
        output_activation: Activation function for output layer (optional)
        dropout_rate: Dropout probability (0.0 = no dropout)
        
    Example:
        >>> mlp = MLP(
        ...     input_size=128,
        ...     hidden_size=256, 
        ...     n_hidden_layers=2,
        ...     hidden_activation='relu',
        ...     output_activation='softmax',
        ...     dropout_rate=0.1
        ... )
        >>> x = torch.randn(32, 128)
        >>> output = mlp(x)  # Shape: (32, 256)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int = 1,
        hidden_activation: Union[str, nn.Module] = 'relu',
        output_activation: Optional[Union[str, nn.Module]] = None,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        if n_hidden_layers < 0:
            raise ValueError(f"n_hidden_layers must be >= 0, got {n_hidden_layers}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout_rate = dropout_rate
        
        # Convert activation functions
        self.hidden_activation = get_torch_activation(hidden_activation)
        self.output_activation = get_torch_activation(output_activation) if output_activation else None
        
        # Build layers
        layers = []
        
        if n_hidden_layers == 0:
            # Direct input to output
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            # Input layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self.hidden_activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(self.hidden_activation)
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
            
            # Output layer (no activation or dropout here)
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Add output activation if specified
        if self.output_activation is not None:
            layers.append(self.output_activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (..., input_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        return self.network(x)
