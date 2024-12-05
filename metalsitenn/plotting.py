# metalsitenn/plotting.py
'''
* Author: Evan Komp
* Created: 11/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
# metalsitenn/plotting.py
'''
* Author: Evan Komp
* Created: 11/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import torch
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from metalsitenn.atom_vocabulary import AtomTokenizer

def plot_atoms_and_vectors(
    positions: torch.Tensor, 
    atom_tokens: Union[List[str], torch.Tensor]=None,
    atom_types: List[int]=None,
    tokenizer: Optional[AtomTokenizer] = None,
    atom_values: Optional[torch.Tensor] = None,
    vectors: Optional[torch.Tensor] = None,
    title: str = "",
    ax = None,
    quiver_multiplier: float=1.0,
    quiver_mask: Optional[torch.Tensor] = None,
    atom_highlight: Optional[torch.Tensor] = None,
    unhighlight_alpha: float=0.4,
    quiver_color: str = 'green'
) -> None:
    """Plot 3D atomic structure with optional vector field for visualizing equivariance.
    
    Color preference hierarchy:
    1. atom_values: Uses viridis colormap with continuous colorbar
    2. atom_tokens: Uses tab20 colormap with discrete colorbar
    3. No color specified: All markers are grey
    
    Marker shapes:
    - atom_types=0: Circle markers
    - atom_types=1: Square markers
    
    Args:
        positions: [N,3] tensor of atomic coordinates
        atom_tokens: List of N atomic tokens or tensor of token indices
        atom_types: List of N atomic types (0 for circle, 1 for square)
        tokenizer: AtomTokenizer object for mapping atom tokens to colors
        atom_values: Optional [N,1] tensor of scalar quantities to color atoms
        vectors: Optional [N,3] tensor of vector quantities to plot as arrows
        title: Title of the plot
        ax: Optional matplotlib axis to plot on
        quiver_multiplier: Multiplier for the length of the vectors
        quiver_mask: Optional mask to plot only a subset of the vectors
        atom_highlight: Optional mask to highlight specific atoms
        unhighlight_alpha: Transparency of unhighlighted atoms
        quiver_color: Color of the vectors
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    positions = positions.detach().numpy()
    vectors = vectors.detach().numpy() if vectors is not None else None
    
    # Determine colors and create appropriate colorbar
    if atom_values is not None:
        # Continuous colormap for scalar values
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=atom_values.min(), vmax=atom_values.max())
        colors = cmap(norm(atom_values))
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Atom Values')
        
    elif atom_tokens is not None:
        # Discrete colors for atom tokens
        assert tokenizer is not None, "Must provide tokenizer to map atom symbols to colors"
        
        # Convert tensor indices to tokens if needed
        if isinstance(atom_tokens, torch.Tensor):
            atom_tokens = [tokenizer.atom_vocab.itos[idx.item()] for idx in atom_tokens]
            
        unique_tokens = sorted(set(atom_tokens))
        cmap = plt.cm.tab20
        
        # Create color mapping
        color_dict = {token: cmap(i/len(unique_tokens)) for i, token in enumerate(unique_tokens)}
        colors = [color_dict[token] for token in atom_tokens]
        
    else:
        # Default grey for all markers
        colors = ['grey'] * len(positions)

    # Create legend elements
    legend_elements = []
    
    # Add color legend if using atom_tokens
    if atom_tokens is not None:
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[token],
                  label=token, markersize=10) 
            for token in unique_tokens
        ])
        
    # Add shape legend if using atom_types
    if atom_types is not None:
        shape_elements = [
            Line2D([0], [0], marker='o', color='grey', label='HETATM', 
                  markersize=10, linestyle='None'),
            Line2D([0], [0], marker='s', color='grey', label='ATM', 
                  markersize=10, linestyle='None')
        ]
        # Add shape elements at the beginning of the legend
        legend_elements = shape_elements + legend_elements

    # Plot atoms with appropriate markers and colors
    for i, pos in enumerate(positions):
        if atom_highlight is not None:
            alpha = 1.0 if i in atom_highlight else unhighlight_alpha
        else:
            alpha = 1.0
            
        # Determine marker shape based on atom_types
        marker = 's' if (atom_types is not None and atom_types[i] == 1) else 'o'
        
        ax.scatter(*pos, c=[colors[i]], s=100, alpha=alpha, marker=marker)

    # Add legend if we have any legend elements
    if legend_elements:
        ax.legend(handles=legend_elements, 
                 bbox_to_anchor=(1.15, 1), 
                 loc='upper left')

    # Plot vectors as arrows if provided
    if vectors is not None:
        end_points = vectors * quiver_multiplier
        if quiver_mask is not None:
            positions = positions[quiver_mask]
            end_points = end_points[quiver_mask]

        ax.quiver(
            positions[:,0], positions[:,1], positions[:,2],
            end_points[:,0], end_points[:,1], end_points[:,2],
            color=quiver_color, alpha=1.0
        )

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_title(title)
    
    return ax