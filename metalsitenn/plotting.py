# metalsitenn/plotting.py
'''
* Author: Evan Komp
* Created: 11/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import torch
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def confusion_from_matrix(
    confusion_matrix: torch.Tensor,
    vocab: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    normalize: bool = True,
    show_values: bool = True,
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix',
    aggregate_no_support: bool = True
) -> plt.Figure:
    """
    Create a confusion matrix plot from a square counts matrix.
    
    Args:
        confusion_matrix: Square tensor of shape (C, C) with confusion matrix counts
        vocab: Optional list of class name strings. If None, uses indices.
        save_path: Optional path to save the figure as PNG
        figsize: Figure size as (width, height)
        normalize: If True, normalize the confusion matrix by row (true labels)
        show_values: If True, show the numeric values in each cell
        cmap: Colormap for the plot
        title: Title for the plot
        aggregate_no_support: If True, aggregate classes with no true examples into "OTHER"
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if it's a tensor
    if isinstance(confusion_matrix, torch.Tensor):
        cm = confusion_matrix.cpu().numpy()
    else:
        cm = np.array(confusion_matrix)
    
    # Create class labels
    if vocab is None:
        labels = [str(i) for i in range(cm.shape[0])]
    else:
        labels = vocab[:cm.shape[0]]  # Trim vocab if it's longer than matrix
    
    # Handle aggregation of classes with no support
    if aggregate_no_support:
        # Find classes with no true examples (row sums are 0)
        row_sums = cm.sum(axis=1)
        no_support_indices = np.where(row_sums == 0)[0]
        
        if len(no_support_indices) > 0:
            # Find indices with support
            support_indices = np.where(row_sums > 0)[0]
            
            # Create new confusion matrix with supported classes + OTHER
            new_size = len(support_indices) + 1  # +1 for OTHER
            new_cm = np.zeros((new_size, new_size))
            new_labels = []
            
            # Copy supported classes
            for i, orig_idx in enumerate(support_indices):
                for j, orig_col_idx in enumerate(support_indices):
                    new_cm[i, j] = cm[orig_idx, orig_col_idx]
                new_labels.append(labels[orig_idx])
            
            # Aggregate predictions for no-support classes into OTHER column
            other_col_sum = cm[:, no_support_indices].sum(axis=1)
            for i, orig_idx in enumerate(support_indices):
                new_cm[i, -1] = other_col_sum[orig_idx]  # Predictions to OTHER classes
            
            # OTHER row: aggregate all true OTHER predictions
            # This represents cases where true label was a no-support class
            other_row_sum = cm[no_support_indices, :].sum(axis=0)
            for j, orig_col_idx in enumerate(support_indices):
                new_cm[-1, j] = other_row_sum[orig_col_idx]  # OTHER true -> supported predictions
            
            # OTHER-to-OTHER: predictions from no-support to no-support
            new_cm[-1, -1] = cm[np.ix_(no_support_indices, no_support_indices)].sum()
            
            # Update variables
            cm = new_cm
            labels = new_labels + ['OTHER']
    
    # Normalize if requested
    if normalize:
        # Add small epsilon to avoid division by zero warnings
        eps = 1e-15
        row_sums = cm.sum(axis=1)
        cm = cm.astype('float') / (row_sums[:, np.newaxis] + eps)
        # Set rows with zero sum to zero (instead of using nan_to_num)
        zero_rows = row_sums == 0
        cm[zero_rows, :] = 0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add text annotations if requested
    if show_values:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig