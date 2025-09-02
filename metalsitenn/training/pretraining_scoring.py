# metalsitenn/training/pretraining_scoring.py
'''
* Author: Evan Komp
* Created: 9/2/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import Dict

import numpy as np
import torch

from ..graph_data import BatchProteinData, ModelOutput
from ..plotting import confusion_from_matrix


######################################################################
# Custom eval functions related to node prediction task
######################################################################

def f1_score_from_cm(cm):
    """Calculate weighted F1 score directly from confusion matrix"""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    # Avoid division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1_per_class = np.where(precision + recall > 0, 
                               2 * precision * recall / (precision + recall), 0)
    
    support = cm.sum(axis=1)
    return (f1_per_class * support).sum() / support.sum()

def custom_eval_batch(batch: BatchProteinData, model_outs: ModelOutput):
    """Compute metrics from outputs logits and loss.
    
    Metrics:
    - loss (easy)
    - cel_loss (eg. minus auxillary losses)
    - correct_preds, incorrect_preds (for accuracy)
    - cm, [1,C,C] confusion matrix of counts - we want to sum these later
    """
    out_metrics = {}
    cel_loss = model_outs.node_loss
    loss = model_outs.loss
    out_metrics['loss'] = loss.detach()
    out_metrics['cel_loss'] = cel_loss.detach()

    # labels and predictions
    atom_masked_mask = batch.atom_masked_mask.detach() # N, 1, sum = M
    labels = batch.element_labels.detach()[atom_masked_mask.squeeze(-1)] # M, 1
    logits = model_outs.node_logits.detach()[atom_masked_mask.squeeze(-1)] # M, num_classes
    preds = logits.argmax(dim=-1).unsqueeze(-1) # M, 1
    total_num_classes = logits.size(1)

    corrects_mask = (preds == labels).squeeze(-1) # M, 1
    incorrects_mask = (preds != labels).squeeze(-1) # M, 1
    out_metrics['correct_preds'] = corrects_mask.sum().unsqueeze(0)
    out_metrics['incorrect_preds'] = incorrects_mask.sum().unsqueeze(0)

    # Efficient confusion matrix computation using advanced indexing
    labels_flat = labels.squeeze(-1)  # M   
    preds_flat = preds.squeeze(-1)    # M

    # now build the counts in a confusion matrix
    cm_indices = labels_flat * total_num_classes + preds_flat  # M
    
    # Use bincount to efficiently count occurrences
    cm_counts = torch.bincount(cm_indices, minlength=total_num_classes * total_num_classes)
    
    # Reshape to confusion matrix format and convert to individual entries
    cm_matrix = cm_counts.view(total_num_classes, total_num_classes).unsqueeze(0)  # 1, C, C such that we can cat them later
    out_metrics['cm'] = cm_matrix
    
    return out_metrics

def custom_eval_logger(trainer, metrics: Dict[str, torch.Tensor]):
    """Custom evaluation logging function."""
    # we can cat them all along first dimension
    cat_metrics = {}
    for key, value in metrics.items():
        cat_metrics[key] = torch.cat(value, dim=0)
        trainer.log_debug(f"Shape of catted eval metric {key} after full val set: {cat_metrics[key].shape}")

    # Compute aggregated metrics
    # losses we just mean
    out_metrics = {}
    out_metrics['loss'] = cat_metrics['loss'].float().mean().item() 
    out_metrics['cel_loss'] = cat_metrics['cel_loss'].float().mean().item()

    # with the pred counts we can get accuracy
    total_correct = cat_metrics['correct_preds'].sum().item()
    total_incorrect = cat_metrics['incorrect_preds'].sum().item()
    out_metrics['accuracy'] = total_correct / (total_correct + total_incorrect) 

    # do metal accuracy
    # we can use the confusion matrix to do this
    cm = cat_metrics['cm'].sum(dim=0) # C, C
    metal_token_index = trainer.collator.featurizer.tokenizers['element'].metal_token_id
    out_metrics['metal_accuracy'] = cm[metal_token_index, metal_token_index].item() / cm[metal_token_index].sum().item()

    # f1 score
    f1 = f1_score_from_cm(cm.numpy())
    out_metrics['f1'] = f1

    # and now we can log the confusion matrix as a sklearn plot with dvc live
    fig = confusion_from_matrix(
        cm, vocab=tuple(trainer.collator.featurizer.tokenizers['element'].get_vocab().keys()),
        normalize=True
    )
    if trainer.accelerator.is_main_process:
        live = trainer.accelerator.get_tracker("dvclive", unwrap=True)
        live.log_image("eval_confusion_matrix.png", fig)

    # also log the metrics
    trainer._log_metrics(out_metrics, prefix='eval/')
    return out_metrics

######################################################################