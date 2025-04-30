# pipeline/2.1_self_supervised_training.py
'''
* Author: Evan Komp
* Created: 1/24/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import logging
from pathlib import Path
import torch
from datasets import load_from_disk
import dvc.api
from e3nn import o3
import joblib
import json
import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from metalsitenn.model import (
    MetalSiteNNConfig, 
    MetalSiteForPretrainingModel,
    get_irreps
)
from metalsitenn.data import AtomicSystemBatchCollator
from metalsitenn.atom_vocabulary import AtomTokenizer
from metalsitenn.trainer import (
    MetalSiteTrainer,
    MetalSiteTrainingArgs,
    COMPUTE_LOSS_SELF_SUPERVISED_TRAINING,
    COMPUTE_EVAL_METRICS_FOUNDATIONAL_TRAINING,
    NO_MASKING_EMBED_AND_CLUSTER
)
from metalsitenn.utils import ParamsObj

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='logs/2.1_self_supervised_training.log')

def get_avg_num_neighbors():
    f = open("data/training_avg_num_neighbors.json", 'r')
    data = json.load(f)
    return data["avg_num_neighbors"]

def load_datasets(data_dir: Path):
    """Load train/val/test datasets."""
    dataset = load_from_disk(data_dir)
    return dataset["train"], dataset["test"]

def get_atom_weights(tokenizer: AtomTokenizer, params: ParamsObj):
    with open("data/metrics/null_model_metrics.json", 'r') as f:
        null_model_metrics = json.load(f)
    bad_keys = []
    for key in null_model_metrics:
        if key not in tokenizer.atom_vocab.stoi:
            bad_keys.append(key)
    for key in bad_keys:
        null_model_metrics.pop(key)
    
    if params.model.imbalance_loss_reweight:
        temperature = params.model.imbalance_loss_temperature
    else:
        temperature = 0.0

    cutoff_token = '<METAL>' if not params.data.metal_known else 'CU'
    atom_weights, freq_dict = tokenizer.get_token_weights(freq_dict=null_model_metrics, temperature=temperature, cutoff_token=cutoff_token)

    logger.info(f"Atom weights: {freq_dict}")

    # make a plot of the atom weights from the dict
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(freq_dict.keys()), y=list(freq_dict.values()), ax=ax)
    plt.savefig("data/plots/atom_weights.png", dpi=300, bbox_inches='tight')
    plt.close()
    return atom_weights


def main(quit_early=False, resume_from_checkpoint=None):
    # torch.cuda.memory._record_memory_history()
    # Load DVC params
    params = ParamsObj(dvc.api.params_show())
    
    # Setup paths and logging
    output_dir = Path("data/model_pretraining")
    
    # Initialize tokenizer
    tokenizer = joblib.load("data/dataset/tokenizer.pkl")
    
    # Load datasets
    logger.info("Loading datasets...")
    if not params.training.debug_use_toy:
        train_dataset, test_dataset = load_datasets(Path("data/dataset/metal_site_dataset"))
    else:
        train_dataset, test_dataset = load_datasets(Path("data/toy_dataset"))
    
    if params.training.debug_sample:
        train_dataset = train_dataset.select(range(params.training.debug_sample))
        test_dataset = test_dataset.select(range(params.training.debug_sample))
        logger.info(f"Debugging mode: using {params.training.debug_sample} samples")

    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Test dataset: {len(test_dataset)} examples")

    # collator
    collator = AtomicSystemBatchCollator(
        tokenizer=tokenizer,
        mask_rate=params.training.mask_rate,
        noise_rate=params.training.noise_rate,
        noise_scale=params.training.noise_scale,
        already_tokenized=True)

    # Create model config
    logger.info("Initializing model...")
    internal_irreps, head_irreps = get_irreps(
        l=params.model.l, 
        scale=params.model.hidden_scale, 
        num_heads=params.model.num_heads, 
        decay=params.model.hidden_scale_decay
    )
    logger.info(f"Internal irreps: {internal_irreps}")
    logger.info(f"Head irreps: {head_irreps}")
    config = MetalSiteNNConfig(
        irreps_node_feats=internal_irreps, # computed by get_irreps
        irreps_sh=o3.Irreps.spherical_harmonics(params.model.l), # determined by user param
        irreps_node_output=internal_irreps, # computed by get_irreps\
        irreps_head=head_irreps, # computed by get_irreps
        atom_vocab_size=tokenizer.atom_vocab.vocab_size, # fixed by tokenizer
        atom_type_vocab_size=tokenizer.record_vocab.vocab_size, # fixed by tokenizer

        # direct user params
        atom_embed_dim=params.model.atom_embed_dim,
        max_radius=params.model.max_radius,
        max_neighbors=params.model.max_neighbors,
        num_basis=params.model.num_basis,
        num_layers=params.model.num_layers,
        num_heads=params.model.num_heads,
        fc_neurons=params.model.fc_neurons,
        alpha_drop=params.model.alpha_drop,
        proj_drop=params.model.proj_drop,
        drop_path_rate=params.model.drop_path_rate,
        avg_num_neighbors=get_avg_num_neighbors(),
        label_smoothing_factor=params.model.label_smoothing_factor,
        model_atom_types=params.model.model_atom_types,

        # not available for training
        output_attentions=False,
        output_hidden_states=False,
        output_initial_embeddings=False
    )
    logger.info(f"Model config: {config}")

    model = MetalSiteForPretrainingModel(config)

    # get atom weights for loss
    atom_weights = get_atom_weights(tokenizer, params)
    model.set_atom_weights(atom_weights)

    logger.info(model)

    # now trainer
    logger.info("Initializing trainer...")
    training_args = MetalSiteTrainingArgs(
        output_dir=output_dir,

        # user params
        eval_steps=params.training.eval_steps,
        logging_steps=params.training.logging_steps,
        load_best_model_at_end=params.training.load_best_model_at_end,

        num_epochs=params.training.num_epochs,  
        per_device_train_batch_size=params.training.per_device_batch_size,
        per_device_eval_batch_size=params.training.per_device_batch_size,
        gradient_accumulation_steps=params.training.gradient_accumulation_steps,
        dataloader_num_workers=params.training.dataloader_num_workers,

        learning_rate=params.training.learning_rate,
        weight_decay=params.training.weight_decay,
        gradient_clipping=params.training.gradient_clipping,
        frac_noise_loss=params.training.frac_noise_loss,

        warmup_pct=params.training.warmup_pct,
        use_early_stopping=params.training.use_early_stopping,
        early_stopping_patience=params.training.early_stopping_patience,
        early_stopping_improvement_fraction=params.training.early_stopping_improvement_fraction
    )
    logger.info(f"Training args: {training_args}")

    # now set up additional metrics
    cluster_kwargs = {
        'min_cluster_size': params.embedding.hdbscan_min_cluster_size,
        'min_samples': params.embedding.hdbscan_min_samples,
        'alpha': params.embedding.hdbscan_alpha,
    }
    tsne_kwargs = {
        'perplexity': params.embedding.tsne_perplexity,
        'n_iter': params.embedding.tsne_n_iter,
        'learning_rate': params.embedding.tsne_learning_rate,
    }
    eval_embed_and_cluster_func = lambda trainer: NO_MASKING_EMBED_AND_CLUSTER(
        trainer, embedding_how=params.embedding.how, cluster_kwargs=cluster_kwargs, tsne_kwargs=tsne_kwargs, hidden_state=params.embedding.hidden_state)

    resume = resume_from_checkpoint is not None
    trainer = MetalSiteTrainer(
        model=model,
        compute_loss_fn=COMPUTE_LOSS_SELF_SUPERVISED_TRAINING,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        eval_metrics=COMPUTE_EVAL_METRICS_FOUNDATIONAL_TRAINING,
        hard_eval_metrics={'cluster': eval_embed_and_cluster_func},
        quit_early=quit_early,
        resume=resume
    )

    # Train
    logger.info("Training model...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training complete.")

    # save final model and args so that it can be reloaded
    model.save_pretrained(os.path.join(output_dir, "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quit-early', action='store_true')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args = parser.parse_args()
    main(quit_early=args.quit_early, resume_from_checkpoint=args.resume_from_checkpoint)


        


        