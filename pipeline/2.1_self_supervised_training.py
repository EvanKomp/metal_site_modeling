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
    COMPUTE_EVAL_METRICS_FOUNDATIONAL_TRAINING
)
from metalsitenn.utils import ParamsObj

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='logs/2.1_self_supervised_training.log')

def get_avg_num_neighbors():
    return 5

def load_datasets(data_dir: Path):
    """Load train/val/test datasets."""
    dataset = load_from_disk(data_dir)
    return dataset["train"], dataset["test"]

def main():
    # Load DVC params
    params = ParamsObj(dvc.api.params_show())
    
    # Setup paths and logging
    output_dir = Path("data/model_pretraining")
    
    # Initialize tokenizer
    tokenizer = AtomTokenizer(
        keep_hydrogen=params.data.model_hydrogens,
        metal_known=params.data.metal_known,
        aggregate_uncommon=params.data.aggregate_uncommon,
        allow_unknown=True
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, test_dataset = load_datasets(Path("data/dataset/metal_site_dataset"))
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
        num_basis=params.model.num_basis,
        num_layers=params.model.num_layers,
        num_heads=params.model.num_heads,
        alpha_drop=params.model.alpha_drop,
        proj_drop=params.model.proj_drop,
        drop_path_rate=params.model.drop_path_rate,
        avg_num_neighbors=get_avg_num_neighbors(),
        label_smoothing_factor=params.model.label_smoothing_factor,

        # not available for training
        output_attentions=False,
        output_hidden_states=False,
        output_initial_embeddings=False
    )
    logger.info(f"Model config: {config}")

    model = MetalSiteForPretrainingModel(config)

    # now trainer
    logger.info("Initializing trainer...")
    training_args = MetalSiteTrainingArgs(
        output_dir=output_dir,
        logging_file="logs/2.1_self_supervised_training.log",

        # user params
        eval_steps=params.training.eval_steps,
        logging_steps=params.training.logging_steps,
        load_best_model_at_end=params.training.load_best_model_at_end,

        num_epochs=params.training.num_epochs,  
        per_device_train_batch_size=params.training.per_device_batch_size,
        per_device_eval_batch_size=params.training.per_device_batch_size,
        gradient_accumulation_steps=params.training.gradient_accumulation_steps,
        dataloader_num_workers=params.training.dataloader_num_workers,
        mixed_precision=params.training.mixed_precision,

        learning_rate=params.training.learning_rate,
        weight_decay=params.training.weight_decay,
        gradient_clipping=params.training.gradient_clipping,
        frac_noise_loss=params.training.frac_noise_loss,

        warmup_steps=params.training.warmup_steps,
        use_early_stopping=params.training.use_early_stopping,
        early_stopping_patience=params.training.early_stopping_patience,
        early_stopping_improvement_fraction=params.training.early_stopping_improvement_fraction
    )
    logger.info(f"Training args: {training_args}")

    trainer = MetalSiteTrainer(
        model=model,
        compute_loss_fn=COMPUTE_LOSS_SELF_SUPERVISED_TRAINING,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        additional_eval_metrics=COMPUTE_EVAL_METRICS_FOUNDATIONAL_TRAINING
    )
    
    # Train
    logger.info("Training model...")
    trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()


        


        