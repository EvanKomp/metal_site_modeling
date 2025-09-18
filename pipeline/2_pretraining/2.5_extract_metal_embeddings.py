# pipeline/2_pretraining/2.5_embed_metals.py
'''
* Author: Evan Komp
* Created: 9/2/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Generate metal embeddings from pretrained model for downstream tasks.
'''
import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import dvc.api
from tqdm import tqdm

from metalsitenn.utils import ParamsObj
from metalsitenn.dataloading import MetalSiteDataset
from metalsitenn.featurizer import MetalSiteCollator
from metalsitenn.nn.model import EquiformerWEdgesForPretraining
from metalsitenn.graph_data import BatchProteinData

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dvc_params() -> ParamsObj:
    """Load DVC parameters for 2_pretraining stage."""
    params = dvc.api.params_show()
    return ParamsObj(params)['2_pretraining']

def initialize_collator_for_embedding(params: ParamsObj, mask_metals: bool = False) -> MetalSiteCollator:
    """
    Initialize collator for embedding generation.
    
    Args:
        params: DVC parameters
        mask_metals: If True, convert all <METAL> tokens to <MASK> tokens before model input
        
    Returns:
        Collator configured for inference
    """
    tokenization_params = params.data.tokenization
    
    # Extract parameters needed for inference
    inference_params = {
        'active_aggregators': tokenization_params.active_aggregators,
        'atom_features': tokenization_params.atom_features,
        'bond_features': tokenization_params.bond_features,
        'k': tokenization_params.k,
        'node_mlm_do': False,  # Disable random masking
        'metal_classification': mask_metals  # This will convert metals to <METAL> then we'll convert to <MASK>
    }
    
    # Remove any None values
    clean_params = {k: v for k, v in inference_params.items() if v is not None}
    
    collator = MetalSiteCollator(**clean_params)
    
    return collator

def load_full_dataset(params: ParamsObj) -> MetalSiteDataset:
    """Load full dataset without splitting."""
    filtering_params = params.data.filtering
    
    cache_folder = './data/1/1.1_parse_sites_metadata'
    dataset = MetalSiteDataset(
        cache_folder=cache_folder,
        overwrite=False,
        save_pdb=False,
        **filtering_params
    )
    
    logging.info(f"Loaded dataset with {len(dataset)} sites after filtering")
    
    # Apply debug size limitation if specified
    if params.data.debug_max_sites:
        from torch.utils.data import Subset
        indices = list(range(min(len(dataset), params.data.debug_max_sites)))
        dataset = Subset(dataset, indices)
        logging.info(f"Limited to {len(dataset)} sites for debugging")
    
    return dataset

def get_metal_masks_from_batch(batch: BatchProteinData, collator: MetalSiteCollator) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Extract boolean masks indicating metal atom positions for each system in batch.
    
    Args:
        batch: Batch of protein data
        collator: Collator to get tokenizers
        
    Returns:
        Tuple of (per_system_masks, global_metal_mask)
        - per_system_masks: List of boolean tensors, one per system
        - global_metal_mask: Single boolean tensor for the entire batch
    """
    element_tokenizer = collator.featurizer.tokenizers['element']
    active_aggregators = collator.active_aggregators or []
    
    # Get metal token IDs
    metal_token_ids = torch.tensor(
        element_tokenizer.get_metal_representing_token_ids(active_aggregators=active_aggregators),
        device=batch.element.device
    )
    
    # Get global metal mask using element labels (original tokens before masking)
    if hasattr(batch, 'element_labels') and batch.element_labels is not None:
        # Use original element tokens if available (before metal anonymization)
        global_metal_mask = torch.isin(batch.element_labels, metal_token_ids)
    else:
        # Fallback to current element tokens
        global_metal_mask = torch.isin(batch.element, metal_token_ids)
    
    # Split into per-system masks using batch indices
    per_system_masks = []
    n_systems = batch.batch.max().item() + 1
    
    for system_idx in range(n_systems):
        system_atom_mask = (batch.batch == system_idx)
        system_metal_mask = global_metal_mask[system_atom_mask]
        per_system_masks.append(system_metal_mask.squeeze(dim=-1))
    
    return per_system_masks, global_metal_mask

def convert_metals_to_mask_tokens(batch: BatchProteinData, metal_mask: torch.Tensor, collator: MetalSiteCollator) -> BatchProteinData:
    """
    Convert <METAL> tokens to <MASK> tokens using featurizer's _mask_atoms method.
    
    Args:
        batch: Batch of protein data
        metal_mask: Boolean mask indicating metal positions globally across batch
        collator: Collator with featurizer for masking
        
    Returns:
        Modified batch with metals converted to mask tokens
    """
    if not metal_mask.any():
        return batch  # No metals to mask
    
    # Convert batch back to list of ProteinData for individual processing
    protein_data_list = batch.to_protein_data_list()
    
    # Process each system individually
    current_idx = 0
    for i, pdata in enumerate(protein_data_list):
        atom_count = batch._atom_counts[i]
        system_metal_mask = metal_mask[current_idx:current_idx + atom_count]
        
        # Get indices of metal atoms in this system
        metal_indices = torch.where(system_metal_mask)[0].tolist()
        
        if metal_indices:
            # Use featurizer's _mask_atoms method to properly mask metals
            current_device = pdata.element.device
            pdata = pdata.to('cpu')
            pdata = collator.featurizer._mask_atoms(
                pdata, 
                indices_to_mask=metal_indices,
                indices_to_tweak=[],
                indices_to_keep=[]
            )
            protein_data_list[i] = pdata.to(current_device)
        
        current_idx += atom_count
    
    # Recreate batch from modified protein data list
    return BatchProteinData(protein_data_list)

def extract_metal_embeddings(
    so3_embeddings: torch.Tensor,
    per_system_masks: List[torch.Tensor], 
    batch: BatchProteinData,
    pooling_method: str = 'mean',
    use_l1_norms: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Extract and pool metal embeddings using precomputed masks.
    
    Args:
        so3_embeddings: SO3 embeddings tensor (N_atoms, l_max+1, embedding_dim)
        per_system_masks: List of boolean masks for metals in each system
        batch: Batch of protein data for pdb_ids
        pooling_method: How to pool multiple metals ['mean', 'sum', 'max']
        use_l1_norms: If True, include L1 vector norms with L0 scalars
    
    Returns:
        Dictionary mapping pdb_id to pooled metal embedding tensor
    """
    results = {}
    
    # Extract L0 (scalar) features
    l0_embeddings = so3_embeddings[:, 0, :]  # (N_atoms, embedding_dim)
    
    # Extract and norm L1 (vector) features if requested
    if use_l1_norms:
        l1_embeddings = so3_embeddings[:, 1:4, :]  # (N_atoms, 3, embedding_dim) - x,y,z components
        l1_norms = torch.norm(l1_embeddings, dim=1)  # (N_atoms, embedding_dim) - norm across spatial dims
        # Concatenate L0 scalars with L1 norms
        combined_embeddings = torch.cat([l0_embeddings, l1_norms], dim=-1)  # (N_atoms, 2*embedding_dim)
    else:
        combined_embeddings = l0_embeddings
    
    # Process each system in the batch using atom counts
    current_idx = 0
    for i, (atom_count, metal_mask) in enumerate(zip(batch._atom_counts, per_system_masks)):
        pdb_id = batch.pdb_id[i][0]  # Extract pdb_id from array structure
        
        # Get embeddings for this system
        system_embeddings = combined_embeddings[current_idx:current_idx + atom_count]
        
        # Extract metal embeddings if any metals found
        if metal_mask.sum() > 0:
            metal_embeddings = system_embeddings[metal_mask]
            
            # Pool multiple metals
            if pooling_method == 'mean':
                pooled = torch.mean(metal_embeddings, dim=0)
            elif pooling_method == 'sum':
                pooled = torch.sum(metal_embeddings, dim=0)
            elif pooling_method == 'max':
                pooled = torch.max(metal_embeddings, dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
            
            results[pdb_id] = pooled
            logging.debug(f"Found {metal_mask.sum()} metals in {pdb_id}")
        else:
            logging.warning(f"No metals found in {pdb_id}")
        
        current_idx += atom_count
    
    return results

def generate_embeddings(
    model: EquiformerWEdgesForPretraining,
    dataloader: DataLoader,
    collator: MetalSiteCollator,
    device: torch.device,
    pooling_method: str = 'mean',
    mask_metals_for_model: bool = False,
    use_l1_norms: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for all systems in dataloader.
    
    Args:
        model: Pretrained model
        dataloader: DataLoader for dataset
        collator: Collator to get tokenizers
        device: Device to run on
        pooling_method: How to pool multiple metals
        mask_metals_for_model: If True, convert metals to mask tokens before model input
        use_l1_norms: If True, include L1 vector norms with L0 scalars
    
    Returns:
        Dictionary mapping pdb_id to embedding array
    """
    model.eval()
    all_embeddings = {}
    i = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            try:
                batch = batch.to(device)
                
                # Get metal masks before any modification
                per_system_masks, global_metal_mask = get_metal_masks_from_batch(batch, collator)
                
                # Convert metals to mask tokens if requested
                if mask_metals_for_model:
                    batch = convert_metals_to_mask_tokens(batch, global_metal_mask, collator)
                
                # Forward pass with return_embeddings=True
                model_outs = model(batch, return_node_embedding_tensor=True, compute_loss=False)
                so3_embeddings = model_outs.node_embeddings  # (N_atoms, l_max+1, embedding_dim)
                
                # Extract metal embeddings using precomputed masks and SO3 embeddings
                batch_embeddings = extract_metal_embeddings(
                    so3_embeddings, per_system_masks, batch, pooling_method, use_l1_norms
                )
                
                # Convert to numpy and store
                for pdb_id, embedding in batch_embeddings.items():
                    all_embeddings[pdb_id] = embedding.cpu().numpy()

                i += len(batch)

                # if i > 5000:
                #     break
            except:
                pass
    
    return all_embeddings

def save_embeddings(
    embeddings: Dict[str, np.ndarray],
    output_path: Path,
    format: str = 'pickle'
) -> None:
    """
    Save embeddings to file.
    
    Args:
        embeddings: Dictionary of embeddings
        output_path: Output file path
        format: Save format ['pickle', 'json', 'npz']
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
    elif format == 'json':
        # Convert arrays to lists for JSON serialization
        json_embeddings = {
            pdb_id: embedding.tolist() 
            for pdb_id, embedding in embeddings.items()
        }
        with open(output_path, 'w') as f:
            json.dump(json_embeddings, f, indent=2)
    elif format == 'npz':
        np.savez_compressed(output_path, **embeddings)
    else:
        raise ValueError(f"Unknown format: {format}")

def main():
    """Main embedding generation function."""
    parser = argparse.ArgumentParser(description="Generate metal embeddings from pretrained model")
    parser.add_argument(
        '--pooling', 
        type=str, 
        default='mean',
        choices=['mean', 'sum', 'max'],
        help='Pooling method for multiple metals'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pickle',
        choices=['pickle', 'json', 'npz'],
        help='Output file format'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/2/2.5_embeddings',
        help='Output directory'
    )
    parser.add_argument(
        '--mask-metals',
        action='store_true',
        help='Convert all <METAL> tokens to <MASK> tokens before model input'
    )
    parser.add_argument(
        '--use-l1-norms',
        action='store_true',
        help='Include L1 vector norms concatenated to L0 scalar features'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/2.5_embed_metals.log')
    
    logger.info(f"Starting metal embedding generation with pooling={args.pooling}, mask_metals={args.mask_metals}, use_l1_norms={args.use_l1_norms}")
    
    # Load DVC parameters
    logger.info("Loading DVC parameters...")
    params = load_dvc_params()
    logger.info("Loaded parameters for 2_pretraining stage")
    
    # Initialize components
    logger.info("Initializing collator for embedding generation...")
    collator = initialize_collator_for_embedding(params, mask_metals=args.mask_metals)
    logger.info(f"Collator initialized with vocab sizes: {collator.featurizer.get_feature_vocab_sizes()}")
    
    # Load full dataset
    logger.info("Loading full dataset...")
    dataset = load_full_dataset(params)
    logger.info(f"Dataset loaded: {len(dataset)} systems")
    
    # Load pretrained model
    logger.info("Loading pretrained model...")
    run_dir = params.training.run_dir
    model_path = Path(run_dir) / "final_model"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Final model not found at {model_path}. Make sure training has completed.")
    
    model = EquiformerWEdgesForPretraining.from_pretrained(model_path)
    logger.info("Pretrained model loaded successfully")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Create dataloader
    batch_size = params.training.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=params.training.num_workers,
        pin_memory=params.training.pin_memory
    )
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(dataset)} systems...")
    embeddings = generate_embeddings(
        model, dataloader, collator, device, args.pooling, args.mask_metals, args.use_l1_norms
    )
    logger.info(f"Generated embeddings for {len(embeddings)} systems with metals")
    
    # Save embeddings
    output_dir = Path(args.output_dir)
    mask_suffix = "_masked" if args.mask_metals else ""
    l1_suffix = "_with_l1" if args.use_l1_norms else ""
    output_file = output_dir / f"metal_embeddings_{args.pooling}{mask_suffix}{l1_suffix}.{args.format}"
    
    logger.info(f"Saving embeddings to {output_file}...")
    save_embeddings(embeddings, output_file, args.format)
    
    # Save metadata
    metadata = {
        'pooling_method': args.pooling,
        'mask_metals': args.mask_metals,
        'use_l1_norms': args.use_l1_norms,
        'n_systems': len(embeddings),
        'embedding_dim': list(embeddings.values())[0].shape[0] if embeddings else 0,
        'model_path': str(model_path),
        'format': args.format,
        'total_dataset_size': len(dataset)
    }
    
    metadata_file = output_dir / f"metal_embeddings_metadata_{args.pooling}{mask_suffix}{l1_suffix}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Completed: {len(embeddings)} embeddings saved from {len(dataset)} total systems")
    logger.info("Metal embedding generation completed successfully!")

if __name__ == "__main__":
    main()