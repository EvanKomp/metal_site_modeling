import torch
import torch.nn as nn
from functools import partial
from fairchem.core.models.equiformer_v2.transformer_block import FeedForwardNetwork
from fairchem.core.models.equiformer_v2.weight_initialization import eqv2_init_weights
from fairchem.core.common import gp_utils


class MetalIDHead(nn.Module):
    """
    Prediction head that outputs a vector of scalars for each protein in the batch representing metal counts. 
    
    Takes node embeddings, processes them through a feedforward network,
    then aggregates by protein to produce one vector per protein.
    
    Args:
        backbone: The backbone model (EquiformerWEdgesBackbone)
        output_dim: Number of scalar outputs per protein
        output_name: Name for the output in returned dictionary
        reduce: How to aggregate node features within each protein ("sum" or "mean")
        use_protein_normalization: Whether to normalize by average protein size
    """
    
    def __init__(
        self, 
        backbone, 
        output_dim: int, 
        output_name: str = "metal_counts",
        reduce: str = "sum"
    ):
        super().__init__()
        
        self.output_name = output_name
        self.output_dim = output_dim
        self.reduce = reduce
        self.avg_num_nodes = backbone.avg_num_nodes # this is k 
        
        # Feedforward network to process node embeddings
        self.final_layer = FeedForwardNetwork(
            sphere_channels=backbone.sphere_channels, # 64
            hidden_channels=backbone.ffn_hidden_channels, # 512
            output_channels=output_dim,
            lmax_list=backbone.lmax_list,
            mmax_list=backbone.mmax_list,  # Fixed: use mmax_list not lmax_list
            SO3_grid=backbone.SO3_grid, # (3,3)
            activation=backbone.ffn_activation, # This has been set to 'scaled_silu' not sure if this is actually being used
            use_gate_act=backbone.use_gate_act, # False
            use_grid_mlp=backbone.use_grid_mlp, # False
            use_sep_s2_act=backbone.use_sep_s2_act, # True
        )
        
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))
    
    def forward(self, data, emb):
        # Process node embeddings through prediction network
        node_output = self.final_layer(emb["node_embedding"])

        # Extract scalar features - get all output_dim channels from l=0
        node_scalars = node_output.embedding[:, 0, :] 
        
        # Handle distributed training across GPUs if needed
        if gp_utils.initialized():
            node_scalars = gp_utils.gather_from_model_parallel_region(node_scalars, dim=0)
        
        # activation to normalize node scalars to be between 0 and 1 
        activated_node_scalars = torch.sigmoid(node_scalars)

        # Create output tensor
        batch_size = len(data._atom_counts)
        output = torch.zeros(
            batch_size,
            self.output_dim,
            device=activated_node_scalars.device,
            dtype=activated_node_scalars.dtype,
        ) # [ N_batch x output_dim]
        
        # Aggregate by protein using index_add_ (handles multi-dim automatically)
        output.index_add_(0, data.batch, activated_node_scalars)
        
        # Apply normalization
        if self.reduce == "mean":
            atom_counts = torch.tensor(
                data._atom_counts, 
                device=output.device, 
                dtype=output.dtype
            ).unsqueeze(-1)
            output = output / atom_counts
        elif self.reduce == "sum":
            output = output / self.avg_num_nodes
        else:
            raise ValueError(f"reduce can only be sum or mean, user provided: {self.reduce}")   
        
        # Don't convert to integers here - keep as continuous for training
        return {self.output_name: output}
    