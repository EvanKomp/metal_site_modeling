# MetalSiteNN

## Description
A foundational model for protein metal binding sites using equivariant neural networks. Key features:
- SO(3) equivariant architecture preserves 3D geometric symmetries, atoms represented by atom identity and atom type (ATM, HETATM)
- Pretraining via atom masking and coordinate noise
- Supports downstream tasks: site prediction, clustering, property prediction, site perturbation

## Pipeline Steps

### 1.1 Parse Site Data
Analyzes metal binding sites to extract core statistics and labels.

**Inputs**:  
- `data/mf_sites/*.pdb`: PDB format files containing metal binding sites

**Outputs**:
- `data/site_labels.csv`: Per-site labels including:
  - Metal identity
  - Coordinating residues
  - Heteroatom presence
  - Completeness metrics
- `data/metrics/site_label_metrics.json`: Aggregated dataset statistics

**Parameters**:
- `max_radius`: Cutoff distance for coordinating residues
- `structure_checks`: Additional validation criteria

### 1.2 Create HuggingFace Dataset
Converts PDB files into a tokenized HuggingFace dataset for efficient training.

**Inputs**:  
- `data/mf_sites/*.pdb`: PDB format files containing metal binding sites
- `data.model_hydrogens`: DVC param for hydrogen inclusion
- `data.metal_known`: DVC param for metal token handling

**Params**:
- `data.model_hydrogens`: Include hydrogen atoms in model
- `data.metal_known`: Use unique tokens for metals vs generic METAL token
- `data.test_frac`: Fraction of sites to reserve for testing

**Outputs**:
- `data/dataset/metal_site_dataset`: HuggingFace dataset dict containing:
  - Atomic coordinates
  - Tokenized atom identities
  - Record types (ATOM/HETATM)
  - System IDs

**Script**: `pipeline/1.2_create_dataset.py`

## DVC Params

data.model_hydrogens: bool=False # Include hydrogen atoms in model
data.metal_known: bool=False # Use unique tokens for metals vs generic METAL token
data.test_frac: float=0.1 # Fraction of sites to reserve for testing

model.max_l: int=3 # Maximum order of irreps to consider within the model
model.hidden_scale: int=128 # base multiplicity for l=0 irreps, eg. Xe0 + Xo0 + ...
model.hidden_scale_decay: float=0.5 # Multiplicative decay factor for each additional l, eg. Xe1 = Xe0 * hidden_scale_decay
model.num_attention_layers: int=3 # Number of transformer blocks
model.num_heads: int=8 # Number of attention heads within each transformer block
model.alpha_drop: float=0.1 # Attention dropout rate
model.proj_drop: float=0.1 # Projection dropout rate
model.drop_path_rate: float=0.1 # Skip connection dropout rate
model.out_drop: float=0.1 # Output layer dropout rate
model.norm_layer: str='layer_norm' # Normalization layer type
model.max_radius: float=6.0 # Maximum edge distance for attention
model.num_radial_basis: int=32 # Number of gaussian radial basis functions to embed edge distances

training.atom_mask_rate: float=0.15 # Fraction of atoms to mask during training
training.coord_noise_rate: float=0.15 # Fraction of coordinates to noise during training
training.zero_noise_in_loss_rate: 0.05 # Additional fraction of coordinates that will not be noised but will be included in loss
> NOTE: coord_noise_rate + zero_noise_in_loss_rate should not exceed 1.0, and will be the fraction of atoms per batch that denoising loss is calculated for
training.noise_scale: float=1.0 # Standard deviation of noise to apply to coordinates - Units of Angstrom. 
> NOTE: Atoms are noised according to this scale, but loss calculation is computed based on vectors from unit scale. Thus model outputs do not need to achieve the same scale as the noise applied.
training.batch_size: int=2 # Number of PDB systems per batch
training.loss_balance: float=0.5 # Weighting factor for balancing atom and position loss, 0.2 = 80% atom loss, 20% position loss


## Components

### AtomTokenizer
Converts atomic identities and record types to integer tokens.

**Parameters**:
- `keep_hydrogen`: Include H atoms in vocabulary
- `metal_known`: Use unique tokens for metals vs generic METAL token

**Methods**:
- `tokenize(atoms, atom_types)`: Convert to integer tokens 
- `decode(tokens)`: Convert back to strings

**Attributes**:
- `atom_vocab`: Vocabulary for atomic elements
- `record_vocab`: Vocabulary for record types (ATOM/HETATM)
- `mask_token`: Integer token for masking
- `oh_size`: Total vocabulary size

### AtomEmbedding
Projects atomic features into irreducible SO(3) representations.

**Parameters**:
- `categorical_features`: List of (vocab_size, embed_dim) tuples
- `continuous_features`: List of continuous feature dimensions
- `irreps_out`: Output irreps specification

**Methods**:
- `forward(categorical_features, continuous_features)`: Embed features
  
**Returns**:
- `embeddings`: Irrep tensor
- `raw`: One-hot encoding

### GraphAttention
Equivariant multi-head attention layer.

**Parameters**:
- `irreps_node_input/output`: Node feature irreps
- `irreps_node_attr`: Node attribute irreps
- `irreps_edge_attr`: Edge geometric irreps
- `irreps_head`: Per-head feature irreps
- `num_heads`: Number of attention heads
- `fc_neurons`: Hidden layer sizes
- `alpha_drop`: Attention dropout rate
- `proj_drop`: Projection dropout rate

**Methods**:
- `forward(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_embedding)`: Apply attention

### TransformerBlock  
Full transformer layer with attention and feedforward.

**Parameters**:
- [Same as GraphAttention]
- `drop_path_rate`: Skip connection dropout
- `norm_layer`: Normalization type

**Methods**:
- `forward()`: [Same inputs as GraphAttention]

### MetalSiteFoundationalModel
Core equivariant model for metal site representation.

**Parameters**:
- `irreps_node_embedding`: Initial node embedding irreps
- `irreps_sh`: Spherical harmonic irreps  
- `irreps_output`: Output irreps
- `max_radius`: Edge cutoff distance
- `num_layers`: Number of transformer blocks
- [Additional transformer params]

**Methods**:
- `forward(atom_identifiers, positions, batch_indices)`: Process protein structure
 
**Returns**:
- `node_features`: Equivariant node representations
- `node_attrs`: Original node attributes

### MetalSiteNodeHead
Prediction head for atom identity and position tasks.

**Parameters**: 
- `irreps_node_input`: Input feature irreps
- `irreps_node_attrs`: Node attribute irreps
- `proj_drop`: Dropout rate
- `tokenizer`: For vocabulary sizes

**Methods**:
- `forward(node_input, node_attrs)`: Generate predictions

**Returns**:
- `atom_logits`: Atom type predictions 
- `record_logits`: Record type predictions
- `position_offsets`: Coordinate adjustments

### AtomicSystemCollator
Batches protein structures with optional masking/noise.

**Parameters**:
- `tokenizer`: For mask tokens
- `mask_rate`: Fraction of atoms to mask
- `noise_rate`: Fraction of positions to noise
- `noise_width`: Noise standard deviation

**Methods**:
- `__call__(batch)`: Create batched data with masking/noise

**Returns**:
PyG Batch with:
- Node features
- Edge connectivity  
- Masking indicators
- Original values for masked items

### PDBReader
Efficiently parses PDB files to extract atomic coordinates and metadata using BioPandas.

**Parameters**:
- `deprotonate`: Remove hydrogen atoms from structures if True

**Methods**:
`read(pdb_path)`: Read single PDB file  
Returns:
- `positions`: [N,3] atomic coordinates 
- `atom_names`: Full atom names (CA, CB, etc)
- `atoms`: Base elements (C, N, etc) 
- `atom_types`: ATOM/HETATM records

`read_dir(pdb_dir)`: Iterator over PDB files in directory  
Returns:
- Same as `read()` plus:
- `id`: PDB identifier from filename

**Implementation Notes**:
- Uses BioPandas for efficient parsing
- Concatenates ATOM and HETATM records
- Optional hydrogen filtering during parsing
- Returns dictionary format for easy integration