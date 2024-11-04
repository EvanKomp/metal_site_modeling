# metalsitenn

This is the README file for the metalsitenn project.

## Overview

DVC tracked pipeline to train a protein metal functional site foundational model. Used for clustering, downstream prediction, finetuning. The model is equivariant and can output scalars and vectors for each atom in the protein. Pooling over the whole protein can produce a single embedding.

## Dependencies
- `e3nn`
- `torch`
- `torch-geometric`

## Notes

- Featurization can ignore metal identity, this atomic number vocabular contains an indicator for 'metal' atoms.
- Pretraining: masked modeling of atom identities, gaussian noise on atom positions

# Pipeline steps


- `1.1_parse_site_data`: Get some labels and statistics for each metal site in the dataset.
    - __Input__: `data/mf_sites/`, contains chunks of PDB files with metal sites.
    - __Output__: `data/site_labels.csv`, per site labels, `data/metrics/site_label_metrics.json` aggregated metrics

# Components

- `metalsitenn.data.SiteFeaturizer` From PDB files, output a graph with atom features and adjacency matrix.
    - __Params__: `metal_known` whether the metal identity will be retained or replced by "metal" indicator.
    
- `metalssitenn.data.SiteCollater` From a list of graphs/dataset, produce batches with expected information for pretraining. The model will take in positions and node features, and we also need the labels for atom type and positions before we noised them, and the indices where we did, to compute losses.
  - __Params__: `atom_mask_rate`, `position_noise_rate`, `position_noise_width`
  - __Returns__: `node_features`, `node_positions`, `label_features`, `label_positions`, `mask_indices`, `noise_indices`
