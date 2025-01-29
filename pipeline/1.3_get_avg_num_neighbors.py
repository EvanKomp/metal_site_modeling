# pipeline/1.3_get_avg_num_neighbors.py
'''
* Author: Evan Komp
* Created: 1/27/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Determine the average number of neighbors in the training set given a cutoff range.
'''
import json
import torch
from datasets import load_from_disk
from torch_geometric.nn import radius_graph
import dvc.api
from metalsitenn.utils import ParamsObj
from tqdm import tqdm

def get_avg_neighbors(positions: torch.Tensor, max_radius: float, batch: torch.Tensor) -> float:
    """Calculate average neighbors per node.
    
    Args:
        positions: [N, 3] atom coordinates
        max_radius: Cutoff radius
        batch: [N] batch indices
        
    Returns:
        Mean neighbors per node
    """
    edge_index = radius_graph(positions, r=max_radius, batch=batch)
    num_nodes = positions.shape[0]
    return edge_index.shape[1] / num_nodes

def main():
    # Load params
    params = ParamsObj(dvc.api.params_show())
    
    # Load dataset
    dataset = load_from_disk("data/dataset/metal_site_dataset")
    train_dataset = dataset["train"]

    total_neighbors = 0
    total_nodes = 0

    # Process each example
    for example in tqdm(train_dataset):
        pos = torch.tensor(example["pos"])
        batch = torch.zeros(pos.shape[0], dtype=torch.long)
        
        avg = get_avg_neighbors(pos, params.model.max_radius, batch)
        total_neighbors += avg * pos.shape[0]
        total_nodes += pos.shape[0]

    global_avg = total_neighbors / total_nodes

    # Save results
    metrics = {"avg_num_neighbors": float(global_avg)}
    with open("data/training_avg_num_neighbors.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()