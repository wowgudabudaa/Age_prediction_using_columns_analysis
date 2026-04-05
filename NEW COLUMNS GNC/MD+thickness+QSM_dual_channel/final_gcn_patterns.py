import torch
import numpy as np
from torch_geometric.data import Batch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader
import pandas as pd
import os


class BrainAgeGNN(nn.Module):
    def __init__(self, input_dim_md, input_dim_qsm):
        super(BrainAgeGNN, self).__init__()

        # GCN layers for MD
        self.md_conv1 = GCNConv(input_dim_md, 64)
        self.md_bn1 = BatchNorm(64)
        self.md_conv2 = GCNConv(64, 128)
        self.md_bn2 = BatchNorm(128)
        self.md_conv3 = GCNConv(128, 128)
        self.md_bn3 = BatchNorm(128)

        # GCN layers for QSM
        self.qsm_conv1 = GCNConv(input_dim_qsm, 64)
        self.qsm_bn1 = BatchNorm(64)
        self.qsm_conv2 = GCNConv(64, 128)
        self.qsm_bn2 = BatchNorm(128)
        self.qsm_conv3 = GCNConv(128, 128)
        self.qsm_bn3 = BatchNorm(128)

        self.dropout = nn.Dropout(p=0.5)

        # Final MLP
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data_md, data_qsm, data_thickness=None):
        # MD stream
        x_md = F.relu(self.md_bn1(self.md_conv1(data_md.x, data_md.edge_index)))
        x_md = F.relu(self.md_bn2(self.md_conv2(x_md, data_md.edge_index)))
        x_md_res = x_md
        x_md = F.relu(self.md_bn3(self.md_conv3(x_md, data_md.edge_index)))
        x_md = x_md + x_md_res
        x_md = global_mean_pool(x_md, data_md.batch)
        x_md = self.dropout(x_md)

        # QSM stream
        x_qsm = F.relu(self.qsm_bn1(self.qsm_conv1(data_qsm.x,
                                                   data_qsm.edge_index)))
        x_qsm = F.relu(self.qsm_bn2(self.qsm_conv2(x_qsm,
                                                   data_qsm.edge_index)))
        x_qsm_res = x_qsm
        x_qsm = F.relu(self.qsm_bn3(self.qsm_conv3(x_qsm,
                                                   data_qsm.edge_index)))
        x_qsm = x_qsm + x_qsm_res
        x_qsm = global_mean_pool(x_qsm, data_qsm.batch)
        x_qsm = self.dropout(x_qsm)

        # Combine MD, QSM, and thickness
        if data_thickness is not None:
            thickness = data_thickness  # already [B, R]
            x = torch.cat([x_md, x_qsm, thickness], dim=1)
        else:
            x = torch.cat([x_md, x_qsm], dim=1)

        return self.fc(x)


def collate_fn_aligend(batch):
    md_data, qsm_data, _ = zip(*batch)  # we’ll re-fetch thickness below

    md_batch = Batch.from_data_list(md_data)
    qsm_batch = Batch.from_data_list(qsm_data)

    # Stack thickness from md_data (which now contains `.thickness`)
    thickness_tensor = torch.stack([d.thickness for d in md_data])

    return md_batch, qsm_batch, thickness_tensor


def extract_last_layer_patterns(model, loader, device='cuda'):
    """
    Extracts node-level and graph-level patterns from the last GCN layer
    of both MD and QSM streams.

    Args:
        model (nn.Module): Trained BrainAgeGNN model.
        loader (DataLoader): PyG DataLoader yielding (data_md, data_qsm).
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: {
            'md_node': list of torch.Tensor (node features per graph),
            'md_graph': np.ndarray (pooled graph features, shape [n_subjects, 128]),
            'qsm_node': list of torch.Tensor,
            'qsm_graph': np.ndarray,
            'thickness': np.ndarray (if provided)
        }
    """
    model.eval()
    md_node_feats = []      # node-level features for each subject (list of tensors)
    qsm_node_feats = []
    md_graph_feats = []     # pooled graph-level features
    qsm_graph_feats = []

    with torch.no_grad():
        for data_md, data_qsm in loader:
            # Move to device
            data_md = data_md.to(device)
            data_qsm = data_qsm.to(device)

            # ---- MD stream (last conv output before pooling) ----
            x_md = F.relu(model.md_bn1(model.md_conv1(data_md.x, data_md.edge_index)))
            x_md = F.relu(model.md_bn2(model.md_conv2(x_md, data_md.edge_index)))
            x_md_res = x_md
            x_md = F.relu(model.md_bn3(model.md_conv3(x_md, data_md.edge_index)))
            x_md = x_md + x_md_res          # node-level features [total_nodes, 128]
            # Store node-level features per graph (split by batch)
            node_mask = data_md.batch
            for i in range(node_mask.max().item() + 1):
                md_node_feats.append(x_md[node_mask == i].cpu())   # keep as tensor
            # Pooled graph-level features
            md_pooled = global_mean_pool(x_md, data_md.batch)      # [batch_size, 128]
            md_graph_feats.append(md_pooled.cpu())

            # ---- QSM stream (last conv output before pooling) ----
            x_qsm = F.relu(model.qsm_bn1(model.qsm_conv1(data_qsm.x, data_qsm.edge_index)))
            x_qsm = F.relu(model.qsm_bn2(model.qsm_conv2(x_qsm, data_qsm.edge_index)))
            x_qsm_res = x_qsm
            x_qsm = F.relu(model.qsm_bn3(model.qsm_conv3(x_qsm, data_qsm.edge_index)))
            x_qsm = x_qsm + x_qsm_res
            # Node-level
            qsm_node_mask = data_qsm.batch
            for i in range(qsm_node_mask.max().item() + 1):
                qsm_node_feats.append(x_qsm[qsm_node_mask == i].cpu())
            # Pooled graph-level
            qsm_pooled = global_mean_pool(x_qsm, data_qsm.batch)    # [batch_size, 128]
            qsm_graph_feats.append(qsm_pooled.cpu())

    # Concatenate across batches
    md_graph = torch.cat(md_graph_feats, dim=0).numpy()   # [n_subjects, 128]
    qsm_graph = torch.cat(qsm_graph_feats, dim=0).numpy()

    return {
        'md_node': md_node_feats,  # list of tensors, each shape [n_nodes_i, 128]
        'md_graph': md_graph,  # numpy array
        'qsm_node': qsm_node_feats,
        'qsm_graph': qsm_graph,
    }


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    save_dir = "extracted_patterns"
    os.makedirs(save_dir, exist_ok=True)

    # Load the graph data lists from the saved files for later use
    graph_data_list_md = torch.load("processed_graph_data/graph_data_list_md.pt")
    graph_data_list_QSM = torch.load("processed_graph_data/graph_data_list_QSM.pt")
    # Create lookup dictionary from subject IDs to their graphs
    md_dict = {data.subject_id: data for data in graph_data_list_md}
    qsm_dict = {data.subject_id: data for data in graph_data_list_QSM}

    # Build aligned list of MD, QSM graphs and thickness
    aligned_graph_list = [
        (md_dict[sid], qsm_dict[sid])
        for sid in md_dict if sid in qsm_dict
    ]

    # Assume you have a trained model, a DataLoader, and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    k = 7
    batch_size = 5
    repeats_per_fold = 10

    for fold in range(k):
        print(f"Processing fold {fold + 1}/{k}...")
        train_idx = np.loadtxt(f"../set_split/train_indices_fold_{fold}.csv",
                               delimiter=",").astype(np.int64)
        test_idx = np.loadtxt(f"../set_split/test_indices_fold_{fold}.csv",
                              delimiter=",").astype(np.int64)

        train_data = [aligned_graph_list[i] for i in train_idx]
        test_data = [aligned_graph_list[i] for i in test_idx]

        for rep in range(repeats_per_fold):

            # Initialize model (with correct input dimensions)
            model = BrainAgeGNN(input_dim_md=22, input_dim_qsm=21)  # e.g., 90 ROIs
            model.load_state_dict(torch.load(f"column_md_model_fold_{fold+1}_rep_{rep+1}_ref.pt"))  # Load saved model
            model = model.to(device)

            # Your dataloader yields (data_md, data_qsm, thickness)
            # thickness can be None if not used
            loader = DataLoader(test_data, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn_aligend)

            # Extract patterns
            patterns = extract_last_layer_patterns(model, loader, device)

            # Save to disk for later analysis
            np.save(f'{save_dir}/md_graph_patterns_fold_{fold+1}_rep_{rep+1}.npy', patterns['md_graph'])      # shape (n_subjects, 128)
            np.save(f'{save_dir}/qsm_graph_patterns_fold_{fold+1}_rep_{rep+1}.npy', patterns['qsm_graph'])

            # Node-level patterns can be saved per subject (e.g., as a list of arrays)
            # Example: save first subject's MD node features
            # torch.save(patterns['md_node'][0], f'{save_dir}/subj0_md_node.pt')

        # compute averege graph-level patterns across repeats for this fold
        md_graphs = []
        qsm_graphs = []
        for rep in range(repeats_per_fold):
            md_graphs.append(np.load(f'{save_dir}/md_graph_patterns_fold_{fold+1}_rep_{rep+1}.npy'))
            qsm_graphs.append(np.load(f'{save_dir}/qsm_graph_patterns_fold_{fold+1}_rep_{rep+1}.npy'))

        avg_md_graph = np.mean(md_graphs, axis=0)  # shape (n_subjects, 128)
        avg_qsm_graph = np.mean(qsm_graphs, axis=0)

        np.save(f'{save_dir}/avg_md_graph_patterns_fold_{fold+1}.npy', avg_md_graph)
        np.save(f'{save_dir}/avg_qsm_graph_patterns_fold_{fold+1}.npy', avg_qsm_graph)

    # Build patterns summary for all subjects
    subject_ids = pd.read_csv("healthy_familial_subject_ids.csv", header=None, dtype=str)[0].tolist()
    patterns_summary = pd.DataFrame(0.0, index=subject_ids,
                                    columns=[f'md_pattern_{i}' for i in range(128)]
                                    + [f'qsm_pattern_{i}' for i in range(128)])
    md_summary = pd.DataFrame(0.0, index=subject_ids, columns=[f'md_pattern_{i}' for i in range(128)])
    qsm_summary = pd.DataFrame(0.0, index=subject_ids, columns=[f'qsm_pattern_{i}' for i in range(128)])

    for fold in range(k):
        test_idx  = np.loadtxt(f"../set_split/test_indices_fold_{fold}.csv",
                               delimiter=",").astype(np.int64)
        subj_ids = [subject_ids[i] for i in test_idx]
        avg_md_graph = np.load(f'{save_dir}/avg_md_graph_patterns_fold_{fold+1}.npy')  # shape (n_subjects, 128)
        avg_qsm_graph = np.load(f'{save_dir}/avg_qsm_graph_patterns_fold_{fold+1}.npy')

        for i, sid in enumerate(subj_ids):
            patterns_summary.loc[sid, [f'md_pattern_{j}' for j in range(128)]] += avg_md_graph[i]
            patterns_summary.loc[sid, [f'qsm_pattern_{j}' for j in range(128)]] += avg_qsm_graph[i]
            md_summary.loc[sid, [f'md_pattern_{j}' for j in range(128)]] += avg_md_graph[i]
            qsm_summary.loc[sid, [f'qsm_pattern_{j}' for j in range(128)]] += avg_qsm_graph[i]

    patterns_summary.to_csv(f'{save_dir}/final_graph_patterns_summary.csv')
    md_summary.to_csv(f'{save_dir}/md_graph_patterns_summary.csv')
    qsm_summary.to_csv(f'{save_dir}/qsm_graph_patterns_summary.csv')

    print("Extraction completed. Graph-level patterns saved.")
