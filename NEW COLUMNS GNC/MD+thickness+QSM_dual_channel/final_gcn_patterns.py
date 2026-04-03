import torch
import numpy as np
from torch_geometric.data import Batch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm


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


def extract_last_layer_patterns(model, loader, device='cuda'):
    """
    Extracts node-level and graph-level patterns from the last GCN layer
    of both MD and QSM streams.

    Args:
        model (nn.Module): Trained BrainAgeGNN model.
        loader (DataLoader): PyG DataLoader yielding (data_md, data_qsm, thickness).
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
    thickness_list = []

    with torch.no_grad():
        for data_md, data_qsm, thickness in loader:
            # Move to device
            data_md = data_md.to(device)
            data_qsm = data_qsm.to(device)
            if thickness is not None:
                thickness = thickness.to(device)

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

            # Thickness (if any)
            if thickness is not None:
                thickness_list.append(thickness.cpu())

    # Concatenate across batches
    md_graph = torch.cat(md_graph_feats, dim=0).numpy()   # [n_subjects, 128]
    qsm_graph = torch.cat(qsm_graph_feats, dim=0).numpy()
    thickness_arr = torch.cat(thickness_list, dim=0).numpy() if thickness_list else None

    return {
        'md_node': md_node_feats,  # list of tensors, each shape [n_nodes_i, 128]
        'md_graph': md_graph,  # numpy array
        'qsm_node': qsm_node_feats,
        'qsm_graph': qsm_graph,
        'thickness': thickness_arr
    }

# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Assume you have a trained model, a DataLoader, and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model (with correct input dimensions)
    model = BrainAgeGNN(input_dim_md=90, input_dim_qsm=90)  # e.g., 90 ROIs
    model.load_state_dict(torch.load('your_model.pth'))
    model = model.to(device)
    
    # Your dataloader yields (data_md, data_qsm, thickness)
    # thickness can be None if not used
    loader = ...   # e.g., torch_geometric.loader.DataLoader
    
    # Extract patterns
    patterns = extract_last_layer_patterns(model, loader, device)
    
    # Save to disk for later analysis
    np.save('md_graph_patterns.npy', patterns['md_graph'])      # shape (n_subjects, 128)
    np.save('qsm_graph_patterns.npy', patterns['qsm_graph'])
    if patterns['thickness'] is not None:
        np.save('thickness_values.npy', patterns['thickness'])
    
    # Node-level patterns can be saved per subject (e.g., as a list of arrays)
    # Example: save first subject's MD node features
    # torch.save(patterns['md_node'][0], 'subj0_md_node.pt')
    
    print("Extraction completed. Graph-level patterns saved.")