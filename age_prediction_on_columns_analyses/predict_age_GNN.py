import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
import torch.nn.functional as F


class BrainGraphDataset(Dataset):
    def __init__(self, subj_ids, base_dir, base_dir2, hemi_list, region_list, transform=None):
        super().__init__()
        self.subj_ids = subj_ids
        self.base_dir = base_dir
        self.base_dir2 = base_dir2
        self.hemi_list = hemi_list
        self.region_list = region_list
        self.metadata = pd.read_csv(os.path.join(base_dir2, 'brain_metadata.csv'))
        self.transform = transform

    def process_subject(self, subj_id):
        node_features = []
        node_coords = []

        for hemi in self.hemi_list:
            for region in self.region_list:
                md_path = os.path.join(self.base_dir, f"S0{subj_id}/columns_1mm/S0{subj_id}_{hemi}_{region}_cols_md.csv")
                coord_path = os.path.join(self.base_dir, f"S0{subj_id}/label_coord_1mm/{hemi}_{region}.csv")
                if not os.path.exists(md_path) or not os.path.exists(coord_path):
                    continue

                md = np.loadtxt(md_path, delimiter=',')  # shape: num_columns x 21
                coords = np.loadtxt(coord_path, delimiter=',')  # shape: 4 x num_points

                for layer in range(md.shape[1]):
                    layer_values = md[:, layer]  # (num_columns,)
                    start_idx = layer * md.shape[0]
                    end_idx = (layer + 1) * md.shape[0]
                    layer_coords = coords[:3, start_idx:end_idx]  # shape: 3 x num_columns

                    avg_coord = np.mean(layer_coords, axis=1)  # shape: (3,)
                    node_coords.append(avg_coord)

                    node_features.append([
                        np.mean(layer_values),
                        np.std(layer_values)
                    ])

        if not node_features:
            return None

        x = torch.tensor(node_features, dtype=torch.float)
        coord_tensor = torch.tensor(node_coords, dtype=torch.float)
        num_nodes = x.shape[0]

        dists = torch.cdist(coord_tensor, coord_tensor, p=2)
        knn = torch.topk(dists, k=min(5, num_nodes), largest=False)
        edge_index = knn.indices.t()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_weight = 1 / (dists[edge_index[0], edge_index[1]] + 1e-8)

        subj_meta = self.metadata[self.metadata['brain_id'] == int(subj_id)]
        y = torch.tensor(subj_meta['age'].values[0], dtype=torch.float).unsqueeze(0)

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

    def len(self):
        return len(self.subj_ids)

    def get(self, idx):
        subj_id = self.subj_ids[idx]
        data = self.process_subject(subj_id)
        return data


class BrainAgeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        self.embeddings = x.clone()
        x = global_mean_pool(x, batch)
        x = self.lin1(x).relu()
        return self.lin2(x)


# ==== Training Setup ====
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            all_preds.append(out.item())
            all_targets.append(data.y.item())
    r2 = r2_score(all_targets, all_preds)
    return total_loss / len(loader), r2


def extract_saliency(model, loader):
    model.eval()
    node_saliency = []
    with torch.no_grad():
        for data in loader:
            _ = model(data.x, data.edge_index, data.edge_weight, data.batch)
            emb = model.embeddings
            saliency = emb.norm(dim=1)
            node_saliency.append(saliency.cpu().numpy())
    return node_saliency

def main():
    # ==== Load Data IDs ====
    base_dir = '/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/output/'
    base_dir2 = '/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/code/'
    low_risk_ids = pd.read_csv(os.path.join(base_dir2, 'low_risk_ids.csv'), header=None)[0].astype(str).tolist()
    high_risk_ids = pd.read_csv(os.path.join(base_dir2, 'high_risk_ids.csv'), header=None)[0].astype(str).tolist()

    hemi_list = ['lh', 'rh']
    region_list = ['region1', 'region2', 'region3']  # replace with actual region names

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    losses = []
    r2_scores = []
    all_saliency = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(low_risk_ids)):
        train_ids = [low_risk_ids[i] for i in train_idx]
        val_ids = [low_risk_ids[i] for i in val_idx]
        test_ids = [high_risk_ids[i] for i in val_idx]

        train_dataset = BrainGraphDataset(train_ids, base_dir, base_dir2, hemi_list, region_list)
        val_dataset = BrainGraphDataset(val_ids, base_dir, base_dir2, hemi_list, region_list)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)

        model = BrainAgeGNN(input_dim=2, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        val_loss_fold = 0
        val_r2_fold = 0

        for epoch in range(20):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, val_r2 = evaluate(model, val_loader, criterion)
            print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {val_r2:.4f}")
            if epoch == 20:
                val_loss_fold = val_loss
                val_r2_fold = val_r2

        losses.append(val_loss_fold)
        r2_scores.append(val_r2_fold)
        fold_saliency = extract_saliency(model, val_loader)
        all_saliency.extend(fold_saliency)

    print("Average Cross-Validation Loss:", np.mean(losses))
    print("Average Cross-Validation R2:", np.mean(r2_scores))

    print("Sample saliency scores from last fold (first subject):", all_saliency[-1])

if __name__ == "__main__":
    main()