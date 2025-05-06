import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import GNNExplainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------- Config ----------
FEATURE_CSV = "column_features.csv"           # columns: brain_id, column_id, fa_0...fa_20
COORD_CSV = "column_coords.csv"               # columns: brain_id, column_id, x, y, z
METADATA_CSV = "brain_metadata.csv"           # columns: brain_id, age, sex, genotype, bmi
EXPORT_DIR = "gnn_results"
K = 8  # k-nearest neighbors
SEED = 42
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3

# ---------- Dataset ----------
class BrainGraphDataset(InMemoryDataset):
    def __init__(self, feature_csv, coord_csv, metadata_csv):
        super().__init__()
        self.df_feat = pd.read_csv(feature_csv)
        self.df_coords = pd.read_csv(coord_csv)
        self.df_meta = pd.read_csv(metadata_csv)

        self.scaler = StandardScaler()
        self.label_encoders = {}

        self.graphs = self.build_graphs()
        self.data, self.slices = self.collate(self.graphs)

    def build_graphs(self):
        graphs = []
        for brain_id in self.df_feat['brain_id'].unique():
            X = self.df_feat[self.df_feat['brain_id'] == brain_id].sort_values('column_id')
            coords = self.df_coords[self.df_coords['brain_id'] == brain_id].sort_values('column_id')
            x = X.drop(columns=['brain_id', 'column_id']).values
            pos = coords[['x', 'y', 'z']].values

            # Scale features
            x = self.scaler.fit_transform(x)
            x = torch.tensor(x, dtype=torch.float32)
            pos = torch.tensor(pos, dtype=torch.float32)

            # Create edges using kNN
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=K+1).fit(pos)
            _, indices = nbrs.kneighbors(pos)
            edge_index = []
            for i, neighbors in enumerate(indices):
                for j in neighbors[1:]:
                    edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Metadata
            meta = self.df_meta[self.df_meta['brain_id'] == brain_id].iloc[0]
            age = torch.tensor([meta['age']], dtype=torch.float32)

            # Encode categorical metadata and attach as graph-level features
            meta_features = []
            for col in ['sex', 'genotype']:
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    self.label_encoders[col] = le.fit(self.df_meta[col])
                encoded = self.label_encoders[col].transform([meta[col]])[0]
                meta_features.append(encoded)

            meta_features.append(meta['bmi'])
            meta_tensor = torch.tensor(meta_features, dtype=torch.float32)

            graph = Data(x=x, edge_index=edge_index, pos=pos, y=age, u=meta_tensor)
            graphs.append(graph)
        return graphs

# ---------- GNN Model ----------
class GCNBrainAge(nn.Module):
    def __init__(self, in_channels, meta_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32 + meta_dim, 1)

    def forward(self, data):
        x, edge_index, batch, u = data.x, data.edge_index, data.batch, data.u
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        u = u.view(x.size(0), -1)
        x = torch.cat([x, u], dim=1)
        return self.fc(x).squeeze()

# ---------- Training Loop ----------
def train(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            out = model(data)
            preds.extend(out.cpu().tolist())
            trues.extend(data.y.cpu().tolist())
    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds, squared=False)
    r2 = r2_score(trues, preds)
    return preds, trues, mae, rmse, r2

# ---------- Saliency (GNNExplainer) ----------
def run_explainer(model, dataset):
    model.eval()
    data = dataset[0].to(DEVICE)
    explainer = GNNExplainer(model, epochs=100)
    explanation = explainer.explain_graph(data)
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    return edge_mask

# ---------- Main ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset = BrainGraphDataset(FEATURE_CSV, COORD_CSV, METADATA_CSV)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = GCNBrainAge(in_channels=21, meta_dim=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        train(model, loader, optimizer, criterion)

    preds, trues, mae, rmse, r2 = evaluate(model, loader)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    pd.DataFrame({"y_true": trues, "y_pred": preds}).to_csv(f"{EXPORT_DIR}/brain_predictions.csv", index=False)
    with open(f"{EXPORT_DIR}/summary.txt", "w") as f:
        f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}\n")

    # Explain first graph
    edge_mask = run_explainer(model, dataset)
    np.save(f"{EXPORT_DIR}/gnn_edge_mask.npy", edge_mask)

if __name__ == "__main__":
    main()
