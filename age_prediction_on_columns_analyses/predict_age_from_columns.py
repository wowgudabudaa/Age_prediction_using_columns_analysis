import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict
from captum.attr import Saliency
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_DIR = '/Volumes/newJetStor/newJetStor/paros//paros_WORK/hanwen/ad_decode_test/output/input_columns'
SUBJ_CSV = "/Volumes/newJetStor/newJetStor/paros//paros_WORK/hanwen/ad_decode_test/code/subject_ids.csv"  # contains 'subj_id' column
AGE_CSV = "/Volumes/newJetStor/newJetStor/paros//paros_WORK/hanwen/ad_decode_test/code/ages.csv"          # contains age values in same order as SUBJ_CSV
HEMIS = ["lh", "rh"]
REGION_NAMES = ['bankssts',
'caudalanteriorcingulate',
'caudalmiddlefrontal',
'cuneus',
'entorhinal',
'fusiform',
'inferiorparietal',
'inferiortemporal',
'isthmuscingulate',
'lateraloccipital',
'lateralorbitofrontal',
'lingual',
'medialorbitofrontal',
'middletemporal',
'parahippocampal',
'paracentral',
'parsopercularis',
'parsorbitalis',
'parstriangularis',
'pericalcarine',
'postcentral',
'posteriorcingulate',
'precentral',
'precuneus',
'rostralanteriorcingulate',
'rostralmiddlefrontal',
'superiorfrontal',
'superiorparietal',
'superiortemporal',
'supramarginal',
'frontalpole',
'temporalpole',
'transversetemporal',
'insula'
]
EXPORT_DIR = "/Volumes/newJetStor/newJetStor/paros//paros_WORK/hanwen/ad_decode_test/output/predict_age_from_column_result"
TOP_K_PATCHES = 10
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------- Dataset ----------
class CorticalDataset(Dataset):
    def __init__(self, base_dir, subj_csv, age_csv):
        subj_df = pd.read_csv(subj_csv)
        age_df = pd.read_csv(age_csv)
        self.subject_ids = subj_df['subj_id'].astype(str).str.zfill(4).tolist()
        self.labels = age_df.values.squeeze().astype(np.float32)
        self.data = []

        for subj_id in self.subject_ids:
            subj_data = []
            subj_dir = os.path.join(base_dir, f"S0{subj_id}_columns_1mm")
            for hemi in HEMIS:
                for region in REGION_NAMES:
                    filename = f"S0{subj_id}_{hemi}_{region}_cols_md.csv"
                    filepath = os.path.join(subj_dir, filename)
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"Missing: {filepath}")
                    mat = pd.read_csv(filepath, header=None).values.astype(np.float32)
                    subj_data.append(mat)
            self.data.append(subj_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        brain_data = self.data[idx]
        tensor_data = [torch.tensor(region, dtype=torch.float32) for region in brain_data]
        return tensor_data, torch.tensor(self.labels[idx], dtype=torch.float32)

# ---------- Model ----------
class RegionLevelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.column_encoder = nn.Sequential(
            nn.Conv1d(21, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(34 * 2 * 32, 1)  # 34 regions × 2 hemis × 32 features

    def forward(self, *regions):
        region_features = []
        for region in regions:
            # Convert single sample to list to maintain 3D tensor after padding
            if isinstance(region, torch.Tensor):
                region = [region]

            padded = torch.nn.utils.rnn.pad_sequence(region, batch_first=True)
            padded = padded.permute(0, 2, 1)  # Now guaranteed to be 3D
            encoded = self.column_encoder(padded).squeeze(-1)
            region_features.append(encoded)

            print(padded.shape)  # Should be [batch_size, 21, num_columns]
            assert len(padded.shape) == 3, f"Expected 3D tensor, got {padded.shape}"

        features = torch.cat(region_features, dim=1)
        return self.fc(features).squeeze(1)

# ---------- Training & Eval ----------
def train(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x = [[r.to(DEVICE) for r in brain] for brain in x]
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(*x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = [[r.to(DEVICE) for r in brain] for brain in x]
            y = y.to(DEVICE)
            pred = model(*x)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    return mae, rmse, r2

# ---------- Saliency ----------
def compute_attributions(model, dataset):
    model.eval()
    saliency = Saliency(model)
    patch_attr = {}

    for idx in range(len(dataset)):
        x, _ = dataset[idx]
        # Convert list of tensors to a tuple of tensors
        x_tuple = tuple(r.to(DEVICE).requires_grad_() for r in x)
        attr = saliency.attribute(x_tuple)  # Pass as tuple
        patch_attr[idx] = [a.squeeze(0).abs().cpu().numpy() for a in attr]
    return patch_attr

def visualize_saliency(attr_dict, save_prefix):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    for brain_idx, region_attrs in attr_dict.items():
        for region_idx, region_attr in enumerate(region_attrs):
            if region_attr.ndim == 2:
                saliency_map = region_attr.sum(axis=0)
                plt.plot(saliency_map)
                plt.title(f"Brain {brain_idx} - Region {region_idx}")
                plt.xlabel("Layer")
                plt.ylabel("Attribution")
                plt.savefig(f"{save_prefix}/brain_{brain_idx}_region_{region_idx}.png")
                plt.close()

# ---------- Nested CV ----------
def nested_cv(dataset):
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    maes, rmses, r2s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(dataset)):
        print(f"Fold {fold+1}")
        model = RegionLevelModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        for epoch in range(EPOCHS):
            train(model, train_loader, optimizer, criterion)

        mae, rmse, r2 = evaluate(model, test_loader)
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        if fold == 0:
            patch_attr = compute_attributions(model, dataset)
            visualize_saliency(patch_attr, EXPORT_DIR)

    return maes, rmses, r2s

def collate_fn(batch):
    regions = list(zip(*[sample[0] for sample in batch]))
    targets = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    return [[r for r in region] for region in regions], targets

# ---------- Main ----------
def main():
    dataset = CorticalDataset(DATA_DIR, SUBJ_CSV, AGE_CSV)
    cnn_maes, cnn_rmses, cnn_r2s = nested_cv(dataset)

    with open(f"{EXPORT_DIR}/summary.txt", "w") as f:
        f.write(f"CNN MAE: {np.mean(cnn_maes):.2f} ± {np.std(cnn_maes):.2f}\n")
        f.write(f"CNN RMSE: {np.mean(cnn_rmses):.2f}\n")
        f.write(f"CNN R²: {np.mean(cnn_r2s):.2f}\n")

if __name__ == "__main__":
    main()
