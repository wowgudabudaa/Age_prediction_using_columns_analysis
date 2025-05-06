import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence  # Add to imports
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict
from captum.attr import Saliency, IntegratedGradients
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_DIR = '/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/output/input_columns'
SUBJ_CSV = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/code/subject_ids.csv"
AGE_CSV = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/code/ages.csv"
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
EXPORT_DIR = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/output/predict_age_from_column_result"
TOP_K_PATCHES = 10
BATCH_SIZE = 8
EPOCHS = 20
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------- Dataset (unchanged) ----------
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
                    mat = pd.read_csv(filepath, header=None).values.astype(np.float32).flatten()
                    subj_data.append(mat)
            self.data.append(subj_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        brain_data = self.data[idx]
        tensor_data = [torch.tensor(region, dtype=torch.float32) for region in brain_data]
        return tensor_data, torch.tensor(self.labels[idx], dtype=torch.float32)

# ---------- Improved Model ----------
class RegionLevelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(21, 32, kernel_size=3, padding=1),
                nn.InstanceNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),  # Additional layer
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ) for _ in REGION_NAMES
        ])
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in REGION_NAMES
        ])
        self.fc = nn.Linear(len(REGION_NAMES) * 2 * 32, 1)

    def forward(self, *regions):
        features = []
        attn_weights = []
        for i, region in enumerate(regions):
            region_idx = i % len(REGION_NAMES)
            encoder = self.region_encoders[region_idx]
            attn = self.attention[region_idx]

            padded = pad_sequence(region, batch_first=True).permute(0, 2, 1)
            encoded = encoder(padded).squeeze(-1)
            weight = attn(encoded)

            # Ensure attention weights are 2D before stacking
            features.append(encoded * weight)
            attn_weights.append(weight.unsqueeze(1))  # Shape: (batch_size, 1)

        features = torch.cat(features, dim=1)
        # Stack along dim=1 and remove last dimension
        attn_matrix = torch.cat(attn_weights, dim=1)  # Shape: (batch_size, 68)
        return self.fc(features).squeeze(1), attn_matrix

# ---------- Training & Eval ----------
def train(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x = [[r.to(DEVICE) for r in brain] for brain in x]
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(*x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    preds, trues, attns = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = [[r.to(DEVICE) for r in brain] for brain in x]
            y = y.to(DEVICE)
            pred, attn = model(*x)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
            attns.append(attn.cpu())
    attns = torch.cat(attns).mean(0).tolist()
    return mean_absolute_error(trues, preds), np.sqrt(mean_squared_error(trues, preds)), r2_score(trues, preds), attns


# ---------- Attribution & Grad-CAM ----------
def compute_attributions(model, dataset):
    ig = IntegratedGradients(model)
    attr_dict = defaultdict(list)

    # Create wrapper to only return predictions
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *inputs):
            pred, _ = self.model(*inputs)
            return pred

    wrapped_model = ModelWrapper(model).to(DEVICE)
    ig = IntegratedGradients(wrapped_model)

    for idx in dataset.indices:  # Only test set samples
        x, _ = dataset.dataset[idx]  # Access original dataset via Subset
        x_tuple = tuple(r.to(DEVICE).unsqueeze(0) for r in x)
        baseline = tuple(torch.zeros_like(r).to(DEVICE) for r in x_tuple)

        # For regression, target=None
        attr = ig.attribute(x_tuple, baseline, target=None)

        summed_attrs = [a.squeeze().abs().sum(0).cpu().numpy() for a in attr]
        attr_dict[idx] = summed_attrs

    return attr_dict

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
    region_attns = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(dataset)):
        model = RegionLevelModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.HuberLoss()

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        for epoch in range(EPOCHS):
            train(model, train_loader, optimizer, criterion)

        mae, rmse, r2, attns = evaluate(model, test_loader)
        region_attns.append(attns)
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        # In nested_cv():
        if fold == 0:
            # Use test_set Subset directly
            attr_dict = compute_attributions(model, test_set)
            visualize_saliency(attr_dict, EXPORT_DIR)

    region_importance = np.mean(region_attns, axis=0)
    print("Region Importance:", region_importance)
    # After region_importance calculation:
    hemi_merged_importance = [
        (region_importance[i] + region_importance[i + 34]) / 2
        for i in range(34)
    ]
    print("Merged Region Importance:", hemi_merged_importance)
    return maes, rmses, r2s

def collate_fn(batch):
    regions = list(zip(*[sample[0] for sample in batch]))
    targets = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    return [[r for r in region] for region in regions], targets

# ---------- Main (unchanged) ----------
def main():
    dataset = CorticalDataset(DATA_DIR, SUBJ_CSV, AGE_CSV)
    cnn_maes, cnn_rmses, cnn_r2s = nested_cv(dataset)

    with open(f"{EXPORT_DIR}/summary.txt", "w") as f:
        f.write(f"CNN MAE: {np.mean(cnn_maes):.2f} ± {np.std(cnn_maes):.2f}\n")
        f.write(f"CNN RMSE: {np.mean(cnn_rmses):.2f}\n")
        f.write(f"CNN R²: {np.mean(cnn_r2s):.2f}\n")


if __name__ == "__main__":
    main()