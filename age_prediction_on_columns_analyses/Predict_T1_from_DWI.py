import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------- UNET 3D Model ---------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature*2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](x)
        return self.final_conv(x)

# --------- Dataset ---------
class DWI_T1_Dataset(Dataset):
    def __init__(self, dwi_paths, t1_paths):
        self.dwi_paths = dwi_paths
        self.t1_paths = t1_paths

    def __len__(self):
        return len(self.dwi_paths)

    def __getitem__(self, idx):
        dwi = nib.load(self.dwi_paths[idx]).get_fdata()
        t1 = nib.load(self.t1_paths[idx]).get_fdata()

        dwi = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
        t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)

        dwi = np.expand_dims(dwi.astype(np.float32), axis=0)
        t1 = np.expand_dims(t1.astype(np.float32), axis=0)

        return torch.from_numpy(dwi), torch.from_numpy(t1)

# --------- Utils ---------
def save_nifti(volume, reference_path, output_path):
    ref_img = nib.load(reference_path)
    new_img = nib.Nifti1Image(volume.squeeze(), affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, output_path)

def evaluate_prediction(pred, target):
    pred = pred.detach().cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()
    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target)**2)
    return mae, mse

def show_slices(dwi, pred_t1, true_t1, slice_idx=None):
    dwi = dwi[0, 0].cpu().numpy()
    pred = pred_t1[0, 0].detach().cpu().numpy()
    true = true_t1[0, 0].cpu().numpy()
    if slice_idx is None:
        slice_idx = dwi.shape[0] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Input DWI")
    plt.imshow(dwi[slice_idx], cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Predicted T1")
    plt.imshow(pred[slice_idx], cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Ground Truth T1")
    plt.imshow(true[slice_idx], cmap="gray"); plt.axis("off")
    plt.tight_layout(); plt.show()

# --------- Main ---------
if __name__ == "__main__":
    # Replace with your actual paths
    dwi_dir = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/input/paired_DWI"
    t1_dir = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/input/paired_T1"
    dwi_only_dir = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/input/DWI_only"
    output_dir = "/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/input/predicted_T1"

    dwi_paths = sorted([os.path.join(dwi_dir, f) for f in os.listdir(dwi_dir) if (f.endswith(".nii") or f.endswith(".nii.gz")) and not f.startswith("._")])
    t1_paths = sorted([os.path.join(t1_dir, f) for f in os.listdir(t1_dir) if (f.endswith(".nii") or f.endswith(".nii.gz")) and not f.startswith("._")])
    dwi_only_paths = sorted([os.path.join(dwi_only_dir, f) for f in os.listdir(dwi_only_dir) if (f.endswith(".nii") or f.endswith(".nii.gz")) and not f.startswith("._")])

    dataset = DWI_T1_Dataset(dwi_paths, t1_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --------- Train ---------
    print("Training model...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for dwi, t1 in dataloader:
            dwi, t1 = dwi.to(device), t1.to(device)
            optimizer.zero_grad()
            pred = model(dwi)
            loss = criterion(pred, t1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                show_slices(dwi, pred, t1)
        print("Epoch Done:", epoch+1)

    # --------- Predict and Save for DWI-only ---------
    print("\nGenerating synthetic T1s for DWI-only and T1 subjects...")
    model.eval()
    with torch.no_grad():
        for path in dwi_only_paths:
            dwi = nib.load(path).get_fdata()
            dwi = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
            dwi = np.expand_dims(dwi.astype(np.float32), axis=(0, 1))
            dwi_tensor = torch.from_numpy(dwi).to(device)

            pred_t1 = model(dwi_tensor)
            pred_np = pred_t1.cpu().numpy()

            fname = os.path.basename(path).replace("dwi", "predicted_t1").replace(".nii.gz", "_T1.nii.gz")
            out_path = os.path.join(output_dir, fname)
            save_nifti(pred_np, path, out_path)
            print(f"Saved: {out_path}")
        for path in dwi_dir:
            dwi = nib.load(path).get_fdata()
            dwi = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
            dwi = np.expand_dims(dwi.astype(np.float32), axis=(0, 1))
            dwi_tensor = torch.from_numpy(dwi).to(device)

            pred_t1 = model(dwi_tensor)
            pred_np = pred_t1.cpu().numpy()

            fname = os.path.basename(path).replace("dwi", "predicted_t1").replace(".nii.gz", "_T1.nii.gz")
            out_path = os.path.join(output_dir, fname)
            save_nifti(pred_np, path, out_path)
            print(f"Saved: {out_path}")

    # --------- Evaluate on Paired Set ---------
    print("\nEvaluating model on paired DWI–T1 cases...")
    mae_list, mse_list = [], []
    with torch.no_grad():
        for dwi, t1 in dataloader:
            dwi, t1 = dwi.to(device), t1.to(device)
            pred = model(dwi)
            mae, mse = evaluate_prediction(pred, t1)
            mae_list.append(mae)
            mse_list.append(mse)

    print(f"\n Evaluation complete:")
    print(f"Mean MAE: {np.mean(mae_list):.4f}")
    print(f"Mean MSE: {np.mean(mse_list):.4f}")