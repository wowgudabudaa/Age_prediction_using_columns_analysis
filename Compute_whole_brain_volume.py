#!/usr/bin/env python3
# python "C:\Users\22679\OneDrive - Duke University\桌面\columns_analyse\Brain_column_analyse\Compute_whole_brain_volume.py" ^
#   --masks_dir "C:\Users\22679\OneDrive - Duke University\桌面\columns_analyse\masks" --recursive --no_mm3

"""
Generate whole_brain_volume.csv with non-zero voxel counts (and mm^3 volumes) for mask NIfTI files.
"""

import os
import argparse
from pathlib import Path

import nibabel as nib
import pandas as pd
import numpy as np

def count_nonzero(mask_path, compute_mm3=True):
    img = nib.load(str(mask_path))
    data = img.get_fdata(dtype=np.float32)
    nonzero_voxels = int(np.count_nonzero(data))
    voxel_volume_mm3 = None
    volume_mm3 = None
    if compute_mm3:
        # affine diagonal can give voxel sizes (abs)
        zooms = img.header.get_zooms()[:3]  # mm per voxel in x,y,z
        voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
        volume_mm3 = nonzero_voxels * voxel_volume_mm3
    return nonzero_voxels, voxel_volume_mm3, volume_mm3

def main(masks_dir, out_csv, pattern="*_mask.nii*",
         recursive=False, compute_mm3=True):
    masks_dir = Path(masks_dir).expanduser()
    if recursive:
        paths = list(masks_dir.rglob(pattern))
    else:
        paths = list(masks_dir.glob(pattern))
    rows = []
    for p in sorted(paths):
        # derive subject id from filename: J01257_mask.nii.gz -> J01257
        name = p.name
        subj = name.split("_mask")[0]
        try:
            nonzero, voxel_vol, vol_mm3 = count_nonzero(p, compute_mm3=compute_mm3)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
            nonzero, voxel_vol, vol_mm3 = None, None, None
        rows.append({
            "subject": subj,
            "mask_path": str(p),
            "nonzero_voxels": nonzero,
            "voxel_volume_mm3": voxel_vol,
            "volume_mm3": vol_mm3
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count non-zero voxels in mask NIfTIs.")
    parser.add_argument("--masks_dir", required=True,
                        help="Directory containing mask files (e.g. C:\\...\\masks)")
    parser.add_argument("--out_csv", default="whole_brain_volume.csv",
                        help="Output CSV filename")
    parser.add_argument("--pattern", default="*_mask.nii*",
                        help="Glob pattern for mask files (default '*_mask.nii*')")
    parser.add_argument("--recursive", action="store_true",
                        help="Search subdirectories recursively")
    parser.add_argument("--no_mm3", dest="compute_mm3", action="store_false",
                        help="Do not compute mm^3 volumes (skip reading voxel sizes)")
    args = parser.parse_args()
    main(args.masks_dir, args.out_csv, pattern=args.pattern,
         recursive=args.recursive, compute_mm3=args.compute_mm3)
