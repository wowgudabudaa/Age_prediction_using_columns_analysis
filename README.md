# Brain_column_analyse

This repository implements a complete pipeline for **cortical column analysis** from brain MR images. It consists of three sequential stages:

1. **MR image processing** → [submit_job](submit_job) — prepares diffusion parametric maps and cortical surfaces from raw MRI data.
2. **Cortical column generation** → [columns_generation_in_MATLAB](columns_generation_in_MATLAB) — extracts cortical columns from the processed images.
3. **GNN model training and evaluation** → [NEW_COLUMNS_GNC](NEW_COLUMNS_GNC) — predicts brain age from the extracted columns.

Each stage consumes the output of the previous one, so the repository is best followed from top to bottom.

## 1. MR Image Processing

This stage turns raw acquisitions into the two inputs required for column extraction: diffusion parametric maps (from DWI) and cortical surfaces (from T1). The two sub-pipelines below can be run independently.

### DWI Processing

The DWI sub-pipeline cleans the raw 4D diffusion acquisition and then fits the diffusion tensor to obtain the parametric maps.

#### Step 1: Extract DWI from the 4D acquisition

[extract_dwi_from_4D_alex3.slurm](submit_job\extract_dwi_from_4D_alex3.slurm)

- **input**: Per-subject upsampled 4D diffusion image (`<subject>_nii4D_upsampled.nii.gz`), subject-space brain mask (`<subject>_subjspace_mask_upsampled.nii.gz`), and a subject list (`subjects.txt`) indexed by the SLURM array.
- **usage**: Submit as a SLURM array job; each task handles one subject. The script extracts the DWI volumes from the 4D image (`mrconvert`), sums them along the volume axis (`mrmath`), and applies the brain mask (`mrcalc`). Intermediate files are staged in local scratch space; only the final result is moved back to shared storage.
- **output**: One masked, summed DWI volume per subject (`<subject>_dwi_sumed_masked_upsampled.nii.gz`), plus per-task timing and memory logs under `logs/`.

#### Step 2: Fit the diffusion tensor and derive parametric maps

[temp_dwi2tensor.sh](submit_job\temp_dwi2tensor.sh)

Building on the cleaned DWI from Step 1, this step fits the diffusion tensor and derives the scalar maps that will be sampled along the cortex in the next stage.

- **input**: The processed 4D DWI (`<subject>_nii4D_upsampled.nii.gz`), the subject-space mask, and the FSL gradient tables (`<subject>_bvecs.txt`, `<subject>_bvals.txt`).
- **usage**: Adjust the subject-specific paths inside the script and run it. `mrgrid` upsamples the mask to 0.5 mm isotropic voxels, `dwi2tensor` fits the diffusion tensor from the DWI, and `tensor2metric` derives scalar parametric maps from the tensor. Additional metrics (ADC, AD, RD, CL, CP, CS, vector/value maps) are available as commented-out `tensor2metric` lines.
- **output**: A diffusion tensor image (`<subject>dt.mif`) and scalar maps such as FA (`<subject>_fa_upsampled.nii.gz`), together with the upsampled mask.

### Cortical Surface Reconstruction

This sub-pipeline reconstructs the cortical surface from the T1 image and registers the DWI into the T1 domain, so that diffusion metrics can be sampled along the cortex.

#### Step 1: Reconstruct the cortical surface from T1

[submit_job_alex.sh](submit_job\submit_job_alex.sh)

- **input**: One folder per subject containing the masked T1 volume (`<subject>_T1_masked.nii.gz`), plus the input/output directories set at the top of the script.
- **usage**: Submit the script as a SLURM master job. It scans the input directory for subject folders and, for each subject, submits a cluster job that runs FreeSurfer's `recon-all -s <subject> -i <subject>_T1_masked.nii.gz -sd <output_dir> -all` to run the full cortical reconstruction pipeline.
- **output**: A per-subject FreeSurfer reconstruction directory under the output directory, containing cortical surfaces, parcellation labels, and thickness maps.

#### Step 2: Register DWI to the T1 domain

[bbregister_hanwen.sh](submit_job\bbregister_hanwen.sh)

Using the surfaces from Step 1 as a reference, this step aligns the DWI volume to the T1 space so that both modalities share a common coordinate system.

- **input**: The masked DWI volume (`<subject>_dwi_masked.nii.gz`) and the FreeSurfer subject built in Step 1.
- **usage**: Edit the subject ID and paths, then run the script. `bbregister` performs boundary-based cross-modal registration of the DWI volume to the T1 space using the DTI cost function (`--dti --init-fsl --12`). The commented-out `tkregister2`/`tkregisterfv` commands can be used to visually verify the alignment.
- **output**: A registration file (`DWI2T1_dti.dat`) containing the affine transform that maps the DWI volume into the T1 (FreeSurfer) domain.

## 2. Cortical Column Generation

With diffusion metrics and cortical surfaces in place, this stage extracts cortical columns — 21-point diffusion profiles running through the cortex — for all subjects in one batch.

[multiple_column_analyse.m](columns_generation_in_MATLAB\multiple_column_analyse.m)

- **input**: A text file listing subject IDs (`subjects_remaining.txt`) and the shared output directory that holds per-subject FreeSurfer reconstructions, label coordinates, and diffusion metric maps.
- **usage**: Open the script in MATLAB, adjust the hard-coded paths, and run it. For each subject it calls `column_analyse_oneMM`, which computes the cortical-column coordinates within anatomical regions, visualizes them, and samples the 21-point column profiles from the MD volume.
- **output**: Per-subject, per-region column features — CSV files (`<subject>_lh_<region>_cols_md.csv` / `<subject>_rh_<region>_cols_md.csv`) holding 21 MD values per column under `columns_1mm/`, with intermediate label coordinates under `label_coord_1mm/`.

## 3. GNN Model Training and Evaluation

In the final stage, a graph neural network is trained to predict brain age from the extracted cortical columns (MD, QSM, and cortical thickness).

[GNN for brain age prediction from cortical columns](NEW_COLUMNS_GNC)

For detailed instructions, see the [GNN model README](NEW_COLUMNS_GNC\README.md).
