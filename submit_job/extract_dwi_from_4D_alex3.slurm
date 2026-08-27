#!/bin/bash

#SBATCH --job-name=upsample_diffusion
#SBATCH --output=./logs/upsample_%A_%a.out
#SBATCH --error=./logs/upsample_%A_%a.err
#SBATCH --array=1-89%32  # Run up to 32 jobs concurrently
#SBATCH --nodes=1  # Each job runs on exactly one node
#SBATCH --ntasks=4  # Each job is a separate task
#SBATCH --cpus-per-task=8  # Each job gets 8 CPUs
#SBATCH --mem-per-cpu=2500M  # Allocate 2.5GB per CPU
#SBATCH --partition=normal
##SBATCH --nodelist=andros,crete,kythira,patmos  # Limit to 4 nodes
##SBATCH --distribution=cyclic  # Ensure equal node distribution
#SBATCH --spread-job  # Spread jobs evenly
##SBATCH --exclusive  # Prevent jobs from sharing nodes







# Define paths
LOCAL_SCRATCH="/tmp/$USER/$SLURM_JOB_ID"
NFS_DIR="$PAROS/paros_WORK/hanwen/ad_decode_test/output"
LOG_DIR="./logs"
mkdir -p $LOCAL_SCRATCH $LOG_DIR

# Set LOCAL_SCRATCH to /scratch if available, otherwise use /tmp
if [[ -d "/scratch" && -w "/scratch" ]]; then
    LOCAL_SCRATCH="/scratch/$USER/$SLURM_JOB_ID"
else
    echo "WARNING: /scratch not available! Using /tmp instead."
    LOCAL_SCRATCH="/tmp/$USER/$SLURM_JOB_ID"
fi

# Ensure LOCAL_SCRATCH exists before using it
mkdir -p "$LOCAL_SCRATCH"

# Check if creation was successful
if [[ ! -d "$LOCAL_SCRATCH" ]]; then
    echo " ERROR: Failed to create $LOCAL_SCRATCH!"
    exit 1
else
    echo " Successfully created: $LOCAL_SCRATCH"
fi



# LOCAL_SCRATCH="/scratch/$USER/$SLURM_JOB_ID"

# #  Ensure the directory is created safely
# if [[ ! -d "$LOCAL_SCRATCH" ]]; then
#     echo "LOCAL_SCRATCH does NOT exist! Creating it now..."
#     mkdir -p "$LOCAL_SCRATCH"
    
#     # Check again after creating
#     if [[ -d "$LOCAL_SCRATCH" ]]; then
#         echo "Successfully created: $LOCAL_SCRATCH"
#     else
#         echo "ERROR: Failed to create $LOCAL_SCRATCH!"
#         exit 1
#     fi
# else
#     echo "LOCAL_SCRATCH already exists: $LOCAL_SCRATCH"
# fi



# Get subject ID
subject_list="$PAROS/paros_WORK/hanwen/ad_decode_test/code/subjects.txt"
subject_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$subject_list")

# Check if valid
if [[ -z "$subject_id" ]]; then
    echo "Error: No subject ID found for task ID ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

# Define log files
TIMING_LOG="$LOG_DIR/timings_${SLURM_ARRAY_TASK_ID}.log"
MEM_LOG="$LOG_DIR/memory_${SLURM_ARRAY_TASK_ID}.log"

# Start Logging
script_start=$(date +%s)
echo "Job $SLURM_ARRAY_TASK_ID started on node $(hostname) at $(date)" | tee -a $TIMING_LOG

# Define file paths
INPUT_FILE="$NFS_DIR/${subject_id}/${subject_id}_nii4D_upsampled.nii.gz"
SCRATCH_DWI="$LOCAL_SCRATCH/${subject_id}_dwi4D_upsampled.nii.gz"
SCRATCH_SUM="$LOCAL_SCRATCH/${subject_id}_dwi_sumed_upsampled.nii.gz"
SCRATCH_MASKED="$LOCAL_SCRATCH/${subject_id}_dwi_sumed_masked_upsampled.nii.gz"
FINAL_DWI="$NFS_DIR/${subject_id}/${subject_id}_dwi_sumed_masked_upsampled.nii.gz"

# Start logging memory usage
vmstat -s > "$MEM_LOG" &
VMSTAT_PID=$!

# -------------------------------
# Step 1: Convert the 4D image and store in local scratch
# -------------------------------
step_start=$(date +%s)
mrconvert "$INPUT_FILE" -coord 3 1:end "$SCRATCH_DWI" -force
step_end=$(date +%s)
echo "Step 1: Image conversion completed in $((step_end - step_start)) seconds" | tee -a $TIMING_LOG

# -------------------------------
# Step 2: Sum up the DWI data and store in local scratch
# -------------------------------
step_start=$(date +%s)
mrmath "$SCRATCH_DWI" sum "$SCRATCH_SUM" -axis 3 -force
step_end=$(date +%s)
echo "Step 2: DWI summation completed in $((step_end - step_start)) seconds" | tee -a $TIMING_LOG

# -------------------------------
# Step 3: Apply mask and store in local scratch
# -------------------------------
step_start=$(date +%s)
MASK_FILE="$NFS_DIR/${subject_id}/${subject_id}_subjspace_mask_upsampled.nii.gz"
mrcalc "$SCRATCH_SUM" "$MASK_FILE" -mult "$SCRATCH_MASKED" -force
step_end=$(date +%s)
echo "Step 3: Mask application completed in $((step_end - step_start)) seconds" | tee -a $TIMING_LOG

# -------------------------------
# Step 4: Move final processed file to NFS
# -------------------------------
step_start=$(date +%s)
mv "$SCRATCH_MASKED" "$FINAL_DWI"
step_end=$(date +%s)
echo "Step 4: Moved final processed file to NFS in $((step_end - step_start)) seconds" | tee -a $TIMING_LOG

# Cleanup local scratch space
rm -rf "$LOCAL_SCRATCH"

# -------------------------------
# Finalizing: Log total runtime
# -------------------------------
script_end=$(date +%s)
total_runtime=$((script_end - script_start))
echo "Job $SLURM_ARRAY_TASK_ID completed in $total_runtime seconds" | tee -a $TIMING_LOG

# Stop Memory Logging
kill $VMSTAT_PID
echo "Final Memory Usage for Job $SLURM_ARRAY_TASK_ID:" >> "$MEM_LOG"
#sstat --format=JobID,MaxRSS,AveRSS,Elapsed $SLURM_JOB_ID >> "$MEM_LOG"
