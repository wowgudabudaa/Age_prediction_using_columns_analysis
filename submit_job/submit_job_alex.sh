#!/bin/bash

# SLURM job settings for the master job that will submit other jobs.
#SBATCH -N 1                       # Only need 1 node for this script
#SBATCH --ntasks=1                 # 1 task for this script
#SBATCH --cpus-per-task=1          # 1 CPU per task for this script
#SBATCH -o /home/alex/job.%j.out   # Output log file
#SBATCH -p normal                  # Partition to submit to (adjust as needed)

# Set the input and output directories
input_dir="$PAROS/paros_WORK/hanwen/ad_decode_test/input/"
output_dir="$PAROS/paros_WORK/hanwen/ad_decode_test/output/"

# List of subjects (or run numbers) to process
cd ${input_dir};

# Scrape runnos from folder names in input directory:
for runno in $(ls -d  S*/ | cut -d '/' -f 1);do
# Looping over the runnos array and submit jobs for each
    # Build the command for this subject
    echo "Processing: ${runno}"
    #BJA, 3 Dec 2024: added -sd option to redirect to local output directory
    cmd="hostname;recon-all -s ${runno} -i ${input_dir}${runno}/${runno}_T1_masked.nii.gz  -sd ${output_dir} -all"
    # Submit the job using your submit_cluster_job.bash script
    bash ${GUNNIES}/submit_cluster_job.bash "$cmd"
done
