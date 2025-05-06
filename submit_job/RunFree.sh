#!/bin/sh
 
# This is a BIAC template script for running FreeSurfer reconstruction on the cluster
# You can use this script with no modification by entering the following command on
# the cluster head-node:
#
# >  qsub -v EXPERIMENT=MyExperiment RunFree.sh SubjID SeriesDir
# where:
#     MyExperiment is your BIAC experiment name (eg. Expt.02)
#     SubjID is the name you want to assign for this subject reconstruction
#         (eg. 1234 or subj21)
#     SeriesDir is the directory path to get to your anatomical DICOM images for the
#        subject you want to reconstruct, within your MyExperiment directory
#        (eg.  Data/Anat/20110101_12345/series300 )
#
# The job will create an output directory for this subject, within your MyExperiment
#  directory  at:
#        Analysis/freesurfer/SubjID
#
# Note: The QSUB command will automatically insert your email address to notify you of job
# status.
#
# If you want to modify this template script, there are 2 USER sections:
#  1. USER DIRECTIVE: If you want mail notifications when
#     your job is completed or fails you need to set the
#     correct email address.
#
#  2. USER SCRIPT: Add the user script in this section.
#     Within this section you can access your experiment
#     folder using $EXPERIMENT. All paths are relative to this variable
#     eg: $EXPERIMENT/Data $EXPERIMENT/Analysis
#     By default all terminal output is routed to the " Analysis "
#     folder under the Experiment directory i.e. $EXPERIMENT/Analysis
#     To change this path, set the OUTDIR variable in this section
#     to another location under your experiment folder
#     eg: OUTDIR=$EXPERIMENT/Analysis/GridOut
#     By default on successful completion the job will return 0
#     If you need to set another return code, set the RETURNCODE
#     variable in this section. To avoid conflict with system return
#     codes, set a RETURNCODE higher than 100.
#     eg: RETURNCODE=110
#     Arguments to the USER SCRIPT are accessible in the usual fashion
#     eg:  $1 $2 $3
# The remaining sections are setup related and don't require
# modifications for most scripts. They are critical for access
# to your data
 
# --- BEGIN GLOBAL DIRECTIVE --
#$ -S /bin/sh
#$ -o $HOME/$JOB_NAME.$JOB_ID.out
#$ -e $HOME/$JOB_NAME.$JOB_ID.out
#$ -m ea
# -- END GLOBAL DIRECTIVE --
 
# -- BEGIN PRE-USER --
#Name of experiment whose data you want to access
EXPERIMENT=${EXPERIMENT:?"Experiment not provided"}
 
EXPERIMENT=`findexp $EXPERIMENT`
EXPERIMENT=${EXPERIMENT:?"Returned NULL Experiment"}
 
if [ $EXPERIMENT = "ERROR" ]
then
        exit 32
else
#Timestamp
echo "----JOB [$JOB_NAME.$JOB_ID] START [`date`] on HOST [$HOSTNAME]----"
# -- END PRE-USER --
# **********************************************************
 
# -- BEGIN USER DIRECTIVE --
# Send notifications to the following address
#$ -M user@somewhere.edu
 
# -- END USER DIRECTIVE --
 
# -- BEGIN USER SCRIPT --
 
SUBJ=$1
SERIESDIR=$2
RAWDIR=${EXPERIMENT}/${SERIESDIR}
FREEOUT=${EXPERIMENT}/Analysis/freesurfer
 
#if out directory doesn't exist, then make it
if [ ! -d "$FREEOUT" ]; then
        mkdir -p $FREEOUT
fi
 
#if the fsaverage subject doesn't exist there, then copy it
for freedirs in fsaverage lh.EC_average rh.EC_average; do
        if [ ! -d "${FREEOUT}/${freedirs}" ]; then
                cp -R $FREESURFER_HOME/subjects/${freedirs} $FREEOUT/
        fi
done
 
export SUBJECTS_DIR=$FREEOUT
 
cd $SUBJECTS_DIR
recon-all -i $RAWDIR/*_00001.dcm -s $SUBJ
 
recon-all -autorecon-all -s $SUBJ
 
# -- END USER SCRIPT -- #
 
# **********************************************************
# -- BEGIN POST-USER --
echo "----JOB [$JOB_NAME.$JOB_ID] STOP [`date`]----"
OUTDIR=${OUTDIR:-$EXPERIMENT/Analysis}
mv $HOME/$JOB_NAME.$JOB_ID.out $OUTDIR/$JOB_NAME.$JOB_ID.out
RETURNCODE=${RETURNCODE:-0}
exit $RETURNCODE
fi
# -- END POST USER--