recon-all -s S00775a -i $PAROS/paros_WORK/hanwen/ad_decode_test/input/S00775_T1_masked.nii -all

qsub -v EXPERIMENT=MyExperiment RunFree.sh SubjID SeriesDir