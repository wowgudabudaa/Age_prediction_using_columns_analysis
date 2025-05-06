cmd="recon-all -s S01470_T1 -i $PAROS/paros_WORK/hanwen/ad_decode_test/input/S01470/S01470_T1_masked.nii -all
recon-all -s S01619_T1 -i $PAROS/paros_WORK/hanwen/ad_decode_test/input/S01619/S01619_T1_masked.nii -all
recon-all -s S01912_T1 -i $PAROS/paros_WORK/hanwen/ad_decode_test/input/S01912/S01912_T1_masked.nii -all
recon-all -s S02110_T1 -i $PAROS/paros_WORK/hanwen/ad_decode_test/input/S02110/S02110_T1_masked.nii.gz -all
recon-all -s S02231_T1 -i $PAROS/paros_WORK/hanwen/ad_decode_test/input/S02231/S02231_T1_masked.nii.gz -all
"
bash ${GUNNIES}/submit_cluster_job.bash "$cmd"