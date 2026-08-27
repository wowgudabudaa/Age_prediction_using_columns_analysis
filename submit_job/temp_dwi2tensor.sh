# File copy
# scp -r AD_DECODE_backup alex@kea.dhe.duke.edu://mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test

# mrconvert -d coreg_in.nii.gz coreg_out.mif
# dwi2tensor coreg.mif difftensor.m bvecs_fix.txt bvals_fix.txt -mask mask.mif tensor2fa 

# dwi2tensor S01912_subjspace_coreg.nii.gz S01912dt.mif -fslgrad S01912_bvecs.txt S01912_bvals.txt -mask S01912_subjspace_mask.nii.gz -force
# tensor2metric -fa S01912hanwen.nii.gz S01912dt.mif -mask S01912_subjspace_mask.nii.gz

# upsample the mask
mrgrid $PAROS/paros_WORK/hanwen/ad_decode_test/AD_DECODE_backup/DWI/S00775_subjspace_mask.nii.gz regrid -voxel 0.5,0.5,0.5 $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775_subjspace_mask_upsampled.nii.gz
# dwi2tensor
dwi2tensor $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775_nii4D_upsampled.nii.gz $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775dt.mif -fslgrad $PAROS/paros_WORK/hanwen/raw_DWI/S00775_bvecs.txt $PAROS/paros_WORK/hanwen/raw_DWI/S00775_bvals.txt -mask $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775_subjspace_mask_upsampled.nii.gz -force
tensor2metric -fa $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775_fa_upsampled.nii.gz $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775dt.mif -mask $PAROS/paros_WORK/hanwen/ad_decode_test/output/S00775/S00775_subjspace_mask_upsampled.nii.gz
# tensor2metric -adc S02967_adc_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -ad S02967_ad_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -rd S02967_rd_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -cl S02967_cl_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -cp S02967_cp_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -cs S02967_cs_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -value S02967_value_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
# tensor2metric -vector S02967_vector_resampled.nii.gz S02967dt.mif -mask S02967_subjspace_mask_resampled.nii.gz
