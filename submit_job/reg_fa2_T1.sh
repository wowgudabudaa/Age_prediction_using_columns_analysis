

#!/bin/bash
runno='S00775'
ANTSPATH='/Applications/ANTS/'
vol_zero='/Users/bass/Desktop/AD_DECODE/masked_data/S00775_T1_masked.nii.gz'
vol_xxx='/Users/bass/Desktop/AD_DECODE/masked_data/S00775_mrtrixfa_masked.nii.gz'
vol_xxx='/Users/bass/Desktop/AD_DECODE/masked_data/S00775_dwi_masked.nii.gz'
out_prefix='/Users/bass/Desktop/HanwenLin/S00775_T1/antsed/S00775'
vol_xxx_out='/Users/bass/Desktop/HanwenLin/S00775_T1/antsed/S00775fa_to_T1.nii.gz'

xform_xxx='/Users/bass/Desktop/HanwenLin/S00775_T1/antsed/0GenericAffine.mat'
firstAffine='/Users/bass/Desktop/HanwenLin/S00775_T1/antsed/S007750GenericAffine.mat'

# reg_cmd="${ANTSPATH}/antsRegistration  --float -d 3 -v  -m Mattes[ ${vol_zero},${vol_xxx},1,32,regular,0.3 ] -t Affine[0.05] -c [ 1000x500x300,1.e-5,15 ] -s 1x1x0vox -f 4x2x1 -u 1 -z 1 -o ${out_prefix}";
# echo $reg_cmd
# $reg_cmd
#--write-composite-transform

reg_cmd="${ANTSPATH}/antsRegistration  --float -d 3 -v  -m CC[ ${vol_zero},${vol_xxx},1,32,regular,0.3 ] -t SyN[0.5,3,0.5] -c [ 1000x1000x1000x60,1e-8,20] -r ${firstAffine} -s 4x2x1x0vox -f 8x4x2x1 -u -z 1 -o ${out_prefix}";
echo $reg_cmd
$reg_cmd

# antsRegistration -v 1 -d 3 -m CC[ /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/median_images/MDT_fa.nii.gz,/mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/preprocess/base_images/S01621_fa_masked.nii.gz,1,4,None]  -o /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_diffeo/S01621_to_MDT_   -c [ 1000x1000x1000x60,1e-8,20] -f 8x4x2x1 -t SyN[0.5,3,0.5] -s 4x2x1x0vox -r  [/mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/S01621_affine.mat,0]  [/mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/S01621_rigid.mat,0]  -u;
# ln -s /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_diffeo/S01621_to_MDT_1Warp.nii.gz /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_diffeo/S01621_to_MDT_warp.nii.gz;
# ln -s /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_diffeo/S01621_to_MDT_1InverseWarp.nii.gz /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_diffeo/MDT_to_S01621_warp.nii.gz;
# rm /mnt/munin2/Badea/Lab/mouse/VBM_21ADDecode03_IITmean_RPI_fullrun-work/dwi/SyN_0p5_3_0p5_fa/faMDT_NoNameYet_n37_i6/reg_diffeo/S01621_to_MDT_0GenericAffine.mat;
 


#apply_cmd="${ANTSPATH}/antsApplyTransforms -d 3 -e 0 -i ${vol_xxx} -r ${vol_zero} -o ${vol_xxx_out} -n Linear -t ${xform_xxx}  -v 0 --float";
#echo $apply_cmd
#$apply_cmd