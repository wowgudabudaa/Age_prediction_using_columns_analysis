clear
clc
close all

%% Load files
output_dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
ID = 'S04491';
lh_white = strcat(output_dir,ID,'/surf/lh.white');
lh_pial = strcat(output_dir,ID,'/surf/lh.pial');
rh_white = strcat(output_dir,ID,'/surf/rh.white');
rh_pial = strcat(output_dir,ID,'/surf/rh.pial');
trans_M_dir = strcat(output_dir,ID,'/DWI2T1_dti_upsampled.dat');

[lh_white_vertices, lh_white_faces] = freesurfer_read_surf(lh_white);
[lh_pial_vertices, lh_pial_faces] = freesurfer_read_surf(lh_pial);
[rh_white_vertices, rh_white_faces] = freesurfer_read_surf(rh_white);
[rh_pial_vertices, rh_pial_faces] = freesurfer_read_surf(rh_pial);
% Connect from W/G to pial
lh_pair = cat(2,lh_white_vertices,lh_pial_vertices);
rh_pair = cat(2,rh_white_vertices,rh_pial_vertices);
% Save paired points
save(strcat(output_dir, ID, '/columns/', ID, '_pair_lh.mat'), "lh_pair");
save(strcat(output_dir, ID, '/columns/', ID, '_pair_rh.mat'), "rh_pair");

%% Generate column points
points_num = 21;% Overall points along each column(two vertices included)
lh_column_points = generate_inner_points(lh_pair, points_num);
rh_column_points = generate_inner_points(rh_pair, points_num);

save(strcat(output_dir, ID, '/columns/',ID, '_column_lh.mat'), "lh_column_points");
save(strcat(output_dir, ID, '/columns/',ID, '_column_rh.mat'), "rh_column_points");

%% Load transformation matrix
data = importdata(trans_M_dir);
trans_M = data.data(4:19);
trans_M = reshape(trans_M,[4, 4])';

% Define Tmov: vox to ras matrix
% Command for this: mri_info --vox2ras-tkr mov.nii
voldim = [512, 512, 272];% [512, 512, 272]
voxres = [0.5, 0.5, 0.5];% [0.5, 0.5, 0.5]
T_mov = vox2ras_tkreg(voldim, voxres);
T_mov = vox2ras_0to1(T_mov);

%% Transform columns in anatomical domain to dwi domain(CRS coordinates)
lh_cp_dwi = columns_reshape(lh_column_points);
lh_cp_dwi = [lh_cp_dwi; ones(1, size(lh_cp_dwi, 2))];%RAS coordinates in anatomical domain
lh_cp_dwi = inv(T_mov) * trans_M * lh_cp_dwi;%CRS coordinates in functional domain, eg. [Cx, Cy, Cz, xxx; Cx, Cy, Cz, xxx]'

lh_white_f = columns_reshape(lh_white_vertices);
lh_white_f = [lh_white_f; ones(1, size(lh_white_f, 2))];
lh_white_f = trans_M * lh_white_f;
lh_white_f = lh_white_f(1:3,:)';

rh_cp_dwi = columns_reshape(rh_column_points);
rh_cp_dwi = [rh_cp_dwi; ones(1, size(rh_cp_dwi, 2))];%RAS coordinates in anatomical domain
rh_cp_dwi = inv(T_mov) * trans_M * rh_cp_dwi;%CRS coordinate in functional domain

rh_white_f = columns_reshape(rh_white_vertices);
rh_white_f = [rh_white_f; ones(1, size(rh_white_f, 2))];
rh_white_f = trans_M * rh_white_f;
rh_white_f = rh_white_f(1:3,:)';

% plot3(lh_cp_dwi(1,1:20000), lh_cp_dwi(2,1:20000), lh_cp_dwi(3,1:20000),'o','Color','r');
% hold on
% plot3(rh_cp_dwi(1,1:20000), rh_cp_dwi(2,1:20000), rh_cp_dwi(3,1:20000),'o','Color','b');
%% Save file
if any(isnan(lh_cp_dwi))
    fprintf('%s has NAN values\n', ID);
end
if any(isnan(rh_cp_dwi))
    fprintf('%s has NAN values\n', ID);
end
save(strcat(output_dir, ID, '/columns/', ID, '_column_lh_dwi.mat'), 'lh_cp_dwi');
save(strcat(output_dir, ID, '/columns/', ID, '_column_rh_dwi.mat'), 'rh_cp_dwi');
write_surf(strcat(output_dir, ID, '/columns/', ID, 'lh_t.white'),lh_white_f, lh_white_faces);
write_surf(strcat(output_dir, ID, '/columns/', ID, 'rh_t.white'),rh_white_f, rh_white_faces);

