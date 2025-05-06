close
clear
clc
%% This script is to generate coordinates in different region
output_dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
ID = 'S04491';
trans_M_dir = strcat(output_dir,ID,'/DWI2T1_dti_upsampled.dat');

%% Define Tmov: vox to ras matrix
% Command for this: mri_info --vox2ras-tkr mov.nii
voldim = [512, 512, 272];% [512, 512, 272]
voxres = [0.5, 0.5, 0.5];% [0.5, 0.5, 0.5]
T_mov = vox2ras_tkreg(voldim, voxres);
T_mov = vox2ras_0to1(T_mov);

%% Load transformation matrix
data = importdata(trans_M_dir);
trans_M = data.data(4:19);
trans_M = reshape(trans_M,[4, 4])';

%% Load filename list
fid = fopen(strcat(output_dir,ID,'/file_list_local.txt'),'rt');
file_list = textscan(fid, '%s');
fclose(fid);
file_list = vertcat(file_list{1,1});
list_size = size(file_list, 1);
load(strcat(output_dir, ID, '/columns/',ID, '_column_lh.mat'));
load(strcat(output_dir, ID, '/columns/',ID, '_column_rh.mat'));

%% Iterate for every label file
for i = 1:list_size
    filename = file_list{i};
    fid = fopen(filename, 'rt');
    vertices = textscan(fid, '%d %f %f %f %f','Headerlines', 2);
    vertx_num = vertices{1}+1;% vertx index, eg. 1,2,3
    last_slash = strfind(filename, '/');
    output_name = filename(last_slash(end)+1:end);
    output_name = erase(output_name, '.label');
    first_dot = strfind(output_name, '.');
    hemi = output_name(1:first_dot(1)-1);
    output_name = strrep(output_name, '.', '_');
    output_name = strcat(output_dir, ID, '/label_coord/', output_name, '.mat');

    if hemi == 'lh'
        lh_region_column_points = lh_column_points(vertx_num,:);
        lh_cp_dwi = columns_reshape(lh_region_column_points);
        lh_cp_dwi = [lh_cp_dwi; ones(1, size(lh_cp_dwi, 2))];%RAS coordinates in anatomical domain
        lh_cp_dwi = inv(T_mov) * trans_M * lh_cp_dwi;%CRS coordinates in functional domain
        save(output_name, 'lh_cp_dwi');
        if any(isnan(lh_cp_dwi))
            fprintf('%s has NAN values\n', ID);
        end
    end

    if hemi == 'rh'
        rh_region_column_points = rh_column_points(vertx_num,:);
        rh_cp_dwi = columns_reshape(rh_region_column_points);
        rh_cp_dwi = [rh_cp_dwi; ones(1, size(rh_cp_dwi, 2))];%RAS coordinates in anatomical domain
        rh_cp_dwi = inv(T_mov) * trans_M * rh_cp_dwi;%CRS coordinates in functional domain
        save(output_name, 'rh_cp_dwi');
        if any(isnan(rh_cp_dwi))
            fprintf('%s has NAN values\n', ID);
        end
    end

%     R_coord = vertices{2};
%     A_coord = vertices{3};
%     S_coord = vertices{4};
%     RAS = [R_coord'; A_coord'; S_coord';ones(1, size(R_coord', 2))];
%     CRS = inv(T_mov) * trans_M * RAS;
end