function coordinates_in_region_oneMM(ID,input_dir,output_dir)

%% This script is to generate coordinates in different region
%PAROS=getenv("PAROS");
%output_dir = [PAROS '/paros_WORK/hanwen/ad_decode_test/output/'];
%ID = 'S00775';
trans_M_dir = strcat(output_dir,ID,'/DWI2T1_dti.dat');

% BJ's note: addng a mkdir command here
mkdir(strcat(output_dir, ID, '/label_coord_1mm/'));

%% Define Tmov: vox to ras matrix
% Command for this: mri_info --vox2ras-tkr mov.nii
voldim = [256, 256, 136];% [512, 512, 272]
voxres = [1, 1, 1];% [0.5, 0.5, 0.5]
T_mov = vox2ras_tkreg(voldim, voxres);
T_mov = vox2ras_0to1(T_mov);

%% Load transformation matrix
if ~isfile(trans_M_dir)
    fprintf('Subject %s doesnt have transformation matrix\n', ID)
    return
end
data = importdata(trans_M_dir);
trans_M = data.data(4:19);
trans_M = reshape(trans_M,[4, 4])';

%% Load filename list
fid = fopen(strcat(output_dir,ID,'/file_list_local.txt'),'rt');
file_list = textscan(fid, '%s');
fclose(fid);
file_list = vertcat(file_list{1,1});
list_size = size(file_list, 1);
if ~isfile(strcat(output_dir, ID, '/columns_1mm/',ID, '_column_lh.mat'))
    fprintf('Subject %s doesnt have columns reslut\n', ID);
    return
end
load(strcat(output_dir, ID, '/columns_1mm/',ID, '_column_lh.mat'));
load(strcat(output_dir, ID, '/columns_1mm/',ID, '_column_rh.mat'));
load(strcat(output_dir, ID, '/columns_1mm/', ID, '_column_lh_dwi.mat'));
load(strcat(output_dir, ID, '/columns_1mm/', ID, '_column_rh_dwi.mat'));
ori_lh_cp_dwi = lh_cp_dwi;
ori_rh_cp_dwi = rh_cp_dwi;

%% Iterate for every label file
for i = 1:list_size
    filename = file_list{i};
    fid = fopen(filename, 'rt');
    vertices = textscan(fid, '%d %f %f %f %f','Headerlines', 2);
    fclose(fid);
    vertx_num = vertices{1}+1;% vertx index, eg. 1,2,3
    last_slash = strfind(filename, '/');
    output_name = filename(last_slash(end)+1:end);
    output_name = erase(output_name, '.label');
    first_dot = strfind(output_name, '.');
    hemi = output_name(1:first_dot(1)-1);
    output_name = strrep(output_name, '.', '_');
    % output_name = strcat(output_dir, ID, '/label_coord/', output_name, '.mat');

    if hemi == 'lh'
        index = vertx_num .* int32(ones(1, 21) .* 21) + int32(repmat(-20:1:0, size(vertx_num, 1), 1));
        index = reshape(index', 1, []);
        max_index = size(ori_lh_cp_dwi, 2);
        index(:, any(index > max_index, 1)) = [];
        lh_cp_dwi = ori_lh_cp_dwi(:, index);
        save(strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.mat'), 'lh_cp_dwi');
        writematrix(lh_cp_dwi, strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.csv'));
    end

    if hemi == 'rh'
        index = vertx_num .* int32(ones(1, 21) .* 21) + int32(repmat(-20:1:0, size(vertx_num, 1), 1));
        index = reshape(index', 1, []);
        max_index = size(ori_rh_cp_dwi, 2);
        index(:, any(index > max_index, 1)) = [];
        rh_cp_dwi = ori_rh_cp_dwi(:, index);
        save(strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.mat'), 'rh_cp_dwi');
        writematrix(rh_cp_dwi, strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.csv'));
    end

%     R_coord = vertices{2};
%     A_coord = vertices{3};
%     S_coord = vertices{4};
%     RAS = [R_coord'; A_coord'; S_coord';ones(1, size(R_coord', 2))];
%     CRS = inv(T_mov) * trans_M * RAS;
end
clearvars;
end