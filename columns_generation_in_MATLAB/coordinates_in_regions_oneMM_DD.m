function coordinates_in_regions_oneMM_DD(ID,input_dir,output_dir)

%% This script is to generate coordinates in different region
%PAROS=getenv("PAROS");
%output_dir = [PAROS '/paros_WORK/hanwen/ad_decode_test/output/'];
%ID = 'S00775';

% BJ's note: addng a mkdir command here
mkdir(strcat(output_dir, ID, '/label_coord_1mm/'));

%% Define Tmov: vox to ras matrix
% Command for this: mri_info --vox2ras-tkr mov.nii
voldim = [256, 256, 136];% [512, 512, 272]
voxres = [1, 1, 1];% [0.5, 0.5, 0.5]
T_mov = vox2ras_tkreg(voldim, voxres);
T_mov = vox2ras_0to1(T_mov);

%% Load transformation matrix
% if ~isfile(trans_M_dir)
%     fprintf('Subject %s doesnt have transformation matrix\n', ID)
%     return
% end
% data = importdata(trans_M_dir);
% trans_M = data.data(4:19);
% trans_M = reshape(trans_M,[4, 4])';

%% Load filename list
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

%% Load DD annotation file
annot_file_lh = 'lh.aparc.annot';
file_name_lh = strcat(output_dir, ID,'/', ID, '/label/', annot_file_lh);
[~,label_lh,colortable_lh]=read_annotation(file_name_lh);

annot_file_rh = 'rh.aparc.annot';
file_name_rh = strcat(output_dir, ID,'/', ID, '/label/', annot_file_rh);
[~,label_rh,colortable_rh]=read_annotation(file_name_rh);

%% Iterate for every label file
for i = 2:36
    if i == 5
        continue
    end
    hemi = 'lh';
    region_name = colortable_lh.struct_names(i,1);
    region_name = region_name{1};
    region_num = colortable_lh.table(i,5);
    vertx_num = find(label_lh == region_num);
    vertx_num = int32(vertx_num);
    index = vertx_num .* int32(ones(1, 21) .* 21) + int32(repmat(-20:1:0, size(vertx_num, 1), 1));
    index = reshape(index', 1, []);
    max_index = size(ori_lh_cp_dwi, 2);
    index(:, any(index > max_index, 1)) = [];
    lh_cp_dwi = ori_lh_cp_dwi(:, index);
    output_name = strcat('lh_', region_name);
    save(strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.mat'), 'lh_cp_dwi');
    writematrix(lh_cp_dwi, strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.csv'));

    hemi = 'rh';
    region_name = colortable_rh.struct_names(i,1);
    region_name = region_name{1};
    region_num = colortable_rh.table(i,5);
    vertx_num = find(label_rh == region_num);
    vertx_num = int32(vertx_num);
    index = vertx_num .* int32(ones(1, 21) .* 21) + int32(repmat(-20:1:0, size(vertx_num, 1), 1));
    index = reshape(index', 1, []);
    max_index = size(ori_rh_cp_dwi, 2);
    index(:, any(index > max_index, 1)) = [];
    rh_cp_dwi = ori_rh_cp_dwi(:, index);
    output_name = strcat('rh_', region_name);
    save(strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.mat'), 'rh_cp_dwi');
    writematrix(rh_cp_dwi, strcat(output_dir, ID, '/label_coord_1mm/', output_name, '.csv'));
end
clearvars;
end