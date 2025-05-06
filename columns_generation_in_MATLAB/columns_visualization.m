%% This is to make a label file that can be read in Freeview to visualize the columns

output_dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
ID = 'S00775';
trans_M_dir = strcat(output_dir,ID,'/DWI2T1_dti_upsampled.dat');

%% Load filename list
fid = fopen(strcat(output_dir,ID,'/file_list_local.txt'),'rt');
file_list = textscan(fid, '%s');
fclose(fid);
file_list = vertcat(file_list{1,1});
list_size = size(file_list, 1);
if ~isfile(strcat(output_dir, ID, '/columns/',ID, '_column_lh.mat'))
    fprintf('Subject %s doesnt have columns reslut', ID);
    return
end
load(strcat(output_dir, ID, '/columns/',ID, '_column_lh.mat'));
load(strcat(output_dir, ID, '/columns/',ID, '_column_rh.mat'));

%% Iterate for every label file
filename = file_list{35};
fid = fopen(filename, 'rt');
vertices = textscan(fid, '%d %f %f %f %f','Headerlines', 2);
vertx_num = vertices{1}+1;% vertx index, eg. 1,2,3
last_slash = strfind(filename, '/');
output_name = filename(last_slash(end)+1:end);
output_name = erase(output_name, '.label');
first_dot = strfind(output_name, '.');
hemi = output_name(1:first_dot(1)-1);
output_name = strrep(output_name, '.', '_');
% output_name = strcat(output_dir, ID, '/label_coord/', output_name, '.mat');

if hemi == 'lh'
    lh_region_column_points = lh_column_points(vertx_num,:);
    lh_points = reshape(lh_region_column_points', 3, [])';
    header = '#!ascii label  , from subject  vox2ras=TkReg';
    lh_cp_dwi = columns_reshape(lh_region_column_points);
    lh_cp_dwi = [lh_cp_dwi; ones(1, size(lh_cp_dwi, 2))];%RAS coordinates in anatomical domain
    lh_cp_dwi = inv(T_mov) * trans_M * lh_cp_dwi;%CRS coordinates in functional domain
end

if hemi == 'rh'
    rh_region_column_points = rh_column_points(vertx_num,:);
    rh_points = reshape(rh_region_column_points', 3, [])';
    header = '#!ascii label  , from subject  vox2ras=TkReg';
    [m, n] = size(rh_points);
    coord_text = [ones([m, 1]) * -1, rh_points, zeros([m, 1])];
    fid = fopen('S00775_visualize.label', 'w');
    fprintf(fid, header);
    fprintf(fid, '\n');
    fprintf(fid, num2str(m));
    fprintf(fid, '\n');
    for i = 1:m
        fprintf(fid, '%d\t', coord_text(i, 1:end-1));
        fprintf(fid, '%d\n', coord_text(i, end));
    end
    fprintf(fid, '\n');
    fclose(fid);
end
