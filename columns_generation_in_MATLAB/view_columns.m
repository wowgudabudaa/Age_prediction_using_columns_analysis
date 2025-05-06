%% This file is to read functional mri image and display columns
close
clear
clc

%% Load functional mri image
ID = 'S04491';
fprintf('The subject is: %s\n', ID);
input_dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
output_dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
dwi = MRIread(strcat(input_dir, ID, '/', ID, '_fa_upsampled.nii.gz'), 0, 1);
volumn = dwi.vol;

% dwi.vol(FSrow+1, FScol+1, FSslice+1)=dwi(FScol, FSrow, FSslice)
load(strcat(output_dir, ID, '/columns/', ID,'_column_lh_dwi.mat'));
load(strcat(output_dir, ID, '/columns/', ID,'_column_rh_dwi.mat'));

%% Write image matrix with columns
points_num = 21;% Should be consistent with the setting from `vertices_connect.m`

% lh_cp_dwi = round(lh_cp_dwi);
% lh_cp_dwi(lh_cp_dwi==0) = 1;
% rh_cp_dwi = round(rh_cp_dwi);
% rh_cp_dwi(rh_cp_dwi==0) = 1;

rl_cp_dwi = [lh_cp_dwi(:,:), rh_cp_dwi(:,:)];
points_size = size(rl_cp_dwi, 2);
rl_values = zeros(1, points_num);

columns_num = points_size / 21;
    for i = 1:21
        index = i:21:(i + 21 * (columns_num-1));
        interp_volumn = interp3(volumn, rl_cp_dwi(1, index), rl_cp_dwi(2, index), rl_cp_dwi(3, index));
        [r, c] = find(isnan(interp_volumn)==true);
        [rl_cp_dwi(1, index(r,c)); rl_cp_dwi(2, index(r,c)); rl_cp_dwi(3, index(r,c))]
        rl_values(1,i) = sum(interp_volumn);
    end

rl_values = rl_values ./ (points_size / points_num);

% figure('Visible', 'off');
plot(0:20,rl_values);
grid on;
xticks([0,20]);
xticklabels({'WM/GM','Pial'});
ylabel('FA');
title(ID);
saveas(gcf, strcat(output_dir, ID, '/columns/', ID, '_fa.jpg'),'jpg');
writematrix(rl_values, strcat(output_dir, ID, '/columns/', ID, '_fa.csv'));
% figure;
% volumn_size = size(volumn, [1,2]);
% dwi_volumn = zeros(volumn_size);
% rows = rl_cp_dwi(1, rl_cp_dwi(3,:)==68);
% cols = rl_cp_dwi(2, rl_cp_dwi(3,:)==68);
% dwi_volumn(cols(21:42), rows(21:42)) = max(max(volumn(:, :, 68)))+1;
% imagesc(volumn(:, :, 68) + dwi_volumn);


