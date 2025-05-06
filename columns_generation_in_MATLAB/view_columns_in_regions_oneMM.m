function view_columns_in_rergions_oneMM(ID,input_dir,output_dir)
%% This file is to read functional mri image and display columns
region_list = {'BA1_exvivo_thresh', 'BA1_exvivo',...
               'BA2_exvivo_thresh', 'BA2_exvivo',...
               'BA3a_exvivo_thresh', 'BA3a_exvivo',...
               'BA3b_exvivo_thresh', 'BA3b_exvivo',...
               'BA4a_exvivo_thresh', 'BA4a_exvivo',...
               'BA4p_exvivo_thresh', 'BA4p_exvivo',...
               'BA6_exvivo_thresh', 'BA6_exvivo',...
               'BA44_exvivo_thresh', 'BA44_exvivo',...
               'BA45_exvivo_thresh', 'BA45_exvivo',...
               'cortex', 'cortex+hipamyg',...
               'entorhinal_exvivo_thresh', 'entorhinal_exvivo',...
               'FG1_mpm_vpnl', 'FG2_mpm_vpnl',...
               'FG3_mpm_vpnl', 'FG4_mpm_vpnl',...
               'hOc1_mpm_vpnl', 'hOc2_mpm_vpnl',...
               'hOc3v_mpm_vpnl', 'hOc4v_mpm_vpnl',...
               'MT_exvivo_thresh', 'MT_exvivo',...
               'nofix_cortex',...
               'perirhinal_exvivo_thresh', 'perirhinal_exvivo',...
               'V1_exvivo_thresh', 'V1_exvivo',...
               'V2_exvivo_thresh', 'V2_exvivo'};

%% Load functional mri image
%ID = 'S00775';
fprintf('The subject is: %s \n', ID);
%input_dir = '$PAROS/paros_WORK/hanwen/ad_decode_test/output/';
%output_dir = '$PAROS/paros_WORK/hanwen/ad_decode_test/output/';
dwi = MRIread(strcat('/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/input/', ID, '/', ID, '_ad_masked.nii.gz'), 0, 1);
volumn = dwi.vol;

for k = 1:size(region_list, 2)
    region_name = region_list{k};
    % dwi.vol(FSrow+1, FScol+1, FSslice+1)=dwi(FScol, FSrow, FSslice)
    lh_region_name = strcat('lh_', region_name, '.mat');
    rh_region_name = strcat('rh_', region_name, '.mat');
    if ~isfile(strcat(output_dir, ID, '/label_coord_1mm/', lh_region_name))
        fprintf('Subject %s doesnt have label files\n', ID)
        return
    end
    load(strcat(output_dir, ID, '/label_coord_1mm/', lh_region_name));
    load(strcat(output_dir, ID, '/label_coord_1mm/', rh_region_name));

    %% Write image matrix with columns
    points_num = 21;% Should be consistent with the setting from `vertices_connect.m`

%     lh_cp_dwi(lh_cp_dwi==0) = 1;
%     rh_cp_dwi(rh_cp_dwi==0) = 1;
    
    rl_cp_dwi = [lh_cp_dwi(:,:), rh_cp_dwi(:,:)];% [Cx, Cy, Cz; Cx, Cy, Cz]'
%    rl_cp_dwi = round(rl_cp_dwi);
    points_size = size(rl_cp_dwi, 2);
    rl_values = zeros(1, points_num);
    lh_values = zeros(1, points_num);
    rh_values = zeros(1, points_num);
    
    % collect values in left hemi
    columns_num = size(lh_cp_dwi, 2) / 21;
    for i = 1:21
        index = i:21:(i + 21 * (columns_num-1));
        interp_volumn = interp3(volumn, lh_cp_dwi(1, index), lh_cp_dwi(2, index), lh_cp_dwi(3, index));
        lh_values(1,i) = sum(interp_volumn);
    end

%     j = 0;
%     for i = 1:size(lh_cp_dwi, 2)
%         j = j + 1;
% %         rl_values(1,j) = rl_values(1,j) + interp3(volumn, rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
% %         lh_values(1,j) = lh_values(1,j) + interp3(volumn, rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
%         rl_values(1,j) = rl_values(1,j) + volumn(rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
%         lh_values(1,j) = lh_values(1,j) + volumn(rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
%         if j == 21
%             j = 0;
%         end
%     end
    % collect values in right hemi
    columns_num = size(rh_cp_dwi, 2) / 21;
    for i = 1:21
        index = i:21:(i + 21 * (columns_num-1));
        rh_values(1,i) = sum(interp3(volumn, rh_cp_dwi(1, index), rh_cp_dwi(2, index), rh_cp_dwi(3, index)));
    end
%     j = 0;
%     for i = size(lh_cp_dwi, 2)+1:points_size
%         j = j + 1;
% %         rl_values(1,j) = rl_values(1,j) + interp3(volumn, rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
% %         rh_values(1,j) = rh_values(1,j) + interp3(volumn, rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
%         rl_values(1,j) = rl_values(1,j) + volumn(rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
%         rh_values(1,j) = rh_values(1,j) + volumn(rl_cp_dwi(2, i), rl_cp_dwi(1, i), rl_cp_dwi(3, i));
%         if j == 21
%             j = 0;
%         end
%     end
    
%     rl_values = rl_values ./ (points_size / points_num);
    rl_values = (lh_values + rh_values) ./ (points_size / points_num);
    lh_values = lh_values ./ (size(lh_cp_dwi, 2) / points_num);
    rh_values = rh_values ./ (size(rh_cp_dwi, 2) / points_num);
    
    fig = figure('Visible', 'off');
    plot(0:20,rl_values, 'LineStyle','--');
    hold on
    plot(0:20, lh_values);
    hold on
    plot(0:20, rh_values);
    legend('lr', 'lh', 'rh');
    grid on;
    xticks([0,20]);
    xticklabels({'WM/GM','Pial'});
    ylabel('AD');
    title(strcat(ID,' ',strrep(region_name, '_', ' ')));
    saveas(gcf, strcat(output_dir, ID, '/columns_1mm/', ID,'_',region_name,'_ad.jpg'),'jpg');
    writematrix(rl_values, strcat(output_dir, ID, '/columns_1mm/', ID,'_rl_',region_name,'_ad.csv'));
    writematrix(lh_values, strcat(output_dir, ID, '/columns_1mm/', ID,'_lh_',region_name,'_ad.csv'));
    writematrix(rh_values, strcat(output_dir, ID, '/columns_1mm/', ID,'_rh_',region_name,'_ad.csv'));
    close(fig);
end
clearvars;
end