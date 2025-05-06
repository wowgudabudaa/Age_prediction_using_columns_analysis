%% This is to get the average plot
clear
close

%% Get the subject list
addpath('/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/Brain_column_analyse');
fid = fopen('/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/subjects.txt','rt');
output_dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
subject_list = textscan(fid, '%s');
fclose(fid);
subject_list = vertcat(subject_list{1,1});
list_size = size(subject_list, 1);
subject_columns = zeros(list_size, 21);

%% Iterate for FA in all regions
for i = 1:list_size
    ID = subject_list{i};
    if isfile(strcat(output_dir, ID, '/columns/', ID, '_fa.csv'))
        columns = readmatrix(strcat(output_dir, ID, '/columns/', ID, '_fa.csv'));
        if any(isnan(columns))
            fprintf('%s has NAN values\n', ID);
            continue
        end
        subject_columns(i,:) = columns;
    end
end
subject_columns = subject_columns(any(subject_columns, 2), :);
average_column = mean(subject_columns);
std_column = std(subject_columns);

plot(0:20,average_column, 'LineStyle','-','Color','r');
hold on
plot(0:20,average_column + std_column, 'LineStyle','--','Color','g');
hold on
plot(0:20,average_column - std_column, 'LineStyle','--','Color','b');
legend('average','average+std','average-std');
grid on;
xticks([0,20]);
xticklabels({'WM/GM','Pial'});
ylabel('FA');
title('');
saveas(gcf, strcat(output_dir, 'ave_fa.jpg'),'jpg');

%% Iterate for FA in different regions
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

for k = 1:size(region_list, 2)
    region_name = region_list{k};
    rl_subject_columns = zeros(list_size, 21);
    rh_subject_columns = zeros(list_size, 21);
    lh_subject_columns = zeros(list_size, 21);
    for i = 1:list_size
        ID = subject_list{i};
        if isfile(strcat(output_dir, ID, '/columns/', ID,'_rl_',region_name,'_fa.csv'))
            rl_columns = readmatrix(strcat(output_dir, ID, '/columns/', ID,'_rl_',region_name,'_fa.csv'));
            if any(isnan(rl_columns))
                fprintf('%s has NAN values in rl file %s\n', ID, strcat(output_dir, ID, '/columns/', ID,'_rl_',region_name,'_fa.csv'));
                continue
            end
            rl_subject_columns(i,:) = rl_columns;
        end
        if isfile(strcat(output_dir, ID, '/columns/', ID,'_rh_',region_name,'_fa.csv'))
            rh_columns = readmatrix(strcat(output_dir, ID, '/columns/', ID,'_rh_',region_name,'_fa.csv'));
            if any(isnan(rh_columns))
                fprintf('%s has NAN values in rh file %s\n', ID, strcat(output_dir, ID, '/columns/', ID,'_rh_',region_name,'_fa.csv'));
                continue
            end
            rh_subject_columns(i,:) = rh_columns;
        end
        if isfile(strcat(output_dir, ID, '/columns/', ID,'_lh_',region_name,'_fa.csv'))
            lh_columns = readmatrix(strcat(output_dir, ID, '/columns/', ID,'_lh_',region_name,'_fa.csv'));
            if any(isnan(lh_columns))
                fprintf('%s has NAN values in lh file %s\n', ID, strcat(output_dir, ID, '/columns/', ID,'_lh_',region_name,'_fa.csv'));
                continue
            end
            lh_subject_columns(i,:) = lh_columns;
        end
    end
    rl_subject_columns = rl_subject_columns(any(rl_subject_columns, 2), :);
    ave_rl_subject_columns = mean(rl_subject_columns);
    std_rl_subject_columns = std(rl_subject_columns);

    rh_subject_columns = rh_subject_columns(any(rh_subject_columns, 2), :);
    ave_rh_subject_columns = mean(rh_subject_columns);
    std_rh_subject_columns = std(rh_subject_columns);

    lh_subject_columns = lh_subject_columns(any(lh_subject_columns, 2), :);
    ave_lh_subject_columns = mean(lh_subject_columns);
    std_lh_subject_columns = std(lh_subject_columns);

    figure('Visible', 'off');
    plot(0:20,ave_rl_subject_columns, 'LineStyle','-','Color','r','LineWidth',0.7);
    hold on
    plot(0:20,ave_rh_subject_columns, 'LineStyle','-','Color','g','LineWidth',0.7);
    hold on
    plot(0:20,ave_lh_subject_columns, 'LineStyle','-','Color','b','LineWidth',0.7);
    legend('rl','rh','lh');
    hold on
    plot(0:20,ave_rl_subject_columns + std_rl_subject_columns, 'LineStyle','--','Color','r');
    hold on
    plot(0:20,ave_rl_subject_columns - std_rl_subject_columns, 'LineStyle','--','Color','r');
    hold on;
    plot(0:20,ave_rh_subject_columns + std_rh_subject_columns, 'LineStyle','--','Color','g');
    hold on
    plot(0:20,ave_rh_subject_columns - std_rh_subject_columns, 'LineStyle','--','Color','g');
    hold on;
    plot(0:20,ave_lh_subject_columns + std_lh_subject_columns, 'LineStyle','--','Color','b');
    hold on
    plot(0:20,ave_lh_subject_columns - std_lh_subject_columns, 'LineStyle','--','Color','b');
    grid on;
    hold off
    xticks([0,20]);
    xticklabels({'WM/GM','Pial'});
    ylabel('FA');
    title(strcat('ave_',strrep(region_name, '_', ' ')));
    saveas(gcf, strcat(output_dir,'ave_',region_name,'_fa.jpg'),'jpg');
end
% figure;
% volumn_size = size(volumn, [1,2]);
% dwi_volumn = zeros(volumn_size);
% rows = rl_cp_dwi(1, rl_cp_dwi(3,:)==68);
% cols = rl_cp_dwi(2, rl_cp_dwi(3,:)==68);
% dwi_volumn(cols(21:42), rows(21:42)) = max(max(volumn(:, :, 68)))+1;
% imagesc(volumn(:, :, 68) + dwi_volumn);
