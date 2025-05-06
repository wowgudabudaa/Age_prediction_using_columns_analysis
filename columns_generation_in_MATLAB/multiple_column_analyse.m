%% This is to excute column_analyse for multiple subjects
addpath('/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/Brain_column_analyse');
fid = fopen('/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/subjects_remaining.txt','rt');
subject_list = textscan(fid, '%s');
fclose(fid);
subject_list = vertcat(subject_list{1,1});
list_size = size(subject_list, 1);
for i = 1:list_size
    subject = subject_list{i};
    column_analyse_oneMM(subject, ...
        '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/', ...
        '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/');
end
