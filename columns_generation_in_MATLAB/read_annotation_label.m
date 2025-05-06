%% This is to read annotation file

dir = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/output/';
subject_id = 'S00775';
annot_file_1 = 'lh.aparc.annot';
file_name_1 = strcat(dir, subject_id, '/', subject_id, '/label/', annot_file_1);
[vertices,label,colortable]=read_annotation(file_name_1);

annot_file_2 = 'lh.aparc.a2009s.annot';
file_name_2 = strcat(dir, subject_id, '/', subject_id, '/label/', annot_file_2);
[vertices_2009,label_2009,colortable_2009]=read_annotation(file_name_2);

annot_file_3 = 'lh.aparc.DKTatlas.annot';
file_name_3 = strcat(dir, subject_id, '/', subject_id, '/label/', annot_file_3);
[vertices_DKT,label_DKT,colortable_DKT]=read_annotation(file_name_3);

annot_file_4 = 'lh.aparc.a2009s.annot';
file_name_4 = strcat(dir, subject_id, '/', subject_id, '/label/', annot_file_4);
[vertices_BA,label_BA,colortable_BA]=read_annotation(file_name_4);
