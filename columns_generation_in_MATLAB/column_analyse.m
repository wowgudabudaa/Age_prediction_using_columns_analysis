function column_analyse(ID,input_dir,output_dir)
%% COLUMN_ANALYSE A wrapper script for the freesurfer-based cortical column analyse
% Tetst for bash/MATLAB variable mixing:
% input_test=strsplit(input_dir,'/')
% topdir=input_test{1}
% if strcmp(topdir(1),'$')
%     topdir_env=topdir(2:end)
%     td=getenv(topdir_env);
%     input_dir=strrep(input_dir,topdir,td)
% end
% 
% output_test=strsplit(output_dir,'/')
% topdir=output_test{1};
% if strcmp(topdir(1),'$')
%     topdir_env=topdir(2:end)
%     td=getenv(topdir_env);
%     output_dir=strrep(output_dir,topdir,td)
% end

% Begin scripts
if isfolder(strcat(output_dir, ID))
    fprintf('Subject %s start \n', ID);
%     vertices_connect_fn(ID,input_dir,output_dir);
%     coordinates_in_regions_fn(ID,input_dir,output_dir);
    view_columns_fn(ID,input_dir,output_dir);
%     view_columns_in_regions_fn(ID,input_dir,output_dir);
    fprintf('Subject %s end \n', ID)
else
    fprintf('Subject %s doesnt exit \n', ID);
end