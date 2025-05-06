function [ avw ] = avw_flip(avw,dims)

% AVW_FLIP: ortho-flip Analyze data image (avw.img)
% 
% [ avw ] = avw_flip(avw,dims)
% 
% where avw is the data struct returned from avw_img_read
% and dims is a cell array of strings, eg: dims = {'x','y','z'}
% with any number of cells, in any order.
% 
% For each cell of dims, the dimension of avw.img is flipped, 
% independently of all other dimensions and the order in which
% they are specified.  There are no 3D rotations, which are 
% not orthogonal flips.
% 
% This function will most likely invalidate the .img orientation, 
% so use it carefully.  There is no attempt in this function
% to maintain the validity of the avw.hdr.hist.orient field.
% 
% see also AVW_IMG_READ, AVW_VIEW
% 


% Licence:  GNU GPL, no express or implied warranties
% History:  01/2003, Darren.Weber@flinders.edu.au
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('\nAVW_FLIP\n');
fprintf('...warning, this function can invalidate the .img orientation\n');

for i = 1:length(dims),
    dim = lower(dims{i});
    switch dim,
    case 'x',
        fprintf('...flipping X\n');
        xdim = size(avw.img,1);
        avw.img = avw.img(xdim:-1:1,:,:);
    case 'y',
        fprintf('...flipping Y\n');
        ydim = size(avw.img,2);
        avw.img = avw.img(:,ydim:-1:1,:);
    case 'z',
        fprintf('...flipping Z\n');
        zdim = size(avw.img,3);
        avw.img = avw.img(:,:,zdim:-1:1);
    end
end

return
