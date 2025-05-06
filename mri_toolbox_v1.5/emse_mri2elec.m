function [V] = emse_mri2elec(vert,reg)

% EMSE_MRI2ELEC: Convert mri coordinates to points in head frame
% 
% [Velec] = emse_mri2elec(vert,reg)
% 
% vert   - the Nx3 (X,Y,Z) MRI coordinates to be converted
% reg    - a structure containing coordinate transform matrices,
%          which is read using emse_open_reg
% 
% Given a point P(x,y,z) in MRI frame (eg, an fMRI activation 
% overlayed onto a high res T1 volume) this function will find 
% the corresponding location in the head space of the electrodes.
% Symbolically we have P(voxel) and want to find P(head).
% 
% 1.  Use the offset between the MRI coordinate frame and 
%     the MRI volume coordinate frame to find P(MRI-voxel).
% 2.  Given P(MRI-voxel) and the voxel size, we can find 
%     P(MRI-mm), which is the MRI coordinates expressed in mm
% 3.  The registration file contains the matrix ImageToHeadMatrix,
%     so P(head) = P(MRI-mm)*reg.mri2elec, where P(MRI-mm) is the 
%     point in MRI coordinates.
% 
% This function performs the last calculation.
% 
% See also: EMSE_OPEN_REG, EMSE_ELEC2MRI
% 

% Licence:  GNU GPL, no express or implied warranties
% History:  06/2002, Darren.Weber@flinders.edu.au
%                    EMSE details thanks to:
%                    Demetrios Voreades, Ph.D.
%                    Applications Engineer, Source Signal Imaging
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('EMSE_MRI2ELEC: In development. Be careful!\n');

% elec input is Nx3 matrix that should be represented 
% in homogenous coordinates:
vert = [ vert ones(size(vert,1),1) ];

Velec = vert * reg.mri2elec;

% Note reg.Melec ~= Velec(:,1:3) due to floating point rounding only.

% reg.mri2elec is a 4x4 matrix, eg:
%
%  -0.9525    0.0452    0.3012         0
%  -0.0522   -0.9985   -0.0154         0
%   0.3000   -0.0304    0.9534         0
%  -0.1295    0.1299    0.0756    1.0000
%
% The first 3x3 cells are the rotations,
% the last row is the translations,
% the last column is projections (usually 0),
% and the value at 4,4 is the homogenous
% coordinate scale unit, usually 1.

% In homogeneous coordinates, the last column
% is the scale factor, usually 1, but in case
% it is ~= 1
V(:,1) = Velec(:,1) ./ Velec(:,4);
V(:,2) = Velec(:,2) ./ Velec(:,4);
V(:,3) = Velec(:,3) ./ Velec(:,4);

return
