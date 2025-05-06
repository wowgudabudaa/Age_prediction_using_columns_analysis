function cp_dwi = columns_reshape(column_points)
size_r = size(column_points, 1);
size_c = size(column_points, 2);
cp_dwi = [];

for i = 1:size_r
    seg = column_points(i, :);
    seg = reshape(seg, [3, size_c/3]);
    cp_dwi = [cp_dwi, seg];
end
return

