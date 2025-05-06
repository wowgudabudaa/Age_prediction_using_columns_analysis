function column_points = generate_inner_points(pair, point_num)
column_points = repmat(pair(:, 1:3), 1, point_num);

vector = (pair(:, 4:6) - pair(:, 1:3)) ./ (point_num - 1);

for i = 0:1:(point_num-1)
    column_points(:, (i*3+1):(i*3+3)) = column_points(:, (i*3+1):(i*3+3)) + vector .* i;
end

return
