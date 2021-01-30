% Function to project points on a line
% NB: the line goes through the origin, hence the points should be
% normalized (mean zero)

function [proj] = fct_project_points_on_line(points_to_be_projected,direction_axis)

% Project activity on axis - just get the scalar value
proj = sum(points_to_be_projected .* direction_axis) / sum(direction_axis .* direction_axis);



end