% Function to determine the direction of the line passing through two
% points, and the projections of points on this line

function [proj1, proj2, proj3, W1, W2, W3, origin1, origin2, origin3, tmpReverse] = fct_lda(Master_activity_DGC,labels)

tmpReverse = zeros(3,1);

% separate the activity for each digit
Master_activity_DGC_digit3 = Master_activity_DGC(:,labels==3);
Master_activity_DGC_digit4 = Master_activity_DGC(:,labels==4);
Master_activity_DGC_digit5 = Master_activity_DGC(:,labels==5);

% compute the center of mass of the representation of each digit
mean_activity_DGC_digit3 = mean(Master_activity_DGC_digit3,2);
mean_activity_DGC_digit4 = mean(Master_activity_DGC_digit4,2);
mean_activity_DGC_digit5 = mean(Master_activity_DGC_digit5,2);

% compute the covariance matrix of the activity for each digit
covMat_activity_DGC_digit3 = cov(Master_activity_DGC_digit3');
covMat_activity_DGC_digit4 = cov(Master_activity_DGC_digit4');
covMat_activity_DGC_digit5 = cov(Master_activity_DGC_digit5');

% Axis1: digits 3 and 4
% Within-class scatter matrix
Sw_axis1 = covMat_activity_DGC_digit3 + covMat_activity_DGC_digit4;
% Between-class scatter matrix
Sb_axis1 = (mean_activity_DGC_digit3-mean_activity_DGC_digit4)*(mean_activity_DGC_digit3-mean_activity_DGC_digit4)';
% Generalized eigenvalue problem: (Sw)'w(Sb)ww = (lambda) w
% Compute LDA projection
[V1, ~] = eig(inv(Sw_axis1)*Sb_axis1);
W1 = V1(:,1);
% Project activity on axis1
proj1_cm3 = fct_project_points_on_line(mean_activity_DGC_digit3,W1);
proj1_cm4 = fct_project_points_on_line(mean_activity_DGC_digit4,W1);
origin1 = mean(fct_project_points_on_line(cat(2,Master_activity_DGC_digit3,Master_activity_DGC_digit4,Master_activity_DGC_digit5),W1));
% Make sure that the axis goes from 3 to 4
if proj1_cm3 > proj1_cm4
    tmpReverse(1) = -1;
else
    tmpReverse(1) = 1;
end
proj1 = tmpReverse(1)*fct_project_points_on_line(Master_activity_DGC,W1) - origin1*ones(1,length(Master_activity_DGC));

% Axis2: digits 3 and 5
% Within-class scatter matrix
Sw_axis2 = covMat_activity_DGC_digit3 + covMat_activity_DGC_digit5;
% Between-class scatter matrix
Sb_axis2 = (mean_activity_DGC_digit3-mean_activity_DGC_digit5)*(mean_activity_DGC_digit3-mean_activity_DGC_digit5)';
% Compute LDA projection
[V2, ~] = eig(inv(Sw_axis2)*Sb_axis2);
W2 = V2(:,1);
% Project activity on axis2
proj2_cm3 = fct_project_points_on_line(mean_activity_DGC_digit3,W2);
proj2_cm5 = fct_project_points_on_line(mean_activity_DGC_digit5,W2);
origin2 = mean(fct_project_points_on_line(cat(2,Master_activity_DGC_digit3,Master_activity_DGC_digit4,Master_activity_DGC_digit5),W2));
if proj2_cm3 > proj2_cm5
    tmpReverse(2) = -1;
else
    tmpReverse(2) = 1;
end
proj2 = tmpReverse(2)*(fct_project_points_on_line(Master_activity_DGC,W2) - origin2*ones(1,length(Master_activity_DGC)));

% Axis3: digits 4 and 5
% Within-class scatter matrix
Sw_axis3 = covMat_activity_DGC_digit4 + covMat_activity_DGC_digit5;
% Between-class scatter matrix
Sb_axis3 = (mean_activity_DGC_digit4-mean_activity_DGC_digit5)*(mean_activity_DGC_digit4-mean_activity_DGC_digit5)';
% Generalized eigenvalue problem: (Sw)'w(Sb)ww = (lambda) w
% Compute LDA projection
[V3, ~] = eig(inv(Sw_axis3)*Sb_axis3);
W3 = V3(:,1);
% Project activity on axis3
proj3_cm4 = fct_project_points_on_line(mean_activity_DGC_digit4,W3);
proj3_cm5 = fct_project_points_on_line(mean_activity_DGC_digit5,W3);
origin3 = mean(fct_project_points_on_line(cat(2,Master_activity_DGC_digit3,Master_activity_DGC_digit4,Master_activity_DGC_digit5),W3));
if proj3_cm4 > proj3_cm5
    tmpReverse(3) = -1;
else
    tmpReverse(3) = 1;
end
proj3 = tmpReverse(3)*(fct_project_points_on_line(Master_activity_DGC,W3) - origin3*ones(1,length(Master_activity_DGC)));


end