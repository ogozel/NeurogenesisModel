% Function to plot histograms of projections on discriminability axes
% labels: all labels
% A, B: labels of patterns we are interested in

function fct_hist_projection(projected_points,labels,A,B,C,tmprange,tmprangeY) 

% Range of histogram - x axis
tmpstep = tmprange(end) - tmprange(end-1);
x = min(tmprange)+(tmpstep/2):tmpstep:max(tmprange);

% Determine colors for plot
if A==3
    tmpColorA = 'blue';
elseif A==4
    tmpColorA = 'green';
elseif A==5
    tmpColorA = 'red';
else
    tmpColorA = 'none';
end

if B==3
    tmpColorB = 'blue';
elseif B==4
    tmpColorB = 'green';
elseif B==5
    tmpColorB = 'red';
else
    tmpColorB = 'none';
end

if C==3
    tmpColorC = 'blue';
elseif C==4
    tmpColorC = 'green';
elseif C==5
    tmpColorC = 'red';
else
    tmpColorC = 'none';
end

% Separate projection points per digit
proj_A = projected_points(labels==A);
proj_B = projected_points(labels==B);
proj_C = projected_points(labels==C);

[nA, xA] = histcounts(proj_A,tmprange);
[nB, xB] = histcounts(proj_B,tmprange);
[nC, xC] = histcounts(proj_C,tmprange);
y = [nA;nB;nC]';

figure()
b = bar(x,y);
b(1).FaceColor = tmpColorA;
b(2).FaceColor = tmpColorB;
b(3).FaceColor = tmpColorC;
box off
set(gca,'TickDir','out')
xlim([min(tmprange), max(tmprange)])
ylim(tmprangeY)
pbaspect([4 1 1])
xlabel('Projection scalar')
title(strcat('Axis ', num2str(A), ' -> ', num2str(B)))



end