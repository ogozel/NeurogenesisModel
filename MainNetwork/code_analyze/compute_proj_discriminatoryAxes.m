% Project end DGC firing rates onto discriminatory axes (dimensionality
% reduction)

clear all
close all
clc


load('../dataset_mnist/mnist144.mat')

% Data at the end of the late phase of neurogenesis case (3,4 + 5)
load('../data_mnist/1epochs_neuroPhase2_3  4  5_onlyFewPlastic.mat')

bool_plotDistributions = 0; % 0 or 1

% Set the colors in RGB format
color3 = [0, 0, 255]/255;
color4 = [0, 128, 0]/255;
color5 = [255, 0, 0]/255;
color3example = [0, 204, 255]/255;
color4example = [0, 204, 0]/255;
color5example = [255, 153, 0]/255;

rangeY = [0, 3000];


%% Discriminability analysis - use the training data to construct the axes...

%%%% !!! Choose the training data at the end of the late phase, or at the
%%%% end of the early phase !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %%%%%%%%%%%%%
load('Master_activity_DGC_neuro345_late_train.mat') % load('Master_activity_DGC_neuro345_late_train.mat') OR load('Master_activity_DGC_neuro345_early_train.mat')


% Select the digits on the training set
idx = ismember(traininglabels,patterns_num);
labels = traininglabels(idx);


Master_activity_DGC = Master_activity_DGC_neuro345_late_train; % Master_activity_DGC_neuro345_late_train; OR Master_activity_DGC_neuro345_early_train;


% Determine the scalar values of the projections of the DGC activity on
% discriminability axis
[proj1, proj2, proj3, W1, W2, W3, origin1, origin2, origin3, tmpReverse] = fct_lda(Master_activity_DGC,labels);


%% ... and then project the test data on these axis

rangeY = [0, 520];

load('Master_activity_DGC_neuro345_late_test.mat')

% Select the digits on the testing set
idx = ismember(testlabels,patterns_num);
labels = testlabels(idx);

Master_activity_DGC = Master_activity_DGC_neuro345_late_test;

proj1_late = tmpReverse(1)*(fct_project_points_on_line(Master_activity_DGC, W1) - origin1*ones(1,length(Master_activity_DGC)));
proj2_late = tmpReverse(2)*(fct_project_points_on_line(Master_activity_DGC, W2) - origin2*ones(1,length(Master_activity_DGC)));
proj3_late = tmpReverse(3)*(fct_project_points_on_line(Master_activity_DGC, W3) - origin3*ones(1,length(Master_activity_DGC)));


% Represent the distributions of the projections on the discriminability
% axes
if bool_plotDistributions
    
    fct_hist_projection(proj1_late,labels,3,4,5,rangeX,rangeY)
    if min(proj1_late) < min(rangeX)
        display('rangeX does not start small enough!')
    end
    if max(proj1_late) > max(rangeX)
        display('rangeX does not finish big enough!')
    end
    fct_hist_projection(proj2_late,labels,3,5,4,rangeX,rangeY)
    if min(proj2_late) < min(rangeX)
        display('rangeX does not start small enough!')
    end
    if max(proj2_late) > max(rangeX)
        display('rangeX does not finish big enough!')
    end
    fct_hist_projection(proj3_late,labels,4,5,3,rangeX,rangeY)
    if min(proj3_late) < min(rangeX)
        display('rangeX does not start small enough!')
    end
    if max(proj3_late) > max(rangeX)
        display('rangeX does not finish big enough!')
    end
    
end



%% Look at the end of the early phase

load('Master_activity_DGC_neuro345_early_test.mat')

Master_activity_DGC = Master_activity_DGC_neuro345_early_test;

% Select the digits on the testing set
idx = ismember(testlabels,patterns_num);
patterns = testimages(:,idx);
labels = testlabels(idx);

proj1_early = tmpReverse(1)*(fct_project_points_on_line(Master_activity_DGC, W1) - origin1*ones(1,length(Master_activity_DGC)));
proj2_early = tmpReverse(2)*(fct_project_points_on_line(Master_activity_DGC, W2) - origin2*ones(1,length(Master_activity_DGC)));
proj3_early = tmpReverse(3)*(fct_project_points_on_line(Master_activity_DGC, W3) - origin3*ones(1,length(Master_activity_DGC)));


% Represent the distributions of the projections on the discriminability
% axis
if bool_plotDistributions
    
    fct_hist_projection(proj1_early,labels,3,4,5,rangeX,rangeY)
    if min(proj1_early) < min(rangeX)
        display('rangeX does not start small enough!')
    end
    if max(proj1_early) > max(rangeX)
        display('rangeX does not finish big enough!')
    end
    fct_hist_projection(proj2_early,labels,3,5,4,rangeX,rangeY)
    if min(proj2_early) < min(rangeX)
        display('rangeX does not start small enough!')
    end
    if max(proj2_early) > max(rangeX)
        display('rangeX does not finish big enough!')
    end
    fct_hist_projection(proj3_early,labels,4,5,3,rangeX,rangeY)
    if min(proj3_early) < min(rangeX)
        display('rangeX does not start small enough!')
    end
    if max(proj3_early) > max(rangeX)
        display('rangeX does not finish big enough!')
    end
    
end



%% Select particular examples of patterns 5 (test patterns and labels)

indices = [909, 718, 2358];
cmap = [color3example; color4example; color5example];


%% Plot the example patterns

figure()
for n=1:length(indices)
    subplot(1,length(indices),n)
    imagesc(reshape(normc(patterns(:,indices(n))),[12,12]))
    box off
    set(gca,'xTick',[],'yTick',[])
    pbaspect([1 1 1])
end
sgtitle('Example patterns')


%% Plot 2D projections on axes 3->4, and 3->5

% Determine how many patterns of each category we have
idx3 = find(labels==3);
idx4 = find(labels==4);
idx5 = find(labels==5);
numPat3 = length(idx3);
numPat4 = length(idx4);
numPat5 = length(idx5);

% Possibility to keep only a certain percentage of them
percKeep = 1;
erasePat3 = randperm(numPat3, numPat3 - round(percKeep*numPat3));
labels(idx3(erasePat3)) = 0;
erasePat4 = randperm(numPat4, numPat4 - round(percKeep*numPat4));
labels(idx4(erasePat4)) = 0;
erasePat5 = randperm(numPat5, numPat5 - round(percKeep*numPat5));
labels(idx5(erasePat5)) = 0;

figure()
rangeX = linspace(-0.4,0.1,50+1);
rangeY = linspace(-0.17,0.17,50+1);

plot( proj1_early(labels==3), proj2_early(labels==3),'o','color',color3,'MarkerSize',1)
hold on
plot( proj1_early(labels==4), proj2_early(labels==4),'o','color',color4,'MarkerSize',1)
plot( proj1_early(labels==5), proj2_early(labels==5),'o','color',color5,'MarkerSize',1)
for i=1:length(indices)
    plot(proj1_early(indices(i)),proj2_early(indices(i)),'marker','x','color',cmap(i,:),'Linewidth',14)
end
xlabel('Projection value on axis1 (3->4)')
ylabel('projection value on axis 2 (3->5)')
title('Early phase')
xlim([min(rangeX) max(rangeX)])
ylim([min(rangeY) max(rangeY)])
box off
set(gca, 'TickDir', 'out')
pbaspect([1 1 1])

figure()
rangeX = linspace(-0.25,0.25,50+1);
rangeY = linspace(-0.17,0.17,50+1);

plot( proj1_late(labels==3), proj2_late(labels==3),'o','color',color3,'MarkerSize',1)
hold on
plot( proj1_late(labels==4), proj2_late(labels==4),'o','color',color4,'MarkerSize',1)
plot( proj1_late(labels==5), proj2_late(labels==5),'o','color',color5,'MarkerSize',1)
for i=1:length(indices)
    plot(proj1_late(indices(i)),proj2_late(indices(i)),'marker','x','color',cmap(i,:),'linewidth',14)
end
xlabel('Projection value on axis 1 (3->4)')
ylabel('Projection value on axis 2 (3->5)')
title('Late phase')
xlim([min(rangeX) max(rangeX)])
ylim([min(rangeY) max(rangeY)])
box off
set(gca, 'TickDir', 'out')
pbaspect([1 1 1])


%% Plot 2D projections on axis 3->5, and 4->5

figure()
rangeX = linspace(-0.17,0.17,50+1);
rangeY = linspace(-0.32,0.38,50+1);

plot( proj2_early(labels==3), proj3_early(labels==3),'o','color',color3,'MarkerSize',1)
hold on
plot( proj2_early(labels==4), proj3_early(labels==4),'o','color',color4,'MarkerSize',1)
plot( proj2_early(labels==5), proj3_early(labels==5),'o','color',color5,'MarkerSize',1)
for i=1:length(indices)
    plot(proj2_early(indices(i)),proj3_early(indices(i)),'marker','x','color',cmap(i,:),'Linewidth',14)
end
xlabel('Projection value on axis2 (3->5)')
ylabel('projection value on axis 3 (4->5)')
title('Early phase')
xlim([min(rangeX) max(rangeX)])
ylim([min(rangeY) max(rangeY)])
box off
set(gca, 'TickDir', 'out')
pbaspect([1 1 1])

figure()
rangeX = linspace(-0.17,0.17,50+1);
rangeY = linspace(-0.32,0.38,50+1);

plot( proj2_late(labels==3), proj3_late(labels==3),'o','color',color3,'MarkerSize',1)
hold on
plot( proj2_late(labels==4), proj3_late(labels==4),'o','color',color4,'MarkerSize',1)
plot( proj2_late(labels==5), proj3_late(labels==5),'o','color',color5,'MarkerSize',1)
for i=1:length(indices)
    plot(proj2_late(indices(i)),proj3_late(indices(i)),'marker','x','color',cmap(i,:),'linewidth',14)
end
xlabel('Projection value on axis2 (3->5)')
ylabel('projection value on axis 3 (4->5)')
title('Late phase')
xlim([min(rangeX) max(rangeX)])
ylim([min(rangeY) max(rangeY)])
box off
set(gca, 'TickDir', 'out')
pbaspect([1 1 1])




%% Compute classification performance based on the LDA fisher discriminability axes [%]

l3 = (labels==3);
l4 = (labels==4);
l5 = (labels==5);


% Confusion matrix
% Rows: predicted label (3,4,5), Columns: real label (3,4,5)
% Last row: unclassified patterns (contradictory results for the projections)


% early phase
confusion_matrix_early = [fct_classPerf_lda(proj1_early,l3,-1,proj2_early,-1), ...
    fct_classPerf_lda(proj1_early,l4,-1,proj2_early,-1), ...
    fct_classPerf_lda(proj1_early,l5,-1,proj2_early,-1); ...
    fct_classPerf_lda(proj1_early,l3,1,proj3_early,-1), ...
    fct_classPerf_lda(proj1_early,l4,1,proj3_early,-1), ...
    fct_classPerf_lda(proj1_early,l5,1,proj3_early,-1); ...
    fct_classPerf_lda(proj2_early,l3,1,proj3_early,1), ...
    fct_classPerf_lda(proj2_early,l4,1,proj3_early,1), ...
    fct_classPerf_lda(proj2_early,l5,1,proj3_early,1);
    1 - (fct_classPerf_lda(proj1_early,l3,-1,proj2_early,-1)+fct_classPerf_lda(proj1_early,l3,1,proj3_early,-1)+fct_classPerf_lda(proj2_early,l3,1,proj3_early,1)), ...
    1 - (fct_classPerf_lda(proj1_early,l4,-1,proj2_early,-1)+fct_classPerf_lda(proj1_early,l4,1,proj3_early,-1)+fct_classPerf_lda(proj2_early,l4,1,proj3_early,1)), ...
    1 - (fct_classPerf_lda(proj1_early,l5,-1,proj2_early,-1)+fct_classPerf_lda(proj1_early,l5,1,proj3_early,-1)+fct_classPerf_lda(proj2_early,l5,1,proj3_early,1))]

% late phase
confusion_matrix_late = [fct_classPerf_lda(proj1_late,l3,-1,proj2_late,-1), ...
    fct_classPerf_lda(proj1_late,l4,-1,proj2_late,-1), ...
    fct_classPerf_lda(proj1_late,l5,-1,proj2_late,-1); ...
    fct_classPerf_lda(proj1_late,l3,1,proj3_late,-1), ...
    fct_classPerf_lda(proj1_late,l4,1,proj3_late,-1), ...
    fct_classPerf_lda(proj1_late,l5,1,proj3_late,-1); ...
    fct_classPerf_lda(proj2_late,l3,1,proj3_late,1), ...
    fct_classPerf_lda(proj2_late,l4,1,proj3_late,1), ...
    fct_classPerf_lda(proj2_late,l5,1,proj3_late,1); ...
    1 - (fct_classPerf_lda(proj1_late,l3,-1,proj2_late,-1)+fct_classPerf_lda(proj1_late,l3,1,proj3_late,-1)+fct_classPerf_lda(proj2_late,l3,1,proj3_late,1)), ...
    1 - (fct_classPerf_lda(proj1_late,l4,-1,proj2_late,-1)+fct_classPerf_lda(proj1_late,l4,1,proj3_late,-1)+fct_classPerf_lda(proj2_late,l4,1,proj3_late,1)), ...
    1 - (fct_classPerf_lda(proj1_late,l5,-1,proj2_late,-1)+fct_classPerf_lda(proj1_late,l5,1,proj3_late,-1)+fct_classPerf_lda(proj2_late,l5,1,proj3_late,1))]



