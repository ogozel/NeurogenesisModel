% Using the assigned readout weights computed previously using the training
% dataset, compute classification performance based on testing set
% Also save the activity of the DGCs for each test pattern

clear all
close all
clc

tic

% Load the data (feedforward weight matrix, patterns_num, ...)
numEpochs = 80;
patterns_num = [3,4];
this_part_of_name = 'epochs_prestorage_mnist_';
this_part2 = '';

dataS = load((strcat('../data_mnist/',num2str(numEpochs),this_part_of_name,num2str(patterns_num),this_part2,'.mat')));
dataR = load(strcat('../data_mnist/',num2str(numEpochs),this_part_of_name,num2str(patterns_num),this_part2,'_Wreadout_100epochs.mat'));




% Load the mnist dataset
load('../dataset_mnist/mnist144.mat')

% Normalize the patterns such that they have an L2-norm of 1
testimages = normc(testimages);

% Select the digits
idx = ismember(testlabels,patterns_num);
patterns = testimages(:,idx); 
labels = testlabels(idx);

% Compute softmax classification
[Total_percentage_error, Master_percentage_error, Master_confusion_matrix, Master_activity_DGC_test] = fct_compute_classification_error(patterns, labels, dataS.W_DG_EC, dataS.BIAS_before, dataS.W_EI, dataS.W_IE, patterns_num, dataR.W_readout, 1); % dataS.BIAS_before

% Save these classification performances and DGC activities for testing
% patterns
save(strcat('../data_mnist/',num2str(numEpochs),this_part_of_name,num2str(patterns_num),this_part2,'_testClassPerf_DGCActivity.mat'), 'Total_percentage_error', 'Master_percentage_error', 'Master_confusion_matrix', 'Master_activity_DGC_test')


toc
