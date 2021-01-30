% Compute the end DGC firing rates for each pattern presentation

clear all
close all
clc

%% Choose which data to load

% Data at the end of prestorage
load('../data_mnist/80epochs_prestorage_mnist_3  4.mat')

% % Data at the end of the early phase of neurogenesis case (3,4 + 5)
% load('../data_mnist/1epochs_neuroPhase1_3  4  5_onlyFewPlastic.mat')

% Data at the end of the late phase of neurogenesis case (3,4 + 5)
%load('../data_mnist/1epochs_neuroPhase2_3  4  5_onlyFewPlastic.mat')



%% Load mnist patterns -> Testing set

% Load the mnist dataset
load('../dataset_mnist/mnist144.mat')

% Normalize the patterns such that they have a L2-norm of 1
trainingimages = normc(trainingimages);
testimages = normc(testimages);

% Select the digits on the training set
idx = ismember(traininglabels,patterns_num);
patterns_train = trainingimages(:,idx); 
labels_train = traininglabels(idx);

% Select the digits on the testing set
idx = ismember(testlabels,patterns_num);
patterns_test = testimages(:,idx); 
labels_test = testlabels(idx);


%% Compute the activities - choose training or testing set!

data = patterns_test; % patterns_train;

Master_activity_DGC_prestore34 = fct_Master_activity(data, W_DG_EC, BIAS, W_EI, W_IE);
save Master_activity_DGC_prestore34.mat Master_activity_DGC_prestore34

%Master_activity_DGC_neuro345_early_train = fct_Master_activity(data, W_DG_EC, BIAS, W_EI, W_IE); % BIAS_before
%save Master_activity_DGC_neuro345_early_train.mat Master_activity_DGC_neuro345_early_train

%Master_activity_DGC_neuro345_late_train = fct_Master_activity(data, W_DG_EC, BIAS_before, W_EI, W_IE);
%save Master_activity_DGC_neuro345_late_train.mat Master_activity_DGC_neuro345_late_train





