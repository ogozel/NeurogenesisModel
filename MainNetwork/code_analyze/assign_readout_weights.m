% Determine the readout weights of the outputs using the training dataset

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


% Load the mnist dataset
load('../dataset_mnist/mnist144.mat')

% Normalize the patterns such that they have a L2-norm of 1
trainingimages = normc(trainingimages);

% Select the digits
idx = ismember(traininglabels,patterns_num);
patterns = trainingimages(:,idx); 
labels = traininglabels(idx);

% Look at the RF
% figure()
% for n=1:size(dataS.W_DG_EC,1)
%     subplot(10,10,n)
%     imagesc(reshape(dataS.W_DG_EC(n,:),[12,12]))
% end

% Determine the readout weights for classification
[W_readout, Master_activity_DGC] = fct_assign_readout_weights(patterns, labels, dataS.W_DG_EC, dataS.BIAS_before, dataS.W_EI, dataS.W_IE, patterns_num); % dataS.BIAS_before

% Save these readout weights
save(strcat('../data_mnist/',num2str(numEpochs),this_part_of_name,num2str(patterns_num),this_part2,'_Wreadout_100epochs.mat'),'W_readout', 'Master_activity_DGC')

toc
