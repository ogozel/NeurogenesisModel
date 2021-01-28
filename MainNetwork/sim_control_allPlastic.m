% Control scenario: keep all feedforward weights plastic

clear all
close all
clc


%% Load the data (feedforward weight matrix, patterns_num, ...)

numEpochs = 80;
patterns_num = [3,4];
dataS = load((strcat('./data_mnist/',num2str(numEpochs),'epochs_prestorage_mnist_',num2str(patterns_num),'.mat')));


% Load the input patterns
load('./dataset_mnist/mnist144.mat')

% Load the network parameters into a structure array
S = load('network_params_easy_mnist.mat');
S.N_EC = 144;

rng(seed)


figure()
for n=1:size(W_DG_EC,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC(n,:),12,12))
    set(gca,'XTickLabel',[],'YTickLabel',[])
    daspect([1 1 1])
end


%% Select the data

% Normalize the patterns such that they have a L2-norm of 1
trainingimages = normc(trainingimages);

% Pick a novel digit cluster
patterns_num = [3,4,5];

% Select the digits
idx = ismember(traininglabels,patterns_num);
patterns = trainingimages(:,idx); 




%% All DGC are still plastic (but they learn as in the prestorage period)

W_DG_EC_before = dataS.W_DG_EC; % values at the end of the prestorage phase
BIAS_before = dataS.BIAS; % values at the end of the prestorage phase



% Learn the feedforward connections of the new DGC
tic
numEpochs = 100;
[W_DG_EC, BIAS] = fct_competitive_dynamics(patterns,W_DG_EC_before,BIAS_before,W_IE,W_EI,numEpochs,patterns_num,seed,'mnist','control_mnist');
toc

figure()
for n=1:size(W_DG_EC,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC(n,:),[12,12]))
    set(gca,'XTickLabel',[],'YTickLabel',[])
    daspect([1 1 1])
end



