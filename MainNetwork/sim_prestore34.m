% Pretraining of the main rate network with MNIST training patterns from
% digits 3 and 4
% NB: Same code can be used to pretrain the network with any number of
% digits


clear all
close all
clc


dataset = 'mnist';

% Choose a combination of two digits
patterns_num = [3 4];


% To make sure that we always start with a different seed
seed = round(now-floor(now),5)*1e5;
rng(seed)



%% Load data

% Load the network parameters into a structure array
S = load('network_params_easy_mnist.mat');

% Load the dataset
load('./dataset_mnist/mnist144.mat')

S.N_EC = 144;
strDataset = 'mnist';
numrows = 12;
numcols = 12;

% Normalize the patterns such that they have a L2-norm of 1
trainingimages = normc(trainingimages);




%% Network

% Initialisation of the recurrent connectivity matrices
W_IE = 1*(rand(S.N_I,S.N_DG)<S.p);
W_EI = -(1/(S.p*S.N_I))*(rand(S.N_DG,S.N_I)<S.p);


% Initialization of the feedforward connectivity matrix
W_DG_EC = rand(S.N_DG, S.N_EC);
% Make sure that the L2-norm of inward weights is equal to 1
W_DG_EC_before = normr(W_DG_EC);


% Plot the receptive fields at initialization (random)
figure()
for n=1:size(W_DG_EC_before,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC_before(n,:),[numrows,numcols]))
end


%% PRETRAINING: Show training patterns from 2 digits and learn all feedforward connections

% Select the digits
idx = ismember(traininglabels,patterns_num);
patterns = trainingimages(:,idx);

BIAS_before = S.minBIAS;


tic
numEpochs = 100;
[W_DG_EC, BIAS] = fct_competitive_dynamics(patterns,W_DG_EC_before,BIAS_before,W_IE,W_EI,numEpochs,patterns_num,seed,dataset,strDataset);
toc


% Plot the receptive fields after pretraining
figure()
for n=1:size(W_DG_EC,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC(n,:),[numrows,numcols]))
    set(gca,'XTickLabel',[],'YTickLabel',[])
    daspect([1 1 1])
end











