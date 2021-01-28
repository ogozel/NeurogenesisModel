% Control scenario: keep feedforward weights of unresponsive cells plastic,
% but no two-step maturation

clear all
close all
clc


%% Load the data (feedforward weight matrix, patterns_num, ...)

numEpochs = 80;
patterns_num = [3,4];
load((strcat('./data_mnist/',num2str(numEpochs),'epochs_prestorage_mnist_',num2str(patterns_num),'.mat')))

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


%% Determine which neurons are dead units

for i=1:100
    tmpnorm(i) = norm(W_DG_EC(i,:));
end
idx_dead = find(tmpnorm<=1);

% Replace N_new dead units by new neurons
N_new = length(idx_dead);
% sel_new = randperm(length(idx_dead),N_new);
% idx_new = idx_dead(sel_new);
idx_new = idx_dead;



%% Only a few DGC are still plastic (but they learn as in the prestorage period)

W_DG_EC_before = W_DG_EC; % values at the end of the prestorage phase
BIAS_before = BIAS; % values at the end of the prestorage phase


% Learn the feedforward connections of the new DGC
tic
numEpochs = 100;
bool_phase1 = 1;
[W_DG_EC_phase1, W_DG_EC_initial_phase1,  BIAS_phase1] = fct_competitive_dynamics_somePlastic(patterns,W_DG_EC_before,BIAS_before,W_IE,W_EI,numEpochs,idx_new,patterns_num,seed,'control_fewPlatic',bool_phase1);
toc

figure()
for n=1:size(W_DG_EC_phase1,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC_phase1(n,:),[12,12]))
    set(gca,'XTickLabel',[],'YTickLabel',[])
    daspect([1 1 1])
end



