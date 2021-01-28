% Neurogenesis case: newborn DGCs learn their feedforward weights through a
% two-step process: first GABAergic input is excitatory, then it switches
% to inhibitory

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
    imagesc(reshape(W_DG_EC(n,:),[12,12]))
    set(gca,'XTickLabel',[],'YTickLabel',[],'XTick',[],'YTick',[])
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

for i=1:S.N_DG
    tmpnorm(i) = norm(W_DG_EC(i,:));
end
idx_dead = find(tmpnorm<=3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Replace N_new dead units by new neurons
% N_new = 10;
% sel_new = randperm(length(idx_dead),N_new);
% idx_new = idx_dead(sel_new);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


idx_new = idx_dead;
N_new = length(idx_new);

%% Early phase of maturation

W_DG_EC_before = W_DG_EC; % values at the end of the prestorage phase
BIAS_before = BIAS; % values at the end of the prestorage phase

W_DG_EC_before(idx_new,:) = 0; % set the weight of the New DGC to 0
BIAS_before(idx_new,:) = 0.0; % set the bias of the New DGC to 0


% Change the recurrent connectivity matrices for the new DGC
W_IE(:,idx_new) = zeros(S.N_I,N_new); % Non-existent
W_EI(idx_new,:) = (1/(S.p*S.N_I))*(rand(N_new,S.N_I)<S.p); % GABA excitatory

% Learn the feedforward connections of the new DGC
tic
numEpochs = 1;
bool_phase1 = 1;
[W_DG_EC_phase1, W_DG_EC_initial_phase1,  BIAS_phase1] = fct_competitive_dynamics_somePlastic(patterns,W_DG_EC_before,BIAS_before,W_IE,W_EI,numEpochs,idx_new,patterns_num,seed,'neuroPhase1',bool_phase1);
toc



figure()
for n=1:size(W_DG_EC_phase1,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC_phase1(n,:),[12,12]))
    set(gca,'XTickLabel',[],'YTickLabel',[],'XTick',[],'YTick',[])
    daspect([1 1 1])
end

disp 'End phase 1'


%% Late phase of maturation

% Values at the end of the early phase of maturation
W_DG_EC_before = W_DG_EC_phase1;
BIAS_before = BIAS_phase1; 

% Change the recurrent connectivity matrices
W_IE(:,idx_new) = (rand(S.N_I,N_new)<S.p); % Now existent
W_EI(idx_new,:) = -W_EI(idx_new,:); % GABA inhibitory

% Learn the feedforward connections of the new DGC
tic
numEpochs = 1;
bool_phase1 = 0;
[W_DG_EC_phase2, W_DG_EC_initial_phase2, BIAS_phase2] = fct_competitive_dynamics_somePlastic(patterns,W_DG_EC_before,BIAS_before,W_IE,W_EI,numEpochs,idx_new,patterns_num,seed,'neuroPhase2',bool_phase1);
toc


figure()
for n=1:size(W_DG_EC_phase2,1)
    subplot(10,10,n)
    imagesc(reshape(W_DG_EC_phase2(n,:),[12,12]))
    set(gca,'XTickLabel',[],'YTickLabel',[],'XTick',[],'YTick',[])
    daspect([1 1 1])
end


