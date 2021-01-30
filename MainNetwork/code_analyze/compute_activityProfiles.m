% Visualize the activity of the neurons upon presentation of a few example 
% patterns

clear all
close all
clc

%% Choose when to look at the activities
% 'endPrestore' : at the end of the prestorage
% 'endLate'     : at the end of the late phase of neurogenesis
% 'endEarly'    : at the end of the early phase of neurogenesis
timeOfInterest = 'endPrestore';


%% Load data

if strcmp(timeOfInterest ,'endPrestore')
    load('../data_mnist/80epochs_prestorage_mnist_3  4.mat') % data at the end of prestorage
    % Select specific patterns
    Master_p = [33, 41, 62];
elseif strcmp(timeOfInterest ,'endLate')
    load('../data_mnist/1epochs_neuroPhase2_3  4  5_onlyFewPlastic.mat') % data at the end of the late phase of neurogenesis case (3,4 + 5)
    % Select specific patterns
    Master_p = [909, 718, 2358];
elseif strcmp(timeOfInterest ,'endEarly')
    load('../data_mnist/1epochs_neuroPhase1_3  4  5_onlyFewPlastic.mat') % data at the end of the early phase of neurogenesis case (3,4 + 5)
    Master_p = [909, 718, 2358];
end

% Naming convention has changed at some point
if ~exist('BIAS')
    BIAS = BIAS_before;
end

% Load the mnist dataset
load('../dataset_mnist/mnist144.mat')

% Load the network parameters into a structure array
S = load('../network_params_easy_mnist.mat');

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


%% Compute the activities for a particular data set and a particular example pattern

% Select a dataset
data = patterns_test;


% Code to select some example patterns
figure()
for i = 1:100
    fr_EC = data(:,i);
    subplot(10,10,i)
    imagesc(reshape(fr_EC,12,12))
    pbaspect([1 1 1])
    hold on
end


%% Plot the receptive fields

if S.N_DG==100
    numrows = 10;
    numcols = 10;
elseif S.N_DG==700
    numrows = 25;
    numcols = 28;
end

figure()
for i=1:S.N_DG
    subplot(numrows,numcols,i)
    imagesc(reshape(W_DG_EC(i,:),12,12))
    pbaspect([1 1 1])
    set(gca,'xTick',[],'yTick',[])
end


Master_activity_DGC = NaN(S.N_DG, length(Master_p));

for i=1:length(Master_p)
    
    p = Master_p(i);
    
    % Current input pattern
    fr_EC = data(:,p);
    
    % Convergence of the firing rate of the DGCs
    t = 0;
    fr_diff = 10; % just need to be big
    fr_DG_before = zeros(S.N_DG,1);
    
    % Firing rate of the inhibitory neurons
    fr_I_before = zeros(S.N_I,1);
    
    while sum(fr_diff < S.convCriterium) < S.N_DG && t<S.maxNumTimesteps
        
        ff_input = W_DG_EC*fr_EC; % feedforward input (from EC to DGCs)
        rec_input = W_EI*fr_I_before;
        total_input_to_DG = ff_input + rec_input;
        input_to_I =  W_IE*fr_DG_before;
        
        rhs_eqDiff_fr_DG = (1/S.tau_m)*(-fr_DG_before + fct_fr_adapt( total_input_to_DG, BIAS));
        fr_DG = fr_DG_before + S.dt* rhs_eqDiff_fr_DG;
        
        rhs_eqDiff_fr_I = (1/S.tau_m)*(-fr_I_before + subplus(input_to_I - 0.1*S.N_DG) );
        fr_I = fr_I_before + S.dt* rhs_eqDiff_fr_I;
        
        fr_diff = abs(fr_DG - fr_DG_before);
        fr_DG_before = fr_DG;
        fr_I_before = fr_I;
        
        t = t+1;
        
    end
    
    % Multiply by 10 to get firing rate between 0 and 10 Hz
    fr_DG = 10*fr_DG;
    
    % Save the activity of the DGC
    Master_activity_DGC(:,i) = fr_DG;
    
    
    %% Plotting
    
    figure()
    imagesc(reshape(fr_EC,12,12))
    set(gca,'xTick',[],'yTick',[])
    pbaspect([1 1 1])
    
    figure()
    imagesc(reshape(fr_DG,numrows,numcols)') % multiply by 10 to get firing rate between 0 and 10 Hz
    set(gca,'tickDir','out')
    xticks(1:1:10)
    yticks(1:1:10)
    pbaspect([1 1 1])
    caxis([0 10])
    colorbar
    
end


%% Plot the distributions reordered according to first pattern

[~, idx_sorted] = sort(Master_activity_DGC(:,3),'descend');

for i=1:length(Master_p)
    
    fr_values = Master_activity_DGC(idx_sorted,i);
    figure()
    if S.N_DG==100
        bar(fr_values,'FaceColor',[0 0 0],'EdgeColor',[1 1 1])
        xticks(10:10:100)
        xlim([0,101])
    elseif S.N_DG==700
        bar(fr_values)
        xticks(50:50:700)
        xlim([0,701])
    end
    yticks(1:1:10)
    xlabel('Neuron index')
    ylabel('Firing rate [Hz]')
    box off
    set(gca,'tickDir','out')
    pbaspect([1 0.3 1])
    
    disp(strcat('Number of active neurons (fr >1 Hz) for this pattern : ',num2str(length(find(fr_values>1)))))
    disp(strcat('Percentage of active neurons (fr >1 Hz) for this pattern : ',num2str(100*length(find(fr_values>1))/length(fr_values)),'%'))
    
end


