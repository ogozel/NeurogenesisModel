% Neurogenesis case with 2 neurons + 1 neuron

% Neurogenesis with in total 3 DGC neurons:
% Prestorage period: network with only 2 output neurons which all have
% plastic feedforward connections. Show only patterns from 2 clusters and 
% learn all feedforward connections.
% Neurogenesis period: add a new neuron to the output layer. Only this new
% neuron has plastic feedforward connections. Show the 2 pretrained
% clusters of patterns and a novel cluster of patterns and learn only the 
% feedforward connections towards the new neuron.


clear all
close all
clc

tic


%%% !!! Choose if normalized RFs or not !!! %%%
normalized = 0;


%% To make sure that we always start with a different seed

seed = round(now*1e5) - 73717*1e6
rng(seed)



%% Load the handmade input patterns

bool_colorbar = 0;

xival = 0.2;
if xival==0.2
    xivalstr = '0p2';
elseif xival==0.8
    xivalstr = '0p8';
end



load(strcat('./dataset_handmade/trainingset_vonMises_xi',xivalstr,'_kappa1e4.mat'))

w_rec = 1.2;


%% Network

% Network parameters
N_EC = 128;
N_Old = 2;
N_New = 1;
N_DG = N_Old + N_New;

% Initialisation of the recurrent connectivity matrix for the prestorage
wOO = -w_rec;
wON = -w_rec;
wNO = -w_rec;
W_DG_DG = fct_recurrentW_DG(N_Old, N_Old, wOO, wON, wNO);



%% Period 1: Prestorage
% Show only 3 patterns and learn the feedforward connections of the Old
% neurons

% Take the first 2 clusters
patterns_num = [1 2];
idx = ismember(traininglabels,patterns_num);
patterns2 = trainingimages(:,idx); 


% Initialization of the feedforward connectivity matrix
example_patterns = cat(1, trainingimages(:,find(traininglabels==1,1))', trainingimages(:,find(traininglabels==2,1))');
W_DG_EC = example_patterns;
% Make sure that the L2-norm of inward weights is equal to 1.5
W_DG_EC_before = 1.5*normr(abs(W_DG_EC));


% Plot the example patterns to which the receptive field of the DGCs were
% initialized
for n=1:N_Old
    figure()
    imagesc(reshape(example_patterns(n,:),[16,8]))
    caxis([0 0.15])
    set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
    filename = strcat('./figures/examplePattern',num2str(n),'_xi',xivalstr);
    saveas(gcf,filename,'epsc')
end


% Learn the feedforward connections of the Old neurons
numTimes = 3;
THETAS = 0.15*ones(2,1);
BIAS = 1.2*ones(2,1);
[W_DG_EC, average_change, some_weights] = fct_learn_ff_connnections_newRule(patterns2,W_DG_EC_before,THETAS,BIAS,W_DG_DG,N_Old,numTimes);


% % Save
% filename = strcat(num2str(seed),'_neurogenesis_',num2str(patterns_num,'%d'),'_period1.mat');
% save(filename, 'W_DG_EC_before', 'W_DG_EC', 'patterns_num', 'average_change', 'some_weights')


% Plot the receptive field of the DGCs
for n=1:N_Old
    figure()
    
    if normalized
        imagesc(reshape((W_DG_EC(n,:)-min(W_DG_EC(n,:)))/max(W_DG_EC(n,:)-min(W_DG_EC(n,:))),[16,8])) % normalized
    else
        imagesc(reshape(normr(W_DG_EC(n,:)),[16,8])) % normalized weight vector to 1
        caxis([0 0.15]) %% to have colorbar axis from 0 to 0.15
    end
    
    if bool_colorbar
        colorbar
        caxis([0 1])
    end
    set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
    if normalized
        filename = strcat('./figures/DGC',num2str(n),'_afterPrestorage_xi',xivalstr,'_L2norm',num2str(norm(W_DG_EC(n,:))));
    else
        filename = strcat('./figures/DGC',num2str(n),'_notnormalized_afterPrestorage_xi',xivalstr,'_L2norm',num2str(norm(W_DG_EC(n,:))));  
    end
    
    saveas(gcf,filename,'epsc')
end




%% Period 2: Neurogenesis, add 1 neuron and learn a novel pattern

% Take the first 3 clusters
patterns_num = [1 2 3];
idx = ismember(traininglabels,patterns_num);
patterns3 = trainingimages(:,idx); 




%% Phase I: GABAergic input towards the New neuron is excitatory, no feedback inhibition

% Add a row in the feedforward connectivity matrix for the New neuron,
% initialized to zero-weights
W_DG_EC_before = cat(1,W_DG_EC,zeros(1,N_EC));


% Modify the recurrent connectivity matrix to take into account the New
% neuron, and the fact that GABA is excitatory
wOO = -w_rec; % weight from any Old neuron to another Old neuron
wON = 0; % weight from the New neuron to the Old neurons ----------> Phase I: no feedback inhibition
wNO = w_rec; % weight from the Old neurons to the New neuron --------> Phase I: GABA excitatory
W_DG_DG = fct_recurrentW_DG(N_DG, N_Old, wOO, wON, wNO);


% Learn only the feedforward weights towards the New neuron
numTimes = 1;
THETAS = 0.15*ones(3,1);
THETAS(3) = 0.15;
BIAS = 1.2*ones(3,1);
BIAS(3,1) = 0.9;
bool_phase1 = 1;
bool_evolution = 1;
[W_DG_EC, average_change, some_weights, Master_normNewborn_phase1, Master_angleNewbornCM3_phase1] = fct_learn_ff_connnections_NewNeur_newRule(patterns3,W_DG_EC_before,THETAS,BIAS,W_DG_DG,N_DG,N_New,numTimes,bool_phase1,bool_evolution,Master_cm(:,3));


% Save
% filename = strcat(num2str(seed),'_neurogenesis_',num2str(patterns_num,'%d'),'_period2_phase1.mat');
% save(filename, 'W_DG_EC_before', 'W_DG_EC', 'patterns_num', 'average_change', 'some_weights')


% Plot the receptive field of the DGCs
for n=1:N_DG
    figure()
    if normalized
        imagesc(reshape((W_DG_EC(n,:)-min(W_DG_EC(n,:)))/max(W_DG_EC(n,:)-min(W_DG_EC(n,:))),[16,8]))
    else
        imagesc(reshape(normr(W_DG_EC(n,:)),[16,8])) % normalized weight vector to 1
        caxis([0 0.15]) %% to have colorbar axis from 0 to 0.15
    end
    if bool_colorbar
        colorbar
        caxis([0 1])
    end
    set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
    if n>2
        if normalized
            filename = strcat('./figures/DGC',num2str(n),'_afterPhase1_xi',xivalstr,'_L2norm',num2str(norm(W_DG_EC(n,:))));
        else
            filename = strcat('./figures/DGC',num2str(n),'_notnormalized_afterPhase1_xi',xivalstr,'_L2norm',num2str(norm(W_DG_EC(n,:))));
        end
        saveas(gcf,filename,'epsc')
    end
end

figure()
plot(Master_normNewborn_phase1)
figure()
plot(Master_angleNewbornCM3_phase1)



%% Phase II: GABAergic input towards the New neuron is inhibitory, feedback inhibition is present

% Modify the recurrent connectivity matrix to take into account the fact
% that GABA is inhibitory, and there is feedback inhibition from the New
% neuron
wOO = -w_rec; % weight from any Old neuron to another Old neuron
wON = -w_rec; % weight from the New neuron to the Old neurons --------> Phase II: feedback inhibition
wNO = -w_rec; % weight from the Old neurons to the New neuron --------> Phase II: GABA inhibitory
W_DG_DG = fct_recurrentW_DG(N_DG, N_Old+N_New, wOO, wON, wNO);


W_DG_EC_before = W_DG_EC;

% Learn only the feedforward weights towards the New neuron
numTimes = 1;
THETAS = 0.15*ones(3,1);
BIAS = 1.2*ones(3,1);
bool_phase1 = 0;
bool_evolution = 1;
[W_DG_EC, average_change, some_weights, Master_normNewborn_phase2, Master_angleNewbornCM3_phase2] = fct_learn_ff_connnections_NewNeur_newRule(patterns3,W_DG_EC_before,THETAS,BIAS,W_DG_DG,N_DG,N_New,numTimes,bool_phase1,bool_evolution,Master_cm(:,3));


% Save
% filename = strcat(num2str(seed),'_neurogenesis_',num2str(patterns_num,'%d'),'_period2_phase2.mat');
% save(filename, 'W_DG_EC_before', 'W_DG_EC', 'patterns_num', 'average_change', 'some_weights')

% Plot the receptive field of the DGCs
for n=1:N_DG
    figure()
    if normalized
        imagesc(reshape((W_DG_EC(n,:)-min(W_DG_EC(n,:)))/max(W_DG_EC(n,:)-min(W_DG_EC(n,:))),[16,8]))
    else
        imagesc(reshape(normr(W_DG_EC(n,:)),[16,8])) % normalized weight vector to 1
        caxis([0 0.15]) %% to have colorbar axis from 0 to 0.15
    end
    if bool_colorbar
        colorbar
        caxis([0 1])
    end
    set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[])
    if n>2
        if normalized
            filename = strcat('./figures/DGC',num2str(n),'_afterPhase2_xi',xivalstr,'_L2norm',num2str(norm(W_DG_EC(n,:))));
        else
            filename = strcat('./figures/DGC',num2str(n),'_notnormalized_afterPhase2_xi',xivalstr,'_L2norm',num2str(norm(W_DG_EC(n,:))));
        end
        saveas(gcf,filename,'epsc')
    end
end

figure()
plot(Master_normNewborn_phase2)
figure()
plot(Master_angleNewbornCM3_phase2)


toc


%% Plot evolution of norm and angle: only beginning of each phase

zT = 1000;

figure()
plot(Master_normNewborn_phase1(1:zT),'linewidth',2)
hold on
ylim([0 1.5])
xlabel('Number of pattern presentations','FontSize',14)
ylabel('L2-norm','FontSize',14)
box off
set(gca, 'TickDir','out','linewidth',2,'FontSize',14)

figure()
plot(Master_normNewborn_phase1(11500:12500),'linewidth',2)
hold on
if xival==0.2
    ylim([1.46 1.5])
elseif xival==0.8
    ylim([1.32 1.36])
end
xlim([0 1000])
xlabel('Number of pattern presentations','FontSize',14)
ylabel('L2-norm','FontSize',14)
box off
set(gca, 'TickDir','out','linewidth',2,'FontSize',14)

figure()
plot(Master_normNewborn_phase2(1:3*zT),'linewidth',2)
hold on
if xival==0.2
    ylim([1.46 1.5])
elseif xival==0.8
    ylim([1.32 1.36])
end
xlabel('Number of pattern presentations','FontSize',14)
ylabel('L2-norm','FontSize',14)
box off
set(gca, 'TickDir','out','linewidth',2,'FontSize',14)

figure()
plot(Master_angleNewbornCM3_phase1(1:zT),'linewidth',2)
hold on
if xival==0.2
    ylim([5 20])
elseif xival==0.8
    ylim([40 55])
end
xlabel('Number of pattern presentations','FontSize',14)
ylabel('Angle [°]','FontSize',14)
box off
set(gca, 'TickDir','out','linewidth',2,'FontSize',14)

figure()
plot(Master_angleNewbornCM3_phase1(11500:12500),'linewidth',2)
hold on
if xival==0.2
    ylim([5 20])
elseif xival==0.8
    ylim([40 55])
end
xlim([0 1000])
xlabel('Number of pattern presentations','FontSize',14)
ylabel('Angle [°]','FontSize',14)
box off
set(gca, 'TickDir','out','linewidth',2,'FontSize',14)

figure()
plot(Master_angleNewbornCM3_phase2(1:3*zT),'linewidth',2)
hold on
if xival==0.2
    ylim([0 10])
elseif xival==0.8
    ylim([45 55])
end
xlabel('Number of pattern presentations','FontSize',14)
ylabel('Angle [°]','FontSize',14)
box off
set(gca, 'TickDir','out','linewidth',2,'FontSize',14)


