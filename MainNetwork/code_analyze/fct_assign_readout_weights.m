% Function that learns the readout weights for classification

function [W_readout, Master_activity_DGC] = fct_assign_readout_weights(data, labels, W_DG_EC, BIAS, W_EI, W_IE, patterns_num)

N_DG = size(W_DG_EC,1);
N_I = size(W_EI,2);

numClasses = length(patterns_num);


% Load the network parameters into a structure array
S = load('../network_params_easy_mnist.mat');


% Initial values for the readout weights
W_readout = 0.1*rand(numClasses,N_DG);

etaRO = 1e-2; % learning rate of the readout weights update

% Save the activity of the DGCs to be able to learn the readout weights on
% several epochs without loss of time
Master_activity_DGC = NaN(N_DG, size(data,2));


for p=1:size(data,2)
    
    % Current input pattern
    fr_EC = data(:,p);
    
    % Label of the pattern
    this_label = labels(p);
    this_label_oneHot = zeros(numClasses,1);
    this_label_oneHot(patterns_num==this_label) = 1;
    
    % Convergence of the firing rate of the DGCs
    t = 0;
    fr_diff = 10; % just need to be big
    fr_DG_before = zeros(N_DG,1);
    
    % Firing rate of the inhibitory neurons
    fr_I_before = zeros(N_I,1);
    
    while sum(fr_diff < S.convCriterium) < N_DG && t<S.maxNumTimesteps
        
        ff_input = W_DG_EC*fr_EC; % feedforward input (from EC to DGCs)
        rec_input = W_EI*fr_I_before;
        total_input_to_DG = ff_input + rec_input;
        input_to_I =  W_IE*fr_DG_before;
        
        rhs_eqDiff_fr_DG = (1/S.tau_m)*(-fr_DG_before + fct_fr_adapt( total_input_to_DG, BIAS));
        fr_DG = fr_DG_before + S.dt* rhs_eqDiff_fr_DG;
        
        rhs_eqDiff_fr_I = (1/S.tau_m)*(-fr_I_before + subplus(input_to_I - 0.1*N_DG) );
        fr_I = fr_I_before + S.dt* rhs_eqDiff_fr_I;
        
        fr_diff = abs(fr_DG - fr_DG_before);
        fr_DG_before = fr_DG;
        fr_I_before = fr_I;
        
        t = t+1;
        
    end
    
    % Save the activity of the DGC
    Master_activity_DGC(:,p) = fr_DG;
    
    % Compute the activity of the readout neurons
    activity_readout = tanh(2*subplus(W_readout*fr_DG));
    activity_derivative_readout = 2*( 1 - (tanh(2*subplus(W_readout*fr_DG))).^2 );
    
    % Update the readout weights
    Wro_update = etaRO * gmultiply( ( this_label_oneHot - activity_readout ) .* activity_derivative_readout, fr_DG' );
    W_readout = W_readout + Wro_update;
    
end

%% Keep learning the readout weights (100 epochs in total)

for pz=0:99*size(data,2)-1
    
    p = 1 + mod(pz, size(data,2));
    
    % Current activity of the DG
    fr_DG = Master_activity_DGC(:,p);
    
    % Label of the pattern
    this_label = labels(p);
    this_label_oneHot = zeros(numClasses,1);
    this_label_oneHot(patterns_num==this_label) = 1;
    
    
    % Save the activity of the DGC
    Master_activity_DGC(:,p) = fr_DG;
    
    % Compute the activity of the readout neurons
    activity_readout = tanh(2*subplus(W_readout*fr_DG));
    activity_derivative_readout = 2*( 1 - (tanh(2*subplus(W_readout*fr_DG))).^2 );
    
    % Update the readout weights
    Wro_update = etaRO * gmultiply( ( this_label_oneHot - activity_readout ) .* activity_derivative_readout, fr_DG' );
    W_readout = W_readout + Wro_update;
    
end




end