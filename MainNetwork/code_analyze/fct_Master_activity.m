% Compute the end DGC firing rates for each pattern presentation


function [Master_activity_DGC] = fct_Master_activity(data, W_DG_EC, BIAS, W_EI, W_IE)


% Load the network parameters into a structure array
S = load('../network_params_easy_mnist.mat');


% Save the activity of the DGCs to be able to learn the readout weights on
% several epochs without loss of time
Master_activity_DGC = NaN(S.N_DG, size(data,2));


for p=1:size(data,2)
    
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
    
    % Save the activity of the DGCs
    Master_activity_DGC(:,p) = fr_DG;
    
end




end