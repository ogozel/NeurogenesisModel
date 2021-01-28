% Function that learns the feedforward connections of only a few neurons

function [W_DG_EC, W_DG_EC_initial, BIAS_before] = fct_competitive_dynamics_somePlastic(patterns,W_DG_EC_before,BIAS_before,W_IE,W_EI,numEpochs,idx_plastic,patterns_num,seed,strProtocol,bool_phase1)

rng(seed)
W_DG_EC_initial = W_DG_EC_before;

% Load the network parameters into a structure array
S = load('network_params_easy_mnist.mat');
S.N_EC = 144;

T = size(patterns,2);

% Initialization
cnt = 0;


while cnt < numEpochs*T
    
    % Current input pattern
    tmp_idx = 1+mod(cnt,size(patterns,2));
    fr_EC = patterns(:,tmp_idx);
    
    % Convergence of the firing rate of the DGCs
    t = 0;
    fr_diff = 10; % just need to be big
    fr_DG_before = zeros(S.N_DG,1);
    
    % Firing rate of the inhibitory neurons
    fr_I_before = zeros(S.N_I,1);
    
    
    while sum(fr_diff < S.convCriterium) < S.N_DG && t<S.maxNumTimesteps
        
        ff_input = W_DG_EC_before*fr_EC; % feedforward input (from EC to DGCs)
        rec_input = W_EI*fr_I_before;
        total_input_to_DG = ff_input + rec_input;
        input_to_I =  W_IE*fr_DG_before;
        
        rhs_eqDiff_fr_DG = (1/S.tau_m)*(-fr_DG_before + fct_fr_adapt( total_input_to_DG, BIAS_before));
        fr_DG = fr_DG_before + S.dt* rhs_eqDiff_fr_DG;
        
        rhs_eqDiff_fr_I = (1/S.tau_mInh)*(-fr_I_before + subplus(input_to_I - 0.1*S.N_DG) );
        fr_I = fr_I_before + S.dt* rhs_eqDiff_fr_I;
        
        fr_diff = abs(fr_DG - fr_DG_before);
        fr_DG_before = fr_DG;
        fr_I_before = fr_I;
        
        t = t+1;
        
    end
    
    
    
    % Learning of the feedforward connections - competitive dynamics
    alpha = (0.05./(S.THETA.^3));
    alpha(~isfinite(alpha)) = 0; % to avoid any problem for the case theta=0
    gamma = S.thetaHat-S.THETA;
    learningUpdate = -alpha.*fr_DG.*subplus(S.THETA-fr_DG)*fr_EC' + gamma.*fr_DG.*subplus(fr_DG-S.THETA)*fr_EC' - gmultiply(S.delta*fr_DG.*subplus(fr_DG-S.THETA).*fr_DG.^2,W_DG_EC_before);
    learningUpdate_plastic = zeros(size(W_DG_EC_before));
    learningUpdate_plastic(idx_plastic,:) = learningUpdate(idx_plastic,:);
    W_DG_EC = max(0, W_DG_EC_before + S.eta* learningUpdate_plastic);
    
    if ~bool_phase1
        biasUpdate = zeros(S.N_DG,1);
        tmp_biasUpdate = (1/S.tau_bias)*(fr_DG-S.FR_target);
        biasUpdate(idx_plastic) = tmp_biasUpdate(idx_plastic);
        
        BIAS = max(S.minBIAS, BIAS_before + biasUpdate);
        BIAS_before = BIAS;
    end
    
    W_DG_EC_before = W_DG_EC;
    
    cnt = cnt +1;
    
    if mod(cnt,T)==0
        cnt/T
    end
    
    % Save intermediate results
    if mod(cnt,1*T)==0
        save(strcat('./data_mnist/',num2str(cnt/T),'epochs_',strProtocol,'_',num2str(patterns_num),'_onlyFewPlastic.mat'),'W_DG_EC','W_DG_EC_initial','BIAS_before','patterns_num','W_EI','W_IE','seed')
    end
    
    
end


end