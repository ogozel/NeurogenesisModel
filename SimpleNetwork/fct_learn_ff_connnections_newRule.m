% Function that learns the feedforward connections

function [W_DG_EC, average_change, some_weights] = fct_learn_ff_connnections_newRule(patterns,W_DG_EC_before,THETAS,BIAS,W_DG_DG,N_DG,numTimes)


% Neurons in DG
tau_m = 0.02;

% Parameters needed for the simulations
maxNumTimesteps = 2000; % maximal number of timesteps allowed in case we do not reach convergence before
convCriterium = 1e-6; % tolerance concerning the firing rate to determine if we reached convergence
dt = 0.001;
eta = 1e-2;

% Initialization
cnt = 1;
fr_DG = 10; % just need to be nonzero

T = size(patterns,2);
some_learningUpdates = NaN(size(W_DG_EC_before,1),size(W_DG_EC_before,2),T);

thatT = 1;
some_weights = NaN(size(W_DG_EC_before,1),size(W_DG_EC_before,2),numTimes);

while cnt <= numTimes*T

    % Current input pattern
    tmp_idx = 1+mod(cnt-1,size(patterns,2));
    fr_EC = patterns(:,tmp_idx);
    
    
    % Convergence of the firing rate of the DGCs
    t = 0;
    fr_diff = 10; % just need to be big
    fr_DG_before = zeros(N_DG,1);
    
    while sum(fr_diff < convCriterium) < N_DG && t<maxNumTimesteps
        
        ff_input = W_DG_EC_before*fr_EC; % feedforward input (from EC to DGCs)
        rec_input = W_DG_DG*fr_DG_before; % local recurrent input (from DGCs to DGCs)
        
        input_before = ff_input + rec_input;
        rhs_eqDiff_fr_DG = (1/tau_m)*(-fr_DG_before + (input_before > BIAS));
        fr_DG = fr_DG_before + dt* rhs_eqDiff_fr_DG;
        
        fr_diff = abs(fr_DG - fr_DG_before);
        fr_DG_before = fr_DG;
        
        t = t+1;
        
    end
    
    % Learning of the feedforward connections
    alpha = (0.03./(THETAS.^3));
    alpha(~isfinite(alpha)) = 0; % to avoid any problem for the case theta=0
    gamma = 1.65-THETAS;
    learningUpdate = -alpha.*fr_DG.*subplus(THETAS-fr_DG)*fr_EC' + gamma.*fr_DG.*subplus(fr_DG-THETAS)*fr_EC' - gmultiply(1*fr_DG.*subplus(fr_DG-THETAS).*fr_DG.^2,W_DG_EC_before);
    W_DG_EC = max(0, W_DG_EC_before + eta* learningUpdate);
    
    % Compute the average change in the feedforward weights
    some_learningUpdates(:,:,1+mod(cnt-1,T)) = learningUpdate;
    if mod(cnt,T)==0
        cnt
        average_change = norm(mean(some_learningUpdates,3))
        some_learningUpdates = NaN(size(W_DG_EC_before,1),size(W_DG_EC_before,2),T); % resetting
        some_weights(:,:,thatT) = W_DG_EC;
        thatT = thatT + 1;
    end
    
    cnt = cnt +1;
    W_DG_EC_before = W_DG_EC;
    
end


end