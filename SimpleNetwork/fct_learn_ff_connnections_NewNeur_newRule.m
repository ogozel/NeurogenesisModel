% Function that learns the feedforward connections ONLY for the New
% neuron(s)

function [W_DG_EC, average_change, some_weights, Master_normNewborn, Master_angleNewbornCM3] = fct_learn_ff_connnections_NewNeur_newRule(patterns,W_DG_EC_before,THETAS,BIAS,W_DG_DG,N_DG,N_New,numTimes,bool_phase1,bool_evolution,Master_cm3)

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
some_learningUpdates = NaN(N_New,size(W_DG_EC_before,2),T);

some_weights = NaN(N_New,size(W_DG_EC_before,2),numTimes+1);
some_weights(:,:,1) = W_DG_EC_before(N_DG-N_New+1:end,:);
thatT = 2;

% To save evolution
deltaNumPat = 1;
zz=2;
Master_normNewborn = zeros(round(numTimes*T/deltaNumPat)+1,1);
Master_normNewborn(1) = norm(W_DG_EC_before(3,:));
Master_angleNewbornCM3 = NaN(round(numTimes*T/deltaNumPat)+1,1);
Master_angleNewbornCM3(1) = (180/pi)*acos((W_DG_EC_before(3,:)*Master_cm3)/(norm(W_DG_EC_before(3,:))*norm(Master_cm3)));

while cnt <= numTimes*T
    
    if bool_phase1
        BIAS(3) = min(1.2, 0.9+(1.2-0.9)*cnt/12000)*ones(1,1);
    end
    
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
        rhs_eqDiff_fr_DG = (1/tau_m)*(-fr_DG_before + (input_before> BIAS));
        fr_DG = fr_DG_before + dt* rhs_eqDiff_fr_DG;
        
        fr_diff = abs(fr_DG - fr_DG_before);
        fr_DG_before = fr_DG;
        
        t = t+1;
        
    end
    
    
    % Learning of the feedforward connections of the New neuron only!
    alpha = (0.03./(THETAS.^3));
    alpha(~isfinite(alpha)) = 0; % to avoid any problem for the case theta=0
    gamma = 1.65-THETAS;
    learningUpdate = -alpha.*fr_DG.*subplus(THETAS-fr_DG)*fr_EC' + gamma.*fr_DG.*subplus(fr_DG-THETAS)*fr_EC' - gmultiply(1*fr_DG.*subplus(fr_DG-THETAS).*fr_DG.^2,W_DG_EC_before);
    learningUpdatePlastic = zeros(3,128);
    learningUpdatePlastic(3,:) = learningUpdate(3,:);
    W_DG_EC = max(0, W_DG_EC_before + eta* learningUpdatePlastic);
    
    % Compute the average change in the feedforward weights
    some_learningUpdates(:,:,1+mod(cnt-1,T)) = learningUpdatePlastic(3,:);
    if mod(cnt,T)==0
        cnt
        average_change = norm(mean(some_learningUpdates,3))
        some_learningUpdates = NaN(N_New,size(W_DG_EC_before,2),T); % resetting
        some_weights(:,:,thatT) = W_DG_EC(N_DG-N_New+1:end,:);
        thatT = thatT + 1;
    end
    
    cnt = cnt +1;
    W_DG_EC_before = W_DG_EC;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save intermediate results to investigate evolution of weight vector
    % onto newborn DGC
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if bool_evolution && mod(cnt,deltaNumPat)==0
        
        Master_normNewborn(zz) = norm(W_DG_EC(3,:));
        Master_angleNewbornCM3(zz) = (180/pi)*acos((W_DG_EC(3,:)*Master_cm3)/(norm(W_DG_EC(3,:))*norm(Master_cm3)));
        zz = zz+1;
        
    end
    
end


end