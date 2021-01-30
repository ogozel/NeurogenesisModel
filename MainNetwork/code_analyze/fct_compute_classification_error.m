% Function that computes classification error

function [Total_percentage_error, Master_percentage_error, Master_confusion_matrix, Master_activity_DGC_test] = fct_compute_classification_error(data, labels, W_DG_EC, BIAS, W_EI, W_IE, patterns_num, W_readout, boolDisplay)

N_DG = size(W_DG_EC,1);
N_I = size(W_EI,2);

numClasses = length(patterns_num);


% Load the network parameters into a structure array
S = load('../network_params_easy_mnist.mat');

% Number of patterns per class
Master_numPat = NaN(length(patterns_num),1);
for c=1:length(patterns_num)
    Master_numPat(c) = length(find(labels==patterns_num(c)));
end

% Matrix where we are going to save the classification errors for each
% class
Master_error = zeros(length(patterns_num),1);

% Confusion matrix - to add up in each elements
Matrix_conf = zeros(length(patterns_num),length(patterns_num));

% Array which contain one Matrix of the activity of the DGCs for each
% digit, each Matrix is the activity in response to all testing patterns
Master_activity_DGC_test = cell(1,length(patterns_num));



for p=1:size(data,2)
    
    % Current input pattern
    fr_EC = data(:,p);
    
    % Label of the pattern
    this_label = labels(p);
    this_label_oneHot = zeros(numClasses,1);
    realI = find(patterns_num==this_label);
    this_label_oneHot(realI) = 1;
    
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
    
    % Save the activity of the DGC in the Matrix associated to the digit of
    % the pattern
    Master_activity_DGC_test{1,patterns_num==this_label} = cat(2, Master_activity_DGC_test{1,patterns_num==this_label}, 10*fr_DG); % multiply by 10 to get results in [Hz]
    
    % Same gain function as for the learning of the readout weights
    activity_readout = tanh(2*subplus(W_readout*fr_DG));
    
    % Compare readout with real label
    [~, I] = max(activity_readout);
    tmp_class_oneHot = zeros(numClasses,1);
    tmp_class_oneHot(I) = 1;
    
    tmp_error = (sum(abs(this_label_oneHot - tmp_class_oneHot)) > 0 );
    
    % Update number of errors
    if tmp_error
        Master_error(patterns_num==this_label) = Master_error(patterns_num==this_label) + 1;
    end
    
    % Update the confusion matrix
    Matrix_conf(I,realI) = Matrix_conf(I,realI) + 1;
    
end

% Compute the percentage of error per class
Master_percentage_error = Master_error./Master_numPat;

% Compute the confusion matrix
Master_confusion_matrix = Matrix_conf./repmat(Master_numPat',length(patterns_num),1);

% Compute the overall percentage of error
Total_percentage_error = sum(Master_error)/sum(Master_numPat);

if boolDisplay
    
    disp(strcat('The overall classification performance is :',num2str(Total_percentage_error)))
    disp(strcat('The percentage of error per digit is :',num2str(Master_percentage_error)))
    disp(strcat('The confusion matrix is :',num2str(Master_confusion_matrix)))
    
end


end