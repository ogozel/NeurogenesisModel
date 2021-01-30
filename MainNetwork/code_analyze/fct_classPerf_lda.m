% Function that computes the classification performance using LDA Fisher
% discrimination

% isBigger:  1 if we consider the projections that are bigger than 0,
%           -1 if we consider the projections that are smaller than 0


function [performance] = fct_classPerf_lda(projections,patternsOfInterest,isBigger,projections2,isBigger2)

if nargin<4
    projections2 = [];
    boolBigger2 = [];
    performance = length(find(isBigger*projections(patternsOfInterest)>0)) / length(find(patternsOfInterest));
        
else
    performance = length(find(isBigger*projections(patternsOfInterest)>0 & isBigger2*projections2(patternsOfInterest)>0)) / length(find(patternsOfInterest));
end




end