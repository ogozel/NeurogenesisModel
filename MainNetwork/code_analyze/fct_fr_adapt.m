% Gain function for the firing rate

function firingRate = fct_fr_adapt(input, bias)

M = 1;
L = 0.5;

firingRate = M*tanh(subplus(input-bias)./L);

end