% Implement the Von Mieses-Fisher distribution (extra part)

% mu: vector of dimension 'dim' that represents the mean of the cluster
% n: number of tangent vectors we want to get
% y: (# dimensions)x(# samples) tangential vectors to mu


function y = fct_sample_tangent_unit(mu,n)

if nargin==1
    n = 1;
end

if size(mu,2) > size(mu,1)
    mu = mu';
end

[U, ~, ~] = svd(mu);

nu = randn(size(mu,1)-1,n);
y = normc(U(:,2:end)*nu);

end

