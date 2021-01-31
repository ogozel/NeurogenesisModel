% Implement the Von Mieses-Fisher distribution (part 2/2)

% n : number of points we sample
% theta : = kappa*direction, with 'kappa' the concentration parameter (a
% scalar) and 'direction' a vector which represents the mean of the cluster
% (should be normalized to 1)


function y = fct_cluster_sampling(n,theta)

dim = length(theta);
kappa = norm(theta);
mu = theta / kappa;

w = fct_rejection_sampling(n, kappa, dim);

v = fct_sample_tangent_unit(mu,n);
y = gmultiply(sqrt(1-w.^2),v') + w*mu';

end


% % Example of use
% dim = 3
% n = 100000;
% kappa = 100000;
% Master_cm = normc(rand(dim,1));
% Master_clusters = fct_cluster_sampling(n, kappa * Master_cm);
% 
% figure()
% scatter3(Master_clusters(:,1),Master_clusters(:,2),Master_clusters(:,3),'b')
% hold on
% scatter3(Master_cm(1),Master_cm(2),Master_cm(3),'r')


