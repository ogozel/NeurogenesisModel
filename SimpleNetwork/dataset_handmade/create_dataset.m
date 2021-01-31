% Create the handmade dataset

clear all
close all
clc

tic

%% Parameters that can be modified

K = 7; % number of clusters
xi = 0.8; % determines the spacing between clusters
S = 6000; % size of the clusters (= how many points there are per cluster)

kappa = 1e4; % concentration parameter


%% Parameters that are fixed

N_EC = 2^K; % number of input neurons

% Normalization factor
NF = 1/sqrt(N_EC*(1+xi^2));


%% Determine the center of mass of the clusters

Master_cm = NaN(N_EC,K);
tmp = NF*(1+xi)*ones(N_EC,1); % now: NF*(1+xi, 1+xi, 1+xi, ....., 1+xi)
for k=1:K
    
    this_cm = tmp;
    
    tmp_start = 1+2^(k-1):2^k:N_EC;
    tmp_length = 2^(k-1);
    for i=1:length(tmp_start)
        this_cm(tmp_start(i):tmp_start(i)+tmp_length-1) = this_cm(tmp_start(i):tmp_start(i)+tmp_length-1) - NF*2*xi;
    end
    
    Master_cm(:,k) = this_cm;
    
end


%% Create the clusters

Master_clusters = NaN(S,N_EC,K);

for k=1:K
    
    k
    
    tmp_mean = Master_cm(:,k);
    Master_clusters(:,:,k) = fct_cluster_sampling(S, kappa * tmp_mean);
        
end




%% Permute randomly the dataset

% Reshape in the same way as the mnist dataset
patterns = reshape(permute(Master_clusters,[2,1,3]), N_EC, K*S);

% Permute the patterns, so that patterns from different clusters are
% intermingled
tmp_shuffle = randperm(size(patterns,2));
patterns = patterns(:,tmp_shuffle);

% Determine the labels and permute them in the same way
labels = [];
for k=1:K
    labels = cat(2, labels, k*ones(1,S));
end
labels = labels(tmp_shuffle);


toc

%% Figure of the scalar product as a function of xi

xis = 0:0.001:1;

figure()
plot(xis, 1./(1+xis.^2),'LineWidth',2)
xlabel('$$ \xi $$','Interpreter','latex')
ylabel('Scalar product')
box off
set(gca, 'TickDir', 'out','LineWidth',2,'FontSize',14)





%% Analyze the produced dataset


for k=1:K
    figure()
    hist(acos(dot(repmat(Master_cm(:,k)',S,1)',Master_clusters(:,:,k)'))*360/(2*pi))
    xlabel('Angle in [°]')
    ylabel('Counts')
    title(strcat('Cluster ',num2str(k)))
end

acos(dot(Master_cm(:,1),Master_cm(:,2)))*360/(2*pi)

figure()
hist(acos(dot(Master_clusters(:,:,1)',Master_clusters(:,:,2)'))*360/(2*pi))
title('Between elements of cluster 1 and elements of cluster 2')

figure()
hist((acos(dot(repmat(Master_cm(:,1)',S,1)',Master_clusters(:,:,1)'))*360/(2*pi))-(acos(dot(repmat(Master_cm(:,2)',S,1)',Master_clusters(:,:,1)'))*360/(2*pi)))
title('Angle from mean of cluster - Angle from mean of other cluster')


C = cov(Master_clusters(:,:,1));

%% Save dataset

trainingimages = patterns;
traininglabels = labels;

filename = strcat('trainingset_vonMises_K',num2str(K),'_xi',num2str(xi),'_kappa',num2str(kappa),'.mat');
save(filename, 'trainingimages', 'traininglabels', 'Master_cm', 'Master_clusters')



%% Plot the condition on xi for having distinct patterns depending on b

b=0.5:0.01:1;
g=2;
y = sqrt(( (g*sqrt(N_EC) + sqrt((g^2)*N_EC+4*b*kappa)) ./ (2*b*sqrt(kappa)) ).^2-1);

figure()
plot(b,y,'b')
hold on

g=1;
y = sqrt(( (g*sqrt(N_EC) + sqrt((g^2)*N_EC+4*b*kappa)) ./ (2*b*sqrt(kappa)) ).^2-1);
plot(b,y,'r')
xlabel('b')
ylabel('xi')
ylim([0 1])
legend('2 std','1 std')
title(strcat('kappa = ',num2str(kappa)))
box off
set(gca, 'TickDir','out')


