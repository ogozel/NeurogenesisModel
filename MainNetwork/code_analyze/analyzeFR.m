% Determine the percentage of patterns that induce DGC activity in 
% different firing rate bins

clear all
close all
clc


load('../data_mnist/1epochs_neuroPhase2_3  4  5_onlyFewPlastic_Wreadout_100epochs.mat')


numEpochs = 80;
patterns_num = [3,4];
load((strcat('../data_mnist/',num2str(numEpochs),'epochs_prestorage_mnist_',num2str(patterns_num),'.mat')))
for i=1:100
    tmpnorm(i) = norm(W_DG_EC(i,:));
end

idx_new = find(tmpnorm<=1.0);
idx_sel = setdiff(1:100,idx_new);

N_new = length(idx_new);
N_sel = length(idx_sel);



%% Analyze firing rates

xi = size(Master_activity_DGC,1);
yi = size(Master_activity_DGC,2);

meanFR = mean(reshape(Master_activity_DGC,xi*yi,1))

semFR = std(reshape(Master_activity_DGC,xi*yi,1),1)/sqrt(xi*yi)

numActiveOverAll = length(find(reshape(Master_activity_DGC,xi*yi,1)>0.1)) / xi*yi % 0.1 corresponds to 1 Hz

for i=1:yi
    numActivePerPattern(i) = length(find(Master_activity_DGC(:,i)>0.1));
end
numActivePerPattern = mean(numActivePerPattern)







%% End of late: Figure distribution of firing rates at the end of late phase, separate selective and newborn DGCs

Master_hist = NaN(size(Master_activity_DGC,2), 10);
for i=1:size(Master_activity_DGC,2)
    Master_hist(i,:) = histcounts(Master_activity_DGC(:,i),10);
end



Master_histsel = NaN(size(Master_activity_DGC,2), 10);
for i=1:size(Master_activity_DGC,2)
    Master_histsel(i,:) = histcounts(Master_activity_DGC(idx_sel,i),10);
end

Master_histnew = NaN(size(Master_activity_DGC,2), 10);
for i=1:size(Master_activity_DGC,2)
    Master_histnew(i,:) = histcounts(Master_activity_DGC(idx_new,i),10);
end



x = [1:1:10];

Master_hist = Master_histsel;
y1 = (100/N_sel)*mean(Master_hist,1);
tmp = (100/N_sel)*quantile(Master_hist,[0.1 0.9],1);
errlowsel = y1 - tmp(1,:);
errhighsel = tmp(2,:) - y1;

Master_hist = Master_histnew;
y2 = (100/N_new)*mean(Master_hist,1);
tmp = (100/N_new)*quantile(Master_hist,[0.1 0.9],1);
errlownew = y2 - tmp(1,:);
errhighnew = tmp(2,:) - y2;


figure()
bar(x,y1)
hold on
er = errorbar(x,y1,errlowsel,errhighsel);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
xlabel('Firing rate bin [Hz]')
ylabel('Percentage [%]')
title('End of late phase - sel')
box off
set(gca,'TickDir','out')
ylim([0 100])

figure()
bar(x,y2)
hold on
er = errorbar(x,y2,errlownew,errhighnew);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
xlabel('Firing rate bin [Hz]')
ylabel('Percentage [%]')
title('End of late phase - new')
box off
set(gca,'TickDir','out')
ylim([0 100])



%% End of early: Figure distribution of firing rates at the end of early phase, separate selective and newborn DGCs

load('Master_activity_DGC_early.mat')


Master_hist_early = NaN(size(Master_activity_DGC_early,2), 10);
for i=1:size(Master_activity_DGC_early,2)
    Master_hist_early(i,:) = histcounts(Master_activity_DGC_early(:,i),10);
end




Master_hist_earlysel = NaN(size(Master_activity_DGC_early,2), 10);
for i=1:size(Master_activity_DGC_early,2)
    Master_hist_earlysel(i,:) = histcounts(Master_activity_DGC_early(idx_sel,i),10);
end

Master_hist_earlynew = NaN(size(Master_activity_DGC_early,2), 10);
for i=1:size(Master_activity_DGC_early,2)
    Master_hist_earlynew(i,:) = histcounts(Master_activity_DGC_early(idx_new,i),10);
end


x = [1:1:10];

Master_hist = Master_hist_earlysel;
y1 = (100/N_sel)*mean(Master_hist,1);
tmp = (100/N_sel)*quantile(Master_hist,[0.1 0.9],1);
errlowsel = y1 - tmp(1,:);
errhighsel = tmp(2,:) - y1;

Master_hist = Master_hist_earlynew;
y2 = (100/N_new)*mean(Master_hist,1);
tmp = (100/N_new)*quantile(Master_hist,[0.1 0.9],1);
errlownew = y2 - tmp(1,:);
errhighnew = tmp(2,:) - y2;


figure()
bar(x,y1)
hold on
er = errorbar(x,y1,errlowsel,errhighsel);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
xlabel('Firing rate bin [Hz]')
ylabel('Percentage [%]')
title('End of early phase - sel')
box off
set(gca,'TickDir','out')
ylim([0 100])

figure()
bar(x,y2)
hold on
er = errorbar(x,y2,errlownew,errhighnew);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
xlabel('Firing rate bin [Hz]')
ylabel('Percentage [%]')
title('End of early phase - new')
box off
set(gca,'TickDir','out')
ylim([0 100])


