clear; close all;


%% Predictions vs True Values
STRAIN{1} = 'EC1118';
STRAIN{2} = 'BMV';
STRAIN{3} = 'CR85';

COLOR{1} = [97 170 192]./252;
COLOR{2} = [231 176 36]./252;
COLOR{3} = [238 121 63]./252;


load('EC1118simData_FINAL_AROMAS.mat');

[row,col] = size(res.sim.exp_data{1});
res.sim.exp_data{1}(res.sim.exp_data{1} <= 0) = Inf;
res.fit.ms{1}(res.fit.ms{1} <= 0) = Inf;
p2 = min(min(min(res.sim.exp_data{1})));
res.sim.exp_data{1}(res.sim.exp_data{1} == Inf) = 0;
res.fit.ms{1}(res.fit.ms{1} == Inf) = 0;
p1 = max(max(max(res.sim.exp_data{1})),max(max(res.fit.ms{1})));

figure;

for count=1:col 

x = res.fit.ms{1}(:,count);
y = res.sim.exp_data{1}(:,count);

subplot(1,3,1)
scatter(x,y,'filled','MarkerFaceColor',COLOR{1})
title('EC1118')
ylabel('Observed Values')
xlabel('Predicted Values')
ylim([0 115])
xlim([0 115])
hold on
line([p2 p1],[p2 p1])

end

load('BMVsimData_FINAL_AROMAS.mat');

[row,col] = size(res.sim.exp_data{1});
res.sim.exp_data{1}(res.sim.exp_data{1} <= 0) = Inf;
res.fit.ms{1}(res.fit.ms{1} <= 0) = Inf;
p2 = min(min(min(res.sim.exp_data{1})));
res.sim.exp_data{1}(res.sim.exp_data{1} == Inf) = 0;
res.fit.ms{1}(res.fit.ms{1} == Inf) = 0;
p1 = max(max(max(res.sim.exp_data{1})),max(max(res.fit.ms{1})));

for count=1:col 
x = res.fit.ms{1}(:,count);
y = res.sim.exp_data{1}(:,count);

subplot(1,3,2)
scatter(x,y,'filled','MarkerFaceColor',COLOR{2})
title('BMV')
xlabel('Predicted Values')
ylim([0 115])
xlim([0 115])
hold on
line([p2 p1],[p2 p1])
end

load('CR85simData_FINAL_AROMAS.mat');

[row,col] = size(res.sim.exp_data{1});
res.sim.exp_data{1}(res.sim.exp_data{1} <= 0) = Inf;
res.fit.ms{1}(res.fit.ms{1} <= 0) = Inf;
p2 = min(min(min(res.sim.exp_data{1})));
res.sim.exp_data{1}(res.sim.exp_data{1} == Inf) = 0;
res.fit.ms{1}(res.fit.ms{1} == Inf) = 0;
p1 = max(max(max(res.sim.exp_data{1})),max(max(res.fit.ms{1})));

for count=1:col 
x = res.fit.ms{1}(:,count);
y = res.sim.exp_data{1}(:,count);

subplot(1,3,3)
scatter(x,y,'filled','MarkerFaceColor',COLOR{3})
ylim([0 115])
xlim([0 115])
title('CR85')
xlabel('Predicted Values')
hold on
line([p2 p1],[p2 p1])
end

% exportgraphics(gcf,'singleSTRAIN_predvsdata.pdf')
