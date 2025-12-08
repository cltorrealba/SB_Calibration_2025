%% SAME PLOT IN SUBPLOT

clear; 

STRAINS{1}='EC1118';
STRAINS{2}='BMV';
STRAINS{3}='CR85';

 COLOR{1} = [97 170 192]./252;
 COLOR{2} = [231 176 36]./252;
 COLOR{3} = [238 121 63]./252;

iexp=1;
 figure; 
%% Biomass
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,1)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,1),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,1),inputs.exps.error_data{1}(:,1),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,1),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,1);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,1);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 4.3])
%     xlabel('Time (h)')
    title('Biomass DW (g/L)')
end




%% CFU
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,2)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,3),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,3),inputs.exps.error_data{1}(:,3),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,3),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,3);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,3);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
%     ylim([0 4.3])
%     xlabel('Time (h)')
    title('CFU (g/L)')
end


%% YAN
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,3)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,2),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,2),inputs.exps.error_data{1}(:,2),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,2),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,2);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,2);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.25])
%     xlabel('Time (h)')
    title('YAN (g/L)')
end


%% Glucose
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,4)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,4),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,4),inputs.exps.error_data{1}(:,4),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,4),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,4);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,4);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
%     ylim([0 4.3])
%     xlabel('Time (h)')
    title('Glucose (g/L)')
end


%% Fructose
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,5)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,5),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,5),inputs.exps.error_data{1}(:,5),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,5),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,5);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,5);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
%     ylim([0 4.3])
%     xlabel('Time (h)')
    title('Fructose (g/L)')
end


%% Ethanol
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,6)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,6),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,6),inputs.exps.error_data{1}(:,6),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,6),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,6);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,6);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 100])
%     xlabel('Time (h)')
    title('Ethanol (g/L)')
end


%% Glycerol
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,7)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,7),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,7),inputs.exps.error_data{1}(:,7),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,7),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,7);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,7);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 8])
    xlabel('Time (h)')
    title('Glycerol (g/L)')
end


%% Acetate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,8)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,8),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,8),inputs.exps.error_data{1}(:,8),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,8),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,8);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,8);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 2])
    xlabel('Time (h)')
    title('Acetate (g/L)')
end


%% Succinate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(3,3,9)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,9),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,9),inputs.exps.error_data{1}(:,9),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,9),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,9);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,9);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 6])
    xlabel('Time (h)')
    title('Succinate (g/L)')    
end

%%

 figure; 
%% EthylAcetate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,1)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,29),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,29),inputs.exps.error_data{1}(:,29),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,29),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,29);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,29);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 0.1])
    xlabel('Time (h)')
    title('Ethyl Acetate (g/L)')
end

%% Phenyl Ethyl Acetate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,2)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,34),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,34),inputs.exps.error_data{1}(:,34),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,34),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,34);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,34);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 6e-3])
    xlabel('Time (h)')
    title('Phenyl Ethyl Acetate (g/L)')
end






%% Isoamyl Acetate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,3)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,31),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,31),inputs.exps.error_data{1}(:,31),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,31),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,31);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,31);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
%     ylim([0 0.25])
    xlabel('Time (h)')
    title('Isoamyl Acetate (g/L)')
end


%% Isobutanol
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,4)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,32),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,32),inputs.exps.error_data{1}(:,32),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,32),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,32);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,32);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 0.2])
    xlabel('Time (h)')
    title('Isobutanol (g/L)')
end


%% Isoamyl Alcohol
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,5)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,33),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,33),inputs.exps.error_data{1}(:,33),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,33),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,33);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,33);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 1])
    xlabel('Time (h)')
    title('Isoamyl Alcohol (g/L)')
end

%% Lactate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,6)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,37),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,37),inputs.exps.error_data{1}(:,37),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,37),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,37);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,37);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 0.6])
    xlabel('Time (h)')
    title('Lactate (g/L)')    
end

%% Phenyl Ethyl Acetate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,7)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,34),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,34),inputs.exps.error_data{1}(:,34),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,34),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,34);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,34);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 6e-3])
    xlabel('Time (h)')
    title('Phenyl Ethyl Acetate (g/L)')
end





%% Phenyl Ethanol
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,8)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,36),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,36),inputs.exps.error_data{1}(:,36),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,36),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,36);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,36);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 1])
    xlabel('Time (h)')
    title('Phenyl Ethanol (g/L)')
end




%% Butanediol
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,9)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,38),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,38),inputs.exps.error_data{1}(:,38),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,38),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,38);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,38);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
    ylim([0 1.2])
    xlabel('Time (h)')
    title('Butanediol (g/L)')   
end

% %% Isobutyl Acetate
% for i=1:length(STRAINS)
%     STRAIN=STRAINS{i};
%     load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
%     subplot(3,4,10)
%     plot(results.sim.tsim{1},results.sim.obs{1}(:,30),'Color',COLOR{i},'LineWidth',1.5);
%     hold on;
%     errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,30),inputs.exps.error_data{1}(:,30),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
% %                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,30),inputs.exps.t_s{iexp}(1:end)');
% %                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
% %                     upper = signal+results.fit.obs_conf_mat{iexp}(:,30);
% %                     lower = signal-results.fit.obs_conf_mat{iexp}(:,30);
% %                     y_coordinates = [upper;  lower(end:-1:1,:)];
% %                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
% %                     if i==1
% %                     alpha(1.0)
% %                     elseif i==2
% %                     alpha(1.0)
% %                     else
% %                     alpha(0.1)
% %                     end
%     xlim([0 310])
%     ylim([0 1e-2])
%     xlabel('Time (h)')
%     title('Isobutyl Acetate (g/L)')
% end

% %% Benzyl Alcohol
% for i=1:length(STRAINS)
%     STRAIN=STRAINS{i};
%     load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
%     subplot(3,4,11)
%     plot(results.sim.tsim{1},results.sim.obs{1}(:,35),'Color',COLOR{i},'LineWidth',1.5);
%     hold on;
%     errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,35),inputs.exps.error_data{1}(:,35),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
% %                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,35),inputs.exps.t_s{iexp}(1:end)');
% %                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
% %                     upper = signal+results.fit.obs_conf_mat{iexp}(:,35);
% %                     lower = signal-results.fit.obs_conf_mat{iexp}(:,35);
% %                     y_coordinates = [upper;  lower(end:-1:1,:)];
% %                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
% %                     if i==1
% %                     alpha(1.0)
% %                     elseif i==2
% %                     alpha(1.0)
% %                     else
% %                     alpha(0.1)
% %                     end
%     xlim([0 310])
% %     ylim([0 8])
%     xlabel('Time (h)')
%     title('Benzyl Alcohol (g/L)')
% end

%% Malate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(2,5,10)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,39),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,39),inputs.exps.error_data{1}(:,39),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
%                     signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,39),inputs.exps.t_s{iexp}(1:end)');
%                     x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
%                     upper = signal+results.fit.obs_conf_mat{iexp}(:,39);
%                     lower = signal-results.fit.obs_conf_mat{iexp}(:,39);
%                     y_coordinates = [upper;  lower(end:-1:1,:)];
%                     patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
%                     if i==1
%                     alpha(1.0)
%                     elseif i==2
%                     alpha(1.0)
%                     else
%                     alpha(0.1)
%                     end
    xlim([0 310])
%     ylim([0 6])
    xlabel('Time (h)')
    title('Malate (g/L)')    
end

%% AAcids

figure;


%% Alanine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,1)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,10),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,10),inputs.exps.error_data{1}(:,10),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,10),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,10);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,10);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.08])
    xlabel('Time (h)')
    title('Alanine (g/L)')
end




%% Arginine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,2)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,11),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,11),inputs.exps.error_data{1}(:,11),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,11),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,11);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,11);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.15])
    xlabel('Time (h)')
    title('Arginine (g/L)')
end


%% Aspartate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,3)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,12),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,12),inputs.exps.error_data{1}(:,12),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,12),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,12);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,12);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.025])
    xlabel('Time (h)')
    title('Aspartate (g/L)')
end


%% Cysteine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,4)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,13),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,13),inputs.exps.error_data{1}(:,13),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,13),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,13);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,13);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 10e-3])
    xlabel('Time (h)')
    title('Cysteine (g/L)')
end


%% Glutamate
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,5)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,14),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,14),inputs.exps.error_data{1}(:,14),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,14),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,14);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,14);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.06])
    xlabel('Time (h)')
    title('Glutamate (g/L)')
end


%% Glutamine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,6)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,15),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,15),inputs.exps.error_data{1}(:,15),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,15),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,15);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,15);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
%     ylim([0 0.06])
    xlabel('Time (h)')
    title('Glutamine (g/L)')
end


%% Glycine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,7)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,16),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,16),inputs.exps.error_data{1}(:,16),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,16),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,16);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,16);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.02])
    xlabel('Time (h)')
    title('Glycine (g/L)')
end


%% Histidine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,8)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,17),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,17),inputs.exps.error_data{1}(:,17),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,17),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,17);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,17);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 20e-3])
    xlabel('Time (h)')
    title('Histidine (g/L)')
end


%% Isoleucine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,9)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,18),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,18),inputs.exps.error_data{1}(:,18),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,18),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,18);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,18);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 20e-3])
    xlabel('Time (h)')
    title('Isoleucine (g/L)')    
end

%% Leucine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,10)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,19),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,19),inputs.exps.error_data{1}(:,19),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,18),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,19);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,19);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.03])
    xlabel('Time (h)')
    title('Leucine (g/L)')    
end

%% Lysine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,11)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,20),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,20),inputs.exps.error_data{1}(:,20),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,20),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,20);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,20);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 6e-3])
    xlabel('Time (h)')
    title('Lysine (g/L)')    
end

%% Methyonine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,12)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,20),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,20),inputs.exps.error_data{1}(:,20),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,20),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,20);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,20);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 6e-3])
    xlabel('Time (h)')
    title('Methyonine (g/L)')    
end

%% NH4Cl
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,13)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,21),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,21),inputs.exps.error_data{1}(:,21),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,21),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,21);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,21);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 4e-3])
    xlabel('Time (h)')
    title('NH4Cl (g/L)')    
end

%%  Phenylalanine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,14)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,22),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,22),inputs.exps.error_data{1}(:,22),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,22),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,22);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,22);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.1])
    xlabel('Time (h)')
    title('Phenylalanine (g/L)')    
end

%%  Serine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,15)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,23),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,23),inputs.exps.error_data{1}(:,23),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,23),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,23);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,23);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.02])
    xlabel('Time (h)')
    title('Serine (g/L)')    
end

%%  Threonine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,16)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,24),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,24),inputs.exps.error_data{1}(:,24),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,24),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,24);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,24);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.04])
    xlabel('Time (h)')
    title('Threonine (g/L)')    
end

%%  Tryptophan
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,17)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,25),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,25),inputs.exps.error_data{1}(:,25),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,25),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,25);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,25);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.04])
    xlabel('Time (h)')
    title('Tryptophan (g/L)')    
end

%%  Tyrosine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,18)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,26),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,26),inputs.exps.error_data{1}(:,26),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,26),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,26);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,26);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.15])
    xlabel('Time (h)')
    title('Tyrosine (g/L)')    
end

%%  Valine
for i=1:length(STRAINS)
    STRAIN=STRAINS{i};
    load([STRAIN 'simData_FINAL_AROMAS_UNCERTAINITY'],'results','inputs');
    subplot(4,5,19)
    plot(results.sim.tsim{1},results.sim.obs{1}(:,27),'Color',COLOR{i},'LineWidth',1.5);
    hold on;
    errorbar(inputs.exps.t_s{1},results.sim.exp_data{1}(:,27),inputs.exps.error_data{1}(:,27),'o','MarkerEdgeColor',COLOR{i},'Color',COLOR{i},'LineWidth',1.5,'MarkerSize',8);
                    signal  = interp1q(results.sim.tsim{iexp}',results.sim.obs{iexp}(:,27),inputs.exps.t_s{iexp}(1:end)');
                    x_coordinates = [inputs.exps.t_s{iexp}(1:end) inputs.exps.t_s{iexp}(end:-1:1)];
                    upper = signal+results.fit.obs_conf_mat{iexp}(:,27);
                    lower = signal-results.fit.obs_conf_mat{iexp}(:,27);
                    y_coordinates = [upper;  lower(end:-1:1,:)];
                    patch(x_coordinates,y_coordinates,COLOR{i},'EdgeColor',COLOR{i})
                    if i==1
                    alpha(1.0)
                    elseif i==2
                    alpha(1.0)
                    else
                    alpha(0.1)
                    end
    xlim([0 310])
    ylim([0 0.02])
    xlabel('Time (h)')
    title('Valine (g/L)')    
end



