
BOOTSTRAP=0;

bootstrap;
inputs.exps.n_exp=length(exps_name);


for i=1:length(exps_name)
    
    inputs.exps.exp_data{i}=nan(length(time_vec),NVARS);
    inputs.exps.error_data{i}=nan(length(time_vec),NVARS);
    
    temperature=nan;
    
    for j=1:length(time_vec)
        
        vec=(Time==time_vec(j) & strcmp(exps_names,exps_name{i}));
        
        if(isnan(temperature))
            temperature=mean(Temp(vec),'omitnan');
        end
        
        for k=1:NVARS
            inputs.exps.exp_data{i}(j,k)=mean(data.data(vec,k),'omitnan');
            inputs.exps.error_data{i}(j,k)=std(data.data(vec,k),'omitnan');
        end
        
    end
    
    inputs.exps.n_s{i}=length(time_vec);
    inputs.exps.obs_names{i}=char(obs_names);
    inputs.exps.obs{i}=char(obs_funcs);
    inputs.exps.n_obs{i}=NVARS;
    inputs.exps.t_s{i}=time_vec';
    
    inputs.exps.exp_y0{i}=inputs.exps.exp_data{i}(1,:);
    inputs.exps.exp_data{i}(inputs.exps.exp_data{i}<0)=NaN;
        
    inputs.exps.t_in{i}=time_vec(1);
    inputs.exps.t_f{i}=time_vec(end);
    inputs.exps.exp_y0{i}(isnan(inputs.exps.exp_y0{i}))=0;
    
    %% 4 CFU From CFU/ml to L
    inputs.exps.exp_data{i}(:,4)=inputs.exps.exp_data{i}(:,4).*1000;
    inputs.exps.error_data{i}(:,4)=inputs.exps.error_data{i}(:,4).*1000;
        
    %% 8 Ethanol is in %Vol-> convert to g/L Ethanol density in 789g/L
    inputs.exps.exp_data{i}(:,8)= inputs.exps.exp_data{i}(:,8).*789/100;
    inputs.exps.error_data{i}(:,8)= inputs.exps.error_data{i}(:,8).*789/100;
    
    %% 16:35 Amino acids is in mg/L -> convert to g/L 
    inputs.exps.exp_data{i}(:,16:35)= inputs.exps.exp_data{i}(:,16:35)./1000;
    inputs.exps.error_data{i}(:,16:35)= inputs.exps.error_data{i}(:,16:35)./1000;    
    
    %% 28 NH4Cl -> Proportion of ammonia
    inputs.exps.exp_data{i}(:,28)= inputs.exps.exp_data{i}(:,28)    .*(18.039/53.491);
    inputs.exps.error_data{i}(:,28)= inputs.exps.error_data{i}(:,28).*(18.039/53.491);
    
    %% Filter undesirable values

    inputs.exps.error_data{i}(inputs.exps.error_data{i}==0)=nan;

    %% 36:43 AROMAS in mg/L -> convert to g/L
    inputs.exps.exp_data{i}(:,36:43)= inputs.exps.exp_data{i}(:,36:43)./1000;
    inputs.exps.error_data{i}(:,36:43)= inputs.exps.error_data{i}(:,36:43)./1000; 
    
end

%% Load Amino acid d

%MW  Alanine	Arginine	Aspartate	Cystein	   Glutamate	Glutamine	Glysine	Histidin IsoLeucine   Leucine  Lysine   Methionine	NH4	     Phenylalanine	Serine	Threonine	Tryptophan	Tyrosine	Valine	
MW=[  89.09     174.2       133.11       121.16     147.13       146.14     75.07   155.15    131.17       131.17  146.19    149.21     18.039   165.19         105.09  119.1192    204.23      181.19      117.151];
NN=[  1         4           2            1           1           2          1        3         1            1      2         1          1        1              1       1           2           1           1      ];


NitrogenData=inputs.exps.exp_data{1}(:,16:35);
NitrogenData_error=inputs.exps.error_data{1}(:,16:35);

%% Exclude Proline, Phenylalanine, Methionine, Lysisne, Leucine, Isoleucine and Histidine
%

NitrogenData(:,[15])=[];
NitrogenData_error(:,[15])=[];


%%
Original=NitrogenData;
Original_error=NitrogenData_error;

for i=1:size(NitrogenData,2)
    Original(:,i)=NN(i).*NitrogenData(:,i)./MW(i);
    NitrogenData(:,i)=NN(i).*NitrogenData(:,i)./MW(i);
    time=inputs.exps.t_s{1};
    
    Original_error(:,i)=NN(i).*NitrogenData_error(:,i)./MW(i);
    NitrogenData_error(:,i)=NN(i).*NitrogenData_error(:,i)./MW(i);
end

NitrogenData(NitrogenData<0)=0;
NitrogenData(isnan(NitrogenData))=0;
YAN=(sum(NitrogenData')).*14.0067; %Nitrogen
YAN(YAN<=0)=nan;


inputs.exps.exp_data{1}(:,end+1)=YAN;

NitrogenData_error(NitrogenData_error<0)=0;
NitrogenData_error(isnan(NitrogenData_error))=0;
YAN_error=(sum(NitrogenData_error')).*14.0067;
YAN_error(YAN_error<=0)=nan;

inputs.exps.error_data{1}(:,end+1)=YAN_error;
