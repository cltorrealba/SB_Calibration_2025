data=importdata(fullfile(STRAIN,'data_hybrids.csv'),',');
    
Time=data.data(:,2);

if(BOOTSTRAP)
    
    T0_indices=find(Time==0);
    bootsrap_indices=datasample(find(Time~=0),length(find(Time~=0)));
    all_indices=[T0_indices' bootsrap_indices'];
    data.data=data.data(all_indices,:);
    data.textdata=data.textdata([1 all_indices+1],:);
    
end


COLNAMES={data.textdata{1,:}};


Temp=data.data(:,1);
Time=data.data(:,2);
replicates={data.textdata{:,2}};
strain1={data.textdata{2:end,1}};


NVARS=length({COLNAMES{3:end}});
COLNAMES={COLNAMES{3:end}};

time_vec=sort(unique(Time));

exps_names=[char(strain1) repmat('-',length(strain1),1)   repmat(' T=',length(strain1),1)  num2str(Temp)];

exps_names=regexprep(cellstr(exps_names),'NaN','single');
exps_name=unique(exps_names);



obs_names=COLNAMES;
obs_funcs=COLNAMES;