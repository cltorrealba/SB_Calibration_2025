%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model = changeBiomass(model,P,GAM,NGAM)
%
% Benjam�n J. S�nchez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [model fullGAM] = changeGAM(model,P,R,GAM)

%Get current contents and calculate conversion factors for proteins and carbs:
[Pbase,Cbase,Rbase] = calculateContent(model);
Pfactor = P/Pbase; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cfactor = (Cbase+Pbase-P-R)/Cbase; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rfactor = R/Rbase; 
fullGAM = GAM +16.965*Pfactor + 1.638*Rfactor + 5.210*Cfactor; %%%%%%%%%%%%%%%%%%%%%%%%%%


%Change biomass composition:
protPos = strcmp(model.metNames,'protein');
carbPos = strcmp(model.metNames,'carbohydrate');
rnaPos = strcmp(model.metNames, 'RNA');

bioRxn  = model.S(protPos,:) == -1;
protRxn = model.S(protPos,:) == 1;
carbRxn = model.S(carbPos,:) == 1;
rnaRxn =  model.S(rnaPos,:) == 1;

for i = 1:length(model.mets)
    
    Sbio  = model.S(i,bioRxn);
    Sprot = model.S(i,protRxn);
    Scarb = model.S(i,carbRxn);
    Srna = model.S(i,rnaRxn);
    
    if Sbio ~= 0
        name  = model.metNames{i};
        isATP = strcmpi(name,'ATP');
        isADP = strcmpi(name,'ADP');
        isH2O = strcmpi(name,'H2O');
        isH   = strcmpi(name,'H+');
        isP   = strcmpi(name,'phosphate');
        
        %Variable ATP growth related maintenance (GAM):
        if isATP || isADP || isH2O || isH || isP
            model.S(i,bioRxn) = sign(Sbio)*round(fullGAM,4);
        end
        
    elseif Sprot ~= 0 && Sprot ~= 1
        %Variable aa content in biomass eq:
        model.S(i,protRxn) = round(Sprot*Pfactor,4);
        
    elseif Scarb ~= 0 && Scarb ~= 1
        %Variable carb content in biomass eq:
        model.S(i,carbRxn) = round(Scarb*Cfactor,4);
        
    elseif Srna ~= 0 && Srna ~= 1
        model.S(i,rnaRxn) = round(Srna*Rfactor,4);
        
    end
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%