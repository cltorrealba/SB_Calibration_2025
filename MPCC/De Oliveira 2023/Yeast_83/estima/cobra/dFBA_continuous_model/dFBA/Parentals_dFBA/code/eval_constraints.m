function  [success fullGAM] =  eval_constraints(t,y,par)

load yeast-GEM.mat;
load inputs_data;

model=addExchangeRxn(model,{'s_3718[c]'});
model.lb(findRxnIDs(model,'EX_s_3718[c]'))=0;
model.ub(findRxnIDs(model,'EX_s_3718[c]'))=0;

D=0;

for i=1:inputs.model.n_st
    namevar=regexprep(inputs.model.st_names(i,:),' ','');
    eval([namevar '=y(i);']);
end

for i=1:inputs.model.n_par
    namepar=regexprep(inputs.model.par_names(i,:),' ','');
    eval([namepar '=par(i);']);
end

%% last N equations are used for addint the constraints
for i=1:size(inputs.model.eqns,1)
    eqn=regexprep(inputs.model.eqns(i,:),' ','');
    eval([eqn ';']);
end

%% Ala	-kAla*Ala*Xprot	ode	r_1873	89.09
model.lb(findRxnIDs(model,'r_1873'))=min(0,dAla/Xprot/optProt/89.09*1000);
model.ub(findRxnIDs(model,'r_1873'))=min(0,dAla/Xprot/optProt/89.09*1000);

%% Arg	-kAla*Arg*Xprot	ode	r_1879	174.2
model.lb(findRxnIDs(model,'r_1879'))=min(0,dArg/Xprot/optProt/174.2*1000);
model.ub(findRxnIDs(model,'r_1879'))=min(0,dArg/Xprot/optProt/174.2*1000);

%% Asp	-kAsp*Asp*Xprot	ode	r_1880	133.11
model.lb(findRxnIDs(model,'r_1880'))=min(0,dAsp/Xprot/optProt/133.11*1000);
model.ub(findRxnIDs(model,'r_1880'))=min(0,dAsp/Xprot/optProt/133.11*1000);

%% Cys	-kCys*Cys*Xprot	ode	r_1883	121.16

%% Glu	-kGlu*Glu*Xprot	ode	r_1889	147.13
model.lb(findRxnIDs(model,'r_1889'))=min(0,dGlu/Xprot/optProt/147.13*1000);
model.ub(findRxnIDs(model,'r_1889'))=min(0,dGlu/Xprot/optProt/147.13*1000);

%% Gln	-kGln*Gln*Xprot	ode	r_1891	146.14
model.lb(findRxnIDs(model,'r_1891'))=min(0,dGln/Xprot/optProt/146.14*1000);
model.ub(findRxnIDs(model,'r_1891'))=min(0,dGln/Xprot/optProt/146.14*1000);

%% Gly	-kGly*Gly*Xprot	ode	r_1810	75.07
model.lb(findRxnIDs(model,'r_1810'))=min(0,dGly/Xprot/optProt/75.07*1000);
model.ub(findRxnIDs(model,'r_1810'))=min(0,dGly/Xprot/optProt/75.07*1000);

%% His	-kHis*His*Xprot	ode	r_1893	155.1546
model.lb(findRxnIDs(model,'r_1893'))=min(0,dHis/Xprot/optProt/155.1546*1000);
model.ub(findRxnIDs(model,'r_1893'))=0;

%% Ile	-kIle*Ile*Xprot	ode	r_1897	131.17
model.lb(findRxnIDs(model,'r_1897'))=min(0,dIle/Xprot/optProt/131.17*1000);
model.ub(findRxnIDs(model,'r_1897'))=min(0,dIle/Xprot/optProt/131.17*1000);

%% Leu	-kLeu*Leu*Xprot	ode	r_1899	131.17
model.lb(findRxnIDs(model,'r_1899'))=min(0,dLeu/Xprot/optProt/131.17*1000);
model.ub(findRxnIDs(model,'r_1899'))=min(0,dLeu/Xprot/optProt/131.17*1000);

% %% Lys	-kLys*Lys*Xprot	ode	r_1900	146.19
model.lb(findRxnIDs(model,'r_1900'))=min(0,dLys/Xprot/optProt/146.19*1000);
% model.ub(findRxnIDs(model,'r_1900'))=min(0,dLys/Xprot/optProt/146.19*1000);

%% Met	-kMet*Met*Xprot	ode	r_1902	149.21
model.lb(findRxnIDs(model,'r_1902'))=min(0,dMet/Xprot/optProt/149.21*1000);
model.ub(findRxnIDs(model,'r_1902'))=min(0,dMet/Xprot/optProt/149.21*1000);

%% NH4Cl	-kNH4Cl*NH4Cl*Xprot	ode	r_1654	18.04
model.lb(findRxnIDs(model,'r_1654'))=min(0,dNH4Cl/Xprot/optProt/14*1000);
model.ub(findRxnIDs(model,'r_1654'))=min(0,dNH4Cl/Xprot/optProt/14*1000);

%% Phe	-kPhe*Phe*Xprot	ode	r_1903	165.19
model.lb(findRxnIDs(model,'r_1903'))=min(0,dPhe/Xprot/optProt/165.19*1000);
model.ub(findRxnIDs(model,'r_1903'))=min(0,dPhe/Xprot/optProt/165.19*1000);

%% Ser	-kSer*Ser*Xprot	ode	r_1906	105.09
model.lb(findRxnIDs(model,'r_1906'))=min(0,dSer/Xprot/optProt/105.09*1000);
model.ub(findRxnIDs(model,'r_1906'))=min(0,dSer/Xprot/optProt/105.09*1000);

%% Thr	-kThr*Thr*Xprot	ode	r_1911	119.1192
model.lb(findRxnIDs(model,'r_1911'))=min(0,dThr/Xprot/optProt/119.1192*1000);
model.ub(findRxnIDs(model,'r_1911'))=min(0,dThr/Xprot/optProt/119.1192*1000);

%% Try	-kTry*Try*Xprot	ode	r_1912	204.23
model.lb(findRxnIDs(model,'r_1912'))=min(0,dTry/Xprot/optProt/204.23*1000);
model.ub(findRxnIDs(model,'r_1912'))=min(0,dTry/Xprot/optProt/204.23*1000);

%% Tyr	-kTyr*Tyr*Xprot	ode	r_1913	181.19
model.lb(findRxnIDs(model,'r_1913'))=min(0,dTyr/Xprot/optProt/204.23*1000);
model.ub(findRxnIDs(model,'r_1913'))=min(0,dTyr/Xprot/optProt/204.23*1000);

%% Val	-kVal*Val*Xprot	ode	r_1914	117.151
model.lb(findRxnIDs(model,'r_1914'))=min(0,dVal/Xprot/optProt/117.151*1000);
model.ub(findRxnIDs(model,'r_1914'))=min(0,dVal/Xprot/optProt/117.151*1000);

%% Citrate	-kcCitrate*(1-repression)*Xprot*Citrate ode	r_1687	192.124
model.lb(findRxnIDs(model,'r_1687'))=min(0,dCitrate/Xprot/optProt/192.124*1000);
model.ub(findRxnIDs(model,'r_1687'))=min(0,dCitrate/Xprot/optProt/192.124*1000);
% 
%% Malate	-kcMalate*(1-repression)*Xprot*Malate	ode	r_1552	134.0874
model.lb(findRxnIDs(model,'r_1552'))=min(0,dMalate/X/134.0874*1000);
model.ub(findRxnIDs(model,'r_1552'))=min(0,dMalate/X/134.0874*1000);

%% Glucose
model.lb(findRxnIDs(model,'r_1714'))=min(0,v_Glx/Xprot/optProt/180.156*1000);
model.ub(findRxnIDs(model,'r_1714'))=min(0,v_Glx/Xprot/optProt/180.156*1000);

%% Fructose
model.lb(findRxnIDs(model,'r_1709'))=min(0,v_F/Xprot/optProt/180.156*1000);
model.ub(findRxnIDs(model,'r_1709'))=min(0,v_F/Xprot/optProt/180.156*1000);

%% Ethanol
model.lb(findRxnIDs(model,'r_1761'))=max(0,dEthanol/Xprot/optProt/46.07*1000);
model.ub(findRxnIDs(model,'r_1761'))=max(0,dEthanol/Xprot/optProt/46.07*1000);

%% Growth
model.lb(findRxnIDs(model,'r_4041'))=dX/X;
model.ub(findRxnIDs(model,'r_4041'))=dX/X;

%% glycerol exchange
model.lb(findRxnIDs(model,'r_1808'))=max(0,dGlycerol/Xprot/optProt/92.09*1000);
model.ub(findRxnIDs(model,'r_1808'))=max(0,dGlycerol/Xprot/optProt/92.09*1000);

%% r_1634	acetate exchange
model.lb(findRxnIDs(model,'r_1634'))=max(0,dAcetate/Xprot/optProt/60.052*1000);
model.ub(findRxnIDs(model,'r_1634'))=max(0,dAcetate/Xprot/optProt/60.052*1000);

%% r_2056	succinate exchange
model.lb(findRxnIDs(model,'r_2056'))=max(0,dSuccinate/Xprot/optProt/118.09*1000);
model.ub(findRxnIDs(model,'r_2056'))=max(0,dSuccinate/Xprot/optProt/118.09*1000);

%% r_1862 isoamyl acetate exchange	isoamyl acetate[e] =>
model.lb(findRxnIDs(model,'r_1862'))=max(0,dAcIsoamilo/Xprot/optProt/130.19*1000);
model.ub(findRxnIDs(model,'r_1862'))=max(0,dAcIsoamilo/Xprot/optProt/130.19*1000);

%% r_1865 isoamylol exchange	isoamylol[e] =>
model.lb(findRxnIDs(model,'r_1865'))=max(0,dAlcIsoam/Xprot/optProt/118.09*1000);
model.ub(findRxnIDs(model,'r_1865'))=max(0,dAlcIsoam/Xprot/optProt/118.09*1000);
 
%% r_1866 isobutanol exchange	isobutanol[e] =>
model.lb(findRxnIDs(model,'r_1866'))=max(0,dIsobutanol/Xprot/optProt/74.122*1000);
model.ub(findRxnIDs(model,'r_1866'))=max(0,dIsobutanol/Xprot/optProt/74.122*1000);

%% r_1867 isobutyl acetate exchange	isobutyl acetate[e] =>
model.lb(findRxnIDs(model,'r_1867'))=max(0,dAcIso/Xprot/optProt/116.16*1000);
model.ub(findRxnIDs(model,'r_1867'))=max(0,dAcIso/Xprot/optProt/116.16*1000);

%% r_1867 2-Phenylethanol
model.lb(findRxnIDs(model,'r_1589'))=max(0,dFenilEtanol/Xprot/optProt/122.16*1000);
model.ub(findRxnIDs(model,'r_1589'))=max(0,dFenilEtanol/Xprot/optProt/122.16*1000);

%% r_2000 phenethyl acetate exchange	phenethyl acetate[e] =>
model.lb(findRxnIDs(model,'r_2000'))=max(0,dFenilEtanol/Xprot/optProt/164.20*1000);
model.ub(findRxnIDs(model,'r_2000'))=max(0,dFenilEtanol/Xprot/optProt/164.20*1000);

%% Define anerobic conditions
model=anaerobicModel(model);
GAM_b=GAM_b*0.99;

try
    
    %% change biomass composition
    [model fullGAM]=changeGAM(model,dXprot/dX,1-dXprot/dX-dXcarb/dX,GAM_b);

    %% Solve pFBA
    [GeneClasses RxnClasses modelIrrevFM MinimizedFlux match2Rev]=compute_minimal_flux(model, 'geneoption',0,'skipclass',1,'tol',1e-3,'inhibitor',1);
    %% pFBA returns an irreversible model, map irreversible back to reversible
    flux=convertIrrevFluxDistribution(MinimizedFlux.full(1:end-1), match2Rev);
    backReacs =  model.lb < 0 & model.ub <= 0;
    flux(backReacs) = -flux(backReacs);
    
    if(MinimizedFlux.stat==1)
        success=1;
    else
        success=0;
    end
    
catch
    
    if(t<TGROWTH)
        success=0;
    else
        %% constrain carbohydrate production
        model.lb(findRxnIDs(model,'EX_s_3718[c]'))=0;
        model.ub(findRxnIDs(model,'EX_s_3718[c]'))=1000;
        
        %% maximize protein synthesis
        model=addExchangeRxn(model, {'s_3717[c]'});
        model.lb(findRxnIDs(model,'EX_s_3717[c]'))=0;
        model.ub(findRxnIDs(model,'EX_s_3717[c]'))=1000;
        model=changeObjective(model,{'EX_s_3717[c]'},1);
        
        %% Growth
        model.lb(findRxnIDs(model,'r_4041'))=0;
        model.ub(findRxnIDs(model,'r_4041'))=1000;

        try
            %% Solve pFBA
            [GeneClasses RxnClasses modelIrrevFM MinimizedFlux match2Rev]=compute_minimal_flux(model, 'geneoption',0,'skipclass',1,'tol',1e-3,'inhibitor',1);
            %% pFBA returns an irreversible model, map irreversible back to reversible
            flux=convertIrrevFluxDistribution(MinimizedFlux.full(1:end-1), match2Rev);
            backReacs =  model.lb < 0 & model.ub <= 0;
            flux(backReacs) = -flux(backReacs);
            
            if(MinimizedFlux.stat==1)
                success=1;
            else
                success=0;
            end
        catch
            success=0;
        end
        
    end
    
end

end


