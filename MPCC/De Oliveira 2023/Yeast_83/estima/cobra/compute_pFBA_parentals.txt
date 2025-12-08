clear;
load_data;
inputs.plotd.plotlevel='min';
res=AMIGO_SModel(inputs);

ltol=1;utol=1;
for ti=1:106
    
    load yeast-GEM_.mat;
    [Pbase,Cbase,Rbase] = calculateContent(model);
   
    %% Exchange for carb
    
    model=addExchangeRxn(model,{'s_3718'});
    model.lb(findRxnIDs(model,'EX_s_3718'))=0;
    model.ub(findRxnIDs(model,'EX_s_3718'))=0;
    
    
    %% maximize protein synthesis
    model=addExchangeRxn(model, {'s_3717'});
    model.lb(findRxnIDs(model,'EX_s_3717'))=0;
    model.ub(findRxnIDs(model,'EX_s_3717'))=0;
    
    
    t=res.sim.tsim{1}(ti);
    D=0;
    
    for i=1:inputs.model.n_st
        namevar=regexprep(inputs.model.st_names(i,:),' ','');
        eval([namevar '=res.sim.states{1}(ti,i);']);
    end
    
    for i=1:inputs.model.n_par
        namepar=regexprep(inputs.model.par_names(i,:),' ','');
        eval([namepar '=inputs.model.par(i);']);
    end
    
    for i=1:size(inputs.model.eqns,1)
        eqn=regexprep(inputs.model.eqns(i,:),' ','');
        eval([eqn ';']);
    end
    
     
    
    
    %% Ala	-kAla*Ala*Xprot	ode	r_1873	89.09
    model.lb(findRxnIDs(model,'r_1873'))=ltol*min(0,dAla/Xprot/optProt/89.09*1000);
    model.ub(findRxnIDs(model,'r_1873'))=utol*min(0,dAla/Xprot/optProt/89.09*1000);
    
    %% Arg	-kAla*Arg*Xprot	ode	r_1879	174.2
    model.lb(findRxnIDs(model,'r_1879'))=ltol*min(0,dArg/Xprot/optProt/174.2*1000);
    model.ub(findRxnIDs(model,'r_1879'))=utol*min(0,dArg/Xprot/optProt/174.2*1000);
    
    %% Asp	-kAsp*Asp*Xprot	ode	r_1880	133.11
    model.lb(findRxnIDs(model,'r_1880'))=ltol*min(0,dAsp/Xprot/optProt/133.11*1000);
    model.ub(findRxnIDs(model,'r_1880'))=utol*min(0,dAsp/Xprot/optProt/133.11*1000);
    
    %% Cys	-kCys*Cys*Xprot	ode	r_1883	121.16
    model.lb(findRxnIDs(model,'r_1883'))=ltol*min(0,dCys/Xprot/optProt/121.16*1000);
    model.ub(findRxnIDs(model,'r_1883'))=utol*min(0,dCys/Xprot/optProt/121.16*1000);
    
    %% Glu	-kGlu*Glu*Xprot	ode	r_1889	147.13
    model.lb(findRxnIDs(model,'r_1889'))=ltol*min(0,dGlu/Xprot/optProt/147.13*1000);
    model.ub(findRxnIDs(model,'r_1889'))=utol*min(0,dGlu/Xprot/optProt/147.13*1000);
    
    %% Gln	-kGln*Gln*Xprot	ode	r_1891	146.14
    model.lb(findRxnIDs(model,'r_1891'))=ltol*min(0,dGln/Xprot/optProt/146.14*1000);
    model.ub(findRxnIDs(model,'r_1891'))=utol*min(0,dGln/Xprot/optProt/146.14*1000);
    
    %% Gly	-kGly*Gly*Xprot	ode	r_1810	75.07
    model.lb(findRxnIDs(model,'r_1810'))=ltol*min(0,dGly/Xprot/optProt/75.07*1000);
    model.ub(findRxnIDs(model,'r_1810'))=utol*min(0,dGly/Xprot/optProt/75.07*1000);
    
    %% His	-kHis*His*Xprot	ode	r_1893	155.1546
    model.lb(findRxnIDs(model,'r_1893'))=ltol*min(0,dHis/Xprot/optProt/155.1546*1000);
    model.ub(findRxnIDs(model,'r_1893'))=0;
    
    %% Ile	-kIle*Ile*Xprot	ode	r_1897	131.17
    model.lb(findRxnIDs(model,'r_1897'))=ltol*min(0,dIle/Xprot/optProt/131.17*1000);
    model.ub(findRxnIDs(model,'r_1897'))=utol*min(0,dIle/Xprot/optProt/131.17*1000);
    
    %% Leu	-kLeu*Leu*Xprot	ode	r_1899	131.17
    model.lb(findRxnIDs(model,'r_1899'))=ltol*min(0,dLeu/Xprot/optProt/131.17*1000);
    model.ub(findRxnIDs(model,'r_1899'))=utol*min(0,dLeu/Xprot/optProt/131.17*1000);
    
    %% NH4Cl	-kNH4Cl*NH4Cl*Xprot	ode	r_1654	18.04
    model.lb(findRxnIDs(model,'r_1654'))=ltol*min(0,dNH4Cl/Xprot/optProt/18.039*1000);
    model.ub(findRxnIDs(model,'r_1654'))=utol*min(0,dNH4Cl/Xprot/optProt/18.039*1000);
    
    %% Lys	-kLys*Lys*Xprot	ode	r_1900	146.19
    model.lb(findRxnIDs(model,'r_1900'))=ltol*min(0,dLys/Xprot/optProt/146.19*1000);
    model.ub(findRxnIDs(model,'r_1900'))=utol*min(0,dLys/Xprot/optProt/146.19*1000);
    
    %% Met	-kMet*Met*Xprot	ode	r_1902	149.21
    model.lb(findRxnIDs(model,'r_1902'))=ltol*min(0,dMet/Xprot/optProt/149.21*1000);
    model.ub(findRxnIDs(model,'r_1902'))=utol*min(0,dMet/Xprot/optProt/149.21*1000);
    
    %% Phe	-kPhe*Phe*Xprot	ode	r_1903	165.19
    model.lb(findRxnIDs(model,'r_1903'))=ltol*min(0,dPhe/Xprot/optProt/165.19*1000);
    model.ub(findRxnIDs(model,'r_1903'))=utol*min(0,dPhe/Xprot/optProt/165.19*1000);
    
    %% Ser	-kSer*Ser*Xprot	ode	r_1906	105.09
    model.lb(findRxnIDs(model,'r_1906'))=ltol*min(0,dSer/Xprot/optProt/105.09*1000);
    model.ub(findRxnIDs(model,'r_1906'))=utol*min(0,dSer/Xprot/optProt/105.09*1000);
    
    %% Thr	-kThr*Thr*Xprot	ode	r_1911	119.1192
    model.lb(findRxnIDs(model,'r_1911'))=ltol*min(0,dThr/Xprot/optProt/119.1192*1000);
    model.ub(findRxnIDs(model,'r_1911'))=utol*min(0,dThr/Xprot/optProt/119.1192*1000);
    
    %% Try	-kTry*Try*Xprot	ode	r_1912	204.23
    model.lb(findRxnIDs(model,'r_1912'))=ltol*min(0,dTry/Xprot/optProt/204.23*1000);
    model.ub(findRxnIDs(model,'r_1912'))=utol*min(0,dTry/Xprot/optProt/204.23*1000);
    
    %% Tyr	-kTyr*Tyr*Xprot	ode	r_1913	181.19
    model.lb(findRxnIDs(model,'r_1913'))=ltol*min(0,dTyr/Xprot/optProt/204.23*1000);
    model.ub(findRxnIDs(model,'r_1913'))=utol*min(0,dTyr/Xprot/optProt/204.23*1000);
    
    %% Val	-kVal*Val*Xprot	ode	r_1914	117.151
    model.lb(findRxnIDs(model,'r_1914'))=ltol*min(0,dVal/Xprot/optProt/117.151*1000);
    model.ub(findRxnIDs(model,'r_1914'))=utol*min(0,dVal/Xprot/optProt/117.151*1000);
    
    
    %% Glucose
    model.lb(findRxnIDs(model,'r_1714'))=min(0,v_Glx/Xprot/optProt/180.156*1000);
    model.ub(findRxnIDs(model,'r_1714'))=min(0,v_Glx/Xprot/optProt/180.156*1000);
    
    %% Fructose
    model.lb(findRxnIDs(model,'r_1709'))=min(0,v_F/Xprot/optProt/180.156*1000);
    model.ub(findRxnIDs(model,'r_1709'))=min(0,v_F/Xprot/optProt/180.156*1000);
    
    %% Ethanol
    model.ub(findRxnIDs(model,'r_1761'))=max(0,dEthanol/Xprot/optProt/46.07*1000);
    model.lb(findRxnIDs(model,'r_1761'))=max(0,dEthanol/Xprot/optProt/46.07*1000); 
    
     %% Growth
     model.lb(findRxnIDs(model,'r_4041'))=dX/X;
     model.ub(findRxnIDs(model,'r_4041'))=dX/X;
    
     %% glycerol exchange
    model.ub(findRxnIDs(model,'r_1808'))=max(0,dGlycerol/Xprot/optProt/92.09*1000);
    
    %% r_1634	acetate exchange
    model.ub(findRxnIDs(model,'r_1634'))=(dAcetate/Xprot/optProt/60.052*1000);
    
    %% NGAM
    rxn=findRxnIDs(model,'r_4046');
    model.lb(rxn)=0;
    model.ub(rxn)=0;
    
    
    %% r_2056	succinate exchange
    model.lb(findRxnIDs(model,'r_2056'))=max(0,dSuccinate/Xprot/optProt/118.09*1000);
   
    
    %% 23But
    rxn=findRxnIDs(model,'r_1549');
    V23But=dButanediol/Xprot/optProt/90.12*1000;
    model.lb(rxn)=  V23But;
    model.ub(rxn)=  V23But;


    %% Lactate MW=90.08
    rxn=findRxnIDs(model,'r_1546');
    vLac=dLactate/Xprot/optProt/90.08*1000;
    model.lb(rxn)=vLac;
    model.ub(rxn)=1000;


    %% Malate MW=59.044
    rxn=findRxnIDs(model,'r_1552');
    model.lb(rxn)=dMalate/Xprot/optProt/134.09*1000;
    model.ub(rxn)=1000;

    %% Acetate Ethyl MW=88.11
    rxn=findRxnIDs(model,'r_1765');
    vEthylAce=dEthylAce/Xprot/optProt/88.11*1000;
    model.lb(rxn)=vEthylAce;
    model.ub(rxn)=vEthylAce;%1000;

    %% Isobutyl Acetate MW=116.16
    rxn=findRxnIDs(model,'r_1867');
    vIsobutylAce=dIsobutylAce/Xprot/optProt/116.16*1000;
    model.lb(rxn)=vIsobutylAce;
    model.ub(rxn)=1000;

    %% Isobutanol MW=74.122
    rxn=findRxnIDs(model,'r_1866');
    vIsoButanol=dIsobutanol/Xprot/optProt/74.12*1000;
    model.lb(rxn)=vIsoButanol;
    model.ub(rxn)=1000;


    %% Isoamyl acetate MW=130.19
    rxn=findRxnIDs(model,'r_1862');
    vIsoamylAce=dIsoamylAce/Xprot/optProt/130.187*1000;
    model.lb(rxn)=vIsoamylAce;
    model.ub(rxn)=vIsoamylAce;%1000;

    %% Isoamyl alcohol  MW=88.148
    rxn=findRxnIDs(model,'r_1865');
    vIsoamylAlc=dIsoamylAlc/Xprot/optProt/88.15*1000;
    model.lb(rxn)=vIsoamylAlc;
    model.ub(rxn)=1000;


    %% 2-Phenylethyl acetate MW= 164.20
    rxn=findRxnIDs(model,'r_2000');
    v2PheAce=dPheAce/Xprot/optProt/164.20*1000;
    model.lb(rxn)=v2PheAce;
    model.ub(rxn)=1000;

    %% 2-fenil-etanol   MW=122.167 g/mol
    rxn=findRxnIDs(model,'r_1589');
    v2PheEth=dPheEth/Xprot/optProt/122.17*1000;
    model.lb(rxn)=v2PheEth;
    model.ub(rxn)=1000;


    %% Define anerobic conditions
     model=anaerobicModel(model);
    model.ub(strcmp(model.rxns,'r_0472')) = 1000;
   
%% Change model objective
        model=changeObjective(model,{'r_0227','EX_s_3717'},[1-phiNS,phiNS]);
        model_nongrowth=model;

   switch growth1+growth2>1e-3
       case 1
           try
            %% change biomass composition
            model=changeGAM(model,dXprot/dX,1-dXprot/dX-dXcarb/dX,GAM_b);
            
            %% Solve pFBA
            [GeneClasses RxnClasses modelIrrevFM MinimizedFlux match2Rev]=compute_minimal_flux(model, 'geneoption',0,'skipclass',1,'tol',1e-3,'inhibitor',1);
            %% pFBA returns an irreversible model, map irreversible back to reversible
            flux=convertIrrevFluxDistribution(MinimizedFlux.x(1:end-1), match2Rev);
            backReacs =  model.lb < 0 & model.ub <= 0;
            flux(backReacs) = -flux(backReacs);
    
           catch
                no_growth
           end
       case 0
            no_growth
   end


        aux_g2=growth2;
        
        end
  