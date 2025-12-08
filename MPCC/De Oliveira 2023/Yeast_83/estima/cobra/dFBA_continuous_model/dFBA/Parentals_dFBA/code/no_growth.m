model=model_nongrowth;
       %% Glucitol MW=118.09
rxn=findRxnIDs(model,'r_1712');
model.lb(rxn)=  0;
model.ub(rxn)=  0;
rxn=findRxnIDs(model,'r_1886');
model.lb(rxn)=  0;
model.ub(rxn)=  0;

%% ATP synthetase block (only in stat)
rxn=findRxnIDs(model,'r_0226');
model.lb(rxn)=0 ;
model.ub(rxn)= 0;

%% block isocitrate lyase
model.lb(findRxnIDs(model,'r_0661'))=0;
model.ub(findRxnIDs(model,'r_0661'))=0;

model.lb(findRxnIDs(model,'r_0662'))=0;
model.ub(findRxnIDs(model,'r_0662'))=0;

%% close acetaldehyde
model.lb(findRxnIDs(model,'r_1633'))=0;
model.ub(findRxnIDs(model,'r_1633'))=0;

%% 1 Octanol
model.lb(findRxnIDs(model,'r_4614'))=0;
model.ub(findRxnIDs(model,'r_4614'))=0;

%% Pyruvate transport
rxn=findRxnIDs(model,'r_1254');
model.lb(rxn)=0 ;
model.ub(rxn)=0;

%% ethanol  MW=88.148
model.ub(findRxnIDs(model,'r_1761'))=1000;

%% Glycerol Out MW=92.09382 g/mol
model.ub(rxn)=1000;

%% Isoamyl alcohol  MW=88.148
model.ub(rxn)=1000;

%% 23But
model.ub(rxn)=  1000;

%% Protein accumulation flux
model.lb(findRxnIDs(model,'EX_s_3717'))=0;
model.ub(findRxnIDs(model,'EX_s_3717'))=1000*phiNS;

%% Carbohydrate accumulation flux
model.lb(findRxnIDs(model,'EX_s_3718'))=phiNS*(growth2/Cbase)/(Xprot/optProt);
model.ub(findRxnIDs(model,'EX_s_3718'))=phiNS*(growth2/Cbase)/(Xprot/optProt);
      
%% MALATE DEHYDROGENASE, REVERSIBLE NECESSARY FOR SUCCINATE production through reductive brach
model.lb(strcmp(model.rxns,'r_0713')) = -1000; %Mithocondria
model.lb(strcmp(model.rxns,'r_0714')) = -1000; %Cytoplasm

%% Growth
model.lb(findRxnIDs(model,'r_2111'))=0;
model.ub(findRxnIDs(model,'r_2111'))=1000;

%% Butyrate
rxn=findRxnIDs(model,'r_2187');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% Decanoate
rxn=findRxnIDs(model,'r_1727');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% Myristate
rxn=findRxnIDs(model,'r_2193');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% (R)-mevalonate exchange
rxn=findRxnIDs(model,'r_1547');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% Panthotenate
rxn=findRxnIDs(model,'r_1548');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% Choline
rxn=findRxnIDs(model,'r_1682');
model.ub(rxn)=1000;

%% hexanoate
rxn=findRxnIDs(model,'r_2187');
model.ub(rxn)=1000;

%% oleate
rxn=findRxnIDs(model,'r_2187');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% ethyl-octanoate exchange
rxn=findRxnIDs(model,'r_4599');
model.lb(rxn)=0;
model.ub(rxn)=1000;

%% 'ethyl-decanoate exchange'
rxn=findRxnIDs(model,'r_4603');
model.lb(rxn)=0;
model.ub(rxn)=1000;

model.ub(strcmp(model.rxns,'r_1757')) = 1000;    %ergosterol
model.ub(strcmp(model.rxns,'r_1915')) = 1000;    %lanosterol
model.ub(strcmp(model.rxns,'r_1994')) = 1000;    %palmitoleate
model.ub(strcmp(model.rxns,'r_2106')) = 1000;    %zymosterol
model.ub(strcmp(model.rxns,'r_2134')) = 1000;    %14-demethyllanosterol
model.ub(strcmp(model.rxns,'r_2137')) = 1000;    %ergosta-5,7,22,24(28)-tetraen-3beta-ol
model.ub(strcmp(model.rxns,'r_2189')) = 1000;    %oleate

%% Growth
model.lb(findRxnIDs(model,'r_4041'))=0;
model.ub(findRxnIDs(model,'r_4041'))=0;


%% Solve pFBA
[GeneClasses RxnClasses modelIrrevFM MinimizedFlux match2Rev]=compute_minimal_flux(model, 'geneoption',0,'skipclass',1,'tol',1e-3,'inhibitor',1);
%% pFBA returns an irreversible model, map irreversible back to reversible
flux=convertIrrevFluxDistribution(MinimizedFlux.x(1:end-1), match2Rev);
backReacs =  model.lb < 0 & model.ub <= 0;
flux(backReacs) = -flux(backReacs);