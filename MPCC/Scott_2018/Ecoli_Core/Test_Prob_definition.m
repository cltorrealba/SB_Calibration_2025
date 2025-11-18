function Test_Prob_definition(Prob_Name)
%%  This file creates a Matlab structure defining the dFBA problem for a given
% value of the barrier parameter. It can be called within or outside the
% main.m execution routine.

clc

% The following modifications were made:
% 1) Matrix S was augmented with a row (C) of all zeros except for a 1 in
% the position 6 (representing uptake flux).
% 2) The vector b was augmented with the value of the glucose uptake flux.

model_name=Prob_Name;
fprintf ('Creating dFBA model structure for problem: %s\n\n', model_name)


%% Load .mat definition if requiered

load('Test_Data.mat')

% Diferential and algebraic variables

DIFF_VARS={'cx','cs'}; 

iODE_VARS={'dcxdt','dcsdt','dvuptdt'};

ALG_VARS={'vupt'};

% Differential equations.
DIFF_EQS={'v_flux(5) * cx'; %v_flux(2985) is biomass flux
          '-vupt*cx';}; 

ALG_EQS ={'vupt' , 'vmax*cs/(Ks+cs)'};

% Derivatives of the algebraic equations w.r.t. time (requiered for the
% R-iODE formulation)
ALG_EQS_TIME_DERIVATIVES ={
     'Ks*vmax/(cs+Ks)^2*dcsdt'}; 

CONSTANTS={ 'Ks', 1;'vmax', 3.8}; 

% Only valid for iODE approach

v_flux_incidence={'cx',5; %derivative of Diffs_EQS wrt v_flux
                    0,0;
                    0,0;}
        

      
%Differential Model Initial Conditions and Time Span

Struct.INITIAL=[1,20]; % set initial conditions, follows the order in DIFF_VARS
Struct.TSPAN=[0,0.9];      % Set the integration time span.



% Define Matrices and vectors of the LP problem

A=Test_Data.A;

if rank(A)<size(A) % if the stoichiometric matrix is not full-rank, reduce it.
    
RSorig=size(A);    

[A_red,idx]=licols(A',1e-10);
A=A_red';
RSred=rank(A);
fprintf('The original matrix had %d rows and was reduced to %d\n',RSorig(1),RSred); 
else
    
end
[nmet, nflux]=size(A);    

b=zeros(nmet,1);
b(nmet)=3.8*20/((1)+20); %Given value of S uptake in mmmol/gDW h
     
% Define bounds
LB=Test_Data.lb;
UB=Test_Data.ub;


LB(LB == -inf) = -1000; % Added by FSCOTT on 6/6/18
UB(UB == inf) = 1000;   % Added by FSCOTT on 6/6/18

% Find elements of UB and LB that are equal and transform to equalities


equalities_pos=find(LB==UB);

fprintf('The following flux had equal LB and UB and was changed to an equality constraint:%d\n ',equalities_pos)

Equalities=zeros(length(equalities_pos),nflux);

b=[b;UB(equalities_pos)];

for i=1:length(equalities_pos)
Equalities(i,equalities_pos(i))=1;
UB(equalities_pos(i))=UB(equalities_pos(i))+10;
LB(equalities_pos(i))=LB(equalities_pos(i))-10;
end

A=[A;Equalities];

c=Test_Data.c';

%Position of the uptake fluxes in X vector 
UPTK_FLUX=[7];

% Positions of the equations connecting the differential and LP model in b.
given=[nmet-length(UPTK_FLUX)+1:nmet]; %Changed due to row reduction



%% Create Matlab Structure
Struct.A=A;
Struct.b=b;
Struct.c=c;
Struct.ub=UB;
Struct.lb=LB;

Struct.VARS.DIFF=DIFF_VARS;
Struct.VARS.ALG=ALG_VARS;
Struct.VARS.iODE_VARS=iODE_VARS;
Struct.EQS.DIFF=DIFF_EQS;
Struct.EQS.ALG=ALG_EQS;
Struct.EQS.ALG_TIME_DERIVATIVE=ALG_EQS_TIME_DERIVATIVES;
Struct.CONSTANTS=CONSTANTS;
Struct.GIVEN=given;
Struct.UPTAKE=UPTK_FLUX;
Struct.v_flux_incidence=v_flux_incidence;





%% Show some useful info. 

fprintf('The condition of the A*A'' matrix is:%4.2d\n', cond(A*A'))
fprintf('There are %i metabolites and %i fluxes\n',size(A))
fprintf('*********************************************************************\n')


 

save(strcat(model_name,'.mat'), 'Struct')

end
