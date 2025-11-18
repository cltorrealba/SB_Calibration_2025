function E_coli_core_Prob_definition_Glucose(Prob_Name)
%%  This file creates a Matlab structure defining the dFBA problem for a given
% value of the barrier parameter. It can be called within or outside the
% main.m execution routine.

clc

% The following modifications were made:
% 1) Matrix S was augmented with a row (C) of all zeros except for a 1 in
% the position 6 (representing uptake flux).
% 2) The vector b was augmented with the value of the glucose uptake flux.

% The Stoichiometric matrix and bounds were taken from:
% http://bigg.ucsd.edu/models/e_coli_core
% The following modifications were made:
% 1) Matrix S was augmented with a row (C) of all zeros except for a 1 in
% the position 28 (representing uptake flux).
% 2) The vector b was augmented with the value of the glucose uptake flux.

model_name=Prob_Name;
fprintf ('Creating dFBA model structure for problem: %s\n\n', model_name)

load('Ecoli_Core_Gluc.mat')

%Variables

DIFF_VARS={'cx','cs','cp'}; %extracelular product is ethanol.

iODE_VARS={'dcxdt','dcsdt','dcpdt','dvuptdt'};

ALG_VARS={'vupt'};

%FLUXES vector of v_flux(number of columns in matrix A)



DIFF_EQS={'v_flux(13) * cx';
          '180/1000*vupt*cx';
          '46/1000*v_flux(24)*cx'};

ALG_EQS ={
    'vupt' , '-1000/180*vmax*cs/(Ks+cs)'  %vupt in mmol/gDW/h
    };

ALG_EQS_TIME_DERIVATIVES ={
     '-50/9*Ks*vmax/(cs+Ks)^2*dcsdt'  %Dif(i) corresponds to the LHS of the ith differential equation
    };

CONSTANTS={ 'Ks', 1;
            'vmax', 1.8}; %set up to achieve 10 mmol/gDW h at 20 g/L of initial glucose
        

% Only valid for iODE approach

v_flux_incidence={'cx',13; %derivative of Diffs_EQS wrt v_flux
                    0,0;
                   '46/1000*cx',24;};
      
      %Differential Model Initial Conditions and Time Span

Struct.INITIAL=[1,20,0];
Struct.TSPAN=[0,3];

% Define Matrices and vectors of the LP problem

A=Ecoli_Core_Gluc.A;

if rank(A)<size(A)
    
RSorig=size(A);    

[A_red,idx]=licols(A',1e-10);
A=A_red';
RSred=rank(A);
fprintf('The original matrix had %d rows and was reduced to %d\n',RSorig(1),RSred); 
else
    
end
[nmet, nflux]=size(A);    

b=zeros(nmet,1);
b(nmet)=-1000/180*1.8*20/(1+20); %Given value of S uptake in mmmol/gDW h
     
% Define bounds
LB=Ecoli_Core_Gluc.lb;
UB=Ecoli_Core_Gluc.ub;

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

c=Ecoli_Core_Gluc.c';


% Positions of the equations connecting the differential and LP model in b.
given=[68]; %Changed due to row reduction

%Position of the uptake fluxes in X vector 
UPTK_FLUX=[28];



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

save(strcat(model_name,'.mat'), 'Struct')


%% Show some useful info. 

fprintf('The condition of the A*A'' matrix is:%4.2d\n', cond(A*A'))
fprintf('There are %i metabolites and %i fluxes\n',size(A))
fprintf('*********************************************************************\n')


 

save(strcat(model_name,'.mat'), 'Struct')
