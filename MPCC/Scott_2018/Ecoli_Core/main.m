 function [t,y]=main

%Depending on solvers selection you might require:

%1. OPTI Toolbox
%2. CPLEX


clc
close all

%% Create & Load Problem structure
 
% Create & save problem structure

Prob_Name='EcoliCoreGlucModel';

E_coli_core_Prob_definition_Glucose(Prob_Name); %call the script where the problem is defined. 
Name_Mat_File=strcat(Prob_Name,'.mat');
load(Name_Mat_File)

%% Use the Method parameter to select among solution methods
 
% Option 1, Implementation of the Direct Approach:
% Method= 'ODEs' ;

% LPMethod = 'CPLEX'; OPTI, linprog, CPLEX

 
% Option 2, DAE (does not require the specification of other fields) [DO NOT USE]
% Method= 'DAE_Solver' ;
 
% Option 3, Reduced Implicit ODE 
% Method= 'DAE_taylored' ;   

% Option 4, Fully Implicit ODE 
% Method= 'implicit_ODE' ;
% Mass_Method='NL'; % 'LIN', 'NL'


% Main routine ODEs/DAE_taylored/DAE/iODE 
%Using the DIRECT APPROACH
%         opts.Method   = 'ODEs';
%         opts.LPMethod ='CPLEX'; %Specify the solver for LP solution during integration 'CPLEX'; OPTI, linprog, CPLEX
%         opts.RigourousTimeIt='No';
%Using the Explicit reduced ODEs method (Eq.18 in Paper after solving the system of Eqs. )
%         opts.Method   = 'DAE_taylored';
%         opts.initializion ='IP'; %IP or CPLEX with solution adjustments to make it interior
%         opts.mu =1e-6; %The penalty parameter, only used in the Implicit Reduced ODE method. 
%         opts.epsilon =0.0001; %the amount that will be substracted or added to the LP solution to make it interior if its in a bound. Only used when  opts.initializion ='CPLEX'; 
%         opts.RigourousTimeIt='No';
        
%Using the fully Implicit reduced ODEs method (Eq.18 in Paper)
        opts.Method   = 'iODE';
        opts.initializion ='IP'; %IP or CPLEX with solution adjustments to make it interior
        opts.mu =1e-6; %The penalty parameter, only used in the Implicit Reduced ODE method. 
        opts.epsilon =0.0001; %the amount that will be substracted or added to the LP solution to make it interior if its in a bound. Only used when  opts.initializion ='CPLEX'; 
        opts.RigourousTimeIt='No';
        

switch opts.Method
    case 'ODEs'
        Method_disp='Direct Approach';
    case 'DAE_taylored'
        Method_disp='Reduced implicit ODEs with forward integration';
    case 'iODE'
        Method_disp='Fully implicit ODEs';
end
fprintf('*******************************************************************\n')       
fprintf('Attemping to solve the dFBA problem using %s \n',Method_disp)
fprintf('Integration statistics:\n')
%% Load data (A, b, c, UB, LB)

A=Struct.A;
b=Struct.b;
c=Struct.c;
UB=Struct.ub;
LB=Struct.lb;


nDif_var=length(Struct.VARS.DIFF); %number of differential variables
nAlg_var=length(Struct.VARS.ALG); %number of pure algebraic variables
nConstants=length(Struct.CONSTANTS); %number of constants and parameters


%Define sets

[nmet, nflux]=size(A); 
flux_set=[1:nflux];
given=Struct.GIVEN; %flows connecting diff and LP model
subset=1:nmet;
subset(given)=[];


initial=Struct.INITIAL; %Vars=[X,S,P];
tspan=Struct.TSPAN;

%% Test model
% Gluc=[0.1:1:20];
% mu=[];
% for i=1:length(Gluc)
%     uptake_gluc=-12.9*Gluc(i)/(Gluc(i)+6.95);
% b(given)=uptake_gluc;
% [v_flux]= cplexlp(c,[],[],A,b,LB,UB,[]);
% mu(i)=v_flux(2985);
% end
% plot(Gluc,mu')



%% Initialization 
if strcmp(opts.Method,'ODEs')
Struct=InitializeIP(opts,Struct); 
else
Struct=InitializeIP(opts,Struct);     
end

% Load initial point.
Xo=Struct.Initial_LP.X;
Lo=Struct.Initial_LP.L;
Zo=Struct.Initial_LP.Z;
Yo=Struct.Initial_LP.Y;

             
switch opts.Method
    case 'ODEs' 
        %Create problem structure to integrate   
        DIFF_MODEL_CREATOR(nDif_var,nConstants,nAlg_var,given,Struct,opts.LPMethod); %Create the .m file containing differential eqs.
        fprintf('Integration begins using the DIRECT APPROACH');
        tic
        options = odeset('AbsTol', 1e-6, 'RelTol', 1e-6,'Stats', 'on');
        [t,y] = ode15s(@(t,y) Equations_Model(t,y,c,A,b,UB,LB,nmet,nflux,Xo) ,tspan,initial,options);
        toc;
        if strcmp(opts.RigourousTimeIt,'Yes')
        options2 = odeset('AbsTol', 1e-6, 'RelTol', 1e-3,'Stats', 'off');
        odes=@()ode45(@(t,y) Equations_Model(t,y,c,A,b,UB,LB,nmet,nflux,Xo) ,tspan,initial,options2);
        avtime=timeit(odes);
        init = cputime;
        [t,y] = ode45(@(t,y) Equations_Model(t,y,c,A,b,UB,LB,nmet,nflux,Xo) ,tspan,initial,options);
        e = cputime-init; 
        fprintf('Time measured using timeit:%f s\n', avtime)
        fprintf('Time measured CPU secs:%f s\n', e)
        end
        
     case 'DAE_taylored'
        mur=opts.mu ;
        Diff_vars=nDif_var;
        TaylorDAE_MODEL_CREATOR2(nDif_var,nConstants,nAlg_var,given,Struct,nflux,nmet,mur)
        %DAE_MODEL_CREATOR3(nDif_var,nConstants,nAlg_var,given,Struct); %Create .m file with DAE equations      
        UPTK_Ini=Xo(Struct.UPTAKE);
        %y0est=[initial';UPTK_Ini;Xo;Yo;Zo;Lo];
        y0est=[initial';UPTK_Ini;Xo;Lo];
        
           
         % Precalculate a part of the matrix involved in the implicit ODEs
         SparseA=sparse(A);
         Zeros3=sparse(nmet,nmet);
         Zeros2=sparse(nflux,nflux);
         PrecalcM= [SparseA Zeros3;Zeros2  -SparseA'];
         
        options = odeset('AbsTol', 1e-6,'RelTol', 1e-3,'Stats','on');
        tic
          [t,y] = ode45(@(t,y) Equations_Model_Index1_DAE(t,y,c,A,b,UB,LB,nmet,nflux,PrecalcM) ,tspan,y0est,options);
       fprintf('Wall-time in integration only: %4.4f sec\n',toc)
        
        if strcmp(opts.RigourousTimeIt,'Yes')
        options2 = odeset('AbsTol', 1e-3,'RelTol', 1e-2,'Stats','off');
        odes=@()ode23(@(t,y) Equations_Model_Index1_DAE(t,y,c,A,b,UB,LB,nmet,nflux,PrecalcM) ,tspan,y0est,options2);
        avtime=timeit(odes);
        fprintf('Time measured using timeit:%f s\n', avtime)
        init = cputime;
        [t,y] = ode45(@(t,y) Equations_Model_Index1_DAE(t,y,c,A,b,UB,LB,nmet,nflux,PrecalcM) ,tspan,y0est,options2);
        e = cputime-init; 
      
        fprintf('Time measured CPU secs:%f s\n', e)
        end
        
            
     case 'iODE'
         mur=opts.mu ;
         UPTK_Ini=Xo(Struct.UPTAKE);
         Yo=[initial';UPTK_Ini;Xo;Lo];
         % Solve the explicit system of ODE to obtain a consistent Yp
         SparseA=sparse(A);
         Zeros3=sparse(nmet,nmet);
         Zeros2=sparse(nflux,nflux);
         PrecalcM= [SparseA Zeros3;Zeros2  -SparseA'];
        TaylorDAE_MODEL_CREATOR2(nDif_var,nConstants,nAlg_var,given,Struct,nflux,nmet,mur)
        [Ypo] = Equations_Model_Index1_DAE(0,Yo,c,A,b,UB,LB,nmet,nflux,PrecalcM); 
         %Create iODE and its Jacobians
         iODE_Model_Creator(Struct,mur)
         Jacobians_iODE_Model_Creator(Struct,mur)
         % TEst Jacobians
            F2 = @(Y)ImplicitODE(0,Y,Ypo,A,b,UB,LB,nmet);
            F2p= @(Yp)ImplicitODE(0,Yo,Yp,A,b,UB,LB,nmet);
            % 
             dFdY = cstepJac(F2,Yo);
            % 
             dFdYp = cstepJac(F2p,Ypo);
             [dFdy_fun_val, dFdyp_val]=ImplicitODE_Jac(0,Yo,Ypo,A,b,UB,LB,Struct);
            % 
             difs=(dFdy_fun_val-dFdY);
             difs2=(dFdyp_val-dFdYp);
           plot(difs)
             
         %solve
         options = odeset('AbsTol', 1e-6, 'RelTol', 1e-3,'Stats', 'on', 'Jacobian',@(t,Y,Yp) ImplicitODE_Jac(t,Y,Yp,A,b,UB,LB,Struct) );

        tic
        [t,y] = ode15i(@(t,Y,Yp) ImplicitODE(t,Y,Yp,A,b,UB,LB,nmet),tspan,Yo,Ypo,options);
        toc
        fprintf('Wall-time in integration only: %4.4f sec\n',toc)

end


plot_vars=[1:nDif_var];
figure; plot(t,y(:,plot_vars));
title(Prob_Name);
xlabel('Time [Hours]');
ylabel('Concentration [g/L]');
legend(Struct.VARS.DIFF);
% 
% 
% v_flux   =  y(:,nDif_var+ nAlg_var+1:nDif_var+ nAlg_var+nflux);
% YUB =  y(:,nDif_var+ nAlg_var+nflux +1:nDif_var+ nAlg_var+2*nflux);
% Z =  y(:,nDif_var+ nAlg_var+2*nflux +1:nDif_var+ nAlg_var+3*nflux);
% L =  y(:,nDif_var+ nAlg_var+3*nflux +1:nDif_var+ nAlg_var+3*nflux+nmet);
% 
% figure; 
% FZ=11;
% subplot(2,2,1)
% semilogy(t,YUB);
% xlabel('Time [Arb. units]','FontWeight','bold');
% ylabel('Y','FontWeight','bold');
% set(gca, 'LineWidth',  1.1);
% 
% subplot(2,2,2)
% semilogy(t,Z);
% hold on
% %semilogy(t,Z(:,64),'LineWidth',1.5)
% xlabel('Time [Arb. units]','FontWeight','bold');
% ylabel('Z','FontWeight','bold');
% set(gca, 'LineWidth',  1.1);
% 
% subplot(2,2,3)
% plot(t,L);
% xlabel('Time [Arb. units]','FontWeight','bold');
% ylabel('\lambda','FontWeight','bold');
% set(gca, 'LineWidth',  1.1);
% 
% subplot(2,2,4)
% figure
% plot(t,v_flux,'color',[0,0,0]+0.8);
% hold on
% plot(t,v_flux(:,34),'LineWidth',1.5)
% plot(t,v_flux(:,64),'LineWidth',1.5)
% plot(t,v_flux(:,26),'LineWidth',1.5)
% xlabel('Time [Arb. units]','FontWeight','bold');
% ylabel('v [mmol/h/gDW]','FontWeight','bold');
% set(gca, 'LineWidth',  1.1);
% 
% set(findall(gcf,'type','axeslables'),'fontsize',FZ+1);
% set(findall(gcf,'type','axes'),'fontsize',FZ);
% set(findall(gcf,'type','text'),'fontSize',11) ;




