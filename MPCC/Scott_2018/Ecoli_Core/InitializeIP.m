function Struct=InitializeIP(opts,Struct)

%options CPLEX, 'IP'. 'CPLEX' uses CPLEX to produce an exact solution. This
%solution is the "interiorized" by adding or substracting and epsilon to
%the solution so its within bounds.

%On the other hand 'IP' uses an in-house implementation of an interior point solution of LP, producing an exact duality gap

% Init_mode ['IP', 'CPLEX']
%% Calculate a valid (interior) initial LP solution.
A=Struct.A;
b=Struct.b;
c=Struct.c;
UB=Struct.ub;
LB=Struct.lb;

[nmet, nflux]=size(A);

if strcmp(opts.Method, 'ODEs')
      fprintf('CPLEX is being used to calculte the initial conditions for the dFBA problem (DIRECT APPROACH)\n') 
     [v_flux_Cplex,fval,exitflag,output,lambda]= cplexlp(c,[],[],A,b,LB,UB);
     gap_dualLP=( -c*v_flux_Cplex-(lambda.eqlin'* b+lambda.lower'*LB-lambda.upper'*UB));
       
      X=sparse(v_flux_Cplex);
      L=sparse(lambda.eqlin);
      Y=sparse(lambda.lower);
      Z=sparse(lambda.upper);
      fprintf('The duality gap at the initial point is %d\n',gap_dualLP)
else
    Initialization=opts.initializion;
    mur=opts.mu;
fprintf('_____________________________________________________________________\n')
switch Initialization
    case 'IP' 
        
      fprintf('An in-house implementation of an interior-point LP solver is being\n used to calculate the initial conditions for the dFBA problem.\n') 
      [Xo,fval,exitflag,output,lambda]= cplexlp(c,[],[],A,b,LB,UB,[]);
      nDif_var=length(Struct.VARS.DIFF);
      nConstants=length(Struct.CONSTANTS);
      UptakeCalculator_CREATOR(nDif_var,nConstants,Struct)
      UPTK_Ini = UptakeCalculator(Struct.INITIAL);
      b(Struct.GIVEN)=UPTK_Ini;
      [NewX, NewL, NewY, NewZ,gapDual,infnormdJdv,infnormUB,infnormLB,infnormBal]=Newton_solver3b(c,A,b,UB,LB,nmet,nflux,Xo,10*mur);
      X=sparse(NewX);
      L=sparse(NewL);
      Y=sparse(NewY);
      Z=sparse(NewZ);
      
          case 'CPLEX' % Added by FSCOTT on 6/6/18
      fprintf('CPLEX is being used to calculte the initial conditions for the dFBA problem\n') 
     [Xo,fval,exitflag,output,lambda]= cplexlp(c,[],[],A,b,LB,UB);
       
     
        eps_int=opts.epsilon;
         for j=1:nflux;
        if(UB(j)-Xo(j))< eps_int;

        Xo2(j)= Xo(j)-eps_int;

        elseif  (Xo(j)-LB(j))< eps_int,
        Xo2(j)= Xo(j)+eps_int;
        else
        Xo2(j)=Xo(j);
        end
        end

        Xo2=Xo2';
      
        %[NewX, NewL, NewY, NewZ,gapDual,infnormdJdv,infnormUB,infnormLB,infnormBal]=Newton_solver3(c,A,b,UB,LB,nmet,nflux,Xo,1e-6);
        muR=mur;
        NewZ=muR./(UB-Xo2);
        NewY=muR./(Xo2-LB);
        NewL=sparse(lambda.eqlin);
        
    
     
       %checks
      X=sparse(Xo2);
      L=sparse(NewL);
      Y=sparse(abs(NewY));
      Z=sparse(NewZ);
      gap_dualLP=( -c*X-(L'* b+Y'*LB-Z'*UB));
      fprintf('The duality gap at the initial point is %d\n',gap_dualLP)
end

end
% Add the initial point to the structure.
Struct.Initial_LP.X=X;
Struct.Initial_LP.L=L;
Struct.Initial_LP.Y=Y;
Struct.Initial_LP.Z=Z;
 


end