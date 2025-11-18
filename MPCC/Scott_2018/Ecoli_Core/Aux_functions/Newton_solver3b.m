% This function is an implementation of an interior point method using a
% reduced matrix (derived by hand from the original system of KKT
% conditions)

function [NewX, NewL, NewY, NewZ,gapDual,infnormdJdv,infnormUB,infnormLB,infnormBal]=Newton_solver3b(c,A,b,UB,LB,nmet,nflux,XLP,mur)


epsilon=1;
tol_Dual=2*nflux*mur;
for j=1:nflux;
if(UB(j)-XLP(j))< epsilon;

Xo(j)= XLP(j)-epsilon;

elseif  (XLP(j)-LB(j))< epsilon,
Xo(j)= XLP(j)+epsilon;
else
Xo(j)=XLP(j);
end
end
Xo=Xo';
Zo=0.1*ones(1,nflux)';
%Xo=ones(1,nflux)';
Yo=1*(1*max(LB)+1)*ones(1,nflux)';
Lo=10*ones(1,nmet)';



Zeros3=zeros(nmet,nmet);

iter=0;
iter_set=100;

Flag_satisfaction=0;

% Initiate Loop for Newton CNS
while (iter<iter_set && Flag_satisfaction==0);

    iter = iter+1;
    Flag=0;
    FlagX=0;
    red=1;
    redX=1;

    DiagZ=diag(Zo);
    DiagY=diag(Yo);

    

    murUB =max(mur,0.0002*sum(Zo.*(UB-Xo))/nflux);
    murLB =max(mur,0.0002*sum(Yo.*(Xo-LB))/nflux);

    r_bal=b-A*Xo;
    r_UB= -1*(-murUB+Zo.*(UB-Xo));
    r_LB=-1*(-murLB+Yo.*(Xo-LB));
    r_dJdv=-1*(-c'-(Lo'*A)'+Yo-Zo);   

       
    invDiagUB=diag((UB- Xo).^-1);
    invDiagLB=diag((Xo-LB).^-1);
    %alpha=(invDiagLB*DiagY+invDiagUB*DiagZ);
    alpha= diag((Xo-LB).^-1.*Yo+(UB- Xo).^-1.*Zo);
    MiniANewton=[A Zeros3;
                  -alpha -A'];
%       
              
   MinibNewton= [r_bal; r_dJdv-invDiagLB*r_LB+invDiagUB*r_UB];
  



    [L,U,p] = lu(MiniANewton,'vector');
   opts.UT = false;
   opts.LT = true;
 [Y,R]=linsolve(L,MinibNewton(p,:),opts);
  opts.UT = true;
 opts.LT = false;
 [XL_N,R]=linsolve(U,Y,opts);

    %k=length(MiniANewton);
% [XL_N] = cplexlp(zeros(k),[],[],MiniANewton,MinibNewton);
    

  
  
    DX = XL_N(1:nflux);
    DL = XL_N(1+nflux:nflux+nmet);
    DZ =invDiagUB*(r_UB+DiagZ*DX);
    DY = invDiagLB*(r_LB-DiagY*DX);
    

    
  % Calculate step sizes to avoid violation of bounds  
        while (Flag==0 ||  FlagX==0) &&  red>1E-8 && redX>1E-8;
        %Calculate new solution
        NewX=redX*DX+Xo;
        NewY=red*DY+Yo;
        NewZ=red*DZ+Zo;
        NewL=red*DL+Lo;

  %adjust the step size if negative or outher bounds values are produced*
        checkY= min(NewY);
        checkZ= min(NewZ);
        checkUB=min(UB-NewX);
        checkLB=min(NewX-LB);

              if(checkUB<mur/10|| checkLB<mur/10);
                redX=redX/1.2;
                FlagX=0;
              else
                FlagX=1;
              end

              if (checkY<0 || checkZ<0);
                red=red/1.2;
                Flag=0;
              else
                Flag=1;
              end %End If
            
        end % End while
  %Update initial point
  Xo=NewX;
  Lo=NewL;
  Yo=NewY;
  Zo=NewZ;
  
  
%gapDual(iter)=(c*Xo+Lo'*b);
gapDual=NewY'*(NewX-LB) + NewZ'*(UB-NewX);
fprintf('In iteration %d, the Duality Gap is %d:\n',iter, gapDual)
%infnormdJdv(iter) = max(r_dJdv);
%infnormUB(iter) =  max(r_UB);
%infnormLB(iter) =max(r_LB);
%infnormBal(iter) = max(r_bal);

infnormdJdv = max(r_dJdv);
infnormUB =  max(r_UB);
infnormLB =max(r_LB);
infnormBal = max(r_bal);

% Termination criteria
% Criteria 1: Dual GAP: (abs(gapDual)<tol_Dual)
        if ((abs(gapDual)<tol_Dual));
            Flag_satisfaction=1;
        end
% Criteria 2: muR value (max(abs(murUB),abs(murLB))<=mur)%
%         if (max(abs(murUB),abs(murLB))<=mur);
%             Flag_satisfaction=1;
%         end

 end %End While Newton Loop
%  
 fprintf('***********************************************************************\n')
 fprintf('Max. Ax-b violation: %d\n',infnormBal );
 fprintf('Max. UB violation: %d\n',infnormUB );
 fprintf('Max. LB violation: %d\n',infnormLB );
 fprintf('Max. dJdv violation: %d\n',infnormdJdv );
 fprintf('Dual Gap is %d at iteration %d\n',gapDual, iter );


