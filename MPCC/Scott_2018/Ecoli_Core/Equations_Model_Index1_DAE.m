function [dy] = Equations_Model_Index1_DAE(t,Y,c,A,b,UB,LB,nmet,nflux,PrecalcM) 
dy = zeros(length(Y),1); 
%************************************************************

cx = Y(1);
cs = Y(2);
cp = Y(3);
%************************************************************

vupt = Y(4);
Ks = 1;
vmax = 1.800000e+00;
v_flux   =  Y(5:99);
L =  Y(100:167);
%************************************************************

muR=1.000000e-06;
b(68)= vupt;
%********* Differential EQS***************
Dif(1) = v_flux(13) * cx;
Dif(2) = 180/1000*vupt*cx;
Dif(3) = 46/1000*v_flux(24)*cx;
dcxdt = Dif(1);
dcsdt = Dif(2);
dcpdt = Dif(3);
%********* Time derivatives of algeb. EQS***************
DAlg(1) = -50/9*Ks*vmax/(cs+Ks)^2*dcsdt;
%********* Bounds EQS***************
alphaZ=muR./(UB-v_flux).^2;
alphaY=muR./(v_flux-LB).^2;
aux=-spdiags(alphaZ+alphaY,-nmet,nflux+nmet,nflux+nmet);
RHS=aux+PrecalcM;
%********* Link of differential and alg. eqs through b***************
dbdt=zeros(1,nmet);
dbdt(68)=DAlg(1);
 LHS=sparse([dbdt.';zeros(nflux,1)]);
%********* Solve the linear EQS in dvdt and dLdt***************
[Sol]=RHS\LHS;
dvdt=Sol(1:nflux);
dLdt=Sol(nflux+1:end);
 dy=[Dif';DAlg';dvdt;dLdt];
