function [dFdy, dFdyp] = ImplicitODE_Jac(t,Y,Yp,A,b,UB,LB,Struct) 
A=Struct.A;
 [nmet, nflux]=size(A); ;
nDif_var=length(Struct.VARS.DIFF);
nConstants=length(Struct.CONSTANTS);
nAlg_var=length(Struct.VARS.ALG);
nDif_EQS=length(Struct.EQS.DIFF);
nDif_ALG=nAlg_var;
dFdy=sparse(nDif_var+nAlg_var+nflux+nmet,nDif_var+nAlg_var+nflux+nmet);
dFdyp=sparse(nDif_var+nAlg_var+nflux+nmet,nDif_var+nAlg_var+nflux+nmet);
%************************************************************

cx = Y(1);
cs = Y(2);
cp = Y(3);
dcxdt = Yp(1);
dcsdt = Yp(2);
dcpdt = Yp(3);
dvuptdt = Yp(4);
vupt = Y(4);
%************************************************************

Ks = 1;
vmax = 1.800000e+00;
v_flux   =  Y(5:99);
L =  Y(100:167);
dvdt   =  Yp(5:99);
dLdt   =  Yp(100:167);
%************************************************************

muR=1.000000e-06;
%********* Bounds EQS***************
alphaZ=muR./(UB-v_flux).^2;
alphaY=muR./(v_flux-LB).^2;
%********* Start with  dFdy, Jacobian of equations wrt differential vars.***************
Dalpha=2*muR./(UB-v_flux).^3+2*muR./(LB-v_flux).^3;
dFdy(1,1) = -v_flux(13);
dFdy(2,1) = -(9*vupt)/50;
dFdy(2,4) = -(9*cx)/50;
dFdy(3,1) = -(23*v_flux(24))/500;
dFdy(4,2) = -(100*Ks*dcsdt*vmax)/(9*(Ks + cs)^3);
dFdy(1,17) = -cx;
dFdy(3,28) = -46/1000*cx;
Range_Rows=nDif_ALG+nDif_EQS+nmet+1:nDif_ALG+nDif_EQS+nmet+nflux;
Range_Col=nDif_var+nAlg_var+1:nDif_var+nAlg_var+nflux;
dFdy(Range_Rows,Range_Col)=diag(Dalpha.*dvdt);
%********* Start with  dFdyP, Jacobian of equations wrt implicit difs vars.***************
dFdyp(1,1) = 1;
dFdyp(2,2) = 1;
dFdyp(3,3) = 1;
dFdyp(4,2) = (50*Ks*vmax)/(9*(Ks + cs)^2);
dFdyp(4,4) = 1;
%********* Mass bal. uptake wrt dvupt_dt ***************
dFdyp(72,4) = -1;
%*********  A ***************
Range_Rows=nDif_ALG+nDif_EQS+1:nDif_ALG+nDif_EQS+nmet;
Range_Col=nDif_var+nAlg_var+1:nDif_var+nAlg_var+nflux;
dFdyp(Range_Rows,Range_Col)=sparse(A);
%********* Diagonal matrix (AlphaY+AlphaZ)***************
Range_Rows=nDif_ALG+nDif_EQS+nmet+1:nDif_ALG+nDif_EQS+nmet+nflux;
Range_Col=nDif_var+nAlg_var+1:nDif_var+nAlg_var+nflux;
dFdyp(Range_Rows,Range_Col)=diag(alphaZ+alphaY);
%********* Lower -A'***************
Range_Rows=nDif_ALG+nDif_EQS+nmet+1:nDif_ALG+nDif_EQS+nmet+nflux;
Range_Col=nDif_var+nAlg_var+nflux+1:nDif_var+nAlg_var+nflux+nmet;
dFdyp(Range_Rows,Range_Col)=sparse(-A.');
