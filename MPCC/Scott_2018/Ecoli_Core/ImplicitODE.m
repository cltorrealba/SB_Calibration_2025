function [dy] = ImplicitODE(t,Y,Yp,A,b,UB,LB,nmet)
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
%********* Differential EQS***************
Dif(1) = dcxdt-v_flux(13) * cx;
Dif(2) = dcsdt-180/1000*vupt*cx;
Dif(3) = dcpdt-46/1000*v_flux(24)*cx;
%********* Time derivatives of algeb. EQS***************
DAlg(1) = dvuptdt--50/9*Ks*vmax/(cs+Ks)^2*dcsdt;
%********* Bounds EQS***************
alphaZ=muR./(UB-v_flux).^2;
alphaY=muR./(v_flux-LB).^2;
%********* Link of differential and alg. eqs through b***************
dbdt=zeros(1,nmet);
 dqsdt(1)= dvuptdt;
dbdt(68)=dqsdt;
 Massbal=A*dvdt-dbdt.';
 Dualbal=-A.'*dLdt+(alphaY+alphaZ).*dvdt;
  dy=sparse([Dif.';DAlg.';Massbal;Dualbal]);
