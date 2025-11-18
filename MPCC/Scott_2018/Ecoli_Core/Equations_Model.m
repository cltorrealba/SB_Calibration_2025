function [dy] = Equations_Model(t,y,c,A,b,UB,LB,nmet,nflux,Xo) 
dy = zeros(3,1); 
%************************************************************

cx = y(1);
cs = y(2);
cp = y(3);
%************************************************************

Ks = 1;
vmax = 1.800000e+00;
vupt = -1000/180*vmax*cs/(Ks+cs);
b(68)= vupt;
%********* OPTIONS FOR SOLVING THE LP PROBLEM ***************
[v_flux] = cplexlp(c,[],[],A,b,LB,UB);
	 ;
%********* Differential EQS***************
dy(1) = v_flux(13) * cx;
dy(2) = 180/1000*vupt*cx;
dy(3) = 46/1000*v_flux(24)*cx;
