function DAE_MODEL_CREATOR3(nDif_var,nConstants,nAlg_var,given,Struct)

EQSFile='Fparametric.m';

[nmet, nflux]=size(Struct.A);
 fid = fopen(EQSFile,'wt');
 
 str1 = ['function Output = Fparametric(t,Y,A,LB,UB,c,b,subset,given,nmet,nflux) \n'];
 str_break=['%%************************************************************\n\n'];
 fprintf(fid,str1);
 
 
 %Write model differential vars

 fprintf(fid,str_break); 
 
 for i=1:nDif_var
 fprintf(fid,'%s = Y(%d);\n',Struct.VARS.DIFF{i},i);  
 end
 
 
fprintf(fid,str_break); 


%Write model ALgebraic vars
 for i=1:nAlg_var
 fprintf(fid,'%s = Y(%d);\n',Struct.VARS.ALG{i},i+nDif_var);  
 end

 % Write Model Constants
 for i=1:nConstants   
 fprintf(fid,'%s = %d;\n',Struct.CONSTANTS{i,1},Struct.CONSTANTS{i,2});  
 end
fprintf(fid,str_break);  
 % Define Uptake vector:
 % Groupping variable structure
% 1:nDif_var                                        Differential Variables
% nDif_var+1: nDif_var+ nAlg_var                    Algebraic Variables
% nDif_var+ nAlg_var+1 :nDif_var+ nAlg_var+nflux    Fluxes
% nDif_var+ nAlg_var+nflux +1: nDif_var+ nAlg_var+2*nflux     Y UB
% nDif_var+ nAlg_var+2*nflux +1: nDif_var+ nAlg_var+3*nflux   Z UB
%  nDif_var+ nAlg_var+3*nflux +1: nDif_var+ nAlg_var+3*nflux+nmet
 
%Write X , YUB,Z and L 
fprintf(fid,'v_flux   =  Y(%d:%d);\n',nDif_var+ nAlg_var+1,nDif_var+ nAlg_var+nflux );
fprintf(fid,'YUB =  Y(%d:%d);\n',nDif_var+ nAlg_var+nflux +1,nDif_var+ nAlg_var+2*nflux  ); 
fprintf(fid,'Z =  Y(%d:%d);\n',nDif_var+ nAlg_var+2*nflux +1,nDif_var+ nAlg_var+3*nflux );
fprintf(fid,'L =  Y(%d:%d);\n',nDif_var+ nAlg_var+3*nflux +1,nDif_var+ nAlg_var+3*nflux+nmet); 
fprintf(fid,'muR =  Y(end);\n');

fprintf(fid,str_break); 


 % Write Model Differential and Pure Algebraic Equations
 
 for i=1:length(Struct.EQS.DIFF)
 fprintf(fid,'DEQS(%d) = %s;\n',i,Struct.EQS.DIFF{i});  
 end
fprintf(fid,str_break);  
 for i=1:length(Struct.EQS.ALG(:,2))
 fprintf(fid,' AEQS(%d)= %s-%s;\n',i,Struct.EQS.ALG{i,1},Struct.EQS.ALG{i,2});  
 end
 fprintf(fid,str_break); 
 fprintf(fid,'AUB= Z.*(UB-v_flux)-muR;\n');
 fprintf(fid,'ALN= YUB.*(v_flux-LB)-muR;\n'); 
 fprintf(fid,'dJdv=-c''-A''*L+YUB-Z;\n'); 
 fprintf(fid,'Bal1= A(subset,:)*v_flux-b(subset);\n'); 
 for i=1:length(given)
 fprintf(fid,'Bal2(%d)= A(%d,:)*v_flux-%s;\n',i,given(i),Struct.VARS.ALG{i}); 
 end
  fprintf(fid,'mur_Def= muR-0.2*sum(Z.*(UB-v_flux)+YUB.*(v_flux-LB)) ;\n'); 
  %
  %muR-((c*v_flux+L.''*b))
 fprintf(fid,str_break); 
 fprintf(fid,'Output=[DEQS.'';AEQS.'';AUB;ALN;dJdv;Bal1;Bal2.'';mur_Def];\n');
   