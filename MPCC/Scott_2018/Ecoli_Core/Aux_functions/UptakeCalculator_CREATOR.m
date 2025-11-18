function UptakeCalculator_CREATOR(nDif_var,nConstants,Struct)

EQSFile='UptakeCalculator.m';


 fid = fopen(EQSFile,'wt');
 
 str1 = ['function Output = UptakeCalculator(initial) \n'];
 str_break=['%%************************************************************\n\n'];
 fprintf(fid,str1);
 
 
 %Write model differential vars

 fprintf(fid,str_break); 
 
 for i=1:nDif_var
 fprintf(fid,'%s = initial(%d);\n',Struct.VARS.DIFF{i},i);  
 end
 
 
fprintf(fid,str_break); 




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


 % Write Model Differential and Pure Algebraic Equations
 

 for i=1:length(Struct.EQS.ALG(:,2))
 fprintf(fid,' Output(%d)= %s;\n',i,Struct.EQS.ALG{i,2});  
 end


   