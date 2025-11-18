function iODE_Model_Creator(Struct,mur)

A=Struct.A;
b=Struct.b;
c=Struct.c;
UB=Struct.ub;
LB=Struct.lb;

[nmet, nflux]=size(A); 
nDif_var=length(Struct.VARS.DIFF);
nConstants=length(Struct.CONSTANTS);
nAlg_var=length(Struct.VARS.ALG);
given=Struct.GIVEN;

EQSFile='ImplicitODE.m';


 fid = fopen(EQSFile,'wt');
 
 str1 = ['function [dy] = ImplicitODE(t,Y,Yp,A,b,UB,LB,nmet)\n'];
 str_break=['%%************************************************************\n\n'];
 
 fprintf(fid,str1);
 
 
 %Write model differential vars

 fprintf(fid,str_break); 
 
 for i=1:nDif_var
 fprintf(fid,'%s = Y(%d);\n',Struct.VARS.DIFF{i},i);  
 end
 
 for i=1:length(Struct.VARS.iODE_VARS)
 fprintf(fid,'%s = Yp(%d);\n',Struct.VARS.iODE_VARS{i},i);  
 end
 

%Write model ALgebraic vars
 for i=1:nAlg_var
 fprintf(fid,'%s = Y(%d);\n',Struct.VARS.ALG{i},i+nDif_var);  
 end

fprintf(fid,str_break); 
  
 % Write Model Constants
 for i=1:nConstants   
 fprintf(fid,'%s = %d;\n',Struct.CONSTANTS{i,1},Struct.CONSTANTS{i,2});  
 end
 
   
 %Write v and L 
fprintf(fid,'v_flux   =  Y(%d:%d);\n',nDif_var+ nAlg_var+1,nDif_var+ nAlg_var+nflux );
fprintf(fid,'L =  Y(%d:%d);\n',nDif_var+ nAlg_var+nflux +1,nDif_var+ nAlg_var+nflux+nmet); 

   
 %Write dvdt and dLdt 
fprintf(fid,'dvdt   =  Yp(%d:%d);\n',nDif_var+ nAlg_var+1,nDif_var+ nAlg_var+nflux );
fprintf(fid,'dLdt   =  Yp(%d:%d);\n',nDif_var+ nAlg_var+nflux +1,nDif_var+ nAlg_var+nflux+nmet);

fprintf(fid,str_break); 
fprintf(fid,'muR=%d;\n',mur); 
   
 
  fprintf(fid,'%%********* Differential EQS***************\n'); 

 for i=1:length(Struct.EQS.DIFF)
 fprintf(fid,'Dif(%d) = %s-%s;\n',i,Struct.VARS.iODE_VARS{i},Struct.EQS.DIFF{i});  
 end
 
   fprintf(fid,'%%********* Time derivatives of algeb. EQS***************\n');
  
  for i=1:length(Struct.EQS.ALG_TIME_DERIVATIVE) %Diff needs to be changed to dsdt.
 fprintf(fid,'DAlg(%d) = %s-%s;\n',i,Struct.VARS.iODE_VARS{i+nDif_var},Struct.EQS.ALG_TIME_DERIVATIVE{i});  
  end    
 
  fprintf(fid,'%%********* Bounds EQS***************\n');
  
   fprintf(fid,'alphaZ=muR./(UB-v_flux).^2;\n'); 
   fprintf(fid,'alphaY=muR./(v_flux-LB).^2;\n'); 
    
    fprintf(fid,'%%********* Link of differential and alg. eqs through b***************\n');
 
 fprintf(fid,'dbdt=zeros(1,nmet);\n');
 aux_counter=0;
for i=nDif_var+1:length(Struct.VARS.iODE_VARS)
    aux_counter=aux_counter+1;
     fprintf(fid,' dqsdt(%d)= %s;\n',aux_counter,Struct.VARS.iODE_VARS{i}); 
end
 fprintf(fid,'dbdt(%d)=dqsdt;\n',Struct.GIVEN);  
 
 fprintf(fid,' Massbal=A*dvdt-dbdt.'';\n');  
 fprintf(fid,' Dualbal=-A.''*dLdt+(alphaY+alphaZ).*dvdt;\n');  
 fprintf(fid,'  dy=sparse([Dif.'';DAlg.'';Massbal;Dualbal]);\n');  

