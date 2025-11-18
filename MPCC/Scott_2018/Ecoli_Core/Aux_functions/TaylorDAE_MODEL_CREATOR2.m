function TaylorDAE_MODEL_CREATOR2(nDif_var,nConstants,nAlg_var,given,Struct,nflux,nmet,mur)

EQSFile='Equations_Model_Index1_DAE.m';


 fid = fopen(EQSFile,'wt');
 
 str1 = ['function [dy] = Equations_Model_Index1_DAE(t,Y,c,A,b,UB,LB,nmet,nflux,PrecalcM) \n'];
 str2 = ['dy = zeros(length(Y),1); \n']; 
 str_break=['%%************************************************************\n\n'];
 
 fprintf(fid,str1);
 fprintf(fid,str2,nDif_var);
 
 
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
 
 %Write X , YUB,Z and L 
fprintf(fid,'v_flux   =  Y(%d:%d);\n',nDif_var+ nAlg_var+1,nDif_var+ nAlg_var+nflux );
%fprintf(fid,'YUB =  Y(%d:%d);\n',nDif_var+ nAlg_var+nflux +1,nDif_var+ nAlg_var+2*nflux  ); 
%fprintf(fid,'Z =  Y(%d:%d);\n',nDif_var+ nAlg_var+2*nflux +1,nDif_var+ nAlg_var+3*nflux );
fprintf(fid,'L =  Y(%d:%d);\n',nDif_var+ nAlg_var+nflux +1,nDif_var+ nAlg_var+nflux+nmet); 

fprintf(fid,str_break); 
fprintf(fid,'muR=%d;\n',mur); 
   
%  % Write Model Pure Algebraic Equations
%  
%  for i=1:length(Struct.EQS.ALG(:,2))
%  fprintf(fid,'%s = %s;\n',Struct.EQS.ALG{i,1},Struct.EQS.ALG{i,2});  
%  end
 
   %Write model algebraic connections
  
  for i=1:nAlg_var
  fprintf(fid,'b(%d)= %s;\n',given(i),Struct.VARS.ALG{i});  
  end
 
%   %Write zeros matrices
%  fprintf(fid,'Zeros3=sparse(nmet,nmet);\n'); 
%  
 
 fprintf(fid,'%%********* Differential EQS***************\n'); 

 for i=1:length(Struct.EQS.DIFF)
 fprintf(fid,'Dif(%d) = %s;\n',i,Struct.EQS.DIFF{i});  
 end
 
 for i=1:length(Struct.EQS.DIFF)
 fprintf(fid,'%s = Dif(%d);\n',Struct.VARS.iODE_VARS{i},i);  
 end
 
  fprintf(fid,'%%********* Time derivatives of algeb. EQS***************\n');
  
  for i=1:length(Struct.EQS.ALG_TIME_DERIVATIVE)
 fprintf(fid,'DAlg(%d) = %s;\n',i,Struct.EQS.ALG_TIME_DERIVATIVE{i});  
  end    
  fprintf(fid,'%%********* Bounds EQS***************\n');
  
   fprintf(fid,'alphaZ=muR./(UB-v_flux).^2;\n'); 
   fprintf(fid,'alphaY=muR./(v_flux-LB).^2;\n'); 
   fprintf(fid,'aux=-spdiags(alphaZ+alphaY,-nmet,nflux+nmet,nflux+nmet);\n'); 
 
   fprintf(fid,'RHS=aux+PrecalcM;\n');   
   
 fprintf(fid,'%%********* Link of differential and alg. eqs through b***************\n');
 
 fprintf(fid,'dbdt=zeros(1,nmet);\n');
 aux_counter=0;
for i=nDif_var+1:length(Struct.VARS.iODE_VARS)
    aux_counter=aux_counter+1;
     fprintf(fid,'dbdt(%d)=DAlg(%d);\n',Struct.GIVEN,aux_counter);  
end

 fprintf(fid,' LHS=sparse([dbdt.'';zeros(nflux,1)]);\n'); 

fprintf(fid,'%%********* Solve the linear EQS in dvdt and dLdt***************\n');
 
 %fprintf(fid,'[Sol]=linsolve(RHS,LHS);\n');
 fprintf(fid,'[Sol]=RHS\\LHS;\n');
 %fprintf(fid,'[Sol] = cplexlp(zeros(nmet+nflux),[],[],RHS,LHS,[],[]);\n');
 
 fprintf(fid,'dvdt=Sol(1:nflux);\n');
 fprintf(fid,'dLdt=Sol(nflux+1:end);\n');
%  fprintf(fid,'dZdt=alphaZ.*dvdt;\n');
%  fprintf(fid,' dYdt=-alphaY.*dvdt;\n');
 
  fprintf(fid,' dy=[Dif'';DAlg'';dvdt;dLdt];\n');   
 %fprintf(fid,' dy=[Dif'';DAlg'';dvdt;dYdt;dZdt;dLdt];\n');   
 %fprintf(fid,' t\n'); 

