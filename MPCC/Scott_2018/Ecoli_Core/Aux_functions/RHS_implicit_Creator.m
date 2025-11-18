function RHS_implicit_Creator(nDif_var,nConstants,nAlg_var,Struct,nflux,nmet,mur)

EQSFile='RHS_Implicit.m';


 fid = fopen(EQSFile,'wt');
 
 str1 = ['function [dy] = RHS_Implicit(t,Y,c,A,b,UB,LB,nmet,nflux) \n'];
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
fprintf(fid,'YUB =  Y(%d:%d);\n',nDif_var+ nAlg_var+nflux +1,nDif_var+ nAlg_var+2*nflux  ); 
fprintf(fid,'Z =  Y(%d:%d);\n',nDif_var+ nAlg_var+2*nflux +1,nDif_var+ nAlg_var+3*nflux );
fprintf(fid,'L =  Y(%d:%d);\n',nDif_var+ nAlg_var+3*nflux +1,nDif_var+ nAlg_var+3*nflux+nmet); 

fprintf(fid,str_break); 
fprintf(fid,'muR=%d;\n',mur); 
   
 
  %Write zeros matrices
 fprintf(fid,'Zeros2=sparse(nflux,1);\n'); 
 
 
 fprintf(fid,'%%********* Differential EQS***************\n'); 

 for i=1:length(Struct.EQS.DIFF)
 fprintf(fid,'Dif(%d) = %s;\n',i,Struct.EQS.DIFF{i});  
 end
 
  fprintf(fid,'%%********* Time derivatives of algeb. EQS***************\n');
  
  for i=1:length(Struct.EQS.ALG_TIME_DERIVATIVE)
 fprintf(fid,'DAlg(%d) = %s;\n',i,Struct.EQS.ALG_TIME_DERIVATIVE{i});  
  end    
 
   
 fprintf(fid,'%%********* Link of differential and alg. eqs through b***************\n');
 
 fprintf(fid,'dbdt=zeros(1,nmet-%d);\n',length(Struct.EQS.ALG_TIME_DERIVATIVE));
 fprintf(fid,'dbdt=[dbdt'';DAlg''];\n');  
 fprintf(fid,'btime=sparse([dbdt]);\n'); 


  fprintf(fid,' dy=[Dif'';DAlg'';btime;Zeros2;Zeros2;Zeros2];\n');   
  
 %fprintf(fid,' t\n');   
  


