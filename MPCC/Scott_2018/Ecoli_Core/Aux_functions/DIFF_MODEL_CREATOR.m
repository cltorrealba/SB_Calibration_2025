function DIFF_MODEL_CREATOR(nDif_var,nConstants,nAlg_var,given,Struct,LPMethod)

EQSFile='Equations_Model.m';


 fid = fopen(EQSFile,'wt');
 
 str1 = ['function [dy] = Equations_Model(t,y,c,A,b,UB,LB,nmet,nflux,Xo) \n'];
 str2 = ['dy = zeros(%d,1); \n']; 
 str_break=['%%************************************************************\n\n'];
 
 fprintf(fid,str1);
 fprintf(fid,str2,nDif_var);
 
 
 %Write model differential vars

 fprintf(fid,str_break); 
 
 for i=1:nDif_var
 fprintf(fid,'%s = y(%d);\n',Struct.VARS.DIFF{i},i);  
 end
 

fprintf(fid,str_break); 
  
 % Write Model Constants
 for i=1:nConstants   
 fprintf(fid,'%s = %d;\n',Struct.CONSTANTS{i,1},Struct.CONSTANTS{i,2});  
 end
 
   
 % Write Model Pure Algebraic Equations
 
 for i=1:length(Struct.EQS.ALG(:,2))
 fprintf(fid,'%s = %s;\n',Struct.EQS.ALG{i,1},Struct.EQS.ALG{i,2});  
 end
 
   %Write model algebraic connections
  
  for i=1:nAlg_var
  fprintf(fid,'b(%d)= %s;\n',given(i),Struct.VARS.ALG{i});  
  end
 
 
 fprintf(fid,'%%********* OPTIONS FOR SOLVING THE LP PROBLEM ***************\n'); 
 
%  strSolverSelection_1= ['switch LPsolver \n'];
%  strSolverSelection_2= ['case ''DirectLP''\n']; 
 if strcmp(LPMethod, 'OPTI')
 strSolverSelection_3= ['opts = optiset(''solver'',''CLP'',''display'',''off'');\n \t Opt = opti(''f'',c,''ineq'',[],[],''eq'',A,b,''bounds'',LB,UB,''options'',opts);\n'];
 strSolverSelection_4= ['\t [v_flux,fval] = solve(Opt);\n'];
 elseif strcmp(LPMethod, 'CPLEX')
 strSolverSelection_3= ['[v_flux] = cplexlp(c,[],[],A,b,LB,UB);\n'];
 strSolverSelection_4= ['\t ;\n'];   
 else
 strSolverSelection_3= ['options = optimoptions(''linprog'',''Algorithm'',''dual-simplex'',''Display'',''none'');\n'];
 strSolverSelection_4=['\t [v_flux,fval,exitflag, output] = linprog(c,[],[],A,b,LB,UB,[],options);\n'];
 end


%  strSolverSelection_5= ['case ''Newton_CNS''\n'];
%  strSolverSelection_6= ['\t [v_flux, NewL, NewY, NewZ,gapDual,infnormdJdv,infnormUB,infnormLB,infnormBal]=Newton_solver3(c,A,b,UB,LB,nmet,nflux,Xo,1E-7);\n ']; 
%  strSolverSelection_7= ['case ''OPTI_CNS''\n'];
%  strSolverSelection_8= ['\t Opt.nlprob.options.rl(%d)=b(%d);\n'];  %rl=ru=beq=[c';b], position corresponds to length(c)+given(i), for example 7+6=13.;
%  strSolverSelection_9= ['\t Opt.nlprob.options.ru(%d)=b(%d);\n'];  
%  strSolverSelection_10= ['\t [X_N ,fval,exitflag,info] = solve(Opt);\n'];
%  strSolverSelection_11= ['\t v_flux = X_N(1:nflux);\n']; 
%  strSolverSelection_12= ['end\n']; 
%     
%  fprintf(fid,strSolverSelection_1); 
%  fprintf(fid,strSolverSelection_2);  
 fprintf(fid,strSolverSelection_3); 
 fprintf(fid,strSolverSelection_4); 
%  fprintf(fid,strSolverSelection_5); 
%  fprintf(fid,strSolverSelection_6); 
%  fprintf(fid,strSolverSelection_7); 
%  for i=1:length(given)
%  fprintf(fid,strSolverSelection_8,given(i)+length(Struct.c),given(i)); 
%  fprintf(fid,strSolverSelection_9,given(i)+length(Struct.c),given(i)); 
%  end
%  fprintf(fid,strSolverSelection_10); 
%  fprintf(fid,strSolverSelection_11); 
%  fprintf(fid,strSolverSelection_12);   
 
 fprintf(fid,'%%********* Differential EQS***************\n'); 

 for i=1:length(Struct.EQS.DIFF)
 fprintf(fid,'dy(%d) = %s;\n',i,Struct.EQS.DIFF{i});  
 end
 % fprintf(fid,'t\n');

    
   

