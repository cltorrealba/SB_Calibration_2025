function Jacobians_iODE_Model_Creator(Struct,mur)


EQSFile='ImplicitODE_Jac.m';


 fid = fopen(EQSFile,'wt');
 
A=Struct.A;
[nmet, nflux]=size(A); 
nDif_var=length(Struct.VARS.DIFF);
nConstants=length(Struct.CONSTANTS);
nAlg_var=length(Struct.VARS.ALG);
nDif_EQS=length(Struct.EQS.DIFF);
nDif_ALG=nAlg_var;

 str1 = ['function [dFdy, dFdyp] = ImplicitODE_Jac(t,Y,Yp,A,b,UB,LB,Struct) \n'];
 str_break=['%%************************************************************\n\n'];

  
 fprintf(fid,str1);
 
 fprintf(fid,'A=Struct.A;\n');
  fprintf(fid,' [nmet, nflux]=size(A); ;\n');

 fprintf(fid,'nDif_var=length(Struct.VARS.DIFF);\n');
 fprintf(fid,'nConstants=length(Struct.CONSTANTS);\n');
 fprintf(fid,'nAlg_var=length(Struct.VARS.ALG);\n');
 fprintf(fid,'nDif_EQS=length(Struct.EQS.DIFF);\n');
 fprintf(fid,'nDif_ALG=nAlg_var;\n');
  
 fprintf(fid,'dFdy=sparse(nDif_var+nAlg_var+nflux+nmet,nDif_var+nAlg_var+nflux+nmet);\n');
 
  fprintf(fid,'dFdyp=sparse(nDif_var+nAlg_var+nflux+nmet,nDif_var+nAlg_var+nflux+nmet);\n');
  
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

  fprintf(fid,'%%********* Bounds EQS***************\n');
  
   fprintf(fid,'alphaZ=muR./(UB-v_flux).^2;\n'); 
   fprintf(fid,'alphaY=muR./(v_flux-LB).^2;\n'); 
  
   
   
 
  fprintf(fid,'%%********* Start with  dFdy, Jacobian of equations wrt differential vars.***************\n'); 

   fprintf(fid,'Dalpha=2*muR./(UB-v_flux).^3+2*muR./(LB-v_flux).^3;\n'); 
   
%% dFdY structure
            %Diff_Vars       %Alg_vars          %v_flux        %L
%Diff_eqs
%Alg_eqs
%Mass_bal
%dLdy
   
   %% obtain Jacobians
for i=1:nDif_var
DifBal{i}= strcat(Struct.VARS.iODE_VARS{i},'-',Struct.EQS.DIFF{i});
end

for i=nDif_var+1:nDif_var+nAlg_var
DifBal{i}= strcat(Struct.VARS.iODE_VARS{i},'-',Struct.EQS.ALG_TIME_DERIVATIVE{i-nDif_var});
end



Dif_vars_Sym=sym([Struct.VARS.DIFF,Struct.VARS.ALG]);

Aux_Mat1=jacobian(DifBal,Dif_vars_Sym);

%% Write elements diferent from zero of dFdDifvars and dFdAlgvars to the sparse matrix
  
for i=1:nDif_var+nAlg_var
    for j=1: nDif_var+nAlg_var
        if strcmp(char(Aux_Mat1(i,j)),'0')
        else
        fprintf(fid,'dFdy(%d,%d) = %s;\n',i,j,char(Aux_Mat1(i,j)));  
        end
    end
end

%% Write elements diferent from zero of dFdv_flux to the sparse matrix

% Elements of diff. EQS  starts at:
%       Column:nDif_var+nAlg_var+1
%       Row:   1
 for i=1:nDif_var
 if Struct.v_flux_incidence{i,2}>0
     j=Struct.v_flux_incidence{i,2}+nDif_var+nAlg_var; %column index position 
 fprintf(fid,'dFdy(%d,%d) = -%s;\n',i,j,Struct.v_flux_incidence{i,1});  
 end
 end
   
 % dMBdt wrt difvars: nothing to write
 % dMBdt wrt Alg_vars: nothing to write
 % dMBdt wrt v_flux: nothing to write
 % dMBdt wrt L: nothing to write
 
 % dLdt wrt difvars: nothing to write
 % dLdt wrt Alg_vars: nothing to write
 % dLdt wrt v_flux:
 



 fprintf(fid,'Range_Rows=nDif_ALG+nDif_EQS+nmet+1:nDif_ALG+nDif_EQS+nmet+nflux;\n');
 fprintf(fid,'Range_Col=nDif_var+nAlg_var+1:nDif_var+nAlg_var+nflux;\n');
 fprintf(fid,'dFdy(Range_Rows,Range_Col)=diag(Dalpha.*dvdt);\n');
 
  

 % dLdt wrt L: nothing to write
 
 %% dFdyp
 fprintf(fid,'%%********* Start with  dFdyP, Jacobian of equations wrt implicit difs vars.***************\n'); 
    %% obtain Jacobians

Dif_vars2_Sym=sym(Struct.VARS.iODE_VARS);

Aux_Mat2=jacobian(DifBal,Dif_vars2_Sym);

for i=1:nDif_var+nAlg_var
    for j=1: nDif_var+nAlg_var
        if strcmp(char(Aux_Mat2(i,j)),'0')
        else
        fprintf(fid,'dFdyp(%d,%d) = %s;\n',i,j,char(Aux_Mat2(i,j)));  
        end
    end
end



 fprintf(fid,'%%********* Mass bal. uptake wrt dvupt_dt ***************\n');
 
 pos=Struct.GIVEN; %uptake positions in mass balances equations. 

      for k=1: length(pos)
          i= pos(k)+nDif_ALG+nDif_EQS;
          for j=nDif_var+1:nDif_var+nAlg_var
        fprintf(fid,'dFdyp(%d,%d) = %d;\n',i,j,-1);  
          end
      end
 
 
 fprintf(fid,'%%*********  A ***************\n');
 fprintf(fid,'Range_Rows=nDif_ALG+nDif_EQS+1:nDif_ALG+nDif_EQS+nmet;\n');
 fprintf(fid,'Range_Col=nDif_var+nAlg_var+1:nDif_var+nAlg_var+nflux;\n');
 fprintf(fid,'dFdyp(Range_Rows,Range_Col)=sparse(A);\n');



 fprintf(fid,'%%********* Diagonal matrix (AlphaY+AlphaZ)***************\n'); 
 fprintf(fid,'Range_Rows=nDif_ALG+nDif_EQS+nmet+1:nDif_ALG+nDif_EQS+nmet+nflux;\n');
 fprintf(fid,'Range_Col=nDif_var+nAlg_var+1:nDif_var+nAlg_var+nflux;\n');
 fprintf(fid,'dFdyp(Range_Rows,Range_Col)=diag(alphaZ+alphaY);\n');
 
 fprintf(fid,'%%********* Lower -A''***************\n');
 fprintf(fid,'Range_Rows=nDif_ALG+nDif_EQS+nmet+1:nDif_ALG+nDif_EQS+nmet+nflux;\n');
 fprintf(fid,'Range_Col=nDif_var+nAlg_var+nflux+1:nDif_var+nAlg_var+nflux+nmet;\n');
 fprintf(fid,'dFdyp(Range_Rows,Range_Col)=sparse(-A.'');\n');

