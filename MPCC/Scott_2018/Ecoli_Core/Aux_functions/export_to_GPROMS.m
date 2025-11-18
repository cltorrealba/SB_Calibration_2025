function export_to_GPROMS


Prob_Name='SCerevisiaeModel';

SCerevisiae_Prob_definition(Prob_Name); %call the script where the problem is defined. 
Name_Mat_File=strcat(Prob_Name,'.mat');
load(Name_Mat_File)

        opts.Method   = 'DAE_taylored';
        opts.initializion ='IP'; %IP or CPLEX with solution adjustments to make it interior
        opts.mu =1e-6; %The penalty parameter, only used in the Implicit Reduced ODE method. 
        opts.epsilon =0.0001; %the amount that will be substracted or added to the LP solution to make it interior if its in a bound. Only used when  opts.initializion ='CPLEX'; 
        opts.RigourousTimeIt='No';

%% Initialization 

Struct=InitializeIP(opts,Struct); 

%Load data
A=Struct.A;
b=Struct.b;
c=Struct.c;
UB=Struct.ub;
LB=Struct.lb;  
X=Struct.Initial_LP.X;
Y=Struct.Initial_LP.Y;
Z=Struct.Initial_LP.Z;
L=Struct.Initial_LP.L;


nflux= length(X);
nmet=length(b);
given=Struct.GIVEN;
UPTK_FLUX= Struct.UPTAKE;
%% Export data in GPROMS format

    Smatrix=sparse(A);
    cvector=sparse(c);

    [i,j,val] = find(cvector);
    data_dump = [j,val];

    [i1,j1,val1] = find(Smatrix);
    data_dump1 = [i1,j1,val1];

    [i2,j2,val2] = find(UB);
    data_dump2 = [i2,val2];

    [i3,j3,val3] = find(LB);
    data_dump3 = [i3,val3];

    fid = fopen('AugMatrixGproms.txt','w');

    %Write fluxes and metabolites sets:

    flux_set=sprintf('''%g'',',[1:nflux]);
    fprintf( fid,'Nflux:= [%s];\n', flux_set );

    fprintf(fid,'\n');

    met_set=sprintf('''%g'',',[1:nmet]);
    fprintf( fid,'Nmetab:= [%s];\n', met_set );

    fprintf(fid,'\n');

    given_set=sprintf('''%g'',',given);
    fprintf( fid,'gvn:= [%s];\n', given_set );

    fprintf(fid,'\n');

    UPTK_set=sprintf('''%g'',',UPTK_FLUX);
    fprintf( fid,'UPTK:= [%s];\n', UPTK_set );

    fprintf(fid,'\n');
    %Objective fun
    fprintf( fid,'c(''%d''):=%g;\n', transpose(data_dump) );

    % A Matrix
    fprintf( fid,'\n' );
    fprintf( fid,'Smatrix(''%d'',''%d''):=%g;\n', transpose(data_dump1) );
    % Upper Bounds
    fprintf( fid,'\n' );
    fprintf( fid,'UB(''%d''):=%g;\n', transpose(data_dump2) );
    %Lower Bounds
    fprintf( fid,'\n' );
    fprintf( fid,'LB(''%d''):=%g;\n', transpose(data_dump3) );




      %% Export initial point


      
    [i,j,val] = find(X);
    data_dump = [i,val];

    %flux values
    fprintf( fid,'\n' );
    fprintf( fid,'v_flux(''%d''):=%g;\n', transpose(data_dump) );


    [i,j,val] = find(L);
    data_dump = [i,val];

    %Lambda values
    fprintf( fid,'\n' );
    fprintf( fid,'lambda(''%d''):=%g;\n', transpose(data_dump) );


    [i,j,val] = find(Z);
    data_dump = [i,val];
    %Z values
    fprintf( fid,'\n' );
    fprintf( fid,'Z(''%d''):=%g;\n', transpose(data_dump) );


    [i,j,val] = find(Y);
    data_dump = [i,val];
    %Y values
    fprintf( fid,'\n' );
    fprintf( fid,'Y(''%d''):=%g;\n', transpose(data_dump) );
    fclose(fid);