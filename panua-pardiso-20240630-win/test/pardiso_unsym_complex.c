/* -------------------------------------------------------------------- */
/*      Example program to show the use of the "PARDISO" routine        */
/*      for a complex unsymmetric linear systems                        */
/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      http://www.panua.ch/products/pardiso/                           */
/*                                                                      */
/*  (C) Olaf Schenk, Panua Technologies, Switzerland.                   */
/*      Email: info@panua.ch                                            */
/* -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pardiso.h"

int main( void ) 
{
    /* Matrix data. */

    int    n = 8;
    int    ia[ 9] = { 0, 4, 7, 9, 11, 12, 15, 17, 20 };
    int    ja[20] = { 0,    2,       5, 6, 
                         1, 2,    4,
                            2,             7,
                               3,       6,
                         1,
                            2,       5,    7,
                         1,             6,
                            2,          6, 7 };

    doublecomplex  a[20];
    
    int      nnz = ia[n];
    int      mtype = 13;        /* Real complex unsymmetric matrix */


    /* RHS and solution vectors. */
    doublecomplex   b[8], x[8];
    int      nrhs = 1;          /* Number of right hand sides. */

    /* Internal solver memory pointer pt,                  */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    /* or void *pt[64] should be OK on both architectures  */ 
    void    *pt[64];

    /* Pardiso control parameters. */
    int      iparm[64];
    int      maxfct, mnum, phase, error, msglvl, solver;
    double   dparm[64];

    /* Number of processors. */
    int      num_procs;

    /* Auxiliary variables. */
    char    *var;
    int      i;

    doublecomplex   ddum;        /* Double dummy */
    int	      idum;              /* Integer dummy. */

    a[0 ].re =  7.0, a[0 ].i = 1.0;
    a[1 ].re =  1.0, a[1 ].i = 1.0;
    a[2 ].re =  2.0, a[2 ].i = 1.0;
    a[3 ].re =  7.0, a[3 ].i = 1.0;
    a[4 ].re = -4.0, a[4 ].i = 0.0;
    a[5 ].re =  8.0, a[5 ].i = 1.0;
    a[6 ].re =  2.0, a[6 ].i = 1.0;
    a[7 ].re =  1.0, a[7 ].i = 1.0;
    a[8 ].re =  5.0, a[8 ].i = 1.0;
    a[9 ].re =  7.0, a[9 ].i = 0.0;
    a[10].re =  9.0, a[10].i = 1.0;
    a[11].re = -4.0, a[11].i = 1.0;
    a[12].re =  7.0, a[12].i = 1.0;
    a[13].re =  3.0, a[13].i = 1.0;
    a[14].re =  8.0, a[14].i = 0.0;
    a[15].re =  1.0, a[15].i = 1.0;
    a[16].re = 11.0, a[16].i = 1.0;
    a[17].re = -3.0, a[17].i = 1.0;
    a[18].re =  2.0, a[18].i = 1.0;
    a[19].re =  5.0, a[19].i = 0.0;

    /* Set right hand side to one. */
    for (i = 0; i < n; i++)
    {
        b[i].re = 1;
	b[i].i	= 1;
    }

/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters und initialize the solvers      */
/*     internal adress pointers. This is only necessary for the FIRST   */
/*     call of the PARDISO solver.                                      */
/* ---------------------------------------------------------------------*/
      

    /* Numbers of processors, value of OMP_NUM_THREADS */
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }
    iparm[2]  = num_procs;
   
    
    maxfct = 1;         /* Maximum number of numerical factorizations. */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 1;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */


/* -------------------------------------------------------------------- */    
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */ 
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }

/* -------------------------------------------------------------------- */    
/* ..  Check license file pardiso.lic                                   */
/* -------------------------------------------------------------------- */ 

    error  = 0;
    solver = 0; /* use sparse direct solver */
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

    if (error != 0)
    {
        if (error == -10 )
           printf("No license file found \n");
        if (error == -11 )
           printf("License is expired \n");
        if (error == -12 )
           printf("Wrong username or hostname \n");
         return 1;
    }
    else
        printf("[PARDISO] License check was successful ... \n");


    if (error != 0) {
        if (error == -10)
           printf("\n No license file found\n\n");
        if (error == -11)
           printf("\n License is expired\n\n");
        if (error == -12)
           printf("\n Wrong username or hostname\n\n");
        exit(1);
    }
    else
       printf("[PARDISO]: License check was successful ... \n");

/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix_z(...)                                        */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */
    
    pardiso_chkmatrix_z (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec_z(...)                                            */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    pardiso_chkvec_z (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR in right hand side: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats_z(...)                                         */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    pardiso_printstats_z (&mtype, &n, a, ia, ja, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */    
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */ 
    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
    printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
   
/* -------------------------------------------------------------------- */    
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */    
    phase = 22;

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    printf("\nFactorization completed ...\n ");

/* -------------------------------------------------------------------- */    
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */    
    phase = 33;
    iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error, dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }

    printf("\nSolve completed ... ");
    printf("\nThe solution of the system is: ");
    for (i = 0; i < n; i++) {
        printf("\n x [%d] = % f % f", i, x[i].re, x[i].i);
    }
    printf ("\n");


/* -------------------------------------------------------------------- */
/* .. pardiso_printstats_z(...)                                         */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    pardiso_chkvec_z (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR in solution: %d", error);
        exit(1);
    }


/* -------------------------------------------------------------------- */    
/* ..  Convert matrix back to 0-based C-notation.                       */
/* -------------------------------------------------------------------- */ 
    for (i = 0; i < n+1; i++) {
        ia[i] -= 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] -= 1;
    }

/* -------------------------------------------------------------------- */    
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */ 
    phase = -1;                 /* Release internal memory. */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);

    printf ("EXIT: Completed\n");
    return 0;
} 
