/* -------------------------------------------------------------------- */
/*      Example program to show the use of the "PARDISO" routine        */
/*      on symmetric linear systems                                     */
/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      http://www.pardiso-project.org                                  */
/*                                                                      */
/*  (C) Olaf Schenk, Institute of Computational Science                 */
/*      Universita della Svizzera italiana, Lugano, Switzerland.        */
/*      Email: olaf.schenk@usi.ch                                       */
/* -------------------------------------------------------------------- */

#include "helmholtz.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "math.h"

#include "pardiso.h"

void * _hh_malloc(size_t  count,  size_t  size)
{
        void * new = malloc (count * size);
        if (new == NULL)
        {
                printf("ERROR during memory allocation!\n");
                exit (7);
        }
        return  new;
}

void * _hh_calloc(size_t  count,  size_t  size)
{
        void * new = calloc (count, size);
        if (new == NULL)
        {
                printf("ERROR during cleared memory allocation!\n");
                exit (7);
        }
        return  new;
}


smat_t *  smat_new (int   m,
		   int    n,
		   int	    type)
{
	smat_t  *new_mat;

	mem_calloc (new_mat,  1,  smat_t);

	new_mat->m   	 = m;
	new_mat->n   	 = n;
	new_mat->nnz 	 = 0;

        if (type == 0 || type == 2)
	   new_mat->sym = 0;
        else
	   new_mat->sym = 1;

        if (type == 0 || type == 1)
	   new_mat->is_complex = 0;
        else
	   new_mat->is_complex = 1;

	mem_calloc (new_mat->ia,  m + 1,  int);

	new_mat->ja = NULL;
	new_mat->a  = NULL;

	return new_mat;
}


smat_t	* smat_new_nnz_struct (int	m,
			       int	n,
			       int	nnz,
			       int	type)
{
        /* type = 0 : unsymmetric, real     */
        /* type = 1 : symmetric,   real     */
        /* type = 2 : unsymmetric, is_complex  */
        /* type = 3 : symmetric,   is_complex  */

	smat_t  *new_mat = smat_new (m, n, type);

        assert(type == 0 || type == 1 || type == 2 || type == 3);

	new_mat->nnz = nnz;
	mem_alloc (new_mat->ja,  nnz+1,  int);

	return new_mat;
}

smat_t	* smat_new_nnz (int	m,
			int	n,
			int	nnz,
			int	type)
{
	smat_t  *new_mat = smat_new_nnz_struct (m, n, nnz, type);

	if (new_mat->is_complex == 0)
		mem_alloc (new_mat->a,  nnz+1,  double);
        else
		mem_alloc (new_mat->a,  2*nnz+1,  double);

	return new_mat;
}

smat_t * tst_gen_poisson_5_sym (int  n, double hx, double hx2, double c, double alpha)
{
        smat_t          *pm = smat_new_nnz(n*n, n*n, ((n - 1)*n) + ((2*n - 1)*n), 3);
        double           fac = hx2;
        double           freq  = 0.10;
        double           k;
        double           valA[2];
	double           valB[2];
	double           valC[2];
	double           PI = 3.14;


        int              i, j, nnz = 0;

 
        k     = 2*PI*freq/c;
        /* For n nxn blocks */
        for (i = 0; i < n; i++)
        {
                int  j;

                /* For each of the n rows */
                for (j = 0; j < n; j++)
                {
                        pm->ia[i*n + j] = nnz;

                        pm->ja[nnz] = i*n + j;
                        pm-> a[2*nnz]   = 4.0 * fac - k*k;
                        pm-> a[2*nnz+1] = alpha*k*k;
                        nnz++;

                        if (j < n-1)
                        {
                                pm->ja[nnz] = i*n + j + 1;
                                pm-> a[2*nnz  ] = -1.0 * fac;
                                pm-> a[2*nnz+1] = 0.0;
                                nnz++;
                        }
                        if (i < n-1)
                        {
                                pm->ja[nnz] = (i+1)*n + j;
                                pm-> a[2*nnz  ] = -1.0 * fac; 
                                pm-> a[2*nnz+1] = 0.0;
                                nnz++;
                        }

                }
        }
        pm->ia[n*n] = nnz;

        /* It is time to  adjust A and b to include boundary conditions */
        /* North: boundary */
        i = 0; 
        /* For each of the n rows */
        for (j = 0; j < n; j++)
        {
            int p = pm->ia[i*n + j];
            valA[0 ] = hx*hx;
            valA[1 ] = k*hx*hx*hx;
	    valB[0 ] = 1;
            valB[1 ] = 0.0;
	    valC[0 ] = 0.0;
            valC[1 ] = 0.0;
            CPX_DIV(valC, &valB,&valA)  
            pm-> a[2*p]     -= valC[0];
            pm-> a[2*p+1]   -= valC[1];
        }

        i = n-1; 
        /* South: boundary */
        for (j = 0; j < n; j++)
        {
            int p = pm->ia[i*n + j];
            valA[0 ] = hx*hx;
            valA[1 ] = k*hx*hx*hx;
	    valB[0 ] = 1;
            valB[1 ] = 0.0;
	    valC[0 ] = 0.0;
            valC[1 ] = 0.0;
            CPX_DIV(valC, &valB,&valA)  
            pm-> a[2*p]     -= valC[0];
            pm-> a[2*p+1]   -= valC[1];
        }
        /* West: boundary */
        j = 0; 
        for (i = 0; i < n; i++)
        {
            int p = pm->ia[i*n + j];
            valA[0 ] = hx*hx;
            valA[1 ] = k*hx*hx*hx;
	    valB[0 ] = 1;
            valB[1 ] = 0.0;
	    valC[0 ] = 0.0;
            valC[1 ] = 0.0;
            CPX_DIV(valC, &valB,&valA)  
            pm-> a[2*p]     -= valC[0];
            pm-> a[2*p+1]   -= valC[1];
        }
        /* East: boundary */
        j = n-1; 
        for (i = 0; i < n; i++)
        {
            int p = pm->ia[i*n + j];
            valA[0 ] = hx*hx;
            valA[1 ] = k*hx*hx*hx;
	    valB[0 ] = 1;
            valB[1 ] = 0.0;
	    valC[0 ] = 0.0;
            valC[1 ] = 0.0;           
            CPX_DIV(valC, &valB,&valA)  
            pm-> a[2*p]     -= valC[0];
            pm-> a[2*p+1]   -= valC[1];
        }

        assert (pm->nnz == nnz);

        return  pm;
}







int main(int argc, char *argv[]  ) 
{

    int    grid = atoi(argv[1]);
    int    n;
    int    nnz;
  
    int    mtype = 6;        /* Real symmetric matrix */


    /* Matlab file */
    char      mat_name[128];
    FILE     *mat_file;

    /* RHS and solution vectors. */
    double*  b      = NULL;
    double*  b_save = NULL;
    double*  x = NULL;
    int      nrhs = 1;          /* Number of right hand sides. */

    /* Internal solver memory pointer pt,                  */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    /* or void *pt[64] should be OK on both architectures  */ 
    void    *pt[64]; 

    /* Pardiso control parameters. */
    int      iparm[64];
    double   dparm[64];
    int      maxfct, mnum, phase, error, msglvl, solver;

    /* Number of processors. */
    int      num_procs;

    /* Auxiliary variables. */
    char    *var;
    int      i, j, k;
    double    norm_r = 0.0;
    double    norm_b = 0.0;
    double    val[2];
  
    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */
 
    double           Xmax  = 6400;
    double           Xmin  =  400;
    double           c;
    double           hx, hx2;
    /* damping of the material */
    double           alpha = 0.05;

    /* Matrix data. */
    smat_t* A = NULL;

    /* mesh size */
    hx    = (Xmax-Xmin)/(grid-1);        
    hx2   = 1/(hx*hx);

    /* todo: define c over the domain */
    /* min c = 1500, max c = 4450 */
    c = 3000;

    n = grid * grid;
    b      = malloc (2*n * sizeof (double));
    if (b == NULL) 
    {
	printf("\nERROR during malloc of b");
        exit(9);
    } 
    /* Set point source in the middle of the boundary */
    for (i = 0; i < n; i++) {
        b[2*i]   = 0;
        b[2*i+1] = 0;
    }
    b[2* (grid / 2 )]   = hx2;

    A = tst_gen_poisson_5_sym(grid, hx, hx2, c, alpha);

    n = A->m;
    nnz = A->ia[n];
 
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

    error = 0;
    solver = 0; /* use sparse direct solver */
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error); 

    iparm[24] = 0;
    iparm[23] = 0;
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
        printf("[PARDISO]: License check was successful ... \n");
    


    /* Numbers of processors, value of OMP_NUM_THREADS */
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }
    iparm[2]  = num_procs;

    maxfct = 1;		/* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 1;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */

/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */
    for (i = 0; i < n+1; i++) {
        A->ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        A->ja[i] += 1;
    }
 
/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */
    
    pardiso_chkmatrix_z  (&mtype, &n, A->a, A->ia, A->ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec(...)                                              */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    pardiso_chkvec_z (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    pardiso_printstats_z (&mtype, &n, A->a, A->ia, A->ja, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR right hand side: %d", error);
        exit(1);
    }
 
/* -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */
    phase = 11; 
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, A->a, A->ia, A->ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
    printf("\nNumber of factorization GFLOPS = %d", iparm[18]);
   
/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */    
    phase = 22;

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, A->a, A->ia, A->ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
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

    b_save = malloc (2*n * sizeof (double));
    x      = malloc (2*n * sizeof (double));
    if (x == NULL) 
    {
	printf("\nERROR during malloc of x");
        exit(9);
    } 

    for (i = 0; i < n; i++) {
        b_save[2*i]   = b[2*i  ];
        b_save[2*i+1] = b[2*i+1];
    }
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, A->a, A->ia, A->ja, &idum, &nrhs,
	     iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }

    /* -------------------------------------------------------------------- */
    /* ..  check residual.                                                  */
    /* -------------------------------------------------------------------- */
 
    for (i = 0; i < n; i++) {
            j = A->ia[i]-1;
            k = A->ja[j]-1;
            CPX_MINUSMUL(&b[2*i], &A->a[2*j], &x[2*k]);
    } 

    for (i = 0; i < n; i++) {
        for (j = A->ia[i]; j < A->ia[i+1]-1; j++) {
            k = A->ja[j] -1;
            CPX_MINUSMUL(&b[2*i], &A->a[2*j], &x[2*k]);
            CPX_MINUSMUL(&b[2*k], &A->a[2*j], &x[2*i]);
        } 
    }

    val[0 ] = 0;
    val[1 ] = 0;
    for (i = 0; i < n; i++) {
        CPX_ADDMUL (&val, &b_save[2*i], &b_save[2*i]);
    }
    norm_b = sqrt(val[0]*val[0] + val[1]*val[1]);

    val[0 ] = 0;
    val[1 ] = 0;
    for (i = 0; i < n; i++) {
        CPX_ADDMUL (&val, &b[2*i], &b[2*i]);
    }
    norm_r = sqrt(val[0]*val[0] + val[1]*val[1]);

    printf("Norm of B         %e \n", norm_b);
    printf("Norm of Residual  %e \n", norm_r/norm_b);

/* -------------------------------------------------------------------- */    
/* ..  Convert matrix back to 0-based C-notation.                       */
/* -------------------------------------------------------------------- */ 
    for (i = 0; i < n+1; i++) {
        A->ia[i] -= 1;
    }
    for (i = 0; i < nnz; i++) {
        A->ja[i] -= 1;
    }

/* -------------------------------------------------------------------- */    
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */    
    phase = -1;                 /* Release internal memory. */
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, A->ia, A->ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
    free(b);
    free(b_save);
    free(x);

    printf ("EXIT: Completed\n");
    return 0;
} 
