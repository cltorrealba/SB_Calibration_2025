/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      http://www.panua.ch/products/pardiso/                           */
/*                                                                      */
/*  (C) Olaf Schenk, Panua Technologies, Switzerland.                   */
/*      Email: info@panua.ch                                            */
/*                                                                      */
/*      Example program to show the use of the "PARDISO" routine        */
/*      on a scalable symmetric linear systems (Laplace equation)       */
/*      The program can be started with "./laplace n nrhs"              */
/*      in which "n" represents the discretization in one spatial       */
/*      direction and "nrhs" are the number of right-hand sides.        */
/*      e.g. ./laplace 100 10                                           */ 
/* -------------------------------------------------------------------- */

#include "laplace.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "pardiso.h"

void * _hh_malloc_laplace(size_t  count,  size_t  size)
{
        void * new = malloc (count * size);
        if (new == NULL)
        {
                printf("ERROR during memory allocation!\n");
                exit (7);
        }
        return  new;
}

void * _hh_calloc_laplace(size_t  count,  size_t  size)
{
        void * new = calloc (count, size);
        if (new == NULL)
        {
                printf("ERROR during cleared memory allocation!\n");
                exit (7);
        }
        return  new;
}


smat_laplace_t *  smat_new_laplace (int   m,
		   int    n,
		   int	    type)
{
	smat_laplace_t  *new_mat;

	mem_calloc (new_mat,  1,  smat_laplace_t);

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


smat_laplace_t	* smat_new_nnz_struct_laplace (int	m,
			       int	n,
			       int	nnz,
			       int	type)
{
        /* type = 0 : unsymmetric, real     */
        /* type = 1 : symmetric,   real     */
        /* type = 2 : unsymmetric, is_complex  */
        /* type = 3 : symmetric,   is_complex  */

	smat_laplace_t  *new_mat = smat_new_laplace (m, n, type);

        assert(type == 0 || type == 1 || type == 2 || type == 1);

	new_mat->nnz = nnz;
	mem_alloc (new_mat->ja,  nnz+1,  int);

	return new_mat;
}

smat_laplace_t	* smat_new_nnz_laplace (int	m,
			int	n,
			int	nnz,
			int	type)
{
	smat_laplace_t  *new_mat = smat_new_nnz_struct_laplace (m, n, nnz, type);

	if (new_mat->is_complex == 0)
		mem_alloc (new_mat->a,  nnz+1,  double);
        else
		mem_alloc (new_mat->a,  2*nnz+1,  double);

	return new_mat;
}

smat_laplace_t * tst_gen_poisson_5_sym_laplace (int  n, double hx, double hx2, double c, double alpha)
{
        smat_laplace_t          *pm = smat_new_nnz_laplace(n*n, n*n, ((n - 1)*n) + ((2*n - 1)*n), 1);
        int              i, j, nnz = 0;

        /* For n nxn blocks */
        for (i = 0; i < n; i++)
        {
                int  j;

                /* For each of the n rows */
                for (j = 0; j < n; j++)
                {
                        pm->ia[i*n + j] = nnz;

                        pm->ja[nnz] = i*n + j;
                        pm-> a[nnz]   = 4.0;
                        nnz++;

                        if (j < n-1)
                        {
                                pm->ja[nnz] = i*n + j + 1;
                                pm-> a[nnz  ] = -1.0;
                                nnz++;
                        }
                        if (i < n-1)
                        {
                                pm->ja[nnz] = (i+1)*n + j;
                                pm-> a[nnz  ] = -1.0; 
                                nnz++;
                        }

                }
        }
        pm->ia[n*n] = nnz;
        assert (pm->nnz == nnz);

        return  pm;
}


int main(int argc, char *argv[]  ) 
{

    int    grid  = atoi(argv[1]); /* Number of discretization points in one spatial direction */
    int    nrhs  = atoi(argv[2]); /* Number of right hand sides. */

    int    n;
    int    nnz;
  
    int    mtype = -2;        /* Real symmetric indefinite matrix */


    /* Matlab file */
    char      mat_name[128];
    FILE     *mat_file;

    /* RHS and solution vectors. */
    double*  b      = NULL;
    double*  b_save = NULL;
    double*  x = NULL;

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
    int      i, j, k, kk;
    double    norm_r = 0.0;
    double    norm_b = 0.0;
    double    val;
  
    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */
 
    double           Xmax  = 6400;
    double           Xmin  =  400;
    double           c;
    double           hx, hx2;
    /* damping of the material */
    double           alpha = 0.05;

    /* Matrix data. */
    smat_laplace_t* A = NULL;

    /* mesh size */
    hx    = (Xmax-Xmin)/(grid-1);        
    hx2   = 1/(hx*hx);

    n = grid * grid;
    b      = malloc (nrhs * n * sizeof (double));
    if (b == NULL) 
    {
	printf("\nERROR during malloc of b");
        exit(9);
    } 
    /* Set point source in the middle of the boundary */
    for (k = 0; k < nrhs; k++) {
    for (i = 0; i < n; i++) {
        b[i + k*n]   = i;
     }
    }

    c = 0;
    A = tst_gen_poisson_5_sym_laplace(grid, hx, hx2, c, alpha);

    n = A->m;
    nnz = A->ia[n];
 
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

    error = 0;
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
    iparm[33] = 0; // bit-by-bit identical results
    /* iparm[50] = 0; */
    /* iparm[51] = 0; */

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
   
    pardiso_chkmatrix  (&mtype, &n, A->a, A->ia, A->ja, &error);
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

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    pardiso_printstats (&mtype, &n, A->a, A->ia, A->ja, &nrhs, b, &error);
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
    printf("\nNumber of factorization GFLOPS = %d\n\n", iparm[18]);
   
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

    iparm[7] = 0;       /* Max numbers of iterative refinement steps. */

    b_save = malloc (nrhs * n * sizeof (double));
    x      = malloc (nrhs * n * sizeof (double));

    if (x == NULL) 
    {
	printf("\nERROR during malloc of x");
        exit(9);
    } 
    for (i = 0; i < n; i++)  /* initialize vector x with zero */
        x[i] = 0;


    for (k = 0; k < nrhs; k++) {
  	  for (i = 0; i < n; i++) {
	        b_save[i + k*n ]   = b[i + k*n ];
    	 } 
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
 
    for (kk = 0; kk < nrhs; kk++) 
    {
    	for (i = 0; i < n; i++) {
            j = A->ia[i]-1;
            k = A->ja[j]-1;
            b[i+ kk*n] -= A->a[j]*x[k + kk*n];
   	} 

	for (i = 0; i < n; i++) {
        	for (j = A->ia[i]; j < A->ia[i+1]-1; j++) {
           	 	k = A->ja[j] -1;
            	 	b[i + kk*n] -= A->a[j]*x[k + kk*n];
            		b[k + kk*n] -= A->a[j]*x[i + kk*n];
        	} 
    	}

    	val = 0;
    	for (i = 0; i < n; i++) {
       	 val += b_save[i+ kk*n]*b_save[i+ kk*n];
    	}
    	norm_b = sqrt(val);

    	val = 0;
    	for (i = 0; i < n; i++) {
        	val += b[i+ kk*n]*b[i+ kk*n];
    	}
    	norm_r = sqrt(val);
    	for (i = 0; i < n; i++) {
        	val += b[i+ kk*n]*b[i+ kk*n];
                /* printf(" x  %32.64e\n", x[i]); */
    	}
	printf("Norm of B         %e Norm of Residual  %e\n", norm_b, norm_r/norm_b);

    }

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
