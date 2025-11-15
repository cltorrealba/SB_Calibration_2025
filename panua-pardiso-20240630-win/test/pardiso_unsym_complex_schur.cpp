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


// C++ compatible

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <cstdlib>
using namespace std;

#include "pardiso.h"

void dumpCSR(const char* filename, int n, int* ia, int* ja, complex<double>* a)
{
  fstream fout(filename, ios::out);
  fout << n << endl;
  fout << n << endl;
  fout << ia[n] << endl;
  
  for (int i = 0; i <= n; i++)
  {
    fout << ia[i] << endl;
  }

  for (int i = 0; i < ia[n]; i++)
  {
    fout << ja[i] << endl;
  }

  for (int i = 0; i < ia[n]; i++)
  {
    fout << a[i] << endl;
  }

  fout.close();
}


void printCSR(int n, int nnz, int* ia, int* ja, complex<double>* a)
{
  cout.setf(ios::scientific, ios::floatfield);
  cout.precision(16);
  cout << "rows: " << setw(10) << n << endl;
  cout << "nnz : " << setw(10) << nnz << endl;

  if (nnz == n*n)
  {
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        cout << setw(10) << a[i*n + j];
      }
      cout << endl;
    }
  }
  else
  {
    for (int i = 0; i < n; i++)
    {
      for (int index = ia[i]; index < ia[i+1]; index++)
      {
        int j = ja[index];
        cout << setw(10) << "(" << i << ", " << j << ") " << a[index];
      }
      cout << endl;
    }
  }
}


void shiftIndices(int n, int nonzeros, int* ia, int* ja, int value)
{
  int i;
    for (i = 0; i < n+1; i++) 
    {
        ia[i] += value;
    }
    for (i = 0; i < nonzeros; i++) 
    {
        ja[i] += value;
    }
}


int main( void ) 
{
    /* Matrix data. */

    int    n = 16;
    int    ia[17] = {0, 9, 17, 24, 31, 38, 46, 53, 61, 63, 65, 67, 69, 71, 73, 75, 77};
    int    ja[77] = {0, 2, 5, 6, 8, 9, 10, 11, 12, 1, 2, 4, 8, 9, 10, 11, 12, 2, 7, 8, 9, 
                     10, 11, 12, 3, 6, 8, 9, 10, 11, 12, 1, 4, 8, 9, 10, 11, 12, 2, 5, 7, 
                     8, 9, 10, 11, 12, 1, 6, 8, 9, 10, 11, 12, 2, 6, 7, 8, 9, 10, 11, 12, 
                     0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}; 

    complex<double> a[77] = { complex<double>(     7.0000000000000000,     1.),
                              complex<double>(     1.0000000000000000,     1.),
                              complex<double>(     2.0000000000000000,     1.),
                              complex<double>(     7.0000000000000000,     1.),
                              complex<double>(     0.8813528146570788,     0.),
                              complex<double>(     0.9693082988664113,     0.),
                              complex<double>(     0.0063764102307857,     0.),
                              complex<double>(     0.0656724982252748,     0.),
                              complex<double>(     0.1285133482338328,     0.),
                              complex<double>(    -4.0000000000000000,     0.),
                              complex<double>(     8.0000000000000000,     1.),
                              complex<double>(     2.0000000000000000,     1.),
                              complex<double>(     0.0246814022448006,     0.),
                              complex<double>(     0.3913069588115609,     0.),
                              complex<double>(     0.9199013631348676,     0.),
                              complex<double>(     0.2300406237580992,     0.),
                              complex<double>(     0.6371356638478847,     0.),
                              complex<double>(     1.0000000000000000,     1.),
                              complex<double>(     5.0000000000000000,     1.),
                              complex<double>(     0.3412064600197927,     0.),
                              complex<double>(     0.3135851048672553,     0.),
                              complex<double>(     0.0179622413055945,     0.),
                              complex<double>(     0.1170403198125443,     0.),
                              complex<double>(     0.7464881699547887,     0.),
                              complex<double>(     7.0000000000000000,     0.),
                              complex<double>(     9.0000000000000000,     1.),
                              complex<double>(     0.4214056775317185,     0.),
                              complex<double>(     0.5533135799374451,     0.),
                              complex<double>(     0.0294164798770650,     0.),
                              complex<double>(     0.9897710614201980,     0.),
                              complex<double>(     0.8052672976354331,     0.),
                              complex<double>(    -4.0000000000000000,     1.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     0.0800111586678662,     0.),
                              complex<double>(     0.7920480624973907,     0.),
                              complex<double>(     0.7114519163682994,     0.),
                              complex<double>(     0.0705164910828937,     0.),
                              complex<double>(     0.2690271315860681,     0.),
                              complex<double>(     7.0000000000000000,     1.),
                              complex<double>(     3.0000000000000000,     1.),
                              complex<double>(     8.0000000000000000,     0.),
                              complex<double>(     0.0793538091733186,     0.),
                              complex<double>(     0.7983140830802098,     0.),
                              complex<double>(     0.5384141865401406,     0.),
                              complex<double>(     0.0600236559152354,     0.),
                              complex<double>(     0.4339797789833593,     0.),
                              complex<double>(     1.0000000000000000,     1.),
                              complex<double>(    11.0000000000000000,     1.),
                              complex<double>(     0.0723762491986621,     0.),
                              complex<double>(     0.8632585405599013,     0.),
                              complex<double>(     0.5245885805189759,     0.),
                              complex<double>(     0.6615771015291297,     0.),
                              complex<double>(     0.4018064872641355,     0.),
                              complex<double>(    -3.0000000000000000,     1.),
                              complex<double>(     2.0000000000000000,     1.),
                              complex<double>(     5.0000000000000000,     0.),
                              complex<double>(     0.9003410060810889,     0.),
                              complex<double>(     0.7979622044473094,     0.),
                              complex<double>(     0.5022399304161738,     0.),
                              complex<double>(     0.3440577694777694,     0.),
                              complex<double>(     0.6081423394349520,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.),
                              complex<double>(     1.0000000000000000,     0.),
                              complex<double>(     0.0000000000000000,     0.) };
    
    int      nnz = ia[n];
    int      mtype = 13;        /* Real complex unsymmetric matrix */

 
    dumpCSR("matrix_sample_mtype_13.csr", n, ia, ja, a);
    printCSR(n, nnz, ia, ja, a);

    /* Internal solver memory pointer pt,                  */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    /* or void *pt[64] should be OK on both architectures  */ 
    void    *pt[64];

    /* Pardiso control parameters. */
    int      iparm[65];
    double   dparm[64];
    int      solver;
    int      maxfct, mnum, phase, error, msglvl;

    /* Number of processors. */
    int      num_procs;

    /* Auxiliary variables. */
    char    *var;
    int      i;

    complex<double>   zdum;              /* Double dummy */
    int               idum;              /* Integer dummy. */

/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters and initialize the solvers      */
/*     internal adress pointers. This is only necessary for the FIRST   */
/*     call of the PARDISO solver.                                      */
/* ---------------------------------------------------------------------*/
      
    error = 0;
    solver = 0; /* use sparse direct solver */
    pardisoinit_z(pt,  &mtype, &solver, &iparm[1], dparm, &error);

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
    else 
    {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }
    iparm[3]  = num_procs;
    iparm[11] = 1;
    iparm[13] = 0;
   
    
    maxfct = 1;         /* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 1;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */


/* -------------------------------------------------------------------- */    
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */ 
    shiftIndices(n, nnz, ia, ja, 1);


/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */
    pardiso_chkmatrix_z(&mtype, &n, a, ia, ja, &error);
    if (error != 0) 
    {
        printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
    }

 
/* -------------------------------------------------------------------- */    
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */ 
    
    int nrows_S = 8;
    phase       = 12;
    iparm[38]   = nrows_S; 

    int nb = 0;

    pardiso_z(pt, &maxfct, &mnum, &mtype, &phase,
               &n, a, ia, ja, &idum, &nb,
               &iparm[1], &msglvl, &zdum, &zdum, &error,  dparm);
    
    if (error != 0) 
    {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    printf("\nReordering completed ...\n");
    printf("Number of nonzeros in factors  = %d\n", iparm[18]);
    printf("Number of factorization GFLOPS = %d\n", iparm[19]);
    printf("Number of nonzeros is   S      = %d\n", iparm[39]);
   
/* -------------------------------------------------------------------- */    
/* ..  allocate memory for the Schur-complement and copy it there.      */
/* -------------------------------------------------------------------- */ 
    int nonzeros_S          = iparm[39];
 
    int* iS                 = new int[nrows_S+1];
    int* jS                 = new int[nonzeros_S];
    complex<double>* S      = new complex<double>[nonzeros_S];
  
    pardiso_get_schur_z(pt, &maxfct, &mnum, &mtype, S, iS, jS);

    printCSR(nrows_S, nonzeros_S, iS, jS, S);

/* -------------------------------------------------------------------- */    
/* ..  Convert matrix from 1-based Fortan notation to 0-based C         */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */ 
    shiftIndices(n, nnz, ia, ja, -1);
    
/* -------------------------------------------------------------------- */    
/* ..  Convert Schur complement from Fortran notation held internally   */
/*     to 0-based C notation                                            */
/* -------------------------------------------------------------------- */ 
    shiftIndices(nrows_S, nonzeros_S, iS, jS, -1);


/* -------------------------------------------------------------------- */    
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */ 
    phase = -1;                 /* Release internal memory. */

    pardiso_z(pt, &maxfct, &mnum, &mtype, &phase,
              &n, &zdum, ia, ja, &idum, &idum,
              &iparm[1], &msglvl, &zdum, &zdum, &error,  dparm);


    delete[] iS;
    delete[] jS;
    delete[] S;

    printf ("EXIT: Completed\n");

    return 0;
} 
