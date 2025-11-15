/* Copyright 2023, Panua Technologies, Switzerland.
 *
 * All rights reserved.
 *
 * This header file defines all functions that can be called in the Pardiso
 * library.
 */

/* TODO: Here should be also the documentation of the arguments for the functions. */

/* Structure for storing double precision complex numbers. */
typedef struct{
	double re; 
	double i;}
doublecomplex;

/*
#ifdef PARDISO_COMPLEX
typedef p_double p_doublecomplex;
#else
typedef p_double double;
#endif
*/

#ifdef __cplusplus
extern "C"
{
#endif

void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
// "void" in the following means that it can be double or doublecomplex
void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                  void   *, int    *,    int *, int *,   int *, int *,
                     int *, void    *, void   *, int *, double *);
void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
void pardiso_chkvec     (int *, int *, double *, int *);
void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

void pardisoinit_z (void   *, int    *,   int *, int *, double *, int *);
// "void" in the following means that it can be double or doublecomplex
void pardiso_z     (void   *, int    *,   int *, int *,    int *, int *,
                  void   *, int    *,    int *, int *,   int *, int *,
                     int *, void    *, void   *, int *, double *);
void pardiso_chkmatrix_z  (int *, int *, void *, int *, int *, int *);
void pardiso_chkvec_z     (int *, int *, void *, int *);
void pardiso_printstats_z (int *, int *, void *, int *, int *, int *,
                           void *, int *);
void pardiso_get_schur_z(void*, int*, int*, int*, void*, int*, int*);

void pardisoinit_d (void   *, int    *,   int *, int *, double *, int *);
// "void" in the following means that it can be double or doublecomplex
void pardiso_d     (void   *, int    *,   int *, int *,    int *, int *,
                  void   *, int    *,    int *, int *,   int *, int *,
                     int *, void    *, void   *, int *, double *);
void pardiso_chkmatrix_d  (int *, int *, void *, int *, int *, int *);
void pardiso_chkvec_d     (int *, int *, void *, int *);
void pardiso_printstats_d (int *, int *, void *, int *, int *, int *,
                           void *, int *);
void pardiso_get_schur_d(void*, int*, int*, int*, void*, int*, int*);
  
#ifdef __cplusplus
}
#endif
