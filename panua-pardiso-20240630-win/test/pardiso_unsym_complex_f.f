C----------------------------------------------------------------------
C       Example program to show the use of the "PARDISO" routine
C       for unsymmetric linear systems
C---------------------------------------------------------------------
C ---------------------------------------------------------------------
C      This program can be downloaded from the following site:         
C      http://www.panua.ch/products/pardiso/                           
C                                                                      
C  (C) Olaf Schenk, Panua Technologies, Switzerland.                   
C      Email: info@panua.ch                                            
C---------------------------------------------------------------------

        PROGRAM pardiso_unsym
        IMPLICIT NONE

C..     Internal solver memory pointer for 64-bit architectures
C..     INTEGER*8 pt(64)
C..     Internal solver memory pointer for 32-bit architectures
C..     INTEGER*4 pt(64)
C..     This is OK in both cases.
        INTEGER*8 pt(64)

C..     All other variables 

        INTEGER     maxfct, mnum, mtype, phase, n, nrhs, error, msglvl
        INTEGER     iparm(64), solver
        INTEGER     ia(9) 
        INTEGER     ja(20)
        REAL*8      dparm(64) 
        COMPLEX*16  a(20) 
        complex*16  b(8)
        complex*16  x(8)

        INTEGER i, idum(1)
        REAL*8  waltime1, waltime2
        complex*16  ddum(1)

C.. Fill all arrays containing matrix data.

        DATA n /8/, nrhs /1/, maxfct /1/, mnum /1/

        DATA ia /1,5,8,10,12,13,16,18,21/

        DATA ja
     1        /1,          3,                 6,    7,
     2               2,    3,          5,                   
     3                     3,                             8,            
     4                          4,                  7,     
     5               2,                                         
     6                     3,                 6,          8,         
     7               2,                             7,      
     8                     3,                       7,    8/
      
        DATA a
     1     /(7.d0, 1.d0), (1.d0,1.d0), (2.d0,1.d0), (7.d0,1.d0),
     2      (-4.d0,0.d0), (8.d0,1.d0), (2.d0,1.d0),                   
     3      (1.d0,1.d0),  (5.d0,1.d0),            
     4      (7.d0,0.d0),  (9.d0,1.d0),     
     5      (-4d0,1.d0),       
     6      (7.d0,1.d0),  (3.d0,1.d0), (8.d0,0.d0),         
     7      (1.d0,1.d0),  (11.d0,1.d0),      
     8      (-3.d0,1.d0), (2.d0,1.d0), (5.d0,0.d0)/

C  .. set right hand side
      do i = 1, n
        b(i) = (1.d0,1.d0)
      end do


      mtype     = 13  ! complex unsymmetric 
      solver    =  0  ! use sparse direct method

C
C  .. Setup Pardiso control parameters und initialize the solvers     
C     internal adress pointers. This is only necessary for the FIRST   
C     call of the PARDISO solver.                                     
C     matrix

      call pardisoinit(pt, mtype, solver, iparm, dparm, error)
      IF (error .NE. 0) THEN
        IF (error.EQ.-10 ) WRITE(*,*) 'No license file found'
        IF (error.EQ.-11 ) WRITE(*,*) 'License is expired'
        IF (error.EQ.-12 ) WRITE(*,*) 'Wrong username or hostname'
        STOP
      ELSE
        WRITE(*,*) '[PARDISO]: License check was successful ... '
      END IF
 
C  .. Numbers of Processors ( value of OMP_NUM_THREADS )
      iparm(3) = 1


c  .. pardiso_chk_matrix(...)
c     Checks the consistency of the given matrix.
c     Use this functionality only for debugging purposes
      CALL pardiso_chkmatrix_z  (mtype, n, a, ia, ja, error);
      IF (error .NE. 0) THEN
         WRITE(*,*) 'The following ERROR was detected: ', error
        STOP
      ENDIF

c ..  pardiso_chkvec(...)
c     Checks the given vectors for infinite and NaN values
c     Input parameters (see PARDISO user manual for a description):
c     Use this functionality only for debugging purposes
      CALL pardiso_chkvec_z (n, nrhs, b, error);
      IF (error .NE. 0) THEN
         WRITE(*,*) 'The following ERROR was detected: ', error
        STOP
      ENDIF
 
c ..  pardiso_printstats(...) 
c     prints information on the matrix to STDOUT.
c     Use this functionality only for debugging purposes

      CALL pardiso_printstats_z (mtype, n, a, ia, ja, nrhs, b, error);
      IF (error .NE. 0) THEN
         WRITE(*,*) 'The following ERROR was detected: ', error
        STOP
       ENDIF
         
C..   Reordering and Symbolic Factorization, This step also allocates
C     all memory that is necessary for the factorization 
C
      phase     = 11      ! only reordering and symbolic factorization
      msglvl    = 1       ! with statistical information
      CALL pardiso (pt, maxfct, mnum, mtype, phase, n, a, ia, ja,
     1              idum, nrhs, iparm, msglvl, ddum, ddum, error, dparm)
     
      WRITE(*,*) 'Reordering completed ... '

      IF (error .NE. 0) THEN
        WRITE(*,*) 'The following ERROR was detected: ', error
        STOP
      END IF

      WRITE(*,*) 'Number of nonzeros in factors   = ',iparm(18)
      WRITE(*,*) 'Number of factorization GFLOPS  = ',iparm(19)

C.. Factorization.
      phase     = 22  ! only factorization
      CALL pardiso (pt, maxfct, mnum, mtype, phase, n, a, ia, ja, 
     1              idum, nrhs, iparm, msglvl, ddum, ddum, error, dparm) 

      WRITE(*,*) 'Factorization completed ...  '
      IF (error .NE. 0) THEN
         WRITE(*,*) 'The following ERROR was detected: ', error
        STOP
      ENDIF 

C.. Back substitution and iterative refinement
      phase     = 33  ! only solve
      iparm(8)  = 1   ! max numbers of iterative refinement steps

      CALL pardiso (pt, maxfct, mnum, mtype, phase, n, a, ia, ja, 
     1              idum, nrhs, iparm, msglvl, b, x, error, dparm) 
      WRITE(*,*) 'Solve completed ... '
     
      WRITE(*,*) 'The solution of the system is '
      DO i = 1, n
        WRITE(*,*) ' x(',i,') = ', x(i)
      END DO

C.. Termination and release of memory
      phase     = -1           ! release internal memory
      CALL pardiso (pt, maxfct, mnum, mtype, phase, n, ddum, idum, idum,
     1              idum, nrhs, iparm, msglvl, ddum, ddum, error, dparm)

      WRITE(*,*) '"EXIT: Completed'
      END



