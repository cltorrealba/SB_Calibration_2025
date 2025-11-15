# COPYRIGHT (C) PANUA TECHNOLOGIES, 2023

Panua Ipopt is a parallel implementation of the well-known nonlinear 
optimization software Ipopt.  It integrates the high-performance parallel 
linear solver package Panua Pardiso.

****** CONTENT ******

This package includes the following directories:

lib    : shared librar(ies) containing the Panua Ipopt solver
include: include files
bin    : binaries (license verification and AMPL solver executable)
test   : test instances

On Linux, Windows and Mac x86, the shared object libipopt.*
includes the necessary BLAS routines for your convenience. 

On Linux and Mac x86, we also include a version of the library 
(libipopt_no_deps.*) that does not include dependencies like BLAS and 
Intel's OpenMP and you will need to specify these libraries when you 
link the library.

On Mac Arm64, BLAS is not included in libipopt.dylib and you need to 
specify a BLAS library (e.g., -framework accelerate)

****** LICENSE KEYS ******

In order to use Panua Ipopt you need to obtain a license key at 
www.panua.ch and put it into a file called panua.lic.  This file may 
contain more than one license key.  Put this file into your home 
directory, or set the environment variable PANUA_LIC_PATH to the 
directory that contains panua.lic .

To verify that your license key is set up properly, run the 
"check_license" binary in the bin directory of the distribution.  Make 
sure that it reports that a valid Ipopt license could be found.

To obtain a license, you need to generate a "fingerprint".  You do this 
by running the "get_fingerprint" executable in the bin subdirectory on 
the machine where you want to use Panua Ipopt.  For single-user 
licenses, you need to be logged in as the user that will use Panua 
Ipopt.  Note that when you run your code on a computing cluster that 
uses a scheduling system, your code will be run on different physical 
machines that all have different fingerprints.  You can either get 
single-user licenses for the individual compute nodes, or you can get a 
single-user cluster license.

****** RUNNING TESTS / COMPILERS ******

To see if the software works on your system, run "make test" in the test 
subdirectory of the distribution.  This compiles some simple C++, C, and 
FORTRAN examples, the AMPL solver binary, and runs Ipopt in parallel for 
up to 4 cores.  Make sure you have a valid license key.

You need to make sure that, at runtime, the ipopt library can be found.
In particular, you need to set the environment variable
  Linux  :  LD_LIBRARY_PATH
  Mac    :  DYLD_LIBRARY_PATH
  Windows:  PATH
so that it includes the directory in which the ipopt library is located.
This is set for you for "make test" but you need to do that yourself in
all other circumstance.

For non-Windows systems, the GNU compilers are chosen, but any other 
compiler that is compatible with the GNU compilers (such as the Intel 
compilers) should work as well.  (Exception: On Mac arm64, you cannot
currently use the ipopt library with the clang compiler for C++ code.)

The Makefile might be a good start for writing your own Makefile to link 
with Ipopt.  (Ignore the part that runs the test suite.)

For Windows systems, you can use the MSVC or Intel compilers and open a
command prompt in which the environment for compilers has been set, such
as the x86_64 command prompt for the compiler.  The script "run_tests.bat"
compiles and executes the examples.  There is also an example MSVS solution
in the MSVS_example directory.

****** DOCUMENTATION ******

Panua Ipopt uses the same interfaces/APIs and options as the open source 
version of Ipopt.  Therefore you can refer to the corresponding 
documentation on COIN-OR:

  https://coin-or.github.io/Ipopt/

Once additional features become available, further documentation will be 
available on the Panua website.

By default, Panua Ipopt will use one core.  In order to exploit 
parallelism, you can increase the number of cores by either

- setting the environment variable OMP_NUM_THREADS or
- setting the Ipopt option "pardiso_num_threads"

to the desired number of threads.

****** COMPATIBILITY WITH OPEN SOURCE IPOPT ******

Panua Ipopt is binary compatible with the open source version of Ipopt.  
If you already have your code integrated with Ipopt, via the shared 
library option, you can simply replace the open-source shared library with 
the one contained in this package.

****** UPDATES ******

We release updated versions of the library from time to time.  Unless 
there is a release of a new feature, you can obtain the most recent 
version at www.panua.ch at any time.

****** LIMITED ACADEMIC LICENSE SUPPORT ******

Academic licenses come without any support.  This software is provided as 
is and no warranty is granted.  We make every attempt to provide software 
that links and runs properly, but we cannot provide individualized support 
for academic licenses and we make no guarantees on the performance of the 
algorithms.  You may report suspected bugs at www.panua.ch and we will 
attempt to fix actual errors in a reasonable response time.  Note, 
however, that improper use or call of the library is the cause of the 
issue in most of the cases.

Before purchasing a non-trial academic license, make sure that you have 
things running with the trial license.

****** BACKGROUND ******

Panua Ipopt is based on the primal-dual interior point method described in 
the paper

  Wächter, A., Biegler, L. On the implementation of an interior-point 
  filter line-search algorithm for large-scale nonlinear programming. 
  Math. Program. 106, 25–57 (2006). 

  https://doi.org/10.1007/s10107-004-0559-y

A technical report version of the paper can be obtained here:

  https://dominoweb.draco.res.ibm.com/reports/RC23149.pdf

We kindly ask you to cite this publication if you use Panua Ipopt in the 
preparation of your own publication.

Panua Ipopt builds on the COIN-OR open source project Ipopt available at 
https://github.com/coin-or/Ipopt.  The part of the Panua Ipopt software 
that is derived from the open source version is available on www.panua.ch.
It is released under the Eclipse Public License.

