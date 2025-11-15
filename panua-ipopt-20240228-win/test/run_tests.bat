
REM Copyright (C) Panua Technologies, 2023
@echo off

REM We need to set the path so that the Ipopt DLL will be found.
set PATH=%PATH%;..\lib

REM ******************************************************************
REM AMPL executable
REM ******************************************************************

echo.
echo Running AMPL example.  Output will be in ampl.log
..\bin\ipopt mytoy > ampl.log 2>&1
>nul findstr /c:"EXIT: Optimal" ampl.log && (
  echo ....Succeeded
) || (
  echo ....FAILED!
)

REM ******************************************************************
REM * C example
REM ******************************************************************
echo.
echo Compiling C example hs071_c.c
cl hs071_c.c -nologo -Ox -I..\include ..\lib\libipopt.lib > hs071_c.log 2>&1

echo Running hs071_c.  Output will be in hs071_c.log

hs071_c.exe >> hs071_c.log 2>&1

>nul findstr /c:"EXIT: Optimal" hs071_c.log && (
  echo ....Succeeded
) || (
  echo ....FAILED!
)
echo.

REM ******************************************************************
REM * C++ example
REM ******************************************************************
echo Compiling C++ example hs071_cpp.cpp
cl -o hs071_cpp.exe hs071_nlp.cpp hs071_main.cpp -nologo -Ox -EHcs -I..\include ..\lib\libipopt.lib > hs071_cpp.log 2>&1

echo Running hs071_cpp.  Output will be in hs071_cpp.log

hs071_cpp.exe >> hs071_cpp.log 2>&1

>nul findstr /c:"EXIT: Optimal" hs071_cpp.log && (
  echo ....Succeeded
) || (
  echo ....FAILED!
)
echo.

REM ******************************************************************
REM * Parallel example
REM ******************************************************************
echo.
echo Compiling parallel example
cl -o solve_example.exe MittelmannBndryCntrlDiri3D.cpp MittelmannBndryCntrlDiri3D_27.cpp MittelmannBndryCntrlDiri3Dsin.cpp RegisteredTNLP.cpp solve_problem.cpp -nologo -Ox -I..\include ..\lib\libipopt.lib > MBndryCntrl_3D_27.log 2>&1

echo Running "solve_example.exe MBndryCntrl_3D_27 20" with 2 cores.
echo Output will be in MBndryCntrl_3D_27.log

set OMP_NUM_THREADS=2
solve_example.exe MBndryCntrl_3D_27 20 >> MBndryCntrl_3D_27.log 2>&1

>nul findstr /c:"EXIT: Optimal" MBndryCntrl_3D_27.log && (
  echo ....Succeeded
) || (
  echo ....FAILED!
)
