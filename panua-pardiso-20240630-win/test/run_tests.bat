@echo off

REM Set C++ compiler and compiler flags for MSVC (cl)
set CXX=cl
set CXX_FLAGS=/Ox /EHsc

REM Set C compiler and compiler flags for MSVC (cl)
set CC=cl
set CC_FLAGS=/Ox

REM Set Fortran compiler and compiler flags for Intel C Compiler (ifx)
set F77=ifx
set F77_FLAGS=-Ox

REM Base directory of the Panua-Pardiso installation
set PARDISO_INSTALL_DIR=..\

REM Pardiso library path
set PARDISO_LIB=%PARDISO_INSTALL_DIR%lib\libpardiso.lib

REM Set the PATH variable so that the DLL can be found
set PATH_BACKUP=%PATH%
set PATH=..\lib;%PATH%

REM Set the number of threads (otherwise Pardiso might throw error)
set OMP_NUM_THREADS=1

REM Executables that will be built for the tests
set EXES=pardiso_sym.exe pardiso_unsym.exe pardiso_unsym_complex.exe ^
    pardiso_sym_f.exe pardiso_unsym_f.exe pardiso_unsym_complex_f.exe ^
    pardiso_sym_schur.exe pardiso_unsym_complex_schur.exe pardiso_unsym_schur.exe ^
    laplace.exe helmholtz.exe

REM Include directories for compilation (contains header)
set INCFLAGS=-I%PARDISO_INSTALL_DIR%include

REM Log files for tests
set TESTLOGS=%EXES:.exe=.log)

REM Build all executables and run tests
call :build_and_test

REM Restore the original value of the PATH variable
set PATH=%PATH_BACKUP%

goto :eof

:build_and_test
REM Compile C source files
for %%f in (*.c) do (
    %CC% %CC_FLAGS% %INCFLAGS% %%f %PARDISO_LIB% -o %%~nf.exe
)

REM Compile C++ source files
for %%f in (*.cpp) do (
    %CXX% %CXX_FLAGS% %INCFLAGS% %%f %PARDISO_LIB% -o %%~nf.exe
)

REM Compile Fortran source files
for %%f in (*.f) do (
    %F77% %F77_FLAGS% %INCFLAGS% %%f %PARDISO_LIB% -o %%~nf.exe
)

REM Run tests
for %%e in (%EXES%) do (
    call :run_test %%~ne
)

REM Clean up object files and executables
del *.obj *.exe
goto :eof

:run_test
echo.
echo Testing executable %1. Output will be in %1.log
.\%1.exe >> %1.log 2>&1

findstr /c:"EXIT: Completed" %1.log >nul
if %errorlevel% equ 0 (
    echo ....Succeeded
) else (
    echo ....FAILED!
)

goto :eof
