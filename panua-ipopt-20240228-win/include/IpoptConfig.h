/* src/Common/config_ipopt.h.  Generated from config_ipopt.h.in by configure.  */
/* src/Common/config_ipopt.h.in. */

#ifndef __CONFIG_IPOPT_H__
#define __CONFIG_IPOPT_H__

/* Version number of project */
#define IPOPT_VERSION "Panua Ipopt 1.0 (2024-02-28)"

/* Define to the debug sanity check level (0 is no test) */
#define IPOPT_CHECKLEVEL 0

/* Define to the debug verbosity level (0 is no output) */
#define IPOPT_VERBOSITY 0

/* Define to 1 if using single precision floating point */
/* #undef IPOPT_SINGLE */

/* Define to 1 if Ipopt index type is int64_t */
/* #undef IPOPT_INT64 */

/* Library Visibility Attribute */
/* #define IPOPTAMPLINTERFACELIB_EXPORT __declspec(dllimport) */
#define IPOPTAMPLINTERFACELIB_EXPORT

/* Library Visibility Attribute */
#define IPOPTLIB_EXPORT __declspec(dllimport)

/* Library Visibility Attribute */
#define SIPOPTLIB_EXPORT __declspec(dllimport)

/** type corresponding to integers in Fortran
 * @deprecated Use ipindex instead.
 */
#define IPOPT_FORTRAN_INTEGER_TYPE ipindex

#endif
