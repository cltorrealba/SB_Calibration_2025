
	/**
	 *  Compressed sparse row matrix structure.
	 */

#define  mem_alloc(var, cnt, typ)        var = (typ *) _hh_malloc_laplace  ((cnt), sizeof(typ))
#define  mem_calloc(var, cnt, typ)       var = (typ *) _hh_calloc_laplace  ((cnt), sizeof(typ))


	typedef struct smat_struct {
		int    m;		/*!< Dimension rows. */
		int    n;		/*!< Dimension columns. */
		int    nnz;		/*!< Number of nonzero elements.  Equal to
					 * <tt>ia[m+1]</tt>. */
		int	*ia;		/*!< Index to rows in a and ja.  Has n+1 entries (length
					 * of last row). */
		int	*ja;		/*!< Column of each value in a. */
		double	*a;		/*!< Values. */
		int	 sym;		/*!< symmetric? */

		/* XXX only used in pardiso right now. */
		int	 is_complex;	/*!< complex? */

	} smat_laplace_t;
#define RE(c)                   ((double*) c)[0]
#define IM(c)                   ((double*) c)[1]

#define CPX_NEG(cv)                                             \
        (RE(cv) = -RE(cv),                                      \
         IM(cv) = -IM(cv))

#define CPX_ASSIGN(c1,c2)                                       \
        (RE(c1) =  RE(c2),                                      \
         IM(c1) =  IM(c2))

#define CPX_SET(c,re,im)                                        \
        (RE(c) = re,                                            \
         IM(c) = im)

#define CPX_NORM(a, b)                                          \
          a = RE(b) * RE(b) + IM(b) * IM(b);                    \
          a = sqrt(a) ;

#define CPX_ADDMUL(cv,a,b)                                      \
        (RE(cv) += RE(a) * RE(b) - IM(a) * IM(b),               \
         IM(cv) += RE(a) * IM(b) + IM(a) * RE(b))

#define CPX_MINUSMUL(cv,a,b)                                    \
        (RE(cv) -= RE(a) * RE(b) - IM(a) * IM(b),               \
         IM(cv) -= RE(a) * IM(b) + IM(a) * RE(b))

#define CPX_MUL(cv,a,b)                                         \
        (RE(cv)  = RE(a) * RE(b) - IM(a) * IM(b),               \
         IM(cv)  = RE(a) * IM(b) + IM(a) * RE(b))

#define CPX_ADD2(cv,a,b)                                                \
        (RE(cv) = RE(a) + RE(b),                                        \
         IM(cv) = IM(a) + IM(b))

#define CPX_ADD(a,b)                                            \
        (RE(a) += RE(b),                                        \
         IM(a) += IM(b))
#define CPX_MINUS(a,b)                                          \
        (RE(a) -= RE(b),                                        \
         IM(a) -= IM(b))

#define CPX_EQUAL(a,b)                                          \
        (RE(a) == RE(b),                                        \
         IM(a) == IM(b))

#define CPX_IS_ZERO(a)                                          \
        (RE(a) == 0.0  &&                                       \
         IM(a) == 0.0)

#define CPX_ADDDIV(c,a,b)                                               \
        if( fabs(RE(b)) <= fabs(IM(b)) )                        \
        {                                                       \
                float  __rat, __den;                             \
                assert (IM(b) != 0);                            \
                __rat = (float) RE(b) / IM(b) ;                  \
                __den = IM(b) * (1 + __rat*__rat);              \
                RE(c) += (RE(a)*__rat + IM(a)) / __den;         \
                IM(c) += (IM(a)*__rat - RE(a)) / __den;         \
        }                                                       \
        else                                                    \
        {                                                       \
                float  __rat, __den;                             \
                __rat = (float) IM(b) / RE(b) ;                  \
                __den = RE(b) * (1 + __rat*__rat);              \
                RE(c) += (RE(a) + IM(a)*__rat) / __den;         \
                IM(c) += (IM(a) - RE(a)*__rat) / __den;         \
        }

#define CPX_DIV(c,a,b)                                          \
        if( fabs(RE(b)) <= fabs(IM(b)) )                        \
        {                                                       \
                float  __rat, __den;                             \
                assert (IM(b) != 0);                            \
                __rat = (float) RE(b) / IM(b) ;                  \
                __den = IM(b) * (1 + __rat*__rat);              \
                RE(c) = (RE(a)*__rat + IM(a)) / __den;          \
                IM(c) = (IM(a)*__rat - RE(a)) / __den;          \
        }                                                       \
        else                                                    \
        {                                                       \
                float  __rat, __den;                             \
                __rat = (float) IM(b) / RE(b) ;                  \
                __den = RE(b) * (1 + __rat*__rat);              \
                RE(c) = (RE(a) + IM(a)*__rat) / __den;          \
                IM(c) = (IM(a) - RE(a)*__rat) / __den;          \
        }

/* Absolute value (modulus, complex norm) is put into `re', `im' is destroyed. */
#define CPX_ABS(a)                                              \
        if (RE(a) < 0)  RE(a) = -RE(a);                         \
        if (IM(a) < 0)  IM(a) = -IM(a);                         \
        if (IM(a) > RE(a))                                      \
        {                                                       \
                float  TMP__ = RE(a);                            \
                RE(a) = IM(a);                                  \
                IM(a) = TMP__;                                  \
        }                                                       \
        if ((RE(a)+IM(a)) != RE(a))                             \
        {                                                       \
                IM(a) = IM(a) / RE(a);                          \
                /* overflow!! */                                \
                RE(a) = RE(a) * sqrt(1.0 + IM(a) * IM(a));      \
        }
/* Complex norm is put into `mag'. */
#define CPX_ABS2(mag,re2,im2)                                   \
        do {                                                    \
                float  _re = fabs (re2);                         \
                float  _im = fabs (im2);                         \
                                                                \
                if (_im > _re)                                  \
                {                                               \
                        mag = _im;                              \
                        _im = _re;                              \
                        _re = mag;                              \
                }                                               \
                                                                \
                if ((_re+_im) == _re)                           \
                {                                               \
                        mag = _re;                              \
                }                                               \
                else                                            \
                {                                               \
                        mag = _im / _re;                        \
                        /* overflow!! */                        \
                        mag = _re * sqrt(1.0 + mag*mag);        \
                }                                               \
        } while (0)

/* `mag' is float norm of `z'.   root of `z' is put into `r'. */
#define CPX_SQRT(mag,r,z)                                       \
        CPX_ABS2 (mag, RE(z), IM(z));                           \
                                                                \
        if      (mag == 0.0)                                    \
        {                                                       \
                RE(r) = IM(r) = 0.0;                            \
        }                                                       \
        else if (RE(z) > 0)                                     \
        {                                                       \
                RE(r) = sqrt (0.5 * (mag + RE(z)));             \
                IM(r) = 0.5 * (IM(z) / RE(r));                  \
        }                                                       \
        else                                                    \
        {                                                       \
                float  _t = sqrt (0.5 * (mag - RE(z)));          \
                if (IM(z) < 0)                                  \
                        _t = -_t;                               \
                IM(r) = _t;                                     \
                RE(r) = 0.5 * (IM(z) / IM(r));                  \
        }

