#ifndef _BLAS_UTIL_H_
#define _BLAS_UTIL_H_

#ifdef __BLAS_LEGACY__
#include <math.h>
#include "cblas.h"
//#include <lapacke.h> //! hasn't used
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#elif defined __APPLE__
#include <Accelerate/Accelerate.h>

#elif defined __USE_MKL__
#include <mkl.h>

#endif


#endif /* _BLAS_UTIL_H_ */
