#ifndef _BLAS_UTIL_H_
#define _BLAS_UTIL_H_

#ifdef __BLAS_LEGACY__
#include <math.h>
#include "utils/cblas.h"
//#include <lapacke.h> //! hasn't used

#elif defined __APPLE__
#include <Accelerate/Accelerate.h>

#elif defined __USE_MKL__
#include <mkl.h>

#endif


#endif /* _BLAS_UTIL_H_ */
