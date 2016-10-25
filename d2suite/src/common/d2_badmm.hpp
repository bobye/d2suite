#ifndef _D2_BADMM_H_
#define _D2_BADMM_H_


#include "common.hpp"
#include "d2.hpp"
#include "cblas.h"
#include "blas_like.h"
#include <random>
namespace d2 {
#define eps (1E-16)
  struct BADMMCache {
    real_t *C;
    real_t *Ctmp;
    real_t *Pi1;
    real_t *Pi2;
    real_t *Lambda;
    real_t *Ltmp;
    real_t *buffer;
  };

  template <typename ElemType1, typename ElemType2>
  void allocate_badmm_cache(const ElemType1 &a, const ElemType2 &b,
			    BADMMCache &cache) {
    const size_t n = a.len * b.len;
    const size_t m = a.len + b.len;
    cache.C      = new real_t[n];
    cache.Ctmp   = new real_t[n];
    cache.Pi1    = new real_t[n];
    cache.Pi2    = new real_t[n];
    cache.Lambda = new real_t[n];
    cache.Ltmp   = new real_t[n];
    cache.buffer = new real_t[m];
  }

  void deallocate_badmm_cache(BADMMCache &cache) {
    delete [] cache.C;
    delete [] cache.Ctmp;
    delete [] cache.Pi1;
    delete [] cache.Pi2;
    delete [] cache.Lambda;
    delete [] cache.Ltmp;
    delete [] cache.buffer;
  }


  template <typename ElemType1, typename ElemType2>
  int EMD_BADMM(const ElemType1 &a, const ElemType2 &b,
		const BADMMCache &cache,
		const size_t niter) {
    const size_t mat_size = a.len * b.len;
    for (size_t i=0; i<mat_size; ++i)
      cache.Ctmp[i] = exp(cache.C[i]);
    for (size_t iter=0; iter < niter; ++iter) {
      for (size_t i=0; i<mat_size; ++i)
	cache.Ltmp[i] = exp(cache.Lambda[i]);
      for (size_t i=0; i<mat_size; ++i)
	cache.Pi1[i] = cache.Pi2[i] / (cache.Ctmp[i] * cache.Ltmp[i]) + eps;
      _D2_FUNC(rnorm)(a.len, b.len, cache.Pi1, cache.buffer);
      _D2_FUNC(gcms)(a.len, b.len, cache.Pi1, a.w);
      for (size_t i=0; i<mat_size; ++i)
	cache.Pi2[i] = cache.Pi1[i] * cache.Ltmp[i] + eps;
      _D2_FUNC(cnorm)(a.len, b.len, cache.Pi2, cache.buffer);
      _D2_FUNC(grms)(a.len, b.len, cache.Pi2, b.w);
      for (size_t i=0; i<mat_size; ++i)
	cache.Lambda[i] = cache.Lambda[i] + cache.Pi1[i] - cache.Pi2[i];
    }
    return 0;
  }

}
#endif /* _D2_BADMM_H_ */
