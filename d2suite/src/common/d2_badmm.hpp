#ifndef _D2_BADMM_H_
#define _D2_BADMM_H_


#include "common.hpp"
#include "d2.hpp"
#include "cblas.h"
#include "blas_like.h"
#include <random>
namespace d2 {
#define eps (1E-16)

  namespace internal {
    struct BADMMCache {
      real_t *C;
      real_t *Ctmp;
      real_t *Pi1;
      real_t *Pi2;
      real_t *Lambda;
      real_t *Ltmp;
      real_t *buffer;
      real_t *Pi_buffer;
    };

    template <typename ElemType1, typename ElemType2>
    void allocate_badmm_cache(const Block<ElemType1> &a, const ElemType2 &b,
			      BADMMCache &cache) {
      const size_t n = a.get_col() * b.len;
      cache.C      = new real_t[n];
      cache.Ctmp   = new real_t[n];
      cache.Pi1    = new real_t[n];
      cache.Pi2    = new real_t[n];
      cache.Lambda = new real_t[n];
      cache.Ltmp   = new real_t[n];
      cache.buffer = new real_t[n];
      cache.Pi_buffer = new real_t[n];
    }

    void deallocate_badmm_cache(BADMMCache &cache) {
      delete [] cache.C;
      delete [] cache.Ctmp;
      delete [] cache.Pi1;
      delete [] cache.Pi2;
      delete [] cache.Lambda;
      delete [] cache.Ltmp;
      delete [] cache.buffer;
      delete [] cache.Pi_buffer;
    }
  }

  /*! 
   * \brief the basic BADMM algorithmic iteration unit
   * \param a the first element
   * \param b the second element
   * \param cache the cached (latent) parameter associated with pair (a,b)
   * \param niter the number of intended iterations
   * \param prim_res the primal residual to be updated. If NULL, it implies no need to compute.
   * \param dual_res the dual residual to be updated. If NULL, it implies no need to compute.
   */
  template <typename ElemType1, typename ElemType2>
  int EMD_BADMM(const ElemType1 &a, const ElemType2 &b,
		const internal::BADMMCache &cache,
		const size_t niter,
		real_t *prim_res, real_t *dual_res) {
    const size_t mat_size = a.len * b.len;
    for (size_t i=0; i<mat_size; ++i)
      cache.Ctmp[i] = exp(cache.C[i]);
    for (size_t iter=0; iter < niter; ++iter) {
      if (dual_res && (iter+1 == niter)) {
	for (size_t i=0; i<mat_size;++i) {
	  cache.Pi_buffer[i] = cache.Pi2[i];
	}
      }

      for (size_t i=0; i<mat_size; ++i)
	cache.Ltmp[i] = exp(cache.Lambda[i]);
      for (size_t i=0; i<mat_size; ++i) {
	cache.Pi1[i] = cache.Pi2[i] / (cache.Ltmp[i]) + eps;
      }
      _D2_FUNC(rnorm)(a.len, b.len, cache.Pi1, cache.buffer);
      _D2_FUNC(gcms)(a.len, b.len, cache.Pi1, a.w);

      for (size_t i=0; i<mat_size; ++i) {	
	cache.Pi2[i] = cache.Pi1[i] * cache.Ltmp[i] / cache.Ctmp[i] + eps;
      }
      
      _D2_FUNC(cnorm)(a.len, b.len, cache.Pi2, cache.buffer);
      _D2_FUNC(grms)(a.len, b.len, cache.Pi2, b.w);

      if (dual_res && (iter+1 == niter)) {
	*dual_res = 0;
	for (size_t i=0; i<mat_size; ++i) {
	  real_t err=cache.Pi_buffer[i] - cache.Pi2[i];
	  *dual_res += fabs(err);
	}
      }
      
      for (size_t i=0; i<mat_size; ++i)
	cache.Lambda[i] += cache.Pi1[i] - cache.Pi2[i];
    }

    real_t Pi2_norm = 0.;
    if (prim_res || dual_res) {
      for (size_t i=0; i<mat_size; ++i)
	Pi2_norm += fabs(cache.Pi2[i]);
    }
    if (prim_res) {
      *prim_res = 0;
      for (size_t i=0; i<mat_size; ++i) {
	real_t err=(cache.Pi1[i] - cache.Pi2[i]);
	*prim_res += fabs(err);
      }
      *prim_res = (*prim_res) / (Pi2_norm);
    }
    if (dual_res) {
      *dual_res = (*dual_res) / (Pi2_norm);
    }
      

    return 0;
  }

}
#endif /* _D2_BADMM_H_ */
