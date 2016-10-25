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
  };

  template <typename ElemType1, typename ElemType2>
  void allocate_badmm_cache(const ElemType1 &a, const ElemType2 &b,
			    BADMMCache &cache) {
  }

  void deallocate_badmm_cache(BADMMCache &cache) {
  }


  template <typename ElemType1, typename ElemType2>
  int EMD_BADMM(const ElemType1 &a, const ElemType2 &b,
		const BADMMCache &cache,
		const size_t niter) {
    return 0;
  }

}
#endif /* _D2_BADMM_H_ */
