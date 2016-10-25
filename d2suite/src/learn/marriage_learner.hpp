#ifndef _MARRIAGE_LEARNER_H_
#define _MARRIAGE_LEARNER_H_

#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_badmm.hpp"

namespace d2 {
  template <typename ElemType1, typename FuncType, size_t dim>
  void ML_BADMM (Block<ElemType1> &data,
		 Elem<def::Function<FuncType>, dim> &learner,
		 const size_t max_iter) {
    BADMMCache *badmm_cache_arr = new BADMMCache[data.get_size()];
    for (size_t i=0; i<data.get_size(); ++i)
      allocate_badmm_cache(data[i], learner, badmm_cache_arr[i]);

    for (size_t iter=0; iter < max_iter; ++iter) {
      for (size_t i=0; i<data.get_size();++i)
	EMD_BADMM(data[i], learner, badmm_cache_arr[i],1);
    }
    
    for (size_t i=0; i<data.get_size(); ++i)
      deallocate_badmm_cache(badmm_cache_arr[i]);
    delete [] badmm_cache_arr;
  }
  
}

#endif /* _MARRIAGE_LEARNER_H_ */
