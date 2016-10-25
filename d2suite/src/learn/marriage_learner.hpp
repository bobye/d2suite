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
		 const size_t max_iter,
		 const real_t rho = 2.0) {    
    BADMMCache *badmm_cache_arr = new BADMMCache[data.get_size()];
    
    for (size_t i=0; i<data.get_size(); ++i)
      allocate_badmm_cache(data[i], learner, badmm_cache_arr[i]);


    for (size_t i=0; i<data.get_size(); ++i) {
      // initialize badmm_cache_arr[i].{Pi2, Lambda}
    }
    
    for (size_t iter=0; iter < max_iter; ++iter) {
      real_t totalC = 0;
      for (size_t i=0; i<data.get_size();++i) {
	_pdist2(data[i].supp, data[i].label, data[i].len,
		learner.supp, learner.len,
		data.meta, badmm_cache_arr[i].C);
	totalC += _D2_CBLAS_FUNC(asum)(data[i].len * learner.len, badmm_cache_arr[i].C, 1);
      }
      totalC /= (data.get_col() * learner.len);
      for (size_t i=0; i<data.get_size();++i) {
	for (size_t j=0; j<data[i].len * learner.len; ++j)
	  badmm_cache_arr[i].C[j] /= (totalC*rho);
      }
      for (size_t i=0; i<data.get_size();++i) {
	EMD_BADMM(data[i], learner, badmm_cache_arr[i],1);	
      }
    }
    
    for (size_t i=0; i<data.get_size(); ++i)
      deallocate_badmm_cache(badmm_cache_arr[i]);
    delete [] badmm_cache_arr;
  }
  
}

#endif /* _MARRIAGE_LEARNER_H_ */
