#ifndef _MARRIAGE_LEARNER_H_
#define _MARRIAGE_LEARNER_H_

#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_badmm.hpp"

namespace d2 {
  template <typename ElemType1, typename FuncType, size_t dim>
  real_t ML_Predict(Block<ElemType1> &data,
		    const Elem<def::Function<FuncType>, dim> &learner,
		    bool write_label = false) {
    real_t *y, *label_cache;
    y = new real_t[data.get_size()];
    label_cache = new real_t[data.get_col()];
    memcpy(label_cache, data.get_label_ptr(), sizeof(real_t) * data.get_col());
    for (size_t i=0; i<data.get_size(); ++i) y[i] = data[i].label[0];

    
    real_t *emds = new real_t [data.get_size() * FuncType::NUMBER_OF_CLASSES];
    for (size_t i=0; i<FuncType::NUMBER_OF_CLASSES; ++i) {
      for (size_t j=0; j<data.get_col(); ++j) data.get_label_ptr()[j] = i;
      EMD(learner, data, emds + data.get_size() * i, NULL, NULL, NULL);
    }

    real_t accuracy = 0.0;    
    if (write_label) {
      
    } else {
      for (size_t i=0; i<data.get_size(); ++i) {
	bool is_correct = true;
	real_t *emd = emds + i;
	for (size_t j=0; j<FuncType::NUMBER_OF_CLASSES; ++j)
	  if (j != y[i]) {
	    is_correct &= emd[j*data.get_size()] > emd[(size_t)y[i]*data.get_size()];
	  }
	accuracy += is_correct;
      }
      accuracy /= data.get_size();
      //      printf("accuracy: %.3lf\n", accuracy);
      memcpy(data.get_label_ptr(), label_cache, sizeof(real_t) * data.get_col());
      
    }

    
    delete [] y;
    delete [] label_cache;
    delete [] emds;

    return accuracy;    
  }
  
  template <typename ElemType1, typename FuncType, size_t dim>
  void ML_BADMM (Block<ElemType1> &data,
		 Elem<def::Function<FuncType>, dim> &learner,
		 const size_t max_iter,
		 const real_t rho = 2.0) {    
    BADMMCache badmm_cache_arr;
    
    allocate_badmm_cache(data, learner, badmm_cache_arr);

    // initialization
    for (size_t j=0; j<data.get_col() * learner.len; ++j)
      badmm_cache_arr.Lambda[j] = 0;

    for (size_t k=0, l=0; k<data.get_col(); ++k)
      for (size_t j=0; j<learner.len; ++j, ++l) {
	badmm_cache_arr.Pi2[l] = data.get_weight_ptr()[k] * learner.w[j];
	badmm_cache_arr.Pi1[l] = badmm_cache_arr.Pi2[l];
      }

    real_t prim_res, dual_res, totalC;
    real_t *X, *y;
    internal::get_dense_if_need(data, &X);
    y = data.get_label_ptr();


    std::cout << "iter    " << "\t"
	      << "loss    " << "\t"
	      << "rho     " << "\t" 
	      << "prim_res" << "\t"
	      << "dual_res" << "\t"
	      << "tr_acc  " << std::endl;

    for (size_t iter=0; iter < max_iter; ++iter) {
      _pdist2(learner.supp, learner.len,
	      data.get_support_ptr(), data.get_label_ptr(), data.get_col(),
	      data.meta, badmm_cache_arr.C);

      if (iter == 0 || true) {
	totalC = _D2_CBLAS_FUNC(asum)(data.get_col() * learner.len, badmm_cache_arr.C, 1);
	totalC /= data.get_col() * learner.len;
      }
      _D2_CBLAS_FUNC(scal)(data.get_col() * learner.len, 1./ (rho*totalC), badmm_cache_arr.C, 1);

      prim_res = 0;
      dual_res = 0;
      BADMMCache badmm_cache_ptr = badmm_cache_arr;
      for (size_t i=0; i<data.get_size();++i) {
	const size_t matsize = data[i].len * learner.len;
	real_t p_res, d_res;
	EMD_BADMM(learner, data[i], badmm_cache_ptr, 20, &p_res, &d_res);

	badmm_cache_ptr.C += matsize;
	badmm_cache_ptr.Ctmp += matsize;
	badmm_cache_ptr.Pi1 += matsize;
	badmm_cache_ptr.Pi2 += matsize;
	badmm_cache_ptr.Lambda += matsize;
	badmm_cache_ptr.Ltmp += matsize;
	badmm_cache_ptr.buffer += matsize;
	badmm_cache_ptr.Pi_buffer += matsize;

	prim_res += p_res;
	dual_res += d_res;
      }
      prim_res /= data.get_size();
      dual_res /= data.get_size();

      real_t loss;
      real_t train_accuracy;
      loss = _D2_CBLAS_FUNC(dot)(data.get_col() * learner.len,
				 badmm_cache_arr.C, 1,
				 badmm_cache_arr.Pi1, 1);
      loss = loss / data.get_size() * totalC * rho;
      
      
      for (size_t i=0; i<learner.len; ++i) {
	real_t *sample_weight = new real_t[data.get_col()];
	_D2_CBLAS_FUNC(copy)(data.get_col(),
			     badmm_cache_arr.Pi1 + i, learner.len,
			     sample_weight, 1);
	//	learner.supp[i].init();
	int err_code = learner.supp[i].fit(X, y, sample_weight, data.get_col());
	assert(err_code >= 0);	
	delete [] sample_weight;
      }

      train_accuracy = ML_Predict(data, learner);
      printf("%zd\t\t%.6lf\t%.6lf\t\%.6lf\t%.6lf\t%.6lf\n", iter, loss, totalC * rho, prim_res, dual_res, train_accuracy);
    }

    
    internal::release_dense_if_need(data, &X);
    

    
    deallocate_badmm_cache(badmm_cache_arr);
  }
  
}

#endif /* _MARRIAGE_LEARNER_H_ */
