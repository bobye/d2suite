#ifndef _MARRIAGE_LEARNER_H_
#define _MARRIAGE_LEARNER_H_

#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_badmm.hpp"

namespace d2 {
  template <typename ElemType1, typename FuncType, size_t dim>
  real_t ML_Predict_ByWinnerTakeAll(Block<ElemType1> &data,
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
	for (size_t j=1; j<FuncType::NUMBER_OF_CLASSES; ++j)
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
  real_t ML_Predict_ByVoting(Block<ElemType1> &data,
		    const Elem<def::Function<FuncType>, dim> &learner,
		    bool write_label = false) {
    real_t *y, *label_cache;
    y = new real_t[data.get_size()];
    label_cache = new real_t[data.get_col()];
    memcpy(label_cache, data.get_label_ptr(), sizeof(real_t) * data.get_col());
    for (size_t i=0; i<data.get_size(); ++i) y[i] = data[i].label[0];

    const size_t mat_size = data.get_col() * learner.len;
    real_t *C    = new real_t [mat_size * FuncType::NUMBER_OF_CLASSES];
    real_t *minC = new real_t [mat_size];
    real_t *Pi   = new real_t [mat_size];
    size_t *index= new size_t [mat_size];
    for (size_t i=0; i<FuncType::NUMBER_OF_CLASSES; ++i) {
      for (size_t j=0; j<data.get_col(); ++j) data.get_label_ptr()[j] = i;
      _pdist2(learner.supp, learner.len,
	      data.get_support_ptr(), data.get_label_ptr(), data.get_col(),
	      data.meta, C + mat_size * i);
    }
    for (size_t i=0; i<mat_size; ++i) {
      real_t minC_value = std::numeric_limits<real_t>::max();
      size_t minC_index = -1;
      for (size_t j=1; j<FuncType::NUMBER_OF_CLASSES; ++j) {
	if (minC_value > C[i+j*mat_size]) {
	  minC_value = C[i+j*mat_size];
	  minC_index = j;
	}
      }
      minC[i] = minC_value - C[i];
      index[i] = minC_index;
    }
    EMD(learner, data, NULL, minC, Pi, NULL, true);        

    real_t accuracy = 0.0;    
    if (write_label) {
      
    } else {
      real_t *Pi_ptr = Pi;
      size_t *index_ptr = index;
      for (size_t i=0; i<data.get_size(); ++i) {
	const size_t ms = learner.len * data[i].len;
	real_t thislabel[FuncType::NUMBER_OF_CLASSES] ={};
	for (size_t j=0; j<ms; ++j) {
	  thislabel[index_ptr[j]] += Pi_ptr[j];
	}
	index_ptr += ms;
	Pi_ptr    += ms;

	bool is_correct = true;
	const real_t w = thislabel[(size_t) y[i]];
	for (size_t j=1; j<FuncType::NUMBER_OF_CLASSES; ++j)
	  if (j!= (size_t) y[i]) {
	    is_correct &= (thislabel[j] < w);
	  }
	accuracy += is_correct;
      }
      accuracy /= data.get_size();
      //printf("accuracy: %.3lf\n", accuracy);
      memcpy(data.get_label_ptr(), label_cache, sizeof(real_t) * data.get_col());
      
    }

    
    delete [] y;
    delete [] label_cache;
    delete [] C;
    delete [] minC;
    delete [] index;
    delete [] Pi;
    
    return accuracy;    
  }
  
  template <typename ElemType1, typename FuncType, size_t dim>
  void ML_BADMM (Block<ElemType1> &data,
		 Elem<def::Function<FuncType>, dim> &learner,
		 const size_t max_iter,
		 const real_t rho = 2.0,
		 Block<ElemType1> *val_data = NULL, size_t val_size = 0) {    
    BADMMCache badmm_cache_arr;
    const real_t beta = 1./(learner.len - 1);
    assert(learner.len > 1);
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
    internal::get_dense_if_need_ec(data, &X);
    y = new real_t[data.get_col() * 2];
    for (size_t i=0; i<data.get_col(); ++i) y[i] = 0;
    memcpy(y+data.get_col(), data.get_label_ptr(), sizeof(real_t)*data.get_col());


    std::cout << "iter    " << "\t"
	      << "loss    " << "\t"
	      << "rho     " << "\t" 
	      << "prim_res" << "\t"
	      << "dual_res" << "\t"
	      << "tr_acc  " << "\t"
	      << "va_acc  " << std::endl;
    

    for (size_t iter=0; iter < max_iter; ++iter) {
      _pdist2(learner.supp, learner.len,
	      data.get_support_ptr(), data.get_label_ptr(), data.get_col(),
	      data.meta, badmm_cache_arr.C);

      _pdist2(learner.supp, learner.len,
	      data.get_support_ptr(), NULL, data.get_col(),
	      data.meta, badmm_cache_arr.Ctmp);

      for (size_t i=0; i<data.get_col() * learner.len; ++i)
	badmm_cache_arr.C[i] -= beta * badmm_cache_arr.Ctmp[i];      

      if (iter == 0) {
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
	EMD_BADMM(learner, data[i], badmm_cache_ptr, 50, &p_res, &d_res);

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
	real_t *sample_weight = new real_t[data.get_col() * 2];
	for (size_t ii=0; ii<data.get_col(); ++ii) sample_weight[ii] = 0;
	_D2_CBLAS_FUNC(axpy)(data.get_col(),
			     - beta,
			     badmm_cache_arr.Pi2 + i, learner.len,
			     sample_weight, 1);
	_D2_CBLAS_FUNC(axpy)(data.get_col(),
			     beta,
			     data.get_weight_ptr(), 1,
			     sample_weight, 1);
	_D2_CBLAS_FUNC(copy)(data.get_col(),
			     badmm_cache_arr.Pi2 + i, learner.len,
			     sample_weight + data.get_col(), 1);
	

	//learner.supp[i].init();
	int err_code = learner.supp[i].fit(X, y, sample_weight, data.get_col() * 2);
	assert(err_code >= 0);	
	delete [] sample_weight;
      }
      
      train_accuracy = ML_Predict_ByWinnerTakeAll(data, learner);
      printf("%zd\t\t%.6lf\t%.6lf\t\%.6lf\t%.6lf\t%.6lf\t", iter, loss, totalC * rho, prim_res, dual_res, train_accuracy);

      if (val_size > 0) {
	real_t validate_accuracy;
	for (size_t i=0; i<val_size; ++i) {
	  validate_accuracy = ML_Predict_ByWinnerTakeAll(val_data[i], learner);
	  printf("%.6lf\t", validate_accuracy);
	}	
      }
      std::cout << std::endl;
    }

    
    internal::release_dense_if_need_ec(data, &X);
    delete [] y;

    
    deallocate_badmm_cache(badmm_cache_arr);
  }
  
}

#endif /* _MARRIAGE_LEARNER_H_ */
