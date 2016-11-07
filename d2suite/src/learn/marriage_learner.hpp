#ifndef _MARRIAGE_LEARNER_H_
#define _MARRIAGE_LEARNER_H_

#include <rabit/rabit.h>
#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_badmm.hpp"

namespace d2 {
  template <typename ElemType1, typename FuncType, size_t dim>
  real_t ML_Predict_ByWinnerTakeAll(Block<ElemType1> &data,
		    const Elem<def::Function<FuncType>, dim> &learner,
		    bool write_label = false) {
    using namespace rabit;
    real_t *y;
    y = new real_t[data.get_size()];
    for (size_t i=0; i<data.get_size(); ++i) y[i] = data[i].label[0];

    real_t *C = new real_t [data.get_col() * learner.len];
    real_t *emds = new real_t [data.get_size() * FuncType::NUMBER_OF_CLASSES];
    for (size_t i=0; i<FuncType::NUMBER_OF_CLASSES; ++i) {
      _pdist2_label(learner.supp, learner.len,
		    data.get_support_ptr(), i, data.get_col(),
		    data.meta, C);
      EMD(learner, data, emds + data.get_size() * i, C, NULL, NULL, true);
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
      size_t global_size = data.get_size();
      Allreduce<op::Sum>(&global_size, 1);
      Allreduce<op::Sum>(&accuracy, 1);
      accuracy /= global_size;      
    }

    
    delete [] y;
    delete [] C;
    delete [] emds;
    
    return accuracy;    
  }

  template <typename ElemType1, typename FuncType, size_t dim>
  real_t ML_Predict_ByVoting(Block<ElemType1> &data,
		    const Elem<def::Function<FuncType>, dim> &learner,
		    bool write_label = false) {
    using namespace rabit;

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

    _pdist2_alllabel(learner.supp, learner.len,
		     data.get_support_ptr(), data.get_col(),
		     data.meta, C);

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
      size_t global_size = data.get_size();

      Allreduce<op::Sum>(&global_size, 1);
      Allreduce<op::Sum>(&accuracy, 1);

      accuracy /= global_size;
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

  static real_t beta; /* an important parameter */

#ifdef _USE_SPARSE_ACCELERATE_
  const static bool sparse = true; /* only possible for def::WordVec */
#endif
  template <typename ElemType>
  void _get_sample_weight(const Block<ElemType> &data,
			  const real_t *Pi, size_t leading,
			  real_t *sample_weight, size_t sample_size) {
    for (size_t ii=0; ii<data.get_col(); ++ii) sample_weight[ii] = 0;
    _D2_CBLAS_FUNC(axpy)(data.get_col(),
			 - beta,
			 Pi, leading,
			 sample_weight, 1);
    _D2_CBLAS_FUNC(axpy)(data.get_col(),
			 beta,
			 data.get_weight_ptr(), 1,
			 sample_weight, 1);
    _D2_CBLAS_FUNC(copy)(data.get_col(),
			 Pi, leading,
			 sample_weight + data.get_col(), 1);    
  }

#ifdef _USE_SPARSE_ACCELERATE_
  template <size_t D>
  void _get_sample_weight(const Block<Elem<def::WordVec, D> > &data,
			  const real_t *Pi, size_t leading,
			  real_t *sample_weight, size_t sample_size) {
    using namespace rabit;
    for (size_t ii=0; ii<sample_size; ++ii) sample_weight[ii] = 0;
    for (size_t ii=0; ii<data.get_col(); ++ii) {
      real_t cur_w = Pi[ii*leading];
      sample_weight[data.get_support_ptr()[ii] +
		    data.meta.size * (size_t) data.get_label_ptr()[ii]] += cur_w;
      sample_weight[data.get_support_ptr()[ii]] += beta * (data.get_weight_ptr()[ii] - cur_w);
    }
    Allreduce<op::Sum>(sample_weight, sample_size);
  }
#endif

  template <typename ElemType>
  size_t _get_sample_size(const Block<ElemType> &data, size_t num_of_copies)
  {return data.get_col() * 2;}

#ifdef _USE_SPARSE_ACCELERATE_
  template <size_t D>
  size_t _get_sample_size(const Block<Elem<def::WordVec, D> > &data, size_t num_of_copies)
  {return data.meta.size * num_of_copies;}
#endif

  template <typename ElemType1, typename FuncType, size_t dim>
  void ML_BADMM (Block<ElemType1> &data,
		 Elem<def::Function<FuncType>, dim> &learner,
		 const size_t max_iter,
		 const real_t rho = 2.0,
		 Block<ElemType1> *val_data = NULL, size_t val_size = 0) {    
    using namespace rabit;

    size_t global_col = data.get_col();
    size_t global_size= data.get_size();    

    Allreduce<op::Sum>(&global_col, 1);
    Allreduce<op::Sum>(&global_size, 1);


#ifdef _USE_SPARSE_ACCELERATE_
    for (size_t i=0; i<learner.len; ++i)
      learner.supp[i].set_communicate(false);
#endif
    
    BADMMCache badmm_cache_arr;
    beta = 1./(learner.len - 1);
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

    real_t prim_res, dual_res, totalC = 0.;
    real_t *X, *y;

#ifdef _USE_SPARSE_ACCELERATE_    
    internal::get_dense_if_need_mapped(data, &X, &y, FuncType::NUMBER_OF_CLASSES);
#else
    internal::get_dense_if_need_ec(data, &X, &y);
#endif

    if (GetRank() == 0) {
      std::cout << "\t"
		<< "iter    " << "\t"
		<< "loss    " << "\t"
		<< "rho     " << "\t" 
		<< "prim_res" << "\t"
		<< "dual_res" << "\t"
		<< "tr_acc  " << "\t"
		<< "va_acc  " << std::endl;
    }    

    real_t loss;
    real_t train_accuracy_1, train_accuracy_2, validate_accuracy_1, validate_accuracy_2;
    real_t old_totalC;
    for (size_t iter=0; iter < max_iter; ++iter) {
      /* ************************************************
       * compute cost matrix
       */

      _pdist2(learner.supp, learner.len,
	      data.get_support_ptr(), data.get_col(),
	      data.meta, badmm_cache_arr.C);

      _pdist2_label(learner.supp, learner.len,
		    data.get_support_ptr(), (real_t) 0, data.get_col(),
		    data.meta, badmm_cache_arr.Ctmp);

      for (size_t i=0; i<data.get_col() * learner.len; ++i)
	badmm_cache_arr.C[i] -= beta * badmm_cache_arr.Ctmp[i];      

      
      /* ************************************************
       * rescale badmm parameters
       */
      if (iter == 0 || true) {
	old_totalC = totalC;
	totalC = _D2_CBLAS_FUNC(asum)(data.get_col() * learner.len, badmm_cache_arr.C, 1);
	Allreduce<op::Sum>(&totalC, 1);
	totalC /= global_col * learner.len;
      }
      if (iter % 10 == 0) {
	// restart badmm
	// old_totalC = 0;
      }
      _D2_CBLAS_FUNC(scal)(data.get_col() * learner.len, 1./ (rho*totalC), badmm_cache_arr.C, 1);
      _D2_CBLAS_FUNC(scal)(data.get_col() * learner.len, old_totalC / totalC, badmm_cache_arr.Lambda, 1);

      /* ************************************************
       * compute current loss
       */
      loss = _D2_CBLAS_FUNC(dot)(data.get_col() * learner.len,
				 badmm_cache_arr.C, 1,
				 badmm_cache_arr.Pi2, 1);
      Allreduce<op::Sum>(&loss, 1);
      loss = loss / global_size * totalC * rho;


      /* ************************************************
       * print status informations 
       */
      if (iter > 0) {
	if (GetRank() == 0) {
	  printf("\t%zd\t\t%.6lf\t%.6lf\t%.6lf\t%.6lf\t", iter, loss, totalC * rho, prim_res, dual_res);
	  printf("%.3lf/", train_accuracy_1);
	  printf("%.3lf\t", train_accuracy_2);
	}
	if (val_size > 0) {
	  for (size_t i=0; i<val_size; ++i) {
	    validate_accuracy_1 = ML_Predict_ByWinnerTakeAll(val_data[i], learner);
	    validate_accuracy_2 = ML_Predict_ByVoting(val_data[i], learner);
	  }	
	}	
	if (GetRank() == 0) {
	  printf("%.3lf/", validate_accuracy_1);
	  printf("%.3lf", validate_accuracy_2);
	  std::cout << std::endl;
	}
      }

      /* ************************************************
       * start badmm iterations
       */
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

      Allreduce<op::Sum>(&prim_res, 1);
      Allreduce<op::Sum>(&dual_res, 1);

      prim_res /= global_size;
      dual_res /= global_size;

      
      /* ************************************************
       * re-fit classifiers
       */
      const size_t sample_size = _get_sample_size(data, FuncType::NUMBER_OF_CLASSES);
#ifdef _USE_SPARSE_ACCELERATE_      
      real_t *sample_weight_local = new real_t[sample_size];
#endif
      for (size_t i=0, old_i=0; i<learner.len; ++i)
      {
	real_t *sample_weight = new real_t[sample_size];
	_get_sample_weight(data, badmm_cache_arr.Pi2 + i, learner.len, sample_weight, sample_size);
	//learner.supp[i].init();
#ifdef _USE_SPARSE_ACCELERATE_      
	if (i % GetWorldSize() == GetRank())
	{
	  std::memcpy(sample_weight_local, sample_weight, sizeof(real_t) * sample_size);
	}
#endif

#ifdef _USE_SPARSE_ACCELERATE_	
	if ((i+1) % GetWorldSize() == 0 || i+1==learner.len) 
	{
	  for (size_t ii=old_i; ii<=i; ++ii) {
	    if (ii % GetWorldSize() == GetRank())
	    {
	      int err_code = 0;
	      err_code = learner.supp[ii].fit(X, y, sample_weight_local, sample_size, sparse);
	      assert(err_code >= 0);
	    }
	  }

	  for (size_t ii=old_i; ii<=i; ++ii) {
	    Broadcast(learner.supp[ii].get_coeff(),
		      learner.supp[ii].get_coeff_size() * sizeof(real_t),
		      ii % GetWorldSize() );
	  }
	  old_i = i+1;	  
	}
#else
	{
	  int err_code = 0;
	  err_code = learner.supp[i].fit(X, y, sample_weight, sample_size);
	  assert(err_code >= 0);
	}	
#endif	


	if (GetRank() == 0)
	{
	  printf("\b\b\b\b\b\b\b%3zd/%3zd", (i+1), learner.len);
	  fflush(stdout);
	}	  	
	delete [] sample_weight;

      }
#ifdef _USE_SPARSE_ACCELERATE_      
      delete [] sample_weight_local;
#endif
      Barrier();
      
      

      train_accuracy_1 = ML_Predict_ByWinnerTakeAll(data, learner);
      train_accuracy_2 = ML_Predict_ByVoting(data, learner);

    }

    
#ifdef _USE_SPARSE_ACCELERATE_    
    internal::release_dense_if_need_mapped(data, &X, &y);
#else
    internal::release_dense_if_need_ec(data, &X, &y);
#endif
    
    deallocate_badmm_cache(badmm_cache_arr);
  }
  
}

#endif /* _MARRIAGE_LEARNER_H_ */
