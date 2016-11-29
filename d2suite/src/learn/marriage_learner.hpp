#ifndef _MARRIAGE_LEARNER_H_
#define _MARRIAGE_LEARNER_H_

#include <rabit/rabit.h>
#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_badmm.hpp"

namespace d2 {
  namespace def {
    /*! \brief all hyper parameters of ML_BADMM
     */
    struct ML_BADMM_PARAM {
      size_t max_iter = 100; ///< the maximum number of iterations
      size_t badmm_iter = 50;///< the number of iterations used in badmm per update
      real_t rho = 10.; ///< the BADMM parameter
      real_t beta = 1.; ///< the relative weight of non-above class
      size_t restart = -1; ///< the number of iterations fulfilled to restart BADMM; -1 means disabled
      real_t termination_tol = 5E-5;
      bool   bootstrap = false; ///< whether using bootstrap samples to initialize classifers
      bool   communicate = false;
    };
  }

  /*!
   * \brief The predicting utility of marriage learning using the winner-take-all method
   * \param data the data block to be predicted
   * \param learner the set of classifiers learnt via marriage learning
   * \param matchmaker the mm classifier learnt via marriage learning
   * \param write_label if false, compute the accuracy, otherwise overwrite labels to data
   */
  template <typename ElemType, typename LearnerType, typename MatchmakerType, size_t dim>
  real_t ML_Predict_ByWinnerTakeAll(Block<ElemType> &data,
				    const Elem<def::Function<LearnerType>, dim> &learner,
				    const MatchmakerType &matchmaker,
				    const def::ML_BADMM_PARAM &param,
				    bool write_label = false,
				    std::vector<real_t> *scores = NULL) {
    using namespace rabit;
    real_t *y;
    y = new real_t[data.get_size()];
    for (size_t i=0; i<data.get_size(); ++i) y[i] = data[i].label[0];

    real_t *C = new real_t [data.get_col() * learner.len];
    real_t *Ctmp = new real_t [data.get_col() * learner.len];
    real_t *emds;
    if (write_label && scores) {
      scores->resize(data.get_col() * learner.len);
      emds = &(*scores)[0];
    } else {
      assert((write_label && scores) || !write_label);
      emds = new real_t [data.get_size() * LearnerType::NUMBER_OF_CLASSES];
    }
    _pdist2_alllabel(&matchmaker, 1, data.get_support_ptr(), data.get_col(), data.meta, Ctmp);
    for (size_t i=0; i<LearnerType::NUMBER_OF_CLASSES; ++i) {
      _pdist2_label(learner.supp, learner.len,
		    data.get_support_ptr(), i, data.get_col(),
		    data.meta, C);
      for (size_t ii=0; ii<learner.len; ++ii)
	for (size_t jj=0; jj<data.get_col(); ++jj) {
	  C[ii + jj*learner.len] += param.beta * Ctmp[jj + ii*data.get_col()];
	}
      EMD(learner, data, emds + data.get_size() * i, C, NULL, NULL, true);
    }


    real_t accuracy = 0.0;    
    if (false) {
      
    } else {
      for (size_t i=0; i<data.get_size(); ++i) {
	real_t *emd = emds + i;
	int label = -1;
	real_t min_emd = std::numeric_limits<real_t>::max();
	for (size_t j=0; j<LearnerType::NUMBER_OF_CLASSES; ++j) {
	  if (emd[j*data.get_size()] < min_emd) {
	    min_emd = emd[j*data.get_size()];
	    label = j;
	  }
	}
	accuracy += label == (int) y[i];
      }
      size_t global_size = data.get_size();
      Allreduce<op::Sum>(&global_size, 1);
      Allreduce<op::Sum>(&accuracy, 1);
      accuracy /= global_size;      
    }

    
    delete [] y;
    delete [] C;
    delete [] Ctmp;
    if (!(write_label && scores))
      delete [] emds;
    
    return accuracy;    
  }

  /*!
   * \brief The predicting utility of marriage learning using the (multimarginal) voting method
   * \param data the data block to be predicted
   * \param learner the set of classifiers learnt via marriage learning
   * \param matchmaker the mm classifier learnt via marriage learning
   * \param write_label if false, compute the accuracy, otherwise overwrite labels to data
   */
  template <typename ElemType, typename LearnerType, typename MatchmakerType, size_t dim>
  real_t ML_Predict_ByVoting(Block<ElemType> &data,
			     const Elem<def::Function<LearnerType>, dim> &learner,
			     const MatchmakerType &matchmaker,
			     const def::ML_BADMM_PARAM &param,
			     bool write_label = false,
			     std::vector<real_t> *class_proportion = NULL) {
    using namespace rabit;
    if (write_label && class_proportion) {
      class_proportion->resize(data.get_size() * LearnerType::NUMBER_OF_CLASSES);
    } else {
      assert((write_label && class_proportion) || !write_label);
    }
    real_t *y, *label_cache;
    y = new real_t[data.get_size()];
    label_cache = new real_t[data.get_col()];
    memcpy(label_cache, data.get_label_ptr(), sizeof(real_t) * data.get_col());
    for (size_t i=0; i<data.get_size(); ++i) y[i] = data[i].label[0];

    const size_t mat_size = data.get_col() * learner.len;
    real_t *C    = new real_t [mat_size * LearnerType::NUMBER_OF_CLASSES];
    real_t *Ctmp = new real_t [mat_size];
    real_t *minC = new real_t [mat_size];
    real_t *Pi   = new real_t [mat_size];
    size_t *index= new size_t [mat_size];

    _pdist2_alllabel(learner.supp, learner.len,
		     data.get_support_ptr(), data.get_col(),
		     data.meta, C);
    _pdist2_alllabel(&matchmaker, 1,
		     data.get_support_ptr(), data.get_col(),
		     data.meta, Ctmp);

    for (size_t i=0; i<mat_size; ++i) {
      real_t minC_value = std::numeric_limits<real_t>::max();
      size_t minC_index = -1;
      real_t addC_value = param.beta * Ctmp[i/learner.len + (i%learner.len)*data.get_col()];
      for (size_t j=0; j<LearnerType::NUMBER_OF_CLASSES; ++j) {
	real_t actual_C = C[i+j*mat_size] + addC_value;
	if (minC_value > actual_C) {
	  minC_value = actual_C;
	  minC_index = j;
	}
      }
      minC[i] = minC_value;
      index[i] = minC_index;
    }
    EMD(learner, data, NULL, minC, Pi, NULL, true);        

    real_t accuracy = 0.0;    
    if (false) {
      
    } else {
      real_t *Pi_ptr = Pi;
      size_t *index_ptr = index;
      for (size_t i=0; i<data.get_size(); ++i) {
	const size_t ms = learner.len * data[i].len;
	real_t thislabel[LearnerType::NUMBER_OF_CLASSES] ={};
	for (size_t j=0; j<ms; ++j) {
	  thislabel[index_ptr[j]] += Pi_ptr[j];	  
	}
	index_ptr += ms;
	Pi_ptr    += ms;

	const real_t w = thislabel[(size_t) y[i]];
	size_t label = std::max_element(thislabel, thislabel+LearnerType::NUMBER_OF_CLASSES) - thislabel;
	if (write_label && class_proportion) {
	  memcpy(&class_proportion[i*LearnerType::NUMBER_OF_CLASSES], thislabel, sizeof(real_t) * LearnerType::NUMBER_OF_CLASSES);
	}
	accuracy += label == (size_t) y[i];
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
    delete [] Ctmp;
    delete [] minC;
    delete [] index;
    delete [] Pi;
    
    return accuracy;    
  }


#ifdef _USE_SPARSE_ACCELERATE_
  const static bool sparse = true; /* only possible for def::WordVec */
#endif

  namespace internal {
    template <typename ElemType>
    void _get_sample_weight(const Block<ElemType> &data,
			    const real_t *Pi, size_t leading,
			    real_t *sample_weight, size_t sample_size) {
      _D2_CBLAS_FUNC(copy)(data.get_col(),
			   Pi, leading,
			   sample_weight, 1);    
    }

    template <typename ElemType>
    void _get_sample_weight_mm(const Block<ElemType> &data,
			       const real_t *Pi, size_t leading,
			       real_t *sample_weight, size_t sample_size) {
      _D2_CBLAS_FUNC(copy)(sample_size,
			   Pi, 1,
			   sample_weight, 1);    
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
      }
      Allreduce<op::Sum>(sample_weight, sample_size);
    }

    template <size_t D>
    void _get_sample_weight_mm(const Block<Elem<def::WordVec, D> > &data,
			       const real_t *Pi, size_t leading,
			       real_t *sample_weight, size_t sample_size) {
      using namespace rabit;
      for (size_t ii=0; ii<sample_size; ++ii) sample_weight[ii] = 0;
      for (size_t ii=0; ii<data.get_col()*leading; ++ii) {
	real_t cur_w = Pi[ii];
	sample_weight[data.get_support_ptr()[ii / leading] +
		      data.meta.size * (ii % leading)] += cur_w;
      }
      Allreduce<op::Sum>(sample_weight, sample_size);
    }
#endif

    template <typename ElemType>
    size_t _get_sample_size(const Block<ElemType> &data, size_t num_of_copies)
    {return data.get_col();}
    
    template <typename ElemType>
    size_t _get_sample_size_mm(const Block<ElemType> &data, size_t num_of_copies)
    {return data.get_col() * num_of_copies;}

#ifdef _USE_SPARSE_ACCELERATE_
    template <size_t D>
    size_t _get_sample_size(const Block<Elem<def::WordVec, D> > &data, size_t num_of_copies)
    {return data.meta.size * num_of_copies;}

    template <size_t D>
    size_t _get_sample_size_mm(const Block<Elem<def::WordVec, D> > &data, size_t num_of_copies)
    {return data.meta.size * num_of_copies;}
#endif
  }


  /*!
   * \brief the marriage learning algorithm enabled by BADMM 
   * \param data a block of elements to train
   * \param learner a d2 element which has several classifiers of type FuncType
   * \param rho the BADMM parameter
   * \param val_data a vector of validation data
   */
  template <typename ElemType,
	    typename LearnerType,
	    typename PredictorType,
	    typename MatchmakerType,
	    size_t dim>
  void ML_BADMM (Block<ElemType> &data,
		 Elem<def::Function<LearnerType>, dim> &learner,
		 Elem<def::Function<PredictorType>, dim> &predictor,
		 MatchmakerType &matchmaker,
		 const def::ML_BADMM_PARAM &param,
		 std::vector<Block<ElemType>* > &val_data) {
    assert(MatchmakerType::NUMBER_OF_CLASSES == learner.len);
    using namespace rabit;
    // basic initialization
    matchmaker.init();
    matchmaker.sync(0);
    matchmaker.set_communicate(param.communicate);
    for (size_t i=0; i<learner.len; ++i) {
      learner.w[i] = 1. / learner.len;
      learner.supp[i].init();
      learner.supp[i].sync(i % rabit::GetWorldSize() );
      learner.supp[i].set_communicate(param.communicate);
      if (std::is_same<LearnerType, PredictorType>::value) {
	assert( (void *) &learner == (void *) &predictor);
      } else {
	predictor.w[i] = 1. / learner.len;
	predictor.supp[i].init();
	predictor.supp[i].sync(i % rabit::GetWorldSize() );
	predictor.supp[i].set_communicate(param.communicate);
      }
    }

    size_t global_col = data.get_col();
    size_t global_size= data.get_size();    

    Allreduce<op::Sum>(&global_col, 1);
    Allreduce<op::Sum>(&global_size, 1);


#ifdef _USE_SPARSE_ACCELERATE_
    for (size_t i=0; i<learner.len; ++i)
      learner.supp[i].set_communicate(false);
#endif
    
    internal::BADMMCache badmm_cache_arr;
    real_t rho = param.rho;
    assert(learner.len > 1);
    allocate_badmm_cache(learner, data, badmm_cache_arr);
    
    // initialization
    for (size_t j=0; j<data.get_col() * learner.len; ++j)
      badmm_cache_arr.Lambda[j] = 0;

    for (size_t k=0, l=0; k<data.get_col(); ++k)
      for (size_t j=0; j<learner.len; ++j, ++l) {
	badmm_cache_arr.Pi2[l] = data.get_weight_ptr()[k] * learner.w[j];
	badmm_cache_arr.Pi1[l] = badmm_cache_arr.Pi2[l];
      }

    real_t prim_res = 1., dual_res = 1., totalC = 0.;
    real_t old_prim_res, old_dual_res, old_totalC;
    real_t *X, *y; // dense data for training learner and predictor
    real_t *X_mm, *y_mm; // dense data for training matchmaker
#ifdef _USE_SPARSE_ACCELERATE_    
    internal::get_dense_if_need_mapped(data, &X, &y, LearnerType::NUMBER_OF_CLASSES);
    internal::get_dense_if_need_mapped(data, &X_mm, &y_mm, MatchmakerType::NUMBER_OF_CLASSES);
#else
    internal::get_dense_if_need(data, &X);
    internal::get_dense_if_need(data, &X_mm, MatchmakerType::NUMBER_OF_CLASSES);
    y = new real_t[data.get_col()];
    y_mm = new real_t[data.get_col() * MatchmakerType::NUMBER_OF_CLASSES];
    for (size_t i=0; i<data.get_col(); ++i) y[i] = data.get_label_ptr()[i];
    for (size_t i=0; i<data.get_col() * MatchmakerType::NUMBER_OF_CLASSES; ++i)
      y_mm[i] = i % MatchmakerType::NUMBER_OF_CLASSES;
#endif


    // initialize classifers using bootstrap samples
    if (param.bootstrap) {
      if (GetRank() == 0) {
	std::cout << "Initializing parameters using bootstrap samples ... " << std::endl;
      }  
      const size_t sample_size = internal::_get_sample_size(data, LearnerType::NUMBER_OF_CLASSES);
      real_t *bootstrap_weight = new real_t[sample_size];
      real_t *sample_weight = new real_t[sample_size];
      internal::_get_sample_weight(data, badmm_cache_arr.Pi2, learner.len, sample_weight, sample_size);
      for (size_t j=0, old_j=0; j<learner.len; ++j) {
	if (j % rabit::GetWorldSize() == rabit::GetRank()) {
	  std::random_device rd;
	  std::uniform_real_distribution<real_t>  unif(0., 1.);
	  std::mt19937 rnd_gen(rd());
	  for (size_t i=0; i<sample_size; ++i) {
	    bootstrap_weight[i] = unif(rnd_gen) * sample_weight[i];
	  }
#ifdef _USE_SPARSE_ACCELERATE_	  
	  learner.supp[j].fit(X, y, bootstrap_weight, sample_size, sparse);
#else
	  learner.supp[j].fit(X, y, bootstrap_weight, sample_size);	  
#endif
	}
	if ((j+1) % rabit::GetWorldSize() == 0 || j+1 == learner.len) {
	  for (size_t jj=old_j; jj <= j; ++jj) {
	    learner.supp[jj].sync(jj % rabit::GetWorldSize() );
	  }
	  old_j = j+1;
	}
      }
      delete [] sample_weight;
      delete [] bootstrap_weight;
    }
    
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
    for (size_t iter=0; iter < param.max_iter; ++iter) {
      /* ************************************************
       * compute cost matrix
       */
      _pdist2(learner.supp, learner.len,
	      data.get_support_ptr(), data.get_col(),
	      data.meta, badmm_cache_arr.C);

      if (iter > 0) {
	real_t *X_tmp;
	internal::get_dense_if_need(data, &X_tmp);
	matchmaker.evals_alllabel(X_tmp, data.get_col(), badmm_cache_arr.Ctmp,
				  MatchmakerType::NUMBER_OF_CLASSES, 1);
	internal::release_dense_if_need(data, &X_tmp);

	for (size_t i=0; i<data.get_col() * learner.len; ++i)
	  badmm_cache_arr.C[i] += param.beta * badmm_cache_arr.Ctmp[i];
      } else {
	for (size_t i=0; i<data.get_col() * learner.len; ++i)
	  badmm_cache_arr.C[i] -= param.beta * log(1./learner.len);
      }
      /* ************************************************
       * rescale badmm parameters
       */
      if (iter == 0 || true) {
	old_totalC = totalC * rho;
	if (prim_res < 0.5 *dual_res) { rho /=2;}
	if (dual_res < 0.5 *prim_res) { rho *=2;}
	totalC = _D2_CBLAS_FUNC(asum)(data.get_col() * learner.len, badmm_cache_arr.C, 1);
	Allreduce<op::Sum>(&totalC, 1);
	totalC /= global_col * learner.len;
      }
      if (param.restart > 0 && iter % param.restart == 0) {
	// restart badmm
	old_totalC = 0;
      }
      
      _D2_CBLAS_FUNC(scal)(data.get_col() * learner.len, 1./ (rho*totalC), badmm_cache_arr.C, 1);
      _D2_CBLAS_FUNC(scal)(data.get_col() * learner.len, old_totalC / (totalC * rho), badmm_cache_arr.Lambda, 1);

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
	  printf("\t%zd", iter);
	  printf("\t\t%.6lf\t%.6lf\t%.6lf\t%.6lf\t", loss, totalC * rho, prim_res, dual_res);
	  printf("%.3lf/", train_accuracy_1);
	  printf("%.3lf\t", train_accuracy_2);
	}
	if (val_data.size() > 0) {
	  for (size_t i=0; i<val_data.size(); ++i) {
	    validate_accuracy_1 = ML_Predict_ByWinnerTakeAll(*val_data[i], predictor, matchmaker, param);
	    validate_accuracy_2 = ML_Predict_ByVoting(*val_data[i], predictor, matchmaker, param);
	  }	
	}	
	if (GetRank() == 0) {
	  printf("%.3lf/", validate_accuracy_1);
	  printf("%.3lf", validate_accuracy_2);
	  std::cout << std::endl;
	}
	if (prim_res < param.termination_tol && dual_res < param.termination_tol) break;
      }

      /* ************************************************
       * start badmm iterations
       */
      old_prim_res = prim_res;
      old_dual_res = dual_res;
      //      while (prim_res >= old_prim_res || dual_res >= old_dual_res) {
      prim_res = 0;
      dual_res = 0;
      for (size_t badmm_ii = 0; badmm_ii < param.badmm_iter; ++badmm_ii) {
	internal::BADMMCache badmm_cache_ptr = badmm_cache_arr;
	for (size_t i=0; i<data.get_size();++i) {
	  const size_t matsize = data[i].len * learner.len;
	  real_t p_res, d_res;
	  EMD_BADMM(learner, data[i], badmm_cache_ptr, 1, &p_res, &d_res);

	  badmm_cache_ptr.C += matsize;
	  badmm_cache_ptr.Ctmp += matsize;
	  badmm_cache_ptr.Pi1 += matsize;
	  badmm_cache_ptr.Pi2 += matsize;
	  badmm_cache_ptr.Lambda += matsize;
	  badmm_cache_ptr.Ltmp += matsize;
	  badmm_cache_ptr.buffer += matsize;
	  badmm_cache_ptr.Pi_buffer += matsize;
	  badmm_cache_ptr.w_sync += learner.len;

	  prim_res += p_res;
	  dual_res += d_res;
	}

	// w update by rule 2
	for (size_t i=0; i<data.get_size() * learner.len; ++i) {
	  badmm_cache_arr.w_sync[i] = sqrt(badmm_cache_arr.w_sync[i]);
	}
	_D2_FUNC(rsum)(learner.len, data.get_size(), badmm_cache_arr.w_sync, learner.w);
	Allreduce<op::Sum>(learner.w, learner.len);
	real_t w_sum = 0;
	for (size_t i=0; i<learner.len; ++i) {
	  learner.w[i] = learner.w[i] * learner.w[i];
	  w_sum += learner.w[i];
	}
	for (size_t i=0; i<learner.len; ++i) {
	  learner.w[i] /= w_sum;
	  predictor.w[i] = learner.w[i];
	}
      }
      Allreduce<op::Sum>(&prim_res, 1);
      Allreduce<op::Sum>(&dual_res, 1);

      prim_res /= global_size * param.badmm_iter;
      dual_res /= global_size * param.badmm_iter;
      //      }

      /* ************************************************
       * re-fit classifiers (matchmaker)
       */
      const size_t sample_size_mm = internal::_get_sample_size_mm(data, MatchmakerType::NUMBER_OF_CLASSES);
      real_t *sample_weight_mm = new real_t[sample_size_mm];
      internal::_get_sample_weight_mm(data, badmm_cache_arr.Pi2, learner.len, sample_weight_mm, sample_size_mm);
      int err_code = 0;
#ifdef _USE_SPARSE_ACCELERATE_	
      err_code = matchmaker.fit(X_mm, y_mm, sample_weight_mm, sample_size_mm, sparse);
#else
      err_code = matchmaker.fit(X_mm, y_mm, sample_weight_mm, sample_size_mm);
#endif
      assert(err_code >= 0);
      delete [] sample_weight_mm;

      
      /* ************************************************
       * re-fit classifiers (learner and predictor)
       */
      const size_t sample_size = internal::_get_sample_size(data, LearnerType::NUMBER_OF_CLASSES);
#ifdef _USE_SPARSE_ACCELERATE_      
      real_t *sample_weight_local = new real_t[sample_size];
#endif
      for (size_t i=0, old_i=0; i<learner.len; ++i)
      {
	real_t *sample_weight = new real_t[sample_size];
	internal::_get_sample_weight(data, badmm_cache_arr.Pi2 + i, learner.len, sample_weight, sample_size);
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
	      if (iter == 0) learner.supp[ii].init();
	      err_code = learner.supp[ii].fit(X, y, sample_weight_local, sample_size, sparse);
	      assert(err_code >= 0);
	      if (!std::is_same<LearnerType, PredictorType>::value) {
		err_code = predictor.supp[ii].fit(X, y, sample_weight_local, sample_size, sparse);
		assert(err_code >= 0);
	      }
	    }
	  }

	  for (size_t ii=old_i; ii<=i; ++ii) {
	    learner.supp[ii].sync(ii % GetWorldSize());
	    if (!std::is_same<LearnerType, PredictorType>::value) {
	      predictor.supp[ii].sync(ii % GetWorldSize());
	    }
	  }
	  old_i = i+1;	  
	}
#else
	{
	  int err_code = 0;
	  err_code = learner.supp[i].fit(X, y, sample_weight, sample_size);
	  assert(err_code >= 0);
	  if (!std::is_same<LearnerType, PredictorType>::value) {
	    err_code = predictor.supp[i].fit(X, y, sample_weight, sample_size);
	    assert(err_code >= 0);
	  }
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
      
      
      train_accuracy_1 = ML_Predict_ByWinnerTakeAll(data, predictor, matchmaker, param);
      train_accuracy_2 = ML_Predict_ByVoting(data, predictor, matchmaker, param);

    }

    
#ifdef _USE_SPARSE_ACCELERATE_    
    internal::release_dense_if_need_mapped(data, &X, &y);
    internal::release_dense_if_need_mapped(data, &X_mm, &y_mm);
#else
    internal::release_dense_if_need(data, &X);
    internal::release_dense_if_need(data, &X_mm, learner.len);
    delete [] y;
    delete [] y_mm;
#endif
    
    deallocate_badmm_cache(badmm_cache_arr);
  }
  
}

#endif /* _MARRIAGE_LEARNER_H_ */
