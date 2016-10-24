#ifndef _WM3_H_
#define _WM3_H_
/* Wasserstein Mixed Membership Model (or Wasserstein NMF) */

#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_sa.hpp"

namespace d2 {

#define eps (1E-16)
  
  template <typename ElemType1, typename ElemType2>
  void WM3_SA (const Block<ElemType1> &model,
	       const Block<ElemType2> &data,
	       const size_t max_epoch,
	       const real_t initT,
	       real_t sigma,
	       real_t gamma,
	       size_t batch_size = 20) {

    size_t K=model.get_size();
    size_t m=model[0].len;
    size_t n=data.get_size();
    size_t global_n = n;    
    bool isGradUse = false;

#ifdef RABIT_RABIT_H_
    using namespace rabit;
    Allreduce<op::Sum>(&global_n, 1);
#endif


    const size_t tau = 5, E=20;
    real_t T=initT, A=0., B=0., D=0., bound;
    Block<ElemType1> mixture_data(n, m);
    mixture_data.initialize(data.get_size(), m);

    SACache sac, sac_b;
    allocate_sa_cache(mixture_data, data, sac, true);
    
    std::vector< Block<ElemType1> * > mbatch;
    std::vector< const Block<ElemType2> * > dbatch;
    for (size_t i=0; i*batch_size < n; ++i) {
      mbatch.push_back(new Block<ElemType1>(mixture_data, i*batch_size, batch_size));
      dbatch.push_back(new const Block<ElemType2>(data, i*batch_size, batch_size));
    }
    // one epoch
#ifdef RABIT_RABIT_H_
    if (GetRank() == 0) 
#endif
    {
    std::cout << getLogHeader() << " logging: start epoch." << std::endl;
    std::cout << getLogHeader()
	      << "\titer"
	      << "\tNesterov"
	      << "\tapprox_primal"
	      << "\tdual_bound"
	      << "\tdual" 
	      << "\t\tt"
	      << "\tgap" << std::endl;
    }
    real_t obj_old=0, obj=0, primal_obj, db_obj, dual_obj, max_obj;
    real_t *gd; //, *md;
    gd = (real_t*) malloc(sizeof(real_t)*std::max(std::max(m,batch_size)*K,n));

    // initialize membership vectors uniformly
    real_t *beta = (real_t*) malloc(sizeof(real_t) * K * n);
    real_t *betaz= (real_t*) malloc(sizeof(real_t) * K * n);
    for (size_t i=0; i<K*n; ++i) beta[i] = rand()%100+1;   
    _D2_FUNC(cnorm)(K, n, beta, gd);
    for (size_t i=0; i<K*n; ++i) betaz[i] = beta[i];
    
    internal::_pdist2(mixture_data.get_support_ptr(), m, data.get_support_ptr(), data.get_col(), data.meta, sac._m);
    real_t mC;
#ifndef RABIT_RABIT_H_    
    mC = _D2_CBLAS_FUNC(asum)(data.get_col() * m, sac._m, 1) / (data.get_col() * m);
#else
    mC = _D2_CBLAS_FUNC(asum)(data.get_col() * m, sac._m, 1);
    Allreduce<op::Sum>(&mC, 1);
    mC /= (data.get_col() * m) * rabit::GetWorldSize();
#endif
    int avg_iterations = 0;
    for (size_t iter=0, accelerator=1; iter < max_epoch; ++iter) {  
      sac_b = sac;
      real_t r = 3;
      real_t lambda = r/(r+accelerator);

      if (accelerator == 20) {
	for (size_t i=0; i<K*n; ++i) betaz[i] = beta[i];
	accelerator=0;
      }
      if (isGradUse) {	  
	_D2_CBLAS_FUNC(scal)(K*n, (1-lambda), beta, 1);
	_D2_CBLAS_FUNC(axpy)(K*n, lambda, betaz, 1, beta, 1);
      }
      
      for (size_t i=0; i*batch_size < n; ++i) {	
	real_t *thisbeta = beta + i*batch_size*K;
	real_t *thisbetaz=betaz + i*batch_size*K;
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
			     m, batch_size, K,
			     1.0,
			     model.get_weight_ptr(), m,
			     thisbeta, K,
			     0.0,
			     mbatch[i]->get_weight_ptr(), m);    
	real_t batchA, batchB, batchD;
	if (iter  == 0 || (iter+1) % E == 0 )
	  avg_iterations += EMD_SA(*mbatch[i], *dbatch[i], T, tau, sac_b, batchA, batchB, batchD, true);
	else
	  avg_iterations += EMD_SA(*mbatch[i], *dbatch[i], T, tau, sac_b, batchA, batchB, batchD, false);	  
	A+=batchA; B+=batchB; D+=batchD;
#ifdef RABIT_RABIT_H_
    Allreduce<op::Sum>(&avg_iterations, 1);	
    Allreduce<op::Sum>(&A, 1);
    Allreduce<op::Sum>(&B, 1);
    Allreduce<op::Sum>(&D, 1);	
#endif

	if (iter > 0) {
	  if (i==0) {++accelerator; isGradUse = true;}
	  _D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
			       m, K, batch_size, 
			       1.0,
			       sac_b._U, m,
			       thisbeta, K,
			       0.0,
			       gd, m);

	  real_t *w=model.get_weight_ptr();
	  _D2_CBLAS_FUNC(scal)(m*K, -gamma/mC, gd, 1);
#ifdef RABIT_RABIT_H_
	  Allreduce<op::Sum>(gd, m*K);
#endif	  
	  for (size_t j=0; j<m*K; ++j) {w[j] = w[j] * exp(gd[j]) + eps;}
	  _D2_FUNC(cnorm)(m, K, w, gd);

	  _D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasTrans, CblasNoTrans,
			       K, batch_size, m,
			       - gamma /mC,
			       model.get_weight_ptr(), m,
			       sac_b._U, m,
			       0.0,
			       gd, K);
	  for (size_t j=0; j<K*batch_size; ++j) {
	    thisbeta[j] = thisbeta[j] * exp(gd[j] * r) + eps;
	    thisbetaz[j] = thisbetaz[j] * exp(accelerator * gd[j] / r) + eps;
	  }
	  _D2_FUNC(cnorm)(K, batch_size, thisbeta, gd);
	  _D2_FUNC(cnorm)(K, batch_size, thisbetaz, gd);
	} else {
	  if (i==0) {isGradUse = false;}
	}
	
	for (size_t j=0; j<batch_size; ++j) {
	  size_t mat_size=(*mbatch[i])[j].len * (*dbatch[i])[j].len;
	  sac_b._m    += mat_size;
	  sac_b._mtmp += mat_size;
	  if (sac._primal) sac_b._primal += mat_size;
	}
	
	sac_b._dual1 += mbatch[i]->get_col();
	sac_b._dual2 += dbatch[i]->get_col();
	sac_b._U += mbatch[i]->get_col();
	sac_b._L += dbatch[i]->get_col();
      }
      
      if (iter % E == 0) {
	real_t *emds, *cache_mat;
	emds = (real_t*) malloc(sizeof(real_t)*n);
	cache_mat = (real_t*) malloc(sizeof(real_t)*m*data.get_max_len());
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
			     m, n, K,
			     1.0,
			     model.get_weight_ptr(), m,
			     beta, K,
			     0.0,
			     mixture_data.get_weight_ptr(), m);    
	for (size_t i=0; i<n; ++i) {    
	  emds[i]=EMD(mixture_data[i], data[i], data.meta, cache_mat);
	}
	obj_old=obj;
	obj = _D2_CBLAS_FUNC(asum)(n, emds, 1);
#ifdef RABIT_RABIT_H_
	Allreduce<op::Sum>(&obj, 1);
#endif

#ifdef RABIT_RABIT_H_	
	if (GetRank() == 0)
#endif
	std::cout << "@obj\t" << obj / global_n << std::endl;
	free(cache_mat);
	free(emds);	
	//model.write("data/orl/mixture_" + std::to_string(K) + "_" + std::to_string(iter) + ".txt");
      }
      

      dual_obj = _D2_CBLAS_FUNC(dot)(n*m, sac._dual1, 1, mixture_data.get_weight_ptr(), 1) - _D2_CBLAS_FUNC(dot)(data.get_col(), sac._dual2, 1, data.get_weight_ptr(), 1);
      db_obj = _D2_CBLAS_FUNC(dot)(n*m, sac._U, 1, mixture_data.get_weight_ptr(), 1) - _D2_CBLAS_FUNC(dot)(data.get_col(), sac._L, 1, data.get_weight_ptr(), 1);
#ifdef RABIT_RABIT_H_
      Allreduce<op::Sum>(&dual_obj, 1);      
      Allreduce<op::Sum>(&db_obj, 1);      
#endif
      //primal_obj = _D2_CBLAS_FUNC(dot)(m*data.get_col(), sac._primal, 1, sac._m, 1);

      //if (dual_obj < 0.1 * db_obj && db_obj > 0.5 * obj)
      if (db_obj > 0.9 * obj)	
	T*=1-1./sqrt(data.get_col()/n + m); //if (T < 0.001) T=0.001;
      

      if (iter  == 0 || (iter+1) % E == 0 ) {
	avg_iterations /= batch_size;
	bound = (obj - db_obj + avg_iterations * A) / (data.get_col() + m*global_n + avg_iterations * B);
	T= std::min(T, bound);
      }
      A=0.;B=0.;

#ifdef RABIT_RABIT_H_
      if (GetRank() == 0)
#endif
      std::cout << getLogHeader() << "\t" << iter
		<< "\t" << accelerator
		<< "\t" << primal_obj / global_n
		<< "\t" << db_obj / global_n
		<< "\t" << dual_obj / global_n
		<< "\t" << T << "\t" << bound 
		<< "\t" << (primal_obj - dual_obj)/global_n << std::endl;
      
    }

    for (size_t i=0; i*batch_size < n; ++i) {
      delete mbatch[i];
      delete dbatch[i];
    }

    free(beta);
    free(gd);
    //    free(md);
    deallocate_sa_cache(sac);
    //mixture_data.write("data/orl/estimate.d2s");    

  }
  
}


#endif /* _WM3_H_ */
