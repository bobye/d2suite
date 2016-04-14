#ifndef _D2_SA_H_
#define _D2_SA_H_

#include "common.hpp"
#include "d2.hpp"
#include "cblas.h"
#include "blas_like.h"
#include <random>
namespace d2 {
  
  /*!
   * SA Gibbs sampler with n iterations
   */
#define eps (1E-10)

  struct SACache {
    real_t *_m;
    real_t *_mtmp;
    real_t *_primal;
    real_t *_dual1;
    real_t *_dual2;
    real_t *_U;
    real_t *_L;
  };

  template <typename ElemType1, typename ElemType2>
  void allocate_sa_cache(const Block<ElemType1> &a,
			 const Block<ElemType2> &b,
			 SACache &sac,
			 const bool hasPrimal = false) {
    assert(a.get_size() == b.get_size());
    size_t max_size = 0;
    for (index_t i=0; i<a.get_size(); ++i) max_size += a[i].len * b[i].len;
    sac._m = (real_t*) malloc(sizeof(real_t) * max_size);
    sac._mtmp = (real_t*) malloc(sizeof(real_t) * max_size);
    if (hasPrimal) {
      sac._primal = (real_t*) malloc(sizeof(real_t) * max_size);
    } else {
      sac._primal = NULL;
    }
    sac._dual1 = (real_t*) calloc(a.get_col(), sizeof(real_t));
    sac._dual2 = (real_t*) calloc(b.get_col(), sizeof(real_t));
    sac._U = (real_t*) malloc(sizeof(real_t) * a.get_col());
    sac._L = (real_t*) malloc(sizeof(real_t) * b.get_col());
  }

  void deallocate_sa_cache(SACache &sac) {
    free(sac._m);
    free(sac._mtmp);
    if (sac._primal) free(sac._primal);
    free(sac._dual1);
    free(sac._dual2);
    free(sac._U);
    free(sac._L);
  }  

  
  template <typename ElemType1, typename ElemType2>
  void EMD_SA (const Block<ElemType1> &a, const Block<ElemType2> &b,
	       real_t &T, const real_t sigma,
	       const size_t niter,
	       const SACache &sac) {
    assert(sac._m && sac._mtmp);
    assert(sac._dual1 && sac._dual2);
    assert(sac._U && sac._L);
    assert(sigma>0 && sigma<1 && T>0);

    std::random_device rd;
    std::exponential_distribution<> rng (1);
    std::mt19937 rnd_gen (rd ());
    
    for (int iter=0; iter < niter; ++iter) {
      real_t *dual1 = sac._dual1;
      real_t *dual2 = sac._dual2;
      real_t *U = sac._U;
      real_t *L = sac._L;
      real_t *M = sac._m;
      real_t *Mtmp = sac._mtmp;
      for (index_t i=0; i < a.get_size(); ++i) {
	// calculate U and sample dual1
	size_t m1=a[i].len;
	size_t m2=b[i].len;
	size_t mat_size=m1*m2;
	_D2_CBLAS_FUNC(copy)(mat_size, M, 1, Mtmp, 1);      
	_D2_FUNC(grmv)(m1, m2, Mtmp, dual2);
	_D2_FUNC(rmin)(m1, m2, Mtmp, U);
	for (index_t j=0; j < m1; ++j) {
	  dual1[j] = U[j] - rng(rnd_gen) * T / (a[i].w[j] + eps);
	}
	// calculate L and sample dual2
	_D2_CBLAS_FUNC(copy)(mat_size, M, 1, Mtmp, 1);      
	for (index_t j=0; j< mat_size; ++j) Mtmp[j] = - Mtmp[j];
	_D2_FUNC(gcmv)(m1, m2, Mtmp, dual1);
	_D2_FUNC(cmax)(m1, m2, Mtmp, L);
	for (index_t j=0; j< m2; ++j) {
	  dual2[j] = L[j] + rng(rnd_gen) * T / (b[i].w[j] + eps);
	}

	M = M + mat_size;
	Mtmp = Mtmp + mat_size;
	dual1 = dual1 + m1;
	dual2 = dual2 + m2;
	U = U + m1;
	L = L + m2;
      }
      T *= sigma;
    }
    if (sac._primal) {
      real_t *primal=sac._primal;
      real_t *U = sac._U;
      real_t *L = sac._L;
      real_t *M = sac._m;
      real_t *Mtmp = sac._mtmp;
      for (index_t i=0; i < b.get_size(); ++i) {
	// calculate U and sample dual1
	size_t m1=a[i].len;
	size_t m2=b[i].len;
	size_t mat_size=m1*m2;
	for (index_t i=0; i< mat_size; ++i) primal[i]=0;

	_D2_CBLAS_FUNC(copy)(mat_size, M, 1, Mtmp, 1);      
	_D2_FUNC(grmv)(m1, m2, Mtmp, L);
	for (index_t j=0; j<m1; ++j) {
	  real_t u = Mtmp[j];
	  index_t idx = j, midx = j;
	  for (index_t k=1; k<m2; ++k, idx+=m1)
	    if (Mtmp[idx] < u) { u = Mtmp[idx]; midx = idx; }
	  primal[midx] += 0.5 * a[i].w[j];
	}
	// calculate L and sample dual2
	_D2_CBLAS_FUNC(copy)(mat_size, M, 1, Mtmp, 1);      
	for (index_t j=0; j< mat_size; ++j) Mtmp[j] = - Mtmp[j];
	_D2_FUNC(gcmv)(m1, m2, Mtmp, U);
	for (index_t j=0; j<m2; ++j) {
	  index_t idx = j*m1; index_t midx = idx;
	  real_t l = Mtmp[idx];
	  for (index_t k=1; k<m1; ++k, ++idx)
	    if (Mtmp[idx] > l) { l = Mtmp[idx]; midx = idx; }
	  primal[midx] += 0.5 * b[i].w[j];
	}
	primal = primal + mat_size;
	M = M + mat_size;
	Mtmp = Mtmp + mat_size;
	U = U + m1;
	L = L + m2;
      }           
    }    
  }

  template <typename ElemType1, typename ElemType2>
  void WM3_SA (const Block<ElemType1> &model,
	       const Block<ElemType2> &data,
	       const size_t max_epoch,
	       const real_t initT,
	       const real_t sigma,
	       const real_t gamma,
	       const size_t batch_size = 50) {
    size_t K=model.get_size();
    size_t m=model[0].len;
    size_t n=data.get_size();
    const size_t inner_iters = 10;
    real_t T=initT;
    Block<ElemType1> mixture_data(data.get_size(), m);
    mixture_data.initialize(data.get_size(), m);
    // initialize membership vectors uniformly
    real_t *beta = (real_t*) malloc(sizeof(real_t) * K * n);
    for (size_t i=0; i<K*n; ++i) beta[i] = 1./K;

    SACache sac, sac_b;
    allocate_sa_cache(mixture_data, data, sac, true);
    internal::_pdist2(mixture_data.get_support_ptr(), m, data.get_support_ptr(), data.get_col(), data.meta, sac._m);

    std::vector< Block<ElemType1> * > mbatch;
    std::vector< const Block<ElemType2> * > dbatch;
    for (size_t i=0; i*batch_size < n; ++i) {
      mbatch.push_back(new Block<ElemType1>(mixture_data, i*batch_size, batch_size));
      dbatch.push_back(new const Block<ElemType2>(data, i*batch_size, batch_size));
    }
    // one epoch
    std::cout << getLogHeader() << " logging: start epoch." << std::endl;
    std::cout << getLogHeader() << "\tapprox primal"
	      << "\tdual" 
	      << "\t\tt" << std::endl;
    
    for (size_t iter=0; iter < max_epoch; ++iter) {
      sac_b = sac;
      for (size_t i=0; i*batch_size < n; ++i) {
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
			     m, batch_size, K,
			     1.0,
			     model.get_weight_ptr(), m,
			     beta + i*batch_size*K, K,
			     0.0,
			     mbatch[i]->get_weight_ptr(), m);    

	EMD_SA(*mbatch[i], *dbatch[i], T, sigma, inner_iters, sac_b);

	for (index_t j=0; j<batch_size; ++j) {
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
      real_t dual_obj = _D2_CBLAS_FUNC(dot)(mixture_data.get_col(), sac._dual1, 1, mixture_data.get_weight_ptr(), 1) - _D2_CBLAS_FUNC(dot)(data.get_col(), sac._dual2, 1, data.get_weight_ptr(), 1);
      real_t primal_obj = _D2_CBLAS_FUNC(dot)(m*data.get_col(), sac._m, 1, sac._primal, 1);      
      std::cout << getLogHeader() << "\t" << primal_obj / n
		<< "\t\t" << dual_obj / n
		<< "\t\t" << T << std::endl;
    }

    for (size_t i=0; i*batch_size < n; ++i) {
      delete mbatch[i];
      delete dbatch[i];
    }
  }
  
}
#endif /* _D2_SA_H_ */
