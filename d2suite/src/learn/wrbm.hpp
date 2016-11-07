#ifndef _WRBM_H_
#define _WRBM_H_

#include "../common/common.hpp"
#include "../common/d2.hpp"
#include "../common/cblas.h"
#include "../common/d2_sa.hpp"

namespace d2 {
  /*! \todo an implementation of Wasserstein RBM using SA */
  template <>
  void WRBM_SA (real_t* W, real_t *a, real_t* b, /* parameters of RBM */
		size_t l, /* latent dimension */
		const Block<Histogram> & data, /* histogram data */
		const size_t max_epoch,
		const real_t initT,
		real_t sigma,
		size_t batch_size = 20) {
    size_t m=data[0].len;
    size_t n=data.get_size();

    std::default_random_engine eng(::time(NULL));
    std::uniform_real_distribution<real_t> rng(0.0, 1.0);
    
    const size_t tau = 5;
    real_t T=initT, A=0., B=0., D=0., bound;
    Block<ElemType1> mixture_data(n, m);
    mcmc_data.initialize(data.get_size(), m);

    SACache sac, sac_b;
    allocate_sa_cache(mcmc_data, data, sac, true);

    std::vector< Block<Histogram> * > mbatch;
    std::vector< const Block<Histogram> * > dbatch;
    for (size_t i=0; i*batch_size < n; ++i) {
      mbatch.push_back(new Block<Histogram>(mcmc_data, i*batch_size, batch_size));
      dbatch.push_back(new const Block<Histogram>(data, i*batch_size, batch_size));
    }

    real_t *latent = (real_t*) malloc(sizeof(real_t) * l * batch_size);
    real_t *pos_assoc_W  = (real_t*) malloc(sizeof(real_t) * l * m);
    real_t *pos_assoc_a = (real_t*) malloc(sizeof(real_t) * m);
    real_t *pos_assoc_b = (real_t*) malloc(sizeof(real_t) * l);
    real_t *neg_assoc_W  = (real_t*) malloc(sizeof(real_t) * l * m);
    real_t *neg_assoc_a = (real_t*) malloc(sizeof(real_t) * m);
    real_t *neg_assoc_b = (real_t*) malloc(sizeof(real_t) * l);
    for (size_t iter=0; iter < max_epoch; ++iter) {
      sac_b = sac;
      for (size_t i=0; i*batch_size < n; ++i) {
	
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
			     l, batch_size, m,
			     1.0,
			     A, l,
			     data.get_weight_ptr(), m,
			     0.0,
			     latent,l);
	_D2_FUNC(gcmv)(l, batch_size, latent, b);
	for (size_t ii=0; ii<l * batch_size ; ++ii)
	  latent[ii] = 1/(1+ exp(-latent[ii]));
	
	_D2_FUNC(rsum)(m, batch_size, data.get_weight_ptr(), pos_assoc_a);
	_D2_FUNC(rsum)(l, batch_size, latent, pos_assoc_b);	
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
			     l, m, batch_size,
			     1.0,
			     latent, l,
			     data.get_weight_ptr(), m,
			     0.0,
			     pos_assoc_W, l);

	for (size_t ii=0; ii<l * batch_size; ++ii)
	  latent[ii] = rng(eng) < latent[ii]? 1.0: 0.0;

	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasTrans, CblasNoTrans,
			     m, batch_size, l,
			     1.0,
			     A, l,
			     latent, l,
			     0.0,
			     mcmc_data.get_weight_ptr(), m);	
	_D2_FUNC(gcmv)(m, batch_size, mcmc_data.get_weight_ptr(), a);

	
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
			     l, batch_size, m,
			     1.0,
			     A, l,
			     mcmc_data.get_weight_ptr(), m,
			     0.0,
			     latent,l);
	_D2_FUNC(gcmv)(l, batch_size, latent, b);
	for (size_t ii=0; ii<l * batch_size ; ++ii)
	  latent[ii] = 1/(1+ exp(-latent[ii]));

	_D2_FUNC(rsum)(m, batch_size, mcmc_data.get_weight_ptr(), neg_assoc_a);
	_D2_FUNC(rsum)(l, batch_size, latent, neg_assoc_b);	
	_D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
			     l, m, batch_size,
			     1.0,
			     latent, l,
			     mcmc_data.get_weight_ptr(), m,
			     0.0,
			     neg_assoc_W, l);
	
	
      }      
    }
    
    
  }
  
}

#endif /* _WRMB_H_ */
