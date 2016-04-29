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
#define eps (1E-16)

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
    size_t mat_size = 0;
    for (size_t i=0; i<a.get_size(); ++i) mat_size += a[i].len * b[i].len;
    sac._m = (real_t*) malloc(sizeof(real_t) * mat_size);
    sac._mtmp = (real_t*) malloc(sizeof(real_t) * mat_size);
    if (hasPrimal) {
      sac._primal = (real_t*) malloc(sizeof(real_t) * mat_size);
    } else {
      sac._primal = NULL;
    }
    sac._dual1 = (real_t*) malloc(a.get_col() * sizeof(real_t));
    sac._dual2 = (real_t*) malloc(b.get_col() * sizeof(real_t));
    for (size_t i=0; i<a.get_col(); ++i) sac._dual1[i] = 0;
    for (size_t i=0; i<b.get_col(); ++i) sac._dual2[i] = 0;    
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
    assert(sigma>0 && sigma<=1 && T>0);
    assert(a.get_size() == b.get_size());

    //    std::random_device rd;
    std::exponential_distribution<real_t> rng (1./T);
    std::mt19937 rnd_gen (1);
    //    auto gen = std::bind(rng, rnd_gen);

    for (int iter=0; iter < niter; ++iter) {
      real_t *dual1 = sac._dual1;
      real_t *dual2 = sac._dual2;
      real_t *U = sac._U;
      real_t *L = sac._L;
      real_t *M = sac._m;
      real_t *Mtmp = sac._mtmp;
      real_t *w1 = a.get_weight_ptr();
      real_t *w2 = b.get_weight_ptr();
      for (size_t i=0; i < b.get_size(); ++i) {
	// calculate U and sample dual1
	const size_t m1=a[i].len;
	const size_t m2=b[i].len;
	const size_t mat_size=m1*m2;

#pragma clang loop vectorize_width(8) interleave_count(8)
	for (size_t j=0; j< mat_size; ++j) Mtmp[j] = M[j];
	_D2_FUNC(grmv)(m1, m2, Mtmp, dual2);
	_D2_FUNC(rmin)(m1, m2, Mtmp, U);
	//      std::generate(dual1, dual1+m1, gen);
	for (size_t j=0; j < m1; ++j, ++dual1, ++U, ++w1) {
	  *dual1 = *U - rng(rnd_gen) / (*w1 + eps);
	}
	// calculate L and sample dual2
#pragma clang loop vectorize_width(8) interleave_count(8)
	for (size_t j=0; j< mat_size; ++j) Mtmp[j] = - M[j];
	_D2_FUNC(gcmv)(m1, m2, Mtmp, dual1 - m1);
	_D2_FUNC(cmax)(m1, m2, Mtmp, L);
	//	std::generate(dual2, dual2+m1, gen);
	for (size_t j=0; j < m2; ++j, ++dual2, ++L, ++w2) {
	  *dual2 = *L + rng(rnd_gen) / (*w2 + eps);
	}
	M = M + mat_size;
	Mtmp = Mtmp + mat_size;
      }
    }
    
    /*
    {
      real_t *dual1 = sac._dual1;
      real_t *dual2 = sac._dual2;
      real_t *U = sac._U;
      real_t *L = sac._L;
      for (size_t i=0; i < b.get_size(); ++i) {
	size_t m1=a[i].len;
	size_t m2=b[i].len;
	real_t baseline = -1000. + _D2_CBLAS_FUNC(asum)(m2, dual2, 1) / m2;
	for (size_t j=0; j < m1; ++j) dual1[j] = dual1[j] - baseline;
	for (size_t j=0; j < m1; ++j) U[j] = U[j] - baseline;
	for (size_t j=0; j < m2; ++j) dual2[j] = dual2[j] - baseline;
	for (size_t j=0; j < m2; ++j) L[j] = L[j] - baseline;      
	dual1 = dual1 + m1;
	dual2 = dual2 + m2;
	U = U + m1;
	L = L + m2;
      }
    }
    */

    real_t cw=0.;
    real_t cost= 0.;
    if (sac._primal) {
      real_t *primal=sac._primal;
      real_t *U = sac._U;
      real_t *L = sac._L;
      real_t *M = sac._m;
      real_t *Mtmp = sac._mtmp;
      real_t *w1 = a.get_weight_ptr();
      real_t *w2 = b.get_weight_ptr();
      for (size_t i=0; i < b.get_size(); ++i) {
	// calculate U and sample dual1
	size_t m1=a[i].len;
	size_t m2=b[i].len;
	const size_t mat_size=m1*m2;
	for (size_t j=0; j< mat_size; ++j) primal[j]=0;

	for (size_t j=0; j< mat_size; ++j) Mtmp[j] = M[j];
	_D2_FUNC(grmv)(m1, m2, Mtmp, L);
	for (size_t j=0; j<m1; ++j) {
	  size_t midx = j;
	  real_t u = Mtmp[midx];
	  size_t idx = midx+m1;
	  for (size_t k=1; k<m2; ++k, idx+=m1)
	    if (Mtmp[idx] < u) { u = Mtmp[idx]; midx = idx; }
	  primal[midx] += 0.5 * w1[j];
	  cw += 0.5 * w1[j] / w2[midx/m1]; 
	}
	// calculate L and sample dual2
	for (size_t j=0; j< mat_size; ++j) Mtmp[j] = - M[j];
	_D2_FUNC(gcmv)(m1, m2, Mtmp, U);
	for (size_t j=0; j<m2; ++j) {
	  size_t midx = j*m1;
	  real_t l = Mtmp[midx];
	  size_t idx = midx+1;
	  for (size_t k=1; k<m1; ++k, ++idx)
	    if (Mtmp[idx] > l) { l = Mtmp[idx]; midx = idx; }
	  primal[midx] += 0.5 * w2[j];
	  cw += 0.5 * w2[j] / w1[midx%m1];
	}
	primal = primal + mat_size;
	M = M + mat_size;
	Mtmp = Mtmp + mat_size;
	U = U + m1;
	L = L + m2;
	w1 = w1 + m1;
	w2 = w2 + m2;
      }
    }

    if (sac._primal) {
      real_t *primal=sac._primal;
      real_t *U = sac._U;
      real_t *L = sac._L;
      real_t *M = sac._m;
      real_t *w1 = a.get_weight_ptr();
      real_t *w2 = b.get_weight_ptr();
      for (size_t i=0; i < b.get_size(); ++i) {
	size_t m1=a[i].len;
	size_t m2=b[i].len;
	const size_t mat_size=m1*m2;

	for (size_t k=0; k< m2; ++k)
	  for (size_t j=0; j< m1; ++j)
	    cost += (M[j+k*m1] + L[k] - U[j])*primal[j+k*m1];
	w1 = w1 + m1;
	w2 = w2 + m2;
	U = U + m1;	
	L = L + m2;
	M = M + mat_size;
	primal = primal + mat_size;
      }
      std::cout << " " << - cost / cw << " ";
    }
    
  }

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
    const size_t inner_iters = 5;
    real_t T=initT;
    real_t obj_old=-1.;
    Block<ElemType1> mixture_data(n, m);
    mixture_data.initialize(data.get_size(), m);

    SACache sac, sac_b;
    allocate_sa_cache(mixture_data, data, sac, false);
    
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
	      << "\t\tt"
	      << "\tgap" << std::endl;

    real_t primal_obj, dual_obj, max_obj;
    real_t *gd, *md;
    gd = (real_t*) malloc(sizeof(real_t)*std::max(m,batch_size)*K);
    md = (real_t*) malloc(sizeof(real_t)*m*K);
    for (size_t i=0; i<m*K; ++i) md[i] = 0.;

    // initialize membership vectors uniformly
    real_t *beta = (real_t*) malloc(sizeof(real_t) * K * n);
    real_t *betaz= (real_t*) malloc(sizeof(real_t) * K * n);
    for (size_t i=0; i<K*n; ++i) beta[i] = rand()%100+1;   
    _D2_FUNC(cnorm)(K, n, beta, gd);
    for (size_t i=0; i<K*n; ++i) betaz[i] = beta[i];
    
    internal::_pdist2(mixture_data.get_support_ptr(), m, data.get_support_ptr(), data.get_col(), data.meta, sac._m);
    real_t mC = _D2_CBLAS_FUNC(asum)(data.get_col() * K, sac._m, 1) / (data.get_col() * K);
    
    for (size_t iter=0, accelerator=1; iter < max_epoch; ++iter, ++accelerator) {  
      sac_b = sac;
      real_t r = 3;
      real_t lambda = r/(r+accelerator);

      if (accelerator == 20) {
	for (size_t i=0; i<K*n; ++i) betaz[i] = beta[i];
	accelerator=1;
      }

      _D2_CBLAS_FUNC(scal)(K*n, (1-lambda), beta, 1);
      _D2_CBLAS_FUNC(axpy)(K*n, lambda, betaz, 1, beta, 1);
      
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

	EMD_SA(*mbatch[i], *dbatch[i], T, sigma, inner_iters, sac_b);

	if (iter > 0 && dual_obj > 0.) {
	  _D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
			       m, K, batch_size, 
			       1.0,
			       sac_b._dual1, m,
			       thisbeta, K,
			       0.0,
			       gd, m);
	  _D2_CBLAS_FUNC(scal)(m*K, 0.95, md, 1);
	  _D2_CBLAS_FUNC(axpy)(m*K, - gamma / mC, gd, 1, md, 1);

	  real_t *w=model.get_weight_ptr();
	  for (size_t j=0; j<m*K; ++j) { w[j] = w[j] * exp(md[j]) + eps; }
	  _D2_FUNC(cnorm)(m, K, w, gd);

	  _D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasTrans, CblasNoTrans,
			       K, batch_size, m,
			       - gamma /mC,
			       model.get_weight_ptr(), m,
			       sac_b._dual1, m,
			       0.0,
			       gd, K);
	  for (size_t j=0; j<K*batch_size; ++j) {
	    thisbeta[j] = thisbeta[j] * exp(gd[j] * r) + eps;
	    thisbetaz[j] = thisbetaz[j] * exp(accelerator * gd[j] / r) + eps;
	  }
	  _D2_FUNC(cnorm)(K, batch_size, thisbeta, gd);
	  _D2_FUNC(cnorm)(K, batch_size, thisbetaz, gd);
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
      
      if ((iter+1) % 20 == 0) {
	real_t *emds, *cache_mat, *cache_dual;
	real_t obj;
	emds = (real_t*) malloc(sizeof(real_t)*n);
	cache_mat = (real_t*) malloc(sizeof(real_t)*m*data.get_max_len());
	//	cache_dual = (real_t*) malloc(sizeof(real_t)*(m+data.get_max_len()));
	real_t *dual1=sac._dual1;
	real_t *dual2=sac._dual2;
	for (size_t i=0; i<n; ++i) {    
	  emds[i]=EMD(mixture_data[i], data[i], data.meta, cache_mat);
	  /*
	  for (size_t j=0; j<m; ++j) dual1[j] = cache_dual[j];
	  for (size_t j=m; j<m+data[i].len; ++j) dual2[j] = -cache_dual[j];
	  dual1 += m;
	  dual2 += data[i].len;
	  */
	}
	obj = _D2_CBLAS_FUNC(asum)(n, emds, 1) / n;
	std::cout << obj << std::endl;
	free(cache_mat);
	free(emds);
	model.write("data/mnist/mixture_5_" + std::to_string(iter+1) + ".txt");
	//free(cache_dual);
	/*
	if (obj_old < 0 || obj <= obj_old) {
	  obj_old = obj;
	} else {
	  break;
	}
	*/
      }

      dual_obj = _D2_CBLAS_FUNC(dot)(n*m, sac._dual1, 1, mixture_data.get_weight_ptr(), 1) - _D2_CBLAS_FUNC(dot)(data.get_col(), sac._dual2, 1, data.get_weight_ptr(), 1);
      if (sac._primal) {
	primal_obj = _D2_CBLAS_FUNC(dot)(m*data.get_col(), sac._m, 1, sac._primal, 1);
      } else {
	primal_obj = _D2_CBLAS_FUNC(dot)(n*m, sac._U, 1, mixture_data.get_weight_ptr(), 1) - _D2_CBLAS_FUNC(dot)(data.get_col(), sac._L, 1, data.get_weight_ptr(), 1);
      }

      if (dual_obj < 0.1*primal_obj)
	T*=1-1./sqrt(2*m); //if (T < 0.001) T=0.001;
      //gamma *= sqrt(1.+iter) / sqrt(2.+iter);
      std::cout << getLogHeader() << "\t" << primal_obj / n
		<< "\t\t" << dual_obj / n
		<< "\t\t" << T
		<< "\t\t" << (primal_obj - dual_obj)/n << std::endl;
      
    }

    for (size_t i=0; i*batch_size < n; ++i) {
      delete mbatch[i];
      delete dbatch[i];
    }

    /*
    for (size_t i=0, k=0; i<n; ++i) {
      for (size_t j=0; j<K; ++j, ++k) {
	std::cout << " " << beta[k];
      }
      std::cout << std::endl;
    }
    */
    free(beta);
    free(gd);
    free(md);
    deallocate_sa_cache(sac);
    mixture_data.write("data/mnist/estimate_5.d2s");
  }
  
}
#endif /* _D2_SA_H_ */
