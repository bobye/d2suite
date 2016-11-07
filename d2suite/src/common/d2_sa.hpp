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

  namespace internal {
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


    real_t sort_and_estimate(real_t *arr, int incr, int elements, real_t *arr2, real_t T, bool is_increasing = true) {
      //  This public-domain C implementation by Darel Rex Finley.
#define  MAX_LEVELS  1000

      real_t piv, piv2;
      int  beg[MAX_LEVELS], end[MAX_LEVELS], i, L, R ;

      if (!is_increasing)
	for (i=0;i<elements*incr; i+=incr) arr[i]=-arr[i];
      i=0; beg[0]=0; end[0]=elements*incr;
      while (i>=0) {
	L=beg[i]; R=end[i]-incr;
	if (L<R) {
	  piv=arr[L]; piv2=arr2[L];
	  if (i==MAX_LEVELS-1) return -1;
	  while (L<R) {
	    while (arr[R]>=piv && L<R) R-=incr;
	    if (L<R) {arr[L]=arr[R]; arr2[L]=arr2[R]; L+=incr;}
	    while (arr[L]<=piv && L<R) L+=incr;
	    if (L<R) {arr[R]=arr[L]; arr2[R]=arr2[L]; R-=incr;}
	  }
	  arr[L]=piv; arr2[L]=piv2;
	  beg[i+1]=L+incr; end[i+1]=end[i]; end[i++]=L; }
	else {
	  i--;
	}
      }

      real_t q=0., lambda=1., lambda2=1., sum=0.;
      for (i=0; i<elements-1; ++i) {
	q+=arr2[i*incr];
	lambda2 *= exp(q*(arr[i*incr] - arr[i*incr+incr])/T);
	sum+=(lambda - lambda2)/q;
	lambda = lambda2;
      }
      sum += lambda;
      return sum;
    }
  }

  /*
   * INPUT/OUTPUT:
   * a, b  -- a block of distributions
   * T     -- temperature
   * niter -- iterations of minimal loop block
   * sac   -- simulated annealing cache
   * A, B, C, hasProposal -- basic relevant statistics to be tracked
   */
  template <typename ElemType1, typename ElemType2>
  int EMD_SA (const Block<ElemType1> &a, const Block<ElemType2> &b,
	      const real_t &T,
	      const size_t niter,
	      const internal::SACache &sac,
	      real_t &A, real_t &B, real_t &D, bool hasProposal = false) {
    assert(sac._m && sac._mtmp);
    assert(sac._dual1 && sac._dual2);
    assert(sac._U && sac._L);
    assert(T>0);
    assert(a.get_size() == b.get_size());

    std::random_device rd;
    std::exponential_distribution<real_t> rng (1./T);
    std::mt19937 rnd_gen (rd());
    //    auto gen = std::bind(rng, rnd_gen);

    real_t upper_bound_old, upper_bound=0;
    int iterations=0;
    
    do {
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

	memcpy(Mtmp, M, sizeof(real_t)*mat_size);
	_D2_FUNC(grmv)(m1, m2, Mtmp, dual2);
	_D2_FUNC(rmin)(m1, m2, Mtmp, U);
	for (size_t j=0; j < m1; ++j, ++dual1, ++U, ++w1) {
	  *dual1 = *U - rng(rnd_gen) / (*w1 + eps);
	}
	// calculate L and sample dual2
	memcpy(Mtmp, M, sizeof(real_t)*mat_size);
	_D2_FUNC(gcmv2)(m1, m2, Mtmp, dual1 - m1);
	_D2_FUNC(cmax)(m1, m2, Mtmp, L);
	for (size_t j=0; j < m2; ++j, ++dual2, ++L, ++w2) {
	  *dual2 = *L + rng(rnd_gen) / (*w2 + eps);
	}
	M = M + mat_size;
	Mtmp = Mtmp + mat_size;
      }
    }
    upper_bound_old = upper_bound;
    upper_bound = _D2_CBLAS_FUNC(dot)(a.get_col(), a.get_weight_ptr(), 1, sac._U, 1)
      - _D2_CBLAS_FUNC(dot)(b.get_col(), b.get_weight_ptr(), 1, sac._L, 1);
    iterations += niter;
    } while (upper_bound - upper_bound_old > 0.001 * upper_bound);    
    
    real_t cost= 0.;
    real_t div = 0.;
    real_t phi=0.;
    if (sac._primal && hasProposal) {
      real_t *primal=sac._primal;
      real_t *w1 = a.get_weight_ptr();
      real_t *w2 = b.get_weight_ptr();
      real_t *M = sac._m;
      real_t *Mtmp = sac._mtmp;
      real_t *U = sac._U;
      real_t *L = sac._L;
      for (size_t i=0; i<b.get_size(); ++i) {
	const size_t m1=a[i].len;
	const size_t m2=b[i].len;
	const size_t mat_size=m1*m2;
	for (size_t j=0; j<m2; ++j)
	  for (size_t k=0; k<m1; ++k)
	    primal[k+j*m1]= w2[j];
	memcpy(Mtmp, M, sizeof(real_t)*mat_size);
	_D2_FUNC(grmv)(m1, m2, Mtmp, L);
	for (size_t j=0; j<m1; ++j) {
	  phi+=internal::sort_and_estimate(Mtmp+j, m1, m2, primal+j, T, true) * w1[j];
	  cost-=(Mtmp[j]-U[j])*w1[j];
	  //	  div+=(_D2_CBLAS_FUNC(dot)(m2, Mtmp+j, m1, primal+j, m1)-Mtmp[j])*w1[j];
	}
	for (size_t j=0; j<m2; ++j)
	  memcpy(primal+j*m1, w1, m1);
	memcpy(Mtmp, M, sizeof(real_t)*mat_size);
	_D2_FUNC(gcmv2)(m1, m2, Mtmp, U);
	for (size_t j=0; j<m2; ++j) {
	  phi+=internal::sort_and_estimate(Mtmp+j*m1, 1, m1, primal+j*m1, T, false)*w2[j];
	  cost-=(Mtmp[j*m1]+L[j])*w2[j];
	  //	  div+=(_D2_CBLAS_FUNC(dot)(m1, Mtmp+j*m1, 1, primal+j*m1, 1)-Mtmp[j*m1])*w2[j];
	}

	primal = primal + mat_size;
	M = M + mat_size;
	Mtmp = Mtmp + mat_size;
	U = U + m1;
	L = L + m2;
	w1 = w1 + m1;
	w2 = w2 + m2;	
      }
      //std::cout << " " << cost << " " << phi << " " << cost/phi << std::endl;      
    }
    A=cost; B=phi; D=div;

    return iterations;
  }

  
}
#endif /* _D2_SA_H_ */
