#ifndef _D2_SERVER_H_
#define _D2_SERVER_H_

#include "solver.h"
#include "blas_like.h"
#include "cblas.h"
#include <algorithm>
#include <queue>

namespace d2 {

  namespace server {
    inline void Init(int argc, char*argv[]) {
#ifdef RABIT_RABIT_H_
      rabit::Init(argc, argv);
#endif    
      d2_solver_setup();
    }
    inline void Finalize() {
#ifdef RABIT_RABIT_H_
      rabit::Finalize();
#endif
      d2_solver_release();
    }
  }


  namespace internal {
    template <size_t dim>
    inline void _pdist2 ( const def::Euclidean::type *s1, const size_t n1,
			  const def::Euclidean::type *s2, const size_t n2,
			  const Meta<Elem<def::Euclidean, dim> > &meta,
			  real_t* mat) {
      _D2_FUNC(pdist2)(dim, n1, n2, s1, s2, mat);
    }

    template <size_t dim>
    inline void _pdist2( const def::Euclidean::type *s1, const size_t n1,
			 const def::WordVec::type   *s2, const size_t n2,
			 const Meta<Elem<def::WordVec, dim> > &meta,
			 real_t* mat) {
      _D2_FUNC(pdist2_sym)(dim, n1, n2, s1, s2, mat, meta.embedding);
    }
    template <typename D2Type, size_t dim>
    inline real_t _EMD(const Elem<D2Type, dim> &e1, const Elem<D2Type, dim> &e2, 
		       const Meta<Elem<D2Type, dim> > &meta, 
		       real_t* cache_mat, real_t* cache_primal, real_t* cache_dual) {
      bool cache_mat_is_null = false;
      real_t val;
      if (cache_mat == NULL) {
	cache_mat_is_null = true;
	cache_mat = (real_t*) malloc(sizeof(real_t) * e1.len * e2.len);
      }
      pdist2<D2Type, dim>(e1.supp, e1.len, 
			  e2.supp, e2.len,
			  meta,
			  cache_mat);
      val = d2_match_by_distmat(e1.len, e2.len, 
				cache_mat, 
				e1.w, e2.w,
				cache_primal, cache_dual, 0);
      
      if (cache_mat_is_null) free(cache_mat);

      return val;
    }


    template<size_t dim>
    inline real_t _LowerThanEMD_v0(const Elem<def::Euclidean, dim> &e1,
				   const Elem<def::Euclidean, dim> &e2,
				   const Meta<Elem<def::Euclidean, dim> > &meta) {
      real_t c1[dim], c2[dim], val=0, d;
      _D2_CBLAS_FUNC(gemv)(CblasColMajor, 
			   CblasNoTrans, 
			   dim, e1.len, 1., e1.supp, dim, 
			   e1.w, 1,
			   0., c1, 1);

      _D2_CBLAS_FUNC(gemv)(CblasColMajor, 
			   CblasNoTrans, 
			   dim, e2.len, 1., e2.supp, dim, 
			   e2.w, 1,
			   0., c2, 1);

      for (size_t i=0; i<dim; ++i) {
	d = (c1[i] - c2[i]);
	d = d*d;
	val += d;
      }
      
      return val;
    }

    template<size_t dim>
    inline real_t _LowerThanEMD_v0(const Elem<def::WordVec, dim> &e1,
				   const Elem<def::WordVec, dim> &e2,
				   const Meta<Elem<def::WordVec, dim> > &meta) {
      real_t c1[dim], c2[dim], val=0, d;
      // implement incomplete !

      for (size_t i=0; i<dim; ++i) {
	d = (c1[i] - c2[i]);
	d = d*d;
	val += d;
      }

      return val;
    }

    template <typename D2Type, size_t dim>
    inline real_t _LowerThanEMD_v1(const Elem<D2Type, dim> &e1, const Elem<D2Type, dim> &e2,
				   const Meta<Elem<D2Type, dim> > &meta,
				   real_t* cache_mat) {
      bool cache_mat_is_null = false;
      real_t val;
      if (cache_mat == NULL) {
	cache_mat_is_null = true;
	cache_mat = (real_t*) malloc(sizeof(real_t) * e1.len * e2.len);	
      }
      // cache_mat is column major
      pdist2<D2Type, dim>(e1.supp, e1.len,
			  e2.supp, e2.len,
			  meta,
			  cache_mat);
      real_t* head=cache_mat;
      real_t min, val1=0, val2=0;

      for (size_t i=0; i<e2.len; ++i) {
	min=std::numeric_limits<real_t>::max();
	for (size_t j=0; j<e1.len; ++j, ++head) 
	  min = std::min(min, *head);
	val1 += min * e2.w[i];
      }

      size_t stride = e1.len;
      for (size_t i=0; i<e1.len; ++i) {
	min=std::numeric_limits<real_t>::max();
	head=cache_mat + i;
	for (size_t j=0; j<e2.len; ++j, head+=stride) 
	  min = std::min(min, *head);
	val2 += min * e1.w[i];
      }
      if (cache_mat_is_null) free(cache_mat);

      return std::max(val1, val2);
    }


    template <size_t dim>
    inline real_t _EMD(const Elem<def::Euclidean, dim> &e1, const Elem<def::WordVec, dim> &e2,
		       const Meta<Elem<def::WordVec, dim> > &meta,
		       real_t* cache_mat, real_t* cache_primal, real_t* cache_dual) {
      bool cache_mat_is_null = false;
      real_t val;
      if (cache_mat == NULL) {
	cache_mat_is_null = true;
	cache_mat = (real_t*) malloc(sizeof(real_t) * e1.len * e2.len);
      }
      _pdist2<dim>(e1.supp, e1.len, 
		   e2.supp, e2.len,
		   meta,
		   cache_mat);
      val = d2_match_by_distmat(e1.len, e2.len, 
				cache_mat, 
				e1.w, e2.w,
				cache_primal, cache_dual, 0);
      
      if (cache_mat_is_null) free(cache_mat);

      return val;
    }


    template <size_t dim>
    inline real_t _LowerThanEMD_v1(const Elem<def::Euclidean, dim> &e1, 
				   const Elem<def::WordVec, dim> &e2,
				   const Meta<Elem<def::WordVec, dim> > &meta,
				   real_t* cache_mat) {
      bool cache_mat_is_null = false;
      real_t val;
      if (cache_mat == NULL) {
	cache_mat_is_null = true;
	cache_mat = (real_t*) malloc(sizeof(real_t) * e1.len * e2.len);	
      }
      pdist2<dim>(e1.supp, e1.len,
		  e2.supp, e2.len,
		  meta,
		  cache_mat);
      real_t* head=cache_mat;
      real_t min, val1=0, val2=0;

      for (size_t i=0; i<e2.len; ++i) {
	min=std::numeric_limits<real_t>::max();
	for (size_t j=0; j<e1.len; ++j, ++head) 
	  min = std::min(min, *head);
	val1 += min * e2.w[i];
      }

      size_t stride = e1.len;
      for (size_t i=0; i<e1.len; ++i) {
	min=std::numeric_limits<real_t>::max();
	head=cache_mat + i;
	for (size_t j=0; j<e2.len; ++j, head+=stride) 
	  min = std::min(min, *head);
	val2 += min * e1.w[i];
      }
      if (cache_mat_is_null) free(cache_mat);

      return std::max(val1, val2);
    }

		

  }

  template <typename D2Type, size_t dim>
  inline void pdist2 (const typename D2Type::type *s1, const size_t n1,
		      const typename D2Type::type *s2, const size_t n2,
		      const Meta<Elem<D2Type, dim> > &meta,
		      real_t* mat) {
    internal::_pdist2(s1, n1, s2, n2, meta, mat);
  }


  template <typename ElemType, typename MetaType>
  inline real_t EMD (const ElemType &e1, const ElemType &e2, const MetaType &meta,
		     real_t* cache_mat,
		     real_t* cache_primal, real_t* cache_dual) {
    return internal::_EMD<typename ElemType::T, ElemType::D>(e1, e2, meta, cache_mat, cache_primal, cache_dual);
  }

  template <size_t dim>
  inline real_t EMD (const Elem<def::Euclidean, dim> &e1, const Elem<def::WordVec, dim> &e2,
		     const Meta<Elem<def::WordVec, dim> > &meta, 
		     real_t* cache_mat,
		     real_t* cache_primal, real_t* cache_dual) {
    return internal::_EMD<dim>(e1, e2, meta, cache_mat, cache_primal, cache_dual);
  }

  template <typename ElemType>
  void EMD (const ElemType &e, const Block<ElemType> &b,
	    __OUT__ real_t* emds,
	    __IN__ real_t* cache_mat,
	    __OUT__ real_t* cache_primal, 
	    __OUT__ real_t* cache_dual) {
    bool cache_mat_is_null = false;
    if (cache_mat == NULL) {
      cache_mat_is_null = true;
      cache_mat = (real_t*) malloc(sizeof(real_t) * e.len * b.get_max_len());	
    }

    real_t *primal_ptr = cache_primal;
    real_t *dual_ptr = cache_dual;

    for (size_t i=0; i<b.get_size(); ++i) {      
      emds[i] = EMD(e, b[i], b.meta, cache_mat, primal_ptr, dual_ptr);
      if (cache_primal) primal_ptr += e.len * b[i].len;
      if (cache_dual) dual_ptr += e.len + b[i].len;
    }

    if (cache_mat_is_null) {
      free(cache_mat);
    }
  }

  template <size_t dim>
  void EMD (const Elem<def::Euclidean, dim> &e, const Block<Elem<def::WordVec, dim> > &b,
	    __OUT__ real_t* emds,
	    __IN__ real_t* cache_mat,
	    __OUT__ real_t* cache_primal, 
	    __OUT__ real_t* cache_dual) {
    // to be implement
  }

  namespace internal {
    template <typename T, typename... Ts>
    void _EMD_impl(const _ElemMultiPhaseConstructor<T, Ts...> &e, 
		   const _BlockMultiPhaseConstructor<T, Ts...> &b,
		   __OUT__ real_t ** emds_arr,
		   __IN__ real_t** cache_mat_arr) {
      EMD(e.head, b.head, *emds_arr, *cache_mat_arr, NULL, NULL);      
      const _ElemMultiPhaseConstructor<Ts...> &e0 = e;
      const _BlockMultiPhaseConstructor<Ts...> &b0 = b;
      _EMD_impl<Ts...>(e0, b0, emds_arr + 1, cache_mat_arr + 1);
    }

    template <>
    void _EMD_impl(const _ElemMultiPhaseConstructor<> &e, 
		   const _BlockMultiPhaseConstructor<> &b,
		   __OUT__ real_t ** emds_arr,
		   __IN__ real_t** cache_mat_arr) {
    }
  }

  template <typename... Ts>
  void EMD(const ElemMultiPhase<Ts...> &e, 
	   const BlockMultiPhase<Ts...> &b,
	   __OUT__ real_t* emds) {
    static const size_t k = internal::tuple_size<Ts...>::value;
    // allocate
    real_t ** cache_mat_arr = (real_t **) malloc(sizeof(real_t *) * k );
    for (size_t i=0; i<k; ++i) cache_mat_arr[i] = NULL;

    real_t ** emds_arr = (real_t **) malloc(sizeof(real_t *) * k);
    for (size_t i=0; i<k; ++i) emds_arr[i] = (real_t *) malloc(sizeof(real_t) * b.get_size());

    internal::_EMD_impl(e, b, emds_arr, cache_mat_arr);    

    // sum to one
    for (size_t j=0; j<b.get_size(); ++j) {
      emds[j] = 0;
    }
    for (size_t i=0; i<k; ++i)
      for (size_t j=0; j<b.get_size(); ++j) {
	emds[j] += emds_arr[i][j];
      }

    // free
    free(cache_mat_arr);

    for (size_t i=0; i<k; ++i) free(emds_arr[i]);
    free(emds_arr);
  }

  template <typename ElemType, typename MetaType>
  inline real_t LowerThanEMD_v0(const ElemType &e1, const ElemType &e2, const MetaType &meta) {
    return internal::_LowerThanEMD_v0<ElemType::D>(e1, e2, meta);
  }

  template <size_t dim>
  inline real_t LowerThanEMD_v0(const Elem<def::Euclidean, dim> &e1, 
				const Elem<def::WordVec, dim> &e2,
				const Meta<Elem<def::WordVec, dim> > &meta) {
    return internal::_LowerThanEMD_v0<dim>(e1, e2, meta);
  }

  template <typename ElemType1, typename ElemType2>
  void LowerThanEMD_v0(const ElemType1 &e, const Block<ElemType2> &b,
		       __OUT__ real_t* emds) {
    for (size_t i=0; i<b.get_size(); ++i) {
      emds[i] = LowerThanEMD_v0(e, b[i], b.meta);
    }
  }


  template <typename ElemType, typename MetaType>
  inline real_t LowerThanEMD_v1(const ElemType &e1, const ElemType &e2, const MetaType &meta,
				real_t* cache_mat) {
    return internal::_LowerThanEMD_v1<typename ElemType::T, ElemType::D>(e1, e2, meta, cache_mat);
  }

  template <size_t dim>
  inline real_t LowerThanEMD_v1(const Elem<def::Euclidean, dim> &e1, 
				const Elem<def::WordVec, dim> &e2,
				const Meta<Elem<def::WordVec, dim> > &meta,
				real_t* cache_mat) {
    return internal::_LowerThanEMD_v1<dim>(e1, e2, meta, cache_mat);
  }


  template <typename ElemType1, typename ElemType2>
  void LowerThanEMD_v1(const ElemType1 &e, const Block<ElemType2> &b,
		       __OUT__ real_t* emds,
		       __IN__ real_t* cache_mat) {
    bool cache_mat_is_null = false;
    if (cache_mat == NULL) {
      cache_mat_is_null = true;
      cache_mat = (real_t*) malloc(sizeof(real_t) * e.len * b.get_col());	
    }
    real_t *mat_ptr = cache_mat;

    for (size_t i=0; i<b.get_size(); ++i) {
      emds[i] = LowerThanEMD_v1(e, b[i], b.meta, mat_ptr);
      mat_ptr += e.len * b[i].len;      
    }

    if (cache_mat_is_null) free(cache_mat);
  }


  namespace internal {
    template <typename ElemType, typename BlockType, typename DistanceFunction>
    void _KNearestNeighbors_Linear_impl(size_t k,
				  const ElemType &e, const BlockType &b,
				  DistanceFunction & lambda,
				  __OUT__ real_t* emds_approx,
				  __OUT__ index_t* rank) {
      lambda(e, b, emds_approx);
      for (size_t i=0; i<b.get_size(); ++i) rank[i] = i;
      std::sort(rank, rank + b.get_size(), 
		[&](size_t i1, size_t i2) {return emds_approx[i1] < emds_approx[i2];});
    }


    template <typename ElemType, typename BlockType, 
	      typename DistanceFunction, typename LowerBoundFunction0, typename LowerBoundFunction1>
    void _KNearestNeighbors_Simple_impl(size_t k,
					const ElemType &e, const BlockType &b,
					DistanceFunction & lambda,
					LowerBoundFunction0 & lower0,
					LowerBoundFunction1 & lower1,
					__OUT__ real_t* emds_approx,
					__OUT__ index_t* rank,
					size_t n) {
      auto compare = [&](size_t i1, size_t i2) {return emds_approx[i1] < emds_approx[i2];};
      for (size_t i=0; i<b.get_size(); ++i) {
	emds_approx[i] = lower0(e, b[i], b.meta);
      }

      for (size_t i=0; i<b.get_size(); ++i) rank[i] = i;
      std::sort(rank, rank + b.get_size(), compare);

      real_t *cache_mat = (real_t*) malloc(sizeof(real_t) * e.len * b.get_max_len());	
      // compute exact distance for the first k
      for (size_t i=0; i<k; ++i) {
	emds_approx[rank[i]] = lambda(e, b[rank[i]], b.meta, cache_mat);
      }

      std::priority_queue<size_t, 
			  std::vector<size_t>, 
			  decltype( compare ) > knn(rank, rank + k, compare);

      size_t idx;
      for (size_t i=k; i<n; ++i) {
	idx = rank[i];
	emds_approx[idx] = std::max(emds_approx[idx], lower1(e, b[idx], b.meta, cache_mat));
	if (emds_approx[idx] < emds_approx[knn.top()]) {
	  emds_approx[idx] = lambda(e, b[idx], b.meta, cache_mat);
	  if (emds_approx[idx] < emds_approx[knn.top()])
	    knn.pop(); knn.push(idx);
	}
      }
      std::sort(rank, rank + n, compare);
      free(cache_mat);
    }

  }


  template <typename ElemType>
  void KNearestNeighbors_Linear(size_t k,
				const ElemType &e, const Block<ElemType> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank) {
    auto lambda =  [](const ElemType& e, const Block<ElemType> &b, real_t* dist) {
      EMD(e, b, dist, NULL, NULL, NULL);
    };
    internal::_KNearestNeighbors_Linear_impl(k, e, b, lambda, emds_approx, rank);
  }

  template <typename... Ts>
  void KNearestNeighbors_Linear(size_t k,
				const ElemMultiPhase<Ts...> &e,
				const BlockMultiPhase<Ts...> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank) {
    auto lambda = [](const ElemMultiPhase<Ts...> &e, 
		     const BlockMultiPhase<Ts...> &b, 
		     real_t* dist) {
      EMD(e, b, dist);
    };
    internal::_KNearestNeighbors_Linear_impl(k, e, b, lambda, emds_approx, rank);
  }

  template <typename ElemType1, typename ElemType2>
  void KNearestNeighbors_Simple(size_t k,
				const ElemType1 &e, const Block<ElemType2> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank,
				size_t n) {
    auto lambda = [](const ElemType1& e1, const ElemType2 &e2, const Meta<ElemType2> &meta, real_t* cache) -> real_t {return EMD(e1, e2, meta, cache);};
    auto lower0 = [](const ElemType1& e1, const ElemType2& e2, const Meta<ElemType2> &meta) -> real_t {return LowerThanEMD_v0(e1, e2, meta);};
    auto lower1 = [](const ElemType1& e1, const ElemType2& e2, const Meta<ElemType2> &meta, real_t* cache) -> real_t {return LowerThanEMD_v1(e1, e2, meta, cache);};
    if (n == 0) n = b.get_size();
    internal::_KNearestNeighbors_Simple_impl(k, e, b, lambda, lower0, lower1, emds_approx, rank, n);
  }



}



#endif /* _D2_SERVER_H_ */
