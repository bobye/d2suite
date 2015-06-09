#ifndef _D2_SERVER_H_
#define _D2_SERVER_H_

#include "solver.h"
#include "blas_like.h"

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
    inline void _pdist2 ( def::Euclidean::type *s1, const size_t n1,
			  def::Euclidean::type *s2, const size_t n2,
			  const Meta<Elem<def::Euclidean, dim> > *meta,
			  real_t* mat) {
      _D2_FUNC(pdist2)(dim, n1, n2, s1, s2, mat);
    }

    template <size_t dim>
    inline void _pdist2( def::Euclidean::type *s1, const size_t n1,
			 def::WordVec::type   *s2, const size_t n2,
			 const Meta<Elem<def::WordVec, dim> > *meta,
			 real_t* mat) {
      _D2_FUNC(pdist2_sym)(dim, n1, n2, s1, s2, mat, meta->embedding);
    }
    template <typename D2Type, size_t dim>
    inline real_t _EMD(Elem<D2Type, dim> &e1, Elem<D2Type, dim> &e2, 
		       const Meta<Elem<D2Type, dim> > * meta, 
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


    template <size_t dim>
    inline real_t _EMD(Elem<def::Euclidean, dim> &e1, Elem<def::WordVec, dim> &e2,
		       const Meta<Elem<def::WordVec, dim> > *meta,
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
		

  }

  template <typename D2Type, size_t dim>
  inline void pdist2 (typename D2Type::type *s1, const size_t n1,
		      typename D2Type::type *s2, const size_t n2,
		      const Meta<Elem<D2Type, dim> > *meta,
		      real_t* mat) {
    internal::_pdist2(s1, n1, s2, n2, meta, mat);
  }


  template <typename ElemType, typename MetaType>
  inline real_t EMD (ElemType &e1, ElemType &e2, const MetaType *meta,
		     real_t* cache_mat,
		     real_t* cache_primal, real_t* cache_dual) {
    return internal::_EMD<typename ElemType::T, ElemType::D>(e1, e2, meta, cache_mat, cache_primal, cache_dual);
  }

  template <size_t dim>
  inline real_t EMD (Elem<def::Euclidean, dim> &e1, Elem<def::WordVec, dim> &e2,
		     const Meta<Elem<def::Euclidean, dim> > *meta, 
		     real_t* cache_mat,
		     real_t* cache_primal, real_t* cache_dual) {
    return internal::_EMD<dim>(e1, e2, meta, cache_mat, cache_primal, cache_dual);
  }

}



#endif /* _D2_SERVER_H_ */
