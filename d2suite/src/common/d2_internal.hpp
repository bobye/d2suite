#ifndef _D2_INTERNAL_H_
#define _D2_INTERNAL_H_


#include "d2.hpp"

namespace d2 {
  namespace internal {

    template <typename D2Type, size_t D>
    class _Meta {};

    template <size_t D>
    class _Meta<def::WordVec, D> {
    public:
      _Meta(): size(0), embedding(NULL) {};
      size_t size;
      real_t *embedding;
      ~_Meta() {
	if (embedding!=NULL) delete [] embedding;
      }
    };

    template <size_t D>
    class _Meta<def::NGram, D> {
    public:
      _Meta(): size(0), dist_mat(NULL) {};
      size_t size;
      real_t *dist_mat;
      index_t vocab[255]; // map from char to index
      ~_Meta() {
	if (dist_mat!=NULL) delete [] dist_mat;
      }
    };

    template <size_t D>
    class _Meta<def::Histogram, D> {
    public:
      _Meta(): size(0), dist_mat(NULL) {};
      size_t size;
      real_t *dist_mat;
      ~_Meta() {
	if (dist_mat!=NULL) delete [] dist_mat;
      }
    };


    template <typename T1=Elem<def::Euclidean, 0>, typename... Ts>
    struct _ElemMultiPhaseConstructor: public _ElemMultiPhaseConstructor<Ts...> {
      _ElemMultiPhaseConstructor (const index_t i=0): 
	ind(i), _ElemMultiPhaseConstructor<Ts...>(i + 1) {}
      index_t ind;
      T1 head;
    };
    template <>
    struct _ElemMultiPhaseConstructor<> {
      _ElemMultiPhaseConstructor (const index_t i) {}
    };

    template <typename T1=Elem<def::Euclidean, 0>, typename... Ts>
    class _BlockMultiPhaseConstructor: public _BlockMultiPhaseConstructor<Ts...> {
    public:
      _BlockMultiPhaseConstructor (const size_t thesize, 
				   const size_t* thelen,
				   const index_t i = 0) : 
	head(thesize, *thelen), ind(i),
	_BlockMultiPhaseConstructor<Ts...>(thesize, thelen+1, i+1) {}
      index_t ind;
      Block<T1> head;
    };

    template <>
    class _BlockMultiPhaseConstructor<> {
    public:
      _BlockMultiPhaseConstructor (const size_t thesize, 
				   const size_t* thelen,
				   const index_t i) {}    
    };

    template <typename T=Elem<def::Euclidean, 0>, typename... Ts>
    struct tuple_size {
      static const size_t value = tuple_size<Ts...>::value + 1;
    };

    template <>
    struct tuple_size<> {
      static const size_t value = 0;
    };

    template <size_t k, typename T, typename... Ts>
    struct _elem_type_holder {
      typedef typename _elem_type_holder<k - 1, Ts...>::type type;
    };
    
    template <typename T, typename... Ts>
    struct _elem_type_holder<0, T, Ts...> {
      typedef T type;
    };


    template <size_t k, typename T, typename... Ts>
    typename std::enable_if<
      k == 0, typename _elem_type_holder<0, T, Ts... >::type & >::type
    _get_phase(_ElemMultiPhaseConstructor<T, Ts...>& t) {
      return t.head;
    }

    template <size_t k, typename T, typename... Ts>
    typename std::enable_if<
      k != 0, typename _elem_type_holder<k, T, Ts...>::type & >::type
    _get_phase(_ElemMultiPhaseConstructor<T, Ts...>& t) {
      _ElemMultiPhaseConstructor<Ts...>& base = t;
      return _get_phase<k - 1>(base);
    }

  
    template <size_t k, typename T, typename... Ts>
    typename std::enable_if<
      k == 0, Block<typename _elem_type_holder<0, T, Ts... >::type> & >::type
    _get_block(_BlockMultiPhaseConstructor<T, Ts...>& t) {
      return t.head;
    }

    template <size_t k, typename T, typename... Ts>
    typename std::enable_if<
      k != 0, Block<typename _elem_type_holder<k, T, Ts...>::type> & >::type
    _get_block(_BlockMultiPhaseConstructor<T, Ts...>& t) {
      _BlockMultiPhaseConstructor<Ts...>& base = t;
      return _get_block<k - 1>(base);
    }


    template <typename T=Elem<def::Euclidean, 0>, typename... Ts>
    void _copy_elem_from_block(_ElemMultiPhaseConstructor<T, Ts...>&e,
			       _BlockMultiPhaseConstructor<T, Ts...>&b,
			       size_t ind) {
      _ElemMultiPhaseConstructor<Ts...> & e_base = e;
      _BlockMultiPhaseConstructor<Ts...> & b_base = b;
      e.head = b.head[ind];      
      _copy_elem_from_block<Ts...>(e_base, b_base, ind);
    }

    template <>
    void _copy_elem_from_block(_ElemMultiPhaseConstructor<>&e,
			       _BlockMultiPhaseConstructor<>&b,
			       size_t ind) {
    }
    
  }

    

}

#endif /* _D2_INTERNAL_H_ */
