#ifndef _D2_INTERNAL_H_
#define _D2_INTERNAL_H_


#include "d2.hpp"
#include <assert.h>
#include <fstream>

namespace d2 {
  namespace internal {

    template <typename D2Type, size_t D>
    class _Meta {
    public:
      void read(const std::string &filename) {};
    };

    template <size_t D>
    class _Meta<def::WordVec, D> : public _Meta<def::Euclidean, D> {
    public:
      _Meta(): size(0), embedding(NULL), _is_allocated(false) {};
      size_t size;
      real_t *embedding;
      void allocate() {
	embedding = new real_t [size*D];
	_is_allocated = true;
      }
      void read(const std::string &filename) {
	std::ifstream fs;
	size_t d;
	fs.open(filename, std::ifstream::in);
	assert(fs.is_open());
	fs >> d; assert(d == D);
	fs >> size;  
	if (!embedding) {
	  allocate();
	}
	for (size_t i=0; i<size*D; ++i)
	  fs >> embedding[i];    
	fs.close();
      }
      ~_Meta() {
	if (embedding!=NULL && _is_allocated) delete [] embedding;
      }
    private:
      bool _is_allocated;
    };

    template <size_t D>
    class _Meta<def::Histogram, D> {
    public:
      _Meta(): size(0), dist_mat(NULL), _is_allocated(false) {};
      size_t size;
      real_t *dist_mat;
      void allocate() {
	dist_mat = new real_t [size*size];
	_is_allocated = true;
      }
      void read(const std::string &filename) {
	std::ifstream fs;
	size_t d;
	fs.open(filename, std::ifstream::in);
	assert(fs.is_open());
	fs >> d >> size; assert(d == 0);
	if (!dist_mat) {
	  allocate();
	}
	for (size_t i=0; i<size*size; ++i)
	  fs >> dist_mat[i];    
	fs.close();
      }
      ~_Meta() {
	if (dist_mat!=NULL && _is_allocated) delete [] dist_mat;
      }
    private:
      bool _is_allocated;
    };

    template <size_t D>
    class _Meta<def::NGram, D>: public _Meta<def::Histogram, D> {
    public:
      using _Meta<def::Histogram, D>::_Meta;
      index_t vocab[255]; // map from char to index
    };

    template <size_t D>
    class _Meta<def::SparseHistogram, D> : public _Meta<def::Histogram, D> {
    public:
      using _Meta<def::Histogram, D>::_Meta;
    };

    /*
     * convert sparse representations into dense representation 
     * for performance with small overheads
     */
    template <typename ElemType>
    void get_dense_if_need(const Block<ElemType> &block, real_t **X);
    
    template <size_t D>
    void get_dense_if_need(const Block<Elem<def::WordVec, D> >&block, real_t **X) {
      *X=new real_t[D * block.get_col()];
      for (size_t i=0; i<block.get_col(); ++i) {
	real_t* p=(*X) + i*D;
	real_t* q=&block.meta.embedding[D*block.get_support_ptr()[i]];
	for (size_t d=0; d<D; ++d)
	  p[d] = q[d];
      }	
    }
   
    template <size_t D>
    void get_dense_if_need(const Block<Elem<def::Euclidean, D> >&block, real_t **X) {
      *X = block.get_support_ptr();
    }

    template <typename ElemType>
    void release_dense_if_need(const Block<ElemType> &block, real_t **X);

    template <size_t D>
    void release_dense_if_need(const Block<Elem<def::WordVec, D> > &block, real_t **X) {
      delete [] (*X);
    }

    template <size_t D>
    void release_dense_if_need(const Block<Elem<def::Euclidean, D> > &block, real_t **X) {
    }


    /*
     * convert sparse representations into dense representation 
     * for performance with small overheads
     *
     * branch ad-hoc code for (extra_class)
     */
    template <typename ElemType>
    void get_dense_if_need_ec(const Block<ElemType> &block, real_t **X);
    
    template <size_t D>
    void get_dense_if_need_ec(const Block<Elem<def::WordVec, D> >&block, real_t **X) {
      *X=new real_t[D * block.get_col() * 2];
      for (size_t i=0; i<block.get_col(); ++i) {
	real_t* p=(*X) + i*D;
	real_t* q=&block.meta.embedding[D*block.get_support_ptr()[i]];
	for (size_t d=0; d<D; ++d)
	  p[d] = q[d];
      }

      std::memcpy(*X + D * block.get_col(), *X, sizeof(real_t) * D * block.get_col());      
    }
   
    template <size_t D>
    void get_dense_if_need_ec(const Block<Elem<def::Euclidean, D> >&block, real_t **X) {
      *X = block.get_support_ptr();
    }

    template <typename ElemType>
    void release_dense_if_need_ec(const Block<ElemType> &block, real_t **X);

    template <size_t D>
    void release_dense_if_need_ec(const Block<Elem<def::WordVec, D> > &block, real_t **X) {
      delete [] (*X);
    }

    template <size_t D>
    void release_dense_if_need_ec(const Block<Elem<def::Euclidean, D> > &block, real_t **X) {
    }
    
    
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
			       const _BlockMultiPhaseConstructor<T, Ts...>&b,
			       size_t ind) {
      _ElemMultiPhaseConstructor<Ts...> & e_base = e;
      const _BlockMultiPhaseConstructor<Ts...> & b_base = b;
      e.head = b.head[ind];      
      _copy_elem_from_block<Ts...>(e_base, b_base, ind);
    }

    template <>
    void _copy_elem_from_block(_ElemMultiPhaseConstructor<>&e,
			       const _BlockMultiPhaseConstructor<>&b,
			       size_t ind) {
    }

    template <typename T=Elem<def::Euclidean, 0>, typename... Ts>
    size_t _get_max_len(const _ElemMultiPhaseConstructor<T, Ts...> &e) {
      const _ElemMultiPhaseConstructor<Ts...> & e_base = e;
      return std::max(e.head.len, _get_max_len<Ts...>(e_base));
    }

    template <>
    size_t _get_max_len(const _ElemMultiPhaseConstructor<> &e) {
      return 0;
    }

    template <typename T=Elem<def::Euclidean, 0>, typename... Ts>
    size_t _get_max_len(const _BlockMultiPhaseConstructor<T, Ts...> &b) {
      const _BlockMultiPhaseConstructor<Ts...> & b_base = b;
      return std::max(b.head.get_max_len(), _get_max_len<Ts...>(b_base));
    }

    template <>
    size_t _get_max_len(const _BlockMultiPhaseConstructor<> &b) {
      return 0;
    }

    
    
  }

    

}

#endif /* _D2_INTERNAL_H_ */
