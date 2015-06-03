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

  }

}

#endif /* _D2_INTERNAL_H_ */
