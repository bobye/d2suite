#ifndef _D2_DATA_H_
#define _D2_DATA_H_

#include "common.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace d2 {
  /*!
   * This header defines basic data structure to work with
   * discrete distribution (d2) data. 
   */

  namespace def {    
    /*!
     * The type of discrete distribution decides how the
     * ground distance is computed. For example, d2::def::Euclidean
     * means the ground distance is computed from the Euclidean 
     * distance between two arbitray vectors. Also, the type
     * of discrete distribution also decides how the data is 
     * stored. For example, d2::def::WordVec means the raw 
     * data is stored with index, aka, the word and the meta part 
     * stores the actual vector w.r.t. individual indexed word. 
     * Thus, different types corresponds to different meta data.
     * In the case of d2::def::Euclidean, the meta data is empty.
     */
    struct Euclidean {
      typedef real_t type;
      static inline size_t step_stride(size_t col, size_t dim) {return col*dim;}
    };

    struct WordVec { // indexed vector
      typedef index_t type;
      static inline size_t step_stride(size_t col, size_t dim) {return col;}
    };

    struct NGram {
      typedef char type;
      static inline size_t step_stride(size_t col, size_t dim) {return col*dim;}
    };

    struct Histogram {
      typedef void type;
      static inline size_t step_stride(size_t col, size_t dim) {return 0;}
    };
  }



  template <typename D2Type>
  class Elem {
  public:
    /* this defines the dimension of supports */
    size_t dim;
    /* this defines the length of supports */
    size_t len;    
    /* this defined the weight array of supports*/
    real_t* w;
    /* this defines the support arrays */
    typename D2Type::type* supp;

  };


  template <typename D2Type>
  class Meta {};

  template <>
  class Meta<def::WordVec> {
  public:
    Meta(): size(0), dim(0), embedding(NULL) {};
    size_t size, dim;
    real_t *embedding;
  };

  template <typename D2Type>
  class Block {
    typedef Elem<D2Type> ElemType;
    typedef typename D2Type::type SuppType;
    typedef Meta<D2Type> MetaType;
  public:

    Block(const size_t thesize, 
	    const size_t thedim,
	    const size_t thelen): 
      dim(thedim), len(thelen), col(0), max_len(0), size(0) {
      // allocate block memory
      p_w = (real_t*) malloc(sizeof(real_t) * thesize * thelen);
      p_supp = (SuppType *) malloc(sizeof(SuppType) * D2Type::step_stride(thesize * thelen, thedim));
      max_col = thesize*thelen;
    };
    
    /* get specific d2 in the block */
    inline Elem<D2Type>& operator[](const size_t ind) {return vec_[ind];}
    
    size_t & get_size() {return size;}
    size_t get_size() const {return size;}
    size_t & get_global_size() {return global_size;}
    size_t get_global_size() const {return global_size;}
    int append(std::istream &is);
    void read_meta(const std::string &filename);
    void realign_vec();

  protected:
    std::vector< Elem<D2Type> > vec_;    
    size_t dim, size;
    size_t len, max_len;
    size_t col, max_col;

    size_t global_size;

    /* actual binary data */
    real_t *p_w;
    SuppType* p_supp;
    MetaType meta;


  };


  template <typename T1=def::Euclidean, typename... Ts>
  struct _BlockMultiPhaseConstructor: _BlockMultiPhaseConstructor<Ts...> {
    _BlockMultiPhaseConstructor (const size_t thesize, 
				 const size_t* thedim,
				 const size_t* thelen,
				 const index_t i = 0) : 
      head(new Block<T1> (thesize, *thedim, *thelen)), ind(i),
      _BlockMultiPhaseConstructor<Ts...>(thesize, thedim+1, thelen+1, i+1) {}
    index_t ind;
    Block<T1> *head;
  };
  template <>
  struct _BlockMultiPhaseConstructor<> {
    _BlockMultiPhaseConstructor (const size_t thesize, 
				 const size_t* thedim,
				 const size_t* thelen,
				 const index_t i) {}    
  };


  template <typename... Ts>
  class BlockMultiPhase {
  public:
    BlockMultiPhase(const size_t thesize, 
		       const size_t* thedim,
		       const size_t* thelen) :
      _constructor (new _BlockMultiPhaseConstructor<Ts...>(thesize, thedim, thelen)) {
    }
    _BlockMultiPhaseConstructor<Ts...> * _constructor;
    size_t size;
    
    void read_meta(const std::string &filename);
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    void write(const std::string &filename) const;
    void split_write(const std::string &filename, const size_t num_copies) const;

  };

  template <typename... Ts>
  class DistributedBlockMultiPhase : BlockMultiPhase<Ts...> {
  public:
    using BlockMultiPhase<Ts...>::BlockMultiPhase; // inherit constructor
    size_t global_size;


    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);
  };

}


#endif /* _D2_DATA_H_ */
