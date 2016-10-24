#ifndef _D2_DATA_H_
#define _D2_DATA_H_

#include "common.hpp"
#include "d2_internal.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include <tuple> 

namespace d2 {
  /*!
   * This header defines basic data structure to work with
   * discrete distribution (d2) data. 
   */

  namespace def {    
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
      typedef char type;
      static inline size_t step_stride(size_t col, size_t dim) {return 0;}
    };

    struct SparseHistogram {
      typedef index_t type;
      static inline size_t step_stride(size_t col, size_t dim) {return col;}
    };
  }



  template <typename D2Type, size_t dim>
  struct Elem {
  public:
    typedef D2Type T;
    static const size_t D = dim;
    /* this defines the length of supports */
    size_t len;    
    /* this defined the weight array of supports*/
    real_t* w;
    /* this defines the support arrays */
    typename D2Type::type* supp;
  };



  template <typename ElemType>
  class Meta : public internal::_Meta<typename ElemType::T, ElemType::D> {
    using internal::_Meta<typename ElemType::T, ElemType::D>::_Meta;
  };

  template <typename ElemType>
  class Block {
    typedef typename ElemType::T::type SuppType;
    typedef Meta<ElemType> MetaType;
  public:
    Block(const size_t thesize, 
	  const size_t thelen): 
      len(thelen), col(0), max_len(0), size(0), isShared(false) {
      // allocate block memory
      p_w = (real_t*) malloc(sizeof(real_t) * thesize * thelen);
      p_supp = (SuppType *) malloc(sizeof(SuppType) * ElemType::T::step_stride(thesize * thelen, ElemType::D));
      assert(p_w && p_supp);
      max_col = thesize*thelen;
    };
    Block(const Block<ElemType> &that, index_t start, size_t thesize) {
      assert(that.get_size() >= start + thesize);
      size = thesize;
      col = 0;
      max_col = -1;
      len = that.len;
      max_len = 0;
      isShared = true;
      p_w = that[start].w;
      p_supp = that[start].supp;
      for (int i=start; i< size+start; ++i) {
	col += that[i].len;
	if (max_len < that[i].len) max_len = that[i].len;
      }
      vec_.resize(thesize);
      for (int i=0; i< size; ++i) {
	vec_[i].len = that[i+start].len;
      }
      realign_vec();
    };
    ~Block() {
      if (!isShared) {
	if (p_w != NULL) free(p_w); 
	if (p_supp != NULL) free(p_supp);
      }
    }
    
    /* get specific d2 in the block */
    inline ElemType& operator[](const size_t ind) {return vec_[ind];}
    inline const ElemType& operator[](const size_t ind) const {return vec_[ind];}
    
    inline size_t & get_size() {return size;}
    inline size_t get_size() const {return size;}
    inline size_t & get_col() {return col;}
    inline size_t get_col() const {return col;}
    inline size_t & get_max_len() {return max_len;}
    inline size_t get_max_len() const {return max_len;}
    inline real_t* &get_weight_ptr() {return p_w;}
    inline real_t* get_weight_ptr() const {return p_w;}
    inline SuppType* &get_support_ptr() {return p_supp;}
    inline SuppType* get_support_ptr() const {return p_supp;}
    inline void initialize(const size_t thesize, const size_t thelen) {
      len = thelen;
      size = thesize;
      col = thesize * thelen;
      max_len = thesize;
      vec_.resize(thesize);
      for (size_t i=0; i<thesize; ++i) vec_[i].len = thelen;
      realign_vec();
    }

    int append(std::istream &is);    
    void realign_vec();
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    void write(const std::string &filename) const;
    void split_write(const std::string &filename, const size_t num_copies) const;

#ifdef RABIT_RABIT_H_
    void sync(const index_t rank) {
      rabit::Broadcast(&size, sizeof(size_t), rank);      
      rabit::Broadcast(&len, sizeof(size_t), rank);
      rabit::Broadcast(&col, sizeof(size_t), rank);
      rabit::Broadcast(&max_len, sizeof(size_t), rank);
      rabit::Broadcast(&max_col, sizeof(size_t), rank);
      rabit::Broadcast(&isShared, sizeof(bool), rank);
      rabit::Broadcast(p_w, sizeof(real_t) * size * len, rank);
      if (ElemType::D > 0) {
	rabit::Broadcast(p_supp, sizeof(SuppType) * ElemType::T::step_stride(size * len, ElemType::D), rank);
      }
      size_t size_of_vec_ = vec_.size();
      rabit::Broadcast(&size_of_vec_, sizeof(size_t), rank);
      if (rabit::GetRank() != rank) vec_.resize(size_of_vec_);
      rabit::Broadcast(&vec_[0], sizeof(ElemType) * size_of_vec_, rank);
    }
#endif
    
    MetaType meta;

    Block<ElemType> & get_subblock(index_t start, size_t thesize) {
      auto ptr=new Block<ElemType>(*this, start, thesize);
      return *ptr;
    }

  protected:
    std::vector< ElemType > vec_;    
    size_t size;
    size_t len, max_len;
    size_t col, max_col;
    bool isShared;

    /* actual binary data */
    real_t *p_w;
    SuppType* p_supp;
  };

  template <typename... Ts>
  struct ElemMultiPhase : public internal::_ElemMultiPhaseConstructor<Ts...> {
    using internal::_ElemMultiPhaseConstructor<Ts...>::_ElemMultiPhaseConstructor;
    template <size_t k>
    typename internal::_elem_type_holder<k, Ts...>::type &
    get_phase() {return internal::_get_phase<k, Ts...>(*this);}

    template <size_t k>
    const typename internal::_elem_type_holder<k, Ts...>::type &
    get_phase() const {return internal::_get_phase<k, Ts...>(*this);}

    size_t get_max_len() const {return internal::_get_max_len(*this);}

  };


  template <typename... Ts>
  class BlockMultiPhase : public internal::_BlockMultiPhaseConstructor<Ts...> {
  public:
    using internal::_BlockMultiPhaseConstructor<Ts...>::_BlockMultiPhaseConstructor;

    size_t & get_size() {return size;}
    size_t get_size() const {return size;}
    
    void read_meta(const std::string &filename);
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    void write(const std::string &filename) const;
    void split_write(const std::string &filename, const size_t num_copies) const;

    template <size_t k> 
    Block<typename internal::_elem_type_holder<k, Ts...>::type> & 
    get_block() { return internal::_get_block<k, Ts...>(*this); }

    template <size_t k>
    typename internal::_elem_type_holder<k, Ts...>::type & 
    get_elem(int ind) { return (internal::_get_block<k, Ts...>(*this))[ind];}

    template <size_t k>
    const typename internal::_elem_type_holder<k, Ts...>::type & 
    get_elem(int ind) const { return (internal::_get_block<k, Ts...>(*this))[ind];}

    // return a multiple phase element by index, meta data are copied
    ElemMultiPhase<Ts...> * get_multiphase_elem(int ind) {
      ElemMultiPhase<Ts...> * ptr = new ElemMultiPhase<Ts...>(0);
      internal::_copy_elem_from_block<Ts...>(*ptr, *this, ind);
      return ptr;
    }

    size_t get_max_len() const {return internal::_get_max_len(*this);}

  protected:
    size_t size;

  };

#ifdef RABIT_RABIT_H_
  template <typename ElemType>
  class DistributedBlock : public Block<ElemType> {
  public:
    DistributedBlock(const size_t thesize, const size_t thelen):
      Block<ElemType>((thesize-1) / rabit::GetWorldSize() + 1, thelen) {};
    
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    size_t & get_global_size() {return global_size;}
    size_t get_global_size() const {return global_size;}
    
  protected:
    size_t global_size;
    
  };

  template <typename... Ts>
  class DistributedBlockMultiPhase : public BlockMultiPhase<Ts...> {
  public:
    DistributedBlockMultiPhase(const size_t thesize, const size_t* thelen):
      BlockMultiPhase<Ts...>((thesize-1) / rabit::GetWorldSize() + 1, thelen, 0) {};
    
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    size_t & get_global_size() {return global_size;}
    size_t get_global_size() const {return global_size;}
    
  protected:
    size_t global_size;

  };
#endif
}


#endif /* _D2_DATA_H_ */
