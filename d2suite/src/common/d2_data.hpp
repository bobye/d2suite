#ifndef _D2_DATA_H_
#define _D2_DATA_H_
/*!
 * \file d2_data.hpp
 * \brief This header defines basic data structure to work with
 * discrete distribution (d2) data. 
 */

#include "common.hpp"
#include "d2_internal.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include <tuple> 

namespace d2 {
  namespace def {
    /*! \brief regular vector */
    struct Euclidean {
      typedef real_t type;
      static inline size_t step_stride(size_t col, size_t dim) {return col*dim;}
    };

    /*! \brief indexed vector */
    struct WordVec { 
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

    template <typename FuncType>
    struct Function {
      typedef FuncType type;
      static inline size_t step_stride(size_t col, size_t dim) {return col;}
    };
  }


  template <typename D2Type, size_t dim>
  struct Elem {
  public:
    /*! the type of support points */
    typedef D2Type T;
    /*! the actual dimension of support points */
    static const size_t D = dim;
    /*! the length of supports */
    size_t len;    
    /*! the weight array of supports*/
    real_t* w;
    /*! the support arrays */
    typename D2Type::type* supp;
    /*! the label array of supports (optional) */
    real_t* label;
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
      size(0), len(thelen), max_len(0), col(0), isShared(false) {
      // allocate block memory
      p_w = (real_t*) malloc(sizeof(real_t) * thesize * thelen);
      p_label = (real_t*) malloc(sizeof(real_t) * thesize * thelen);      
      p_supp = (SuppType *) malloc(sizeof(SuppType) * ElemType::T::step_stride(thesize * thelen, ElemType::D));
      assert(p_w && p_label && p_supp);
      max_col = thesize*thelen;
    };
    Block(const Block<ElemType> &that, index_t start, size_t thesize, bool isview = true) {
      //      assert(that.get_size() >= start + thesize);
      size = 0;
      col = 0;
      max_col = -1;
      len = that.len;
      max_len = 0;
      size_t end = std::min(thesize+start, that.get_size());
      for (size_t i=start; i< end; ++i) {
	size+= 1;
	col += that[i].len;
	if (max_len < that[i].len) max_len = that[i].len;
      }
      vec_.resize(thesize);
      for (size_t i=0; i< size; ++i) {
	vec_[i].len = that[i+start].len;
      }
      if (isview) {
	isShared = true;
	p_w = that[start].w;
	p_label = that[start].label;
	p_supp = that[start].supp;
      } else {
	isShared = false;
	p_w = (real_t*) malloc(sizeof(real_t) * col);
	p_label = (real_t*) malloc(sizeof(real_t) * col);
	p_supp = (SuppType *) malloc(sizeof(SuppType) * ElemType::T::step_stride(col, ElemType::D));
	memcpy(p_w, &(that[start].w), sizeof(real_t) * col);
	memcpy(p_label, &(that[start].label), sizeof(real_t) * col);
	memcpy(p_supp, &(that[start].supp), sizeof(SuppType) * ElemType::T::step_stride(col, ElemType::D));
      }
      realign_vec();
      meta = that.meta; // shallow copy
      meta.to_shared();
    };
    ~Block() {
      if (!isShared) {
	if (p_w != NULL) free(p_w); 
	if (p_label != NULL) free(p_label); 
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
    inline real_t* &get_label_ptr() {return p_label;}
    inline real_t* get_label_ptr() const {return p_label;}
    inline SuppType* &get_support_ptr() {return p_supp;}
    inline SuppType* get_support_ptr() const {return p_supp;}
    inline MetaType &get_meta() {return meta;}
    inline MetaType get_meta() const {return meta;}
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
    void read(const std::string &filename, const size_t size, const std::string &filename_meta);
    void read(const std::string &filename, const size_t size, const MetaType &meta);    
    void read_label(const std::string &filename, const size_t start = 0);
      
    void write(const std::string &filename) const;
    void split_write(const std::string &filename, const size_t num_copies) const;
    void train_test_split_write(const std::string &filename, const real_t train_ratio = 0.7, const size_t start = 0, const unsigned int seed = 0) const;

#ifdef RABIT_RABIT_H_
    void sync(const index_t rank) {
      rabit::Broadcast(&size, sizeof(size_t), rank);      
      rabit::Broadcast(&len, sizeof(size_t), rank);
      rabit::Broadcast(&col, sizeof(size_t), rank);
      rabit::Broadcast(&max_len, sizeof(size_t), rank);
      rabit::Broadcast(&max_col, sizeof(size_t), rank);
      rabit::Broadcast(&isShared, sizeof(bool), rank);
      rabit::Broadcast(p_w, sizeof(real_t) * size * len, rank);
      rabit::Broadcast(p_label, sizeof(real_t) * size * len, rank);
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
    real_t *p_w, *p_label;
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





