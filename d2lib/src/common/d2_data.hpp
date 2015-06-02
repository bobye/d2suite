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

  class ElemBase {
  public:
    virtual inline void put_(std::ostream &os) const = 0;
  };

  template <typename D2Type>
  class Elem : public ElemBase {
  public:
    /* this defines the dimension of supports */
    size_t dim;
    /* this defines the length of supports */
    size_t len;    
    /* this defined the weight array of supports*/
    real_t* w;
    /* this defines the support arrays */
    typename D2Type::type* supp;

    inline void put_(std::ostream &os) const;    
  };


  class BlockBase { // interface
  public:

    virtual inline ElemBase& operator[](const size_t ind) = 0;

    virtual void read_meta(const std::string &filename) = 0;
    /* read from input stream and append a new d2 to current block */
    virtual int append(std::istream &is) = 0;
    /* post processing to enforce vec[] of d2 aligning well with
     * inner data blocks (aka. p_w and p_supp)
     */
    virtual void realign_vec() = 0;
    virtual size_t & get_size() = 0;
    virtual size_t get_size() const = 0;
    virtual size_t & get_global_size() = 0;
    virtual size_t get_global_size() const = 0;
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
  class Block : public BlockBase {
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

    int append(std::istream &is);
    void read_meta(const std::string &filename);
    void realign_vec();

  };


  template<typename D1=def::Euclidean, typename...Ds>
  void _block_push_back_recursive(std::vector< BlockBase*> &phase,
				  const size_t n, 
				  const size_t* dim_arr,
				  const size_t* len_arr,
				  const index_t ind) {
    phase.push_back(new Block<D1>(n, dim_arr[ind], len_arr[ind]));
    _block_push_back_recursive<Ds...>(phase, n, dim_arr, len_arr, ind+1);
  }

  template<>
  void _block_push_back_recursive(std::vector< BlockBase*> &phase,
				  const size_t n, 
				  const size_t* dim_arr,
				  const size_t* len_arr,
				  const index_t ind) {};

  template <typename... Ts>
  class BlockMultiPhase {
  public:
    size_t size;
    std::vector< index_t > label;

    BlockMultiPhase(){};    
    BlockMultiPhase(const size_t n, 
		      const size_t* dim_arr,
		      const size_t* len_arr) {      
      _block_push_back_recursive<Ts...>(phase_, n, dim_arr, len_arr, 0);
      label.resize(n);
    };

    inline BlockBase & operator[](size_t ind) {return *phase_[ind];}
    inline BlockBase & operator[](size_t ind) const {return *phase_[ind];}

    /* file io */
    void read_meta(const std::string &filename);
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    void write(const std::string &filename) const;
    void split_write(const std::string &filename, const int num_of_copies) const;
    
    void write_split(const std::string &filename);    

    size_t get_phase_size() const { return phase_.size(); }
  private:
    std::vector< BlockBase* > phase_;

  };


  template <typename... Ts>
  class DistributedBlockMultiPhase : public BlockMultiPhase<Ts...> {    
  public:
    using BlockMultiPhase<Ts...>::BlockMultiPhase; // inherit constructor
    size_t global_size;


    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);
   
  };


}


#endif /* _D2_DATA_H_ */
