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

  class d2_base {
  public:
    virtual inline void put(std::ostream &os) const = 0;
  };

  template <typename D2Type>
  class d2 : public d2_base {
  public:
    /* this defines the dimension of supports */
    size_t dim;
    /* this defines the length of supports */
    size_t len;    
    /* this defined the weight array of supports*/
    real_t* w;
    /* this defines the support arrays */
    typename D2Type::type* supp;

    inline void put(std::ostream &os) const;    
  };


  class d2_block_base { // interface
  public:

    virtual inline d2_base& operator[](const size_t ind) = 0;

    virtual void read_meta(const std::string &filename) = 0;
    /* read from input stream and append a new d2 to current block */
    virtual int append(std::istream &is) = 0;
    /* post processing to enforce vec[] of d2 aligning well with
     * inner data blocks (aka. p_w and p_supp)
     */
    virtual void align_d2vec() = 0;
    virtual size_t & get_size() = 0;
    virtual size_t get_size() const = 0;
    virtual size_t & get_global_size() = 0;
    virtual size_t get_global_size() const = 0;
  };


  template <typename D2Type>
  class meta {};

  template <>
  class meta<def::WordVec> {
  public:
    meta(): dict_size(0), dict_dim(0), dict_embedding(NULL) {};
    size_t dict_size, dict_dim;
    real_t *dict_embedding;
  };

  template <typename D2Type>
  class d2_block : public d2_block_base {
    typedef typename D2Type::type SuppType;
  public:

    d2_block(const size_t thesize, 
	     const size_t thedim,
	     const size_t thelen): 
      dim(thedim), len(thelen), col(0), max_len(0), size(0) {
      // allocate block memory
      p_w = (real_t*) malloc(sizeof(real_t) * thesize * thelen);
      p_supp = (SuppType *) malloc(sizeof(SuppType) * D2Type::step_stride(thesize * thelen, thedim));
      max_col = thesize*thelen;
    };
    std::vector< d2<D2Type> > vec;    
    
    /* get specific d2 in the block */
    inline d2<D2Type>& operator[](const size_t ind) {return vec[ind];}
    
    size_t & get_size() {return size;}
    size_t get_size() const {return size;}
    size_t & get_global_size() {return global_size;}
    size_t get_global_size() const {return global_size;}
  protected:
    size_t dim, size;
    size_t len, max_len;
    size_t col, max_col;

    size_t global_size;

    /* actual binary data */
    typedef meta<D2Type> MetaType;
    real_t *p_w;
    SuppType* p_supp;
    MetaType meta;

    int append(std::istream &is);
    void align_d2vec();
    void read_meta(const std::string &filename);

  };


  template<typename D1=def::Euclidean, typename...Ds>
  void block_push_back_recursive(std::vector< d2_block_base*> &phase,
				 const size_t n, 
				 const size_t* dim_arr,
				 const size_t* len_arr,
				 const index_t ind) {
    phase.push_back(new d2_block<D1>(n, dim_arr[ind], len_arr[ind]));
    block_push_back_recursive<Ds...>(phase, n, dim_arr, len_arr, ind+1);
  }

  template<>
  void block_push_back_recursive(std::vector< d2_block_base*> &phase,
				 const size_t n, 
				 const size_t* dim_arr,
				 const size_t* len_arr,
				 const index_t ind) {};

  template <typename... Ts>
  class md2_block {
  public:
    size_t size;
    std::vector< index_t > label;
    std::vector< d2_block_base* > phase;

    md2_block(){};    
    md2_block(const size_t n, 
	      const size_t* dim_arr,
	      const size_t* len_arr)
    {      
      block_push_back_recursive<Ts...>(phase, n, dim_arr, len_arr, 0);
      label.resize(n);
    };

    inline d2_block_base & operator[](size_t ind) {return *phase[ind];}
    inline d2_block_base & operator[](size_t ind) const {return *phase[ind];}

    /* file io */
    void read_meta(const std::string &filename);
    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);

    void write(const std::string &filename) const;
    void split_write(const std::string &filename, const int num_of_copies) const;
    
    void write_split(const std::string &filename);    

  private:

  };


  template <typename... Ts>
  class parallel_md2_block : public md2_block<Ts...> {    
  public:
    using md2_block<Ts...>::md2_block; // inherit constructor
    size_t global_size;


    void read_main(const std::string &filename, const size_t size);
    void read(const std::string &filename, const size_t size);
   
  };


}


#endif /* _D2_DATA_H_ */
