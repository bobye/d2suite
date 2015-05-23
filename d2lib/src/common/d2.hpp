#ifndef _D2_H_
#define _D2_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <assert.h>

namespace d2 {

  /* this defines the float point that will be used to 
   * store real numbers.
   */
  typedef float real_t;

  /* this defines the unsigned integer type that will
   * be used to store index 
   */
  typedef unsigned index_t;


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
    D2Type* supp;

    inline void put(std::ostream &os) const { os << *this; }
  };

  template <typename D2Type1, typename D2Type2>
  inline real_t GetDistance( const d2<D2Type1>& op1, 
			     const d2<D2Type2>& op2, 
			     real_t* cache);



  class d2_block_base {
  public:
    friend class mult_d2_block;
    d2_block_base() {};
    d2_block_base(const size_t thesize, 
		  const size_t thedim,
		  const size_t thelen,
		  const std::string &thetype = std::string("euclidean")): 
      dim(thedim), len(thelen), type(thetype), col(0), max_len(0), size(0) {};

    virtual inline d2_base& operator[](size_t ind) = 0;


    size_t dim, size;
    size_t len, max_len;
    size_t col, max_col;

    std::string type;

  protected:
    virtual int append(std::istream &is) = 0;
    virtual void align_d2vec() = 0;

  };

  template <typename D2Type>
  class d2_block : public d2_block_base {
  public:
    typedef d2_block<D2Type> T;
    d2_block(const size_t thesize, 
	     const size_t thedim,
	     const size_t thelen,
	     const std::string &thetype = std::string("euclidean")): 
      d2_block_base(thesize, thedim, thelen, thetype) {
      // allocate block memory
      p_w = (real_t*) malloc(sizeof(real_t) * thesize * thelen);
      if (type == "euclidean") {
	p_supp = (real_t *) malloc(sizeof(real_t) * thesize * thelen * thedim);
      }
      max_col = thesize*thelen;
    };
    std::vector< d2<D2Type> > vec;    
    
    /* get specific d2 in the block */
    inline d2<D2Type>& operator[](size_t ind) {return vec[ind];}

  protected:
    /* actual binary data */
    real_t *p_w;
    D2Type* p_supp;

    int append(std::istream &is);
    void align_d2vec();

  };


  class mult_d2_block {
  public:
    size_t size;
    std::vector< index_t > label;
    std::vector< d2_block_base* > phase;
    std::vector< std::string > type;

    mult_d2_block(const size_t n, 
		  const size_t* dim_arr,
		  const size_t* len_arr,
		  const std::string* str_arr,
		  const size_t num_of_phases = 1)
    {      
      for (size_t i=0; i<num_of_phases; ++i) {
	if (str_arr[i] == "euclidean")
	  phase.push_back( new d2_block<real_t>(n, dim_arr[i], len_arr[i], "euclidean"));
	label.resize(n);
      }
    };

    /* file io */
    void read(const std::string &filename, const size_t size);
    void write(const std::string &filename);

    
    void write_split(const std::string &filename);    
  };


}

#endif /* _D2_H_ */
