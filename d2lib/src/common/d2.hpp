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



  template <typename D2Type>
  struct d2 {
    /* this defines the dimension of supports */
    size_t dim;
    /* this defines the length of supports */
    size_t len;    
    /* this defined the weight array of supports*/
    real_t* w;
    /* this defines the support arrays */
    D2Type* supp;
  };

  template <typename D2Type1, typename D2Type2>
  inline real_t GetDistance( const d2<D2Type1>& op1, 
			     const d2<D2Type2>& op2, 
			     real_t* cache);


  template <typename D2Type>
  std::istream& operator>> (std::istream& is, d2<D2Type> & op);
  template <typename D2Type>
  std::ostream& operator<< (std::ostream& os, const d2<D2Type> &op);

  // specifications
  template <>
  std::istream& operator>> (std::istream& is, d2<real_t> & op) {
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    for (size_t i=0; i<op.len * op.dim; ++i) is >> op.supp[i];
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const d2<real_t> & op) {
    os << op.dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i]; os << std::endl;
    for (size_t i=0; i<op.len; ++i) {
      for (size_t j=0; j<op.dim; ++j) 
	os << op.supp[i*op.dim + j];
      os << std::endl;
    }
    return os;
  }


  class d2_block_base {
  public:
    d2_block_base() {};
    d2_block_base(const size_t thesize, 
		  const size_t thedim,
		  const size_t thelen,
		  const std::string &thetype = std::string("euclidean")): 
      dim(thedim), len(thelen), type(thetype), col(0), max_len(0), size(0) {};

    virtual int append(std::istream &is) = 0;

    size_t dim, size;
    size_t len, max_len;
    size_t col, max_col;

    std::string type;
  };

  template <typename D2Type>
  class d2_block : public d2_block_base {
  public:
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
    inline d2<D2Type> operator[](size_t ind) const {return vec[ind];}

    int append(std::istream &is);
  private:
    /* actual binary data */
    real_t *p_w;
    D2Type* p_supp;

  };

  /* append one d2 */
  template <typename D2Type>
  int d2_block<D2Type>::append(std::istream &is) {
    d2<D2Type> theone;
    is >> theone.dim >> theone.len;
    if (is.fail() || is.eof()) return 1;
    assert(theone.dim == dim);
    if (theone.len + col > max_col) {
      if (type == "euclidean") {
	max_col *=2;
	p_w = (real_t*) realloc(p_w, sizeof(real_t)*max_col);
	p_supp = (real_t*) realloc(p_supp, sizeof(real_t)*max_col*dim);
      } else {
	std::cerr << "Unrecognized type!" << std::endl;
      }
    }
    size += 1;
    col += theone.len;
    if (theone.len > max_len) max_len = theone.len;
    theone.w = p_w + col;
    if (type == "euclidean") {
      theone.supp = p_supp + col*dim;
    } else {
      std::cerr << "Unrecognized type!" << std::endl;
    }
    is >> theone;

    return is.eof();
  }
  

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
    void read(std::string filename, size_t size);
    void write(std::string filename);


    void write_split(std::string filename);    
  };

  void mult_d2_block::read(std::string filename, size_t size) {
    std::ifstream fs;
    int checkEnd = 0;
    fs.open(filename, std::ifstream::in);
    assert(fs.is_open());

    for (size_t i=0; i<size; ++i) {
      for (size_t j=0; j<phase.size(); ++j) {
	checkEnd += phase[j]->append(fs);
      }
      if (checkEnd > 0) break;
    }
    this->size = phase.back()->size;
    if (this->size < size) 
      std::cerr << "Warning: only read " << this->size << " instances." << std::endl; 

    fs.close();
  }

}

#endif /* _D2_H_ */
