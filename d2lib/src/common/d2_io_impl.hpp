#ifndef _D2_IO_IMPL_H_
#define _D2_IO_IMPL_H_


#include "d2_data.hpp"

#include <assert.h>

namespace d2 {

  template <typename D2Type>
  inline void d2<D2Type>::put(std::ostream &os) const { os << *this; }

  std::ostream& operator<< (std::ostream&os, const d2_base &op) {
    op.put(os); return os;
  }

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
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) {
      for (size_t j=0; j<op.dim; ++j) 
	os << op.supp[i*op.dim + j] << " ";
      os << std::endl;
    }
    return os;
  }

  template <>
  std::istream& operator>> (std::istream& is, d2<index_t> & op) {
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    for (size_t i=0; i<op.len; ++i) is >> op.supp[i];
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const d2<index_t> & op) {
    os << op.dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.supp[i] << " "; os << std::endl;
    return os;
  }

  template <>
  std::istream& operator>> (std::istream& is, d2<char> & op) {
    int c;
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    for (size_t i=0; i<op.len; ++i) {      
      while ((c =is.peek()) == ' ' || c == '\n') is.ignore();
      is.get(op.supp + i*op.dim, ' ');
    }
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const d2<char> & op) {
    os << op.dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) {
      for (size_t j=0; j<op.dim; ++j) 
	os << op.supp[i*op.dim + j];
      os << " ";
    }
    os << std::endl;
    return os;
  }

  template <>
  std::istream& operator>> (std::istream& is, d2<void> & op) {
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const d2<void> & op) {
    os << op.dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    return os;
  }



  /* append one d2 */
  template <typename D2Type>
  int d2_block<D2Type>::append(std::istream &is) {
    d2<D2Type> theone;
    is >> theone.dim >> theone.len;
    if (is.fail() || is.eof()) return 1;
    assert(theone.dim == dim);
    if (theone.len + col > max_col) {
      std::cerr << "Warning: memory insufficient, reallocate!" << std::endl;
      if (type == "euclidean") {
	max_col *=2;
	p_w = (real_t*) realloc(p_w, sizeof(real_t)*max_col);
	p_supp = (real_t*) realloc(p_supp, sizeof(real_t)*max_col*dim);
      } else {
	std::cerr << "Error: unrecognized type!" << std::endl;
	exit(1);
      }
    }
    if (theone.len > max_len) max_len = theone.len;
    theone.w = p_w + col;
    if (type == "euclidean") {
      theone.supp = p_supp + col*dim;
    } else {
      std::cerr << "Error: unrecognized type!" << std::endl;
      exit(1);
    }
    is >> theone;
    vec.push_back(theone);

    size += 1;
    col += theone.len;

    return is.eof();
  }

  template <typename D2Type>
  void d2_block<D2Type>::align_d2vec() {
    if (type == "euclidean") {
      assert(size > 0);
      vec[0].w = p_w;
      vec[0].supp = p_supp;
      for (size_t i=1; i<size; ++i) {
	vec[i].w = vec[i-1].w + vec[i-1].len;
	vec[i].supp = vec[i-1].supp + vec[i-1].len * dim;
      }
    }
  }

  template <>
  void d2_block<real_t>::read_meta(const std::string &filename) {}

  template <>
  void d2_block<index_t>::read_meta(const std::string &filename) {
    std::ifstream fs;
    fs.open(filename, std::ifstream::in);
    assert(fs.is_open());
    fs >> meta.dict_size >> meta.dict_dim;
    for (size_t i=0; i<meta.dict_size; ++i)
      for (size_t j=0; j<meta.dict_dim; ++j)
	fs >> meta.dict_embedding[i*meta.dict_dim + j];    
    fs.close();
  }


  void md2_block::read(const std::string &filename, const size_t size) {
    std::ifstream fs;
    int checkEnd = 0;

    for (size_t i=0; i<phase.size(); ++i) 
      phase[i]->read_meta(filename + ".meta" + std::to_string(i));

    /* read main file */
    fs.open(filename, std::ifstream::in);
    assert(fs.is_open());

    for (size_t i=0; i<size; ++i) {
      for (size_t j=0; j<phase.size(); ++j) {
	checkEnd += phase[j]->append(fs);
      }
      if (checkEnd > 0) break;
    }
    this->size = phase.back()->size;
    for (size_t j =0; j<phase.size(); ++j) {
      phase[j]->align_d2vec();
    }
    if (this->size < size) 
      std::cerr << "Warning: only read " << this->size << " instances." << std::endl; 
   
    fs.close();
  }

  void md2_block::write(const std::string &filename) {
    if (filename != "") {
      std::ofstream fs;
      fs.open(filename, std::ofstream::out);
      assert(fs.is_open());

      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase.size(); ++j) {
	  fs << (*phase[j])[i];
	}
      }
      fs.close();    
    } else {
      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase.size(); ++j) {
	  std::cout << (*phase[j])[i];
	}
      }

    }
  }

}

#endif /* _D2_IO_IMPL_H_ */
