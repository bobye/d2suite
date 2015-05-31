#ifndef _D2_IO_IMPL_H_
#define _D2_IO_IMPL_H_


#include "d2_data.hpp"
#include "timer.h"
#include <string>
#include <assert.h>
#include <algorithm>

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
      std::cerr << getLogHeader() << " warning: memory insufficient, reallocate!" << std::endl;
      if (type == "euclidean") {
	max_col *=2;
	p_w = (real_t*) realloc(p_w, sizeof(real_t)*max_col);
	p_supp = (D2Type*) realloc(p_supp, sizeof(D2Type)*max_col*dim);
      }
      else if (type == "wordid") {
	max_col *=2;
	p_w = (real_t*) realloc(p_w, sizeof(real_t)*max_col);
	p_supp = (D2Type*) realloc(p_supp, sizeof(D2Type)*max_col);
      }
      else {
	std::cerr << getLogHeader() << " error: unrecognized type!" << std::endl;
	exit(1);
      }
    }
    if (theone.len > max_len) max_len = theone.len;
    theone.w = p_w + col;
    if (type == "euclidean") {
      theone.supp = p_supp + col*dim;
    } 
    else if (type == "wordid") {
      theone.supp = p_supp + col;
    }
    else {
      std::cerr << getLogHeader() << " error: unrecognized type!" << std::endl;
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
    assert(size > 0);
    vec[0].w = p_w;
    vec[0].supp = p_supp;
    if (type == "euclidean") {
      for (size_t i=1; i<size; ++i) {
	vec[i].w = vec[i-1].w + vec[i-1].len;
	vec[i].supp = vec[i-1].supp + vec[i-1].len * dim;
      }
    } else if (type == "wordid") {
      for (size_t i=1; i<size; ++i) {
	vec[i].w = vec[i-1].w + vec[i-1].len;
	vec[i].supp = vec[i-1].supp + vec[i-1].len;
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
    if (!meta.dict_embedding) {
      meta.dict_embedding = (real_t*) malloc(sizeof(real_t)* meta.dict_size * meta.dict_dim);
    }
    for (size_t i=0; i<meta.dict_size*meta.dict_dim; ++i)
	fs >> meta.dict_embedding[i];    
    fs.close();
  }


  void md2_block::read_meta(const std::string &filename) {
    using namespace std;
    double startTime = getRealTime();
    for (size_t i=0; i<phase.size(); ++i) 
      phase[i]->read_meta(filename + ".meta" + to_string(i));
    cerr << getLogHeader() << " logging: read meta data in " 
	 << (getRealTime() - startTime) << " seconds." << endl;    
  }

  void md2_block::read_main(const std::string &filename, const size_t size) {
    using namespace std;
    ifstream fs;
    int checkEnd = 0;

    double startTime = getRealTime();

    /* read main file */
    { fs.open(filename, ifstream::in); }
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

    if (this->size < size) {
      cerr << getLogHeader() << " warning: only read " 
	   << this->size << " instances." << endl; 
    }
    fs.close();
    cerr << getLogHeader() << " logging: read data in " 
	 << (getRealTime() - startTime) << " seconds." << endl;
  }

  void md2_block::read(const std::string &filename, const size_t size) {
    read_meta(filename);
    read_main(filename, size);
  }

  void md2_block::write(const std::string &filename) const {
    using namespace std;

    if (filename != "") {
      ofstream fs;
      fs.open(filename, ofstream::out);
      assert(fs.is_open());

      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase.size(); ++j) {
	  fs << (*this)[j][i];
	}
      }
      fs.close();    
    } else {
      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase.size(); ++j) {
	  cout << (*this)[j][i];
	}
      }

    }
  }

  void md2_block::split_write(const std::string &filename, const int num_of_copies) const {
    using namespace std;
    
    vector<size_t> rand_ind(size);
    for (size_t i=0; i<size; ++i) rand_ind[i] = i;
    random_shuffle(rand_ind.begin(), rand_ind.end());

    if (filename != "") {
      assert(num_of_copies > 1);
      double startTime = getRealTime();
      vector<ofstream> fs(num_of_copies);
      size_t batch_size = (size-1) / num_of_copies + 1;

      for (int j=0; j<num_of_copies; ++j) {
	fs[j].open(filename + ".part" + to_string(j), ofstream::out);
	assert(fs[j].is_open());
      }
      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase.size(); ++j) {
	  fs[i/batch_size] << (*this)[j][rand_ind[i]];
	}
      }
      for (int j=0; j<num_of_copies; ++j) fs[j].close();
      cerr << getLogHeader() << " logging: write data into part0.." << num_of_copies << " in " 
	   << (getRealTime() - startTime) << " seconds." << endl;
      
    } else {
      cerr << getLogHeader() << " error: empty filename specified." << endl;
    }
  }

}

#endif /* _D2_IO_IMPL_H_ */
