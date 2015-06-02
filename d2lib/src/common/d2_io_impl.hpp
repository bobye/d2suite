#ifndef _D2_IO_IMPL_H_
#define _D2_IO_IMPL_H_


#include "d2_data.hpp"
#include "timer.h"
#include <string>
#include <assert.h>
#include <algorithm>

namespace d2 {

  template <typename D2Type>
  inline void Elem<D2Type>::put_(std::ostream &os) const { os << *this; }

  std::ostream& operator<< (std::ostream&os, const ElemBase &op) {
    op.put_(os); return os;
  }

  template <typename D2Type>
  std::istream& operator>> (std::istream& is, Elem<D2Type> & op);
  template <typename D2Type>
  std::ostream& operator<< (std::ostream& os, const Elem<D2Type> &op);

  // specifications
  template <>
  std::istream& operator>> (std::istream& is, Elem<def::Euclidean> & op) {
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    for (size_t i=0; i<op.len * op.dim; ++i) is >> op.supp[i];
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const Elem<def::Euclidean> & op) {
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
  std::istream& operator>> (std::istream& is, Elem<def::WordVec> & op) {
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    for (size_t i=0; i<op.len; ++i) is >> op.supp[i];
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const Elem<def::WordVec> & op) {
    os << op.dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.supp[i] << " "; os << std::endl;
    return os;
  }

  template <>
  std::istream& operator>> (std::istream& is, Elem<def::NGram> & op) {
    int c;
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    for (size_t i=0; i<op.len; ++i) {      
      while ((c =is.peek()) == ' ' || c == '\n') is.ignore();
      is.get(op.supp + i*op.dim, ' ');
    }
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const Elem<def::NGram> & op) {
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
  std::istream& operator>> (std::istream& is, Elem<def::Histogram> & op) {
    for (size_t i=0; i<op.len; ++i) is >> op.w[i];
    return is;
  }

  template <>
  std::ostream& operator<< (std::ostream& os, const Elem<def::Histogram> & op) {
    os << op.dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    return os;
  }



  /* append one d2 */
  template <typename D2Type>
  int Block<D2Type>::append(std::istream &is) {
    Elem<D2Type> theone;
    is >> theone.dim >> theone.len;
    if (is.fail() || is.eof()) return 1;
    assert(theone.dim == dim);
    if (theone.len + col > max_col) {
      std::cerr << getLogHeader() << " warning: memory insufficient, reallocate!" << std::endl;
      max_col *=2;
      p_w = (real_t*) realloc(p_w, sizeof(real_t)*max_col);
      p_supp = (SuppType*) realloc(p_supp, sizeof(SuppType)*D2Type::step_stride(max_col,dim));
    }
    if (theone.len > max_len) max_len = theone.len;
    theone.w = p_w + col;
    theone.supp = p_supp + D2Type::step_stride(col,dim);
    is >> theone;
    vec_.push_back(theone);

    size += 1;
    col += theone.len;

    return is.eof();
  }

  template <typename D2Type>
  void Block<D2Type>::realign_vec() {
    assert(size > 0);
    vec_[0].w = p_w;
    vec_[0].supp = p_supp;

    for (size_t i=1; i<size; ++i) {
      vec_[i].w = vec_[i-1].w + vec_[i-1].len;
      vec_[i].supp = vec_[i-1].supp + D2Type::step_stride(vec_[i-1].len, dim);
    }
  }

  template <>
  void Block<def::Euclidean>::read_meta(const std::string &filename) {}

  template <>
  void Block<def::WordVec>::read_meta(const std::string &filename) {
    std::ifstream fs;
    fs.open(filename, std::ifstream::in);
    assert(fs.is_open());
    fs >> meta.size >> meta.dim;
    if (!meta.embedding) {
      meta.embedding = (real_t*) malloc(sizeof(real_t)* meta.size * meta.dim);
    }
    for (size_t i=0; i<meta.size*meta.dim; ++i)
	fs >> meta.embedding[i];    
    fs.close();
  }


  template<typename... Ts>
  void BlockMultiPhase<Ts...>::read_meta(const std::string &filename) {
    using namespace std;
    double startTime = getRealTime();
    for (size_t i=0; i<phase_.size(); ++i) 
      phase_[i]->read_meta(filename + ".meta" + to_string(i));
    cerr << getLogHeader() << " logging: read meta data in " 
	 << (getRealTime() - startTime) << " seconds." << endl;    
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::read_main(const std::string &filename, const size_t size) {
    using namespace std;
    ifstream fs;
    int checkEnd = 0;

    double startTime = getRealTime();

    /* read main file */
    { fs.open(filename, ifstream::in); }
    assert(fs.is_open());

    for (size_t i=0; i<size; ++i) {
      for (size_t j=0; j<phase_.size(); ++j) {
	checkEnd += phase_[j]->append(fs);
      }
      if (checkEnd > 0) break;
    }
    this->size = phase_.back()->get_size();
    for (size_t j =0; j<phase_.size(); ++j) {
      phase_[j]->realign_vec();
    }

    if (this->size < size) {
      cerr << getLogHeader() << " warning: only read " 
	   << this->size << " instances." << endl; 
    }
    fs.close();
    cerr << getLogHeader() << " logging: read data in " 
	 << (getRealTime() - startTime) << " seconds." << endl;
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::read(const std::string &filename, const size_t size) {
    read_meta(filename);
    read_main(filename, size);
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::write(const std::string &filename) const {
    using namespace std;

    if (filename != "") {
      ofstream fs;
      fs.open(filename, ofstream::out);
      assert(fs.is_open());

      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase_.size(); ++j) {
	  fs << (*this)[j][i];
	}
      }
      fs.close();    
    } else {
      for (size_t i=0; i<size; ++i) {
	for (size_t j=0; j<phase_.size(); ++j) {
	  cout << (*this)[j][i];
	}
      }

    }
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::split_write(const std::string &filename, const int num_of_copies) const {
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
	for (size_t j=0; j<phase_.size(); ++j) {
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

