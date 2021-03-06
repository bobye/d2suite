#ifndef _D2_IO_IMPL_H_
#define _D2_IO_IMPL_H_


#include "d2_data.hpp"
#include "timer.h"
#include <string>
#include <assert.h>
#include <algorithm>
#include <random>

namespace d2 {

  /*! \brief overloading >> operator for input stream */
  template <typename D2Type, size_t dim>
  std::istream& operator>> (std::istream& is, Elem<D2Type, dim> & op);
  /*! \brief overloading << operator for output stream */
  template <typename D2Type, size_t dim>
  std::ostream& operator<< (std::ostream& os, const Elem<D2Type, dim> &op);


  // specifications
  template <size_t dim>
  std::istream& operator>> (std::istream& is, Elem<def::Euclidean, dim> & op) {
    real_t sum = 0.;
    for (size_t i=0; i<op.len; ++i) {is >> op.w[i]; sum += op.w[i];}
    for (size_t i=0; i<op.len; ++i) op.w[i] /= sum;
    for (size_t i=0; i<op.len * dim; ++i) is >> op.supp[i];
    return is;
  }

  template <size_t dim>
  std::ostream& operator<< (std::ostream& os, const Elem<def::Euclidean, dim> & op) {
    os << dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) {
      for (size_t j=0; j<dim; ++j) 
	os << op.supp[i*dim + j] << " ";
      os << std::endl;
    }
    return os;
  }

  template <size_t dim>
  std::istream& operator>> (std::istream& is, Elem<def::WordVec, dim> & op) {
    real_t sum = 0.;
    for (size_t i=0; i<op.len; ++i) {is >> op.w[i]; sum += op.w[i];}
    for (size_t i=0; i<op.len; ++i) op.w[i] /= sum;
    for (size_t i=0; i<op.len; ++i) {is >> op.supp[i];
      if (op.supp[i]>0) op.supp[i]--;}
    return is;
  }

  template <size_t dim>
  std::ostream& operator<< (std::ostream& os, const Elem<def::WordVec, dim> & op) {
    os << dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.supp[i]+1 << " "; os << std::endl;
    return os;
  }

  template <size_t dim>
  std::istream& operator>> (std::istream& is, Elem<def::NGram, dim> & op) {
    int c;
    real_t sum = 0.;
    for (size_t i=0; i<op.len; ++i) {is >> op.w[i]; sum += op.w[i];}
    for (size_t i=0; i<op.len; ++i) op.w[i] /= sum;
    for (size_t i=0; i<op.len; ++i) {      
      while ((c =is.peek()) == ' ' || c == '\n') is.ignore();
      is.get(op.supp + i*dim, ' ');
    }
    return is;
  }

  template <size_t dim>
  std::ostream& operator<< (std::ostream& os, const Elem<def::NGram, dim> & op) {
    os << dim << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) {
      for (size_t j=0; j<dim; ++j) 
	os << op.supp[i*dim + j];
      os << " ";
    }
    os << std::endl;
    return os;
  }

  std::istream& operator>> (std::istream& is, Elem<def::Histogram, 0> & op) {
    real_t sum = 0.;
    for (size_t i=0; i<op.len; ++i) {is >> op.w[i]; sum += op.w[i];}
    for (size_t i=0; i<op.len; ++i) op.w[i] /= sum;
    return is;
  }

  std::ostream& operator<< (std::ostream& os, const Elem<def::Histogram, 0> & op) {
    os << 0 << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    return os;
  }

  std::istream& operator>> (std::istream& is, Elem<def::SparseHistogram, 0> & op) {
    real_t sum = 0.;
    for (size_t i=0; i<op.len; ++i) {is >> op.w[i]; sum += op.w[i];}
    for (size_t i=0; i<op.len; ++i) op.w[i] /= sum;
    for (size_t i=0; i<op.len; ++i) {is >> op.supp[i]; op.supp[i]--;}
    return is;
  }

  std::ostream& operator<< (std::ostream& os, const Elem<def::SparseHistogram, 0> & op) {
    os << 0 << std::endl << op.len << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.w[i] << " "; os << std::endl;
    for (size_t i=0; i<op.len; ++i) os << op.supp[i]+1 << " "; os << std::endl;
    return os;
  }

  template<size_t length>
  std::ostream& operator<<= (std::ostream& os, const Elem<def::SparseHistogram, 0> &op) {
    real_t w[length]={0};
    for (size_t i=0; i<op.len; ++i) w[op.supp[i]] = op.w[i];
    os << 0 << std::endl << length << std::endl;
    for (size_t i=0; i<length; ++i) os << w[i] << " "; os << std::endl;
    return os;
  }
  /* append one d2 */
  template <typename ElemType>
  int Block<ElemType>::append(std::istream &is) {
    ElemType theone;
    int dim;
    is >> dim >> theone.len; 
    if (is.fail() || is.eof()) return 1;
    assert(dim == ElemType::D || 0 == ElemType::D);
    if (theone.len + col > max_col) {
      std::cerr << getLogHeader() << " warning: memory insufficient, reallocate!" << std::endl;
      max_col *=2;
      p_w = (real_t*) realloc(p_w, sizeof(real_t)*max_col);
      p_label = (real_t*) realloc(p_label, sizeof(real_t)*max_col);
      p_supp = (SuppType*) realloc(p_supp, sizeof(SuppType)*ElemType::T::step_stride(max_col, ElemType::D));
    }
    if (theone.len > max_len) max_len = theone.len;
    theone.w = p_w + col;
    theone.label = p_label + col;
    theone.supp = p_supp + ElemType::T::step_stride(col,ElemType::D);
    is >> theone;
    vec_.push_back(theone);

    size += 1;
    col += theone.len;

    return is.eof();
  }

  template <typename ElemType>
  void Block<ElemType>::realign_vec() {
    assert(size > 0);
    vec_[0].w = p_w;
    vec_[0].label = p_label;
    vec_[0].supp = p_supp;

    for (size_t i=1; i<size; ++i) {
      vec_[i].w = vec_[i-1].w + vec_[i-1].len;
      vec_[i].label = vec_[i-1].label + vec_[i-1].len;
      vec_[i].supp = vec_[i-1].supp + ElemType::T::step_stride(vec_[i-1].len, ElemType::D);
    }
  }


  template <typename ElemType>
  void Block<ElemType>::read(const std::string &filename, const size_t size) {
    meta.read(filename + ".meta0");
    read_main(filename, size);
  }

  template <typename ElemType>
  void Block<ElemType>::read(const std::string &filename, const size_t size, const std::string &filename_meta) {
    meta.read(filename_meta);
    read_main(filename, size);
  }

  template <typename ElemType>
  void Block<ElemType>::read(const std::string &filename, const size_t size, const MetaType &meta) {
    this->meta = meta;
    this->meta.to_shared();
    read_main(filename, size);
  }
  
  template <typename ElemType>
  void Block<ElemType>::read_label(const std::string &filename, const size_t start) {
    std::ifstream fs;
    fs.open(filename, std::ifstream::in);
    assert(fs.is_open());
    for (size_t i=0; i<size; ++i) {
      real_t label;
      fs >> label;
      for (size_t j=0; j<vec_[i].len; ++j)
	vec_[i].label[j] = label - start;
    }
  }

  namespace internal {
    template<typename T1=Elem<def::Euclidean, 0>, typename... Ts>
    int _append(std::istream &is, _BlockMultiPhaseConstructor<T1, Ts...> &t) {
      int tag = (t.head).append(is);
      _BlockMultiPhaseConstructor<Ts...> & base = t.tail;
      return tag || _append(is, base);
    }

    template<>
    int _append(std::istream &is, _BlockMultiPhaseConstructor<> &t) {
      return 0;
    }

    template<typename ElemType>
    int _append(std::istream &is, Block<ElemType> &t) {
      return t.append(is);
    }
  

    template<typename T1=Elem<def::Euclidean, 0>, typename... Ts>
    void _read_meta(const std::string &filename, _BlockMultiPhaseConstructor<T1, Ts...> &t) {
      t.head.meta.read(filename + ".meta" + std::to_string(t.ind));
      _BlockMultiPhaseConstructor<Ts...> & base = t.tail;
      _read_meta<Ts...>(filename, base);
    } 
    template<>
    void _read_meta(const std::string &filename, _BlockMultiPhaseConstructor<> &t) {}

    template<typename T1=Elem<def::Euclidean, 0>, typename... Ts>
    void _realign_vec(_BlockMultiPhaseConstructor<T1, Ts...> &t) {
      (t.head).realign_vec();
      _BlockMultiPhaseConstructor<Ts...> & base = t.tail;
      _realign_vec<Ts...>(base);
    }
    template<>
    void _realign_vec(_BlockMultiPhaseConstructor<> &t) {}

    template <typename ElemType>
    void _realign_vec(Block<ElemType> &t) {
      t.realign_vec();
    }

    template<typename T1=Elem<def::Euclidean, 0>, typename... Ts>
    void _append_to(std::ostream &os, const _BlockMultiPhaseConstructor<T1, Ts...> &t, size_t i) {
      os << t.head[i];
      const _BlockMultiPhaseConstructor<Ts...> &base = t.tail;
      _append_to(os, base, i);
    }
    template<>
    void _append_to(std::ostream &os, const _BlockMultiPhaseConstructor<> &t, size_t i) {}

    template<typename ElemType>
    void _append_to(std::ostream &os, const Block<ElemType> &t, size_t i) {
      os << t[i];
    }

  template<typename BlockType>
  void _read_main(BlockType &block, const std::string &filename, const size_t size) {
    using namespace std;
    ifstream fs;
    double startTime = getRealTime();
    /* read main file */
    int checkEnd = 0;
    { fs.open(filename, ifstream::in); }
    assert(fs.is_open());
    size_t i;
    for (i=0; i<size; ++i) {
      if( _append(fs, block) > 0) {
	cerr << getLogHeader() << " warning: read only " << i << " instances." << endl;
	break;
      }
    }
    block.get_size() = i;

    _realign_vec(block);

    fs.close();
    cerr << getLogHeader() << " logging: read data in " 
    << (getRealTime() - startTime) << " seconds." << endl;
  }


    template<typename BlockType>
    void _write(BlockType &block, const std::string &filename) {
      using namespace std;
      const size_t size = block.get_size();
      if (filename != "") {
	ofstream fs;
	fs.open(filename, ofstream::out);
	assert(fs.is_open());

	for (size_t i=0; i<size; ++i) {
	  _append_to(fs, block, i);
	}
	fs.close();    
      } else {
	for (size_t i=0; i<size; ++i) {
	  _append_to(cout, block, i);
	}

      }
    }

    template<typename BlockType> 
    void _split_write(BlockType &block, const std::string &filename, const size_t num_copies) {
      using namespace std;
      const size_t size = block.get_size();
      vector<size_t> rand_ind(size);
      for (size_t i=0; i<size; ++i) rand_ind[i] = i;
      random_shuffle(rand_ind.begin(), rand_ind.end());

      if (filename != "") {
	assert(num_copies > 1);
	double startTime = getRealTime();
	vector<ofstream> fs(num_copies);
	size_t batch_size = (size-1) / num_copies + 1;

	for (int j=0; j<num_copies; ++j) {
	  fs[j].open(filename + ".part" + to_string(j), ofstream::out);
	  assert(fs[j].is_open());
	}
	for (size_t i=0; i<size; ++i) {
	  _append_to(fs[i/batch_size], block, rand_ind[i]);
	}
	for (int j=0; j<num_copies; ++j) fs[j].close();
	cerr << getLogHeader() << " logging: write data into part0.." << num_copies-1 << " in " 
	     << (getRealTime() - startTime) << " seconds." << endl;
      
      } else {
	cerr << getLogHeader() << " error: empty filename specified." << endl;
      }
    }
    template<typename BlockType> 
    void _train_test_split_write(const BlockType &block, const std::string &filename, const real_t train_ratio, const size_t start, const unsigned int seed, const size_t tr_size = 0) {
      using namespace std;
      const size_t size = block.get_size();
      vector<size_t> rand_ind(size);
      for (size_t i=0; i<size; ++i) rand_ind[i] = i;
      if (seed > 0) {
	std::mt19937 r{seed};
	std::shuffle(std::begin(rand_ind), std::end(rand_ind), r);
      }
      
      if (filename != "") {
	double startTime = getRealTime();
	ofstream fs_train, fs_test;	
	size_t train_size;
	assert(tr_size < block.get_size());
	if (tr_size == 0)
	  train_size = std::round(size * train_ratio);
	else {
	  assert(seed == 0);
	  train_size = tr_size;
	}
	
	size_t test_size  = size - train_size;

	// write data
	fs_train.open(filename + ".train", ofstream::out);
	fs_test.open (filename + ".test",  ofstream::out);
	assert(fs_train.is_open() && fs_test.is_open());

	for (size_t i=0; i<train_size; ++i) {
	  _append_to(fs_train, block, rand_ind[i]);
	}
	for (size_t i=train_size; i<size; ++i) {
	  _append_to(fs_test, block, rand_ind[i]);
	}

	fs_train.close();
	fs_test.close();

	// write label
	fs_train.open(filename + ".train.label", ofstream::out);
	fs_test.open (filename + ".test.label",  ofstream::out);
	assert(fs_train.is_open() && fs_test.is_open());

	for (size_t i=0; i<train_size; ++i) {
	  fs_train << block[rand_ind[i]].label[0] + start << std::endl;
	}
	for (size_t i=train_size; i<size; ++i) {
	  fs_test << block[rand_ind[i]].label[0] + start << std::endl;
	}

	fs_train.close();
	fs_test.close();

	  
	cerr << getLogHeader() << " logging: write data into train,test" << " in " 
	     << (getRealTime() - startTime) << " seconds." << endl;
      
      } else {
	cerr << getLogHeader() << " error: empty filename specified." << endl;
      }
    }
  }

  
  template<typename... Ts>
  void BlockMultiPhase<Ts...>::read_meta(const std::string &filename) {
    using namespace std;
    double startTime = getRealTime();
    internal::_read_meta<Ts...>(filename, *this);
    cerr << getLogHeader() << " logging: read meta data in " 
	 << (getRealTime() - startTime) << " seconds." << endl;    
  }


  template<typename... Ts>
  void BlockMultiPhase<Ts...>::read(const std::string &filename, const size_t size) {
    read_meta(filename);
    read_main(filename, size);
  }

  template <typename ElemType>
  void Block<ElemType>::read_main(const std::string &filename, const size_t size) {
    internal::_read_main(*this, filename, size);
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::read_main(const std::string &filename, const size_t size) {
    internal::_read_main(*this, filename, size);
  }
  

  template<typename ElemType>
  void Block<ElemType>::write(const std::string &filename) const {
    internal::_write(*this, filename);
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::write(const std::string &filename) const {
    internal::_write(*this, filename);
  }

  template<typename ElemType>
  void Block<ElemType>::split_write(const std::string &filename, const size_t num_copies) const {
    internal::_split_write(*this, filename, num_copies);
  }

  template<typename ElemType>
  void Block<ElemType>::train_test_split_write(const std::string &filename, const real_t train_ratio, const size_t start, const unsigned int seed, const size_t tr_size) const {
    internal::_train_test_split_write(*this, filename, train_ratio, start, seed, tr_size);
  }

  template<typename... Ts>
  void BlockMultiPhase<Ts...>::split_write(const std::string &filename, const size_t num_copies) const {
    internal::_split_write(*this, filename, num_copies);
  }


}

#endif /* _D2_IO_IMPL_H_ */

