#ifndef _D2_DECISION_TREE_H_
#define _D2_DECISION_TREE_H_

#include "../common/common.hpp"
#include "../common/blas_like.h"
#include "../common/cblas.h"
#include <assert.h>
#include <vector>
#include <stack>
#include <tuple>
#include <array>
namespace d2 {
  
  namespace internal {
    /*! \brief base class for decision tree nodes
     * which includes shared functions and data members of both leaf and branch
    */
    template <size_t dim, size_t n_class>
    class _DTNode {
    public:
      constexpr static real_t prior_class_weight = 0.;
      std::array<real_t, n_class> class_histogram = {}; ///< histogram of sample weights 
      virtual real_t predict(real_t *X); ///< recursive prediction function
    };
    
    template <size_t dim, size_t n_class>
    class _DTLeaf : public _DTNode<dim, n_class> {
    public:
      real_t predict(real_t *X) {return label;}
      real_t label;
    };

    template <size_t dim, size_t n_class>
    class _DTBranch : public virtual _DTNode<dim, n_class> {
    public:
      ~_DTBranch() {// recursive destructor
	if (left) delete left;
	if (right) delete right;
      }
      real_t predict(real_t *X) {
	if (X[index]<cutoff) {return left->predict(X);}
	else {return right->predict(X);}
      }
      _DTNode<dim, n_class> *left = nullptr, *right = nullptr;
      size_t index;
      real_t cutoff;      
    };


    /*! \brief node assignment data structure stores
     * the indexes of sample data
     */
    struct node_assignment {
      size_t * ptr; ///< index array
      size_t size; ///< size of index array      
      size_t cache_offset; ///< offset to the cache array head, aka (ptr - cache_offset) should be constant
    };

    struct sample {
      real_t X;
      size_t y;
      real_t sample_weight;
      size_t index;
    };

    /*! \brief the whole data structure used in building the decision trees
     */
    template <size_t dim, size_t n_class>    
    struct buf_tree_constructor {      
      std::vector<std::vector<real_t> > X; ///< store data in coordinate-order
      std::vector<size_t> y;
      std::vector<real_t> sample_weight;
      size_t max_depth;
      std::vector<sample> sample_cache;
      std::stack<std::tuple<node_assignment,
			    _DTBranch<dim, n_class>*> > tree_stack;
    };


    real_t best_split(sample *sample,
		      size_t n,
		      real_t &cutoff,
		      size_t &left_count) {
      return 0;
    }
    void inplace_split(sample *sample,
		       node_assignment &assignment,
		       real_t cutoff,
		       size_t left_count) {
    }
    template <size_t dim, size_t n_class>
    _DTNode<dim, n_class> *build_dtnode(node_assignment &assignment,
					node_assignment &aleft,
					node_assignment &aright,
					buf_tree_constructor<dim, n_class> &buf) {
      // default: return leaf node
      aleft.ptr = NULL;
      aright.ptr= NULL;

      // make sure there is at least one sample
      assert(assignment.size > 0);

      // compute the class histogram on the sample
      std::array<real_t, n_class> class_hist = {};
      size_t *index=assignment.ptr;
      for (size_t ii = 0; ii < assignment.size; ++ii) {
	class_hist[buf.y[index[ii]]] += buf.sample_weight[index[ii]];
      }

      // basic statistics regarding class histogram
      auto max_class_w = std::max_element(class_hist.begin(), class_hist.end());
      auto all_class_w = std::accumulate(class_hist.begin(), class_hist.end(), 0);      

      
      if (assignment.size == 1 || buf.tree_stack.size() > buf.max_depth || (1 - *max_class_w / all_class_w) < 0.01 ) {
	// if the condtion to create a leaf node is satisfied
	_DTLeaf<dim, n_class> *leaf = new _DTLeaf<dim, n_class>();
	leaf->class_histogram = class_hist;
	leaf->label = max_class_w - class_hist.begin();
	return leaf;
      } else {
	// if it is possible to create a branch node
	std::array<real_t, dim> goodness = {};
	std::array<real_t, dim> cutoff   = {};
	std::array<size_t, dim> left_count = {};
	// compute goodness split score across different dimensions
	for (size_t ii = 0; ii < dim; ++ii) {
	  sample * sample_cache = &buf.sample_cache[0] + assignment.cache_offset;
	  for (size_t jj = 0; jj < assignment.size; ++jj) {
	    size_t index = assignment.ptr[jj];
	    sample &sample = sample_cache[jj];
	    sample.X = buf.X[ii][index];
	    sample.y = buf.y[index];
	    sample.sample_weight = buf.sample_weight[index];
	    sample.index = index;
	  }
	  goodness[ii] = best_split(sample_cache,
				    assignment.size,
				    cutoff[ii],
				    left_count[ii]);
	}
	// pick the best goodness 
	auto best_goodness = std::max_element(goodness.begin(), goodness.end());	
	if (*best_goodness < 1E-3) {
	  // if the best goodness is not good enough, a leaf node is still created
	  _DTLeaf<dim, n_class> *leaf = new _DTLeaf<dim, n_class>();
	  leaf->class_histogram = class_hist;
	  leaf->label = max_class_w - class_hist.begin();
	  return leaf;
	} else {
	  // otherwise, create a branch node subject to the picked dimension/goodness
	  _DTBranch<dim, n_class> *branch = new _DTBranch<dim, n_class>();
	  size_t ii = best_goodness - goodness.begin();
	  branch->index = ii;
	  branch->cutoff = cutoff[ii];
	  sample *sample_cache = &buf.sample_cache[0] + assignment.cache_offset;
	  for (size_t jj=0; jj<assignment.size; ++jj) {
	    sample_cache[jj].X = buf.X[ii][assignment.ptr[jj]];
	  }
	  // split assignment
	  inplace_split(sample_cache,
			assignment,
			branch->cutoff,
			left_count[branch->index]);
	  // create branched assignment
	  aleft.ptr = assignment.ptr;
	  aleft.size = left_count[ii];
	  aleft.cache_offset = assignment.cache_offset;
	  aleft.ptr = assignment.ptr + left_count[ii];
	  aleft.size = assignment.size - left_count[ii];
	  aleft.cache_offset = assignment.cache_offset + left_count[ii];
	  return branch;
	}
	
      }
    }

    template <size_t dim, size_t n_class>
    _DTNode<dim, n_class>* build_tree(size_t sample_size,
				      buf_tree_constructor<dim, n_class> &_buf) {
      auto &tree_stack = _buf.tree_stack;

      // create index array at root node
      std::vector<size_t> root_index(sample_size);
      for (size_t i=0; i<sample_size; ++i) root_index[i] = i;
      // create the node_assignment at root node and push into stack
      node_assignment root_assignment = {&root_index[0], sample_size, 0};
      tree_stack.push(std::make_tuple(root_assignment,
				      nullptr));

      // allocate cache memory
      _buf.sample_cache.resize(sample_size);
      // to be returned
      _DTNode<dim, n_class> *root = NULL;

      // start to travel a tree construction using a stack
      while (!tree_stack.empty()) { 
	auto cur_tree = tree_stack.top(); 
	auto cur_assignment = std::get<0>(cur_tree);
	auto cur_parent = std::get<1>(cur_tree);

	node_assignment assignment_left, assignment_right;
	_DTNode<dim, n_class> *node = build_dtnode(cur_assignment,
						   assignment_left,
						   assignment_right,
						   _buf);
	if (root) { root = node; assert(!cur_parent); }
	else if (cur_parent) {
	  if (cur_parent->right)
	    cur_parent->right = node;
	  else
	    cur_parent->left = node;
	}	
	tree_stack.pop();
	if (assignment_left.ptr && assignment_right.ptr) {// spanning the tree
	  tree_stack.push(std::make_tuple(assignment_left,
					  dynamic_cast<_DTBranch<dim, n_class> *>(node)));
	  tree_stack.push(std::make_tuple(assignment_right,
					  dynamic_cast<_DTBranch<dim, n_class> *>(node)));
	}	
      }
      return root;
    }

  }
  
  /*! \brief the decision tree class that is currently used in marriage learning framework 
   */
  template <size_t dim, size_t n_class>
  class Decision_Tree {
  public:
    static const size_t NUMBER_OF_CLASSES = n_class;

    void init() {}
    void predict(const real_t *X, const size_t n, real_t *y) const {};
    real_t eval(const real_t *X, const real_t y) const { return 0.;}
    void eval_alllabel(const real_t *X, real_t *loss, const size_t stride) const {}
    real_t eval_min(const real_t *X) const {}
    void evals(const real_t *X, const real_t *y, const size_t n, real_t *loss, const size_t leading, const size_t stride = 1) const {}
    void evals_alllabel(const real_t *X, const size_t n, real_t *loss, const size_t leading, const size_t stride) const {}
    void evals_min(const real_t *X, const size_t n, real_t *loss, const size_t leading) const {}
    
    int fit(const real_t *X, const real_t *y, const real_t *sample_weight, const size_t n, bool sparse = false) {
      using namespace internal;
      sample_size = n;
      
      buf_tree_constructor<dim, n_class> buf;
      buf.X.resize(dim);
      for (size_t i=0, j=0; i<sample_size; ++i) {
	for (size_t k=0; k<dim; ++k, ++j) {
	  buf.X[k].push_back(X[j]);
	}
	buf.y.push_back((size_t) y[i]);
	buf.sample_weight.push_back(sample_weight[i]);	
      }
      _DTNode<dim, n_class> *root = build_tree(sample_size, buf);
      return 0;
    }

    inline real_t* &get_coeff()  { return coeff; }
    inline real_t* get_coeff() const { return coeff; }
    inline size_t get_coeff_size() {return n_class*dim+n_class;}
    inline void set_communicate(bool bval) { communicate = bval; }

  private:
    real_t *coeff;
    size_t sample_size;
    bool communicate = true;
    
  };    
}

#endif /* _D2_DECISION_TREE_H_ */

