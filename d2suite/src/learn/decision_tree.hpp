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
#include <algorithm>
#include <numeric>
#include <cmath>
#include <utility>

#ifdef RABIT_RABIT_H
#include <dmlc/io.h>
#endif

namespace d2 {
  
  namespace internal {
    template <size_t dim, size_t n_class> class _DTLeaf;
    template <size_t dim, size_t n_class> class _DTBranch;
    
    
    /*! \brief base class for decision tree nodes
     * which includes shared functions and data members of both leaf and branch
    */
    template <size_t dim, size_t n_class>
    class _DTNode {
    public:
      std::array<real_t, n_class> class_histogram = {}; ///< histogram of sample weights
      /*! \brief get pointer to the leaf node by a given sample */
      virtual _DTLeaf<dim, n_class>* get_leafnode(const real_t *X) = 0;
      /*! \brief get resubstitution error */
      virtual real_t get_R() = 0;
#ifdef RABIT_RABIT_H_
      /*! \brief write data into a stream buffer */
      virtual void write(dmlc::Stream *fo) const = 0;
      /*! \brief read data from a stream buffer */
      virtual void read(dmlc::Stream *fi) = 0;
#endif
      real_t score; ///< uncertainty score
      real_t weight; ///< node sample weight
      real_t r;///< resubstituion error
      constexpr static real_t prior_weight = 0.1;
    };    

    /*! \brief lead node in decision tree
     */
    template <size_t dim, size_t n_class>
    class _DTLeaf : public _DTNode<dim, n_class> {
    public:
      _DTLeaf<dim, n_class>* get_leafnode(const real_t *X) {
	return this;
      }
      real_t get_R() {return this->r;}
#ifdef RABIT_RABIT_H_
      void write(dmlc::Stream *fo) const {
	fo->Write(&this->class_histogram[0], sizeof(real_t) * n_class);
	fo->Write(&this->score, sizeof(real_t));
	fo->Write(&this->weight, sizeof(real_t));
	fo->Write(&this->label, sizeof(size_t));
	fo->Write(&this->r, sizeof(real_t));
      }
      void read(dmlc::Stream *fi) {
	fi->Read(&this->class_histogram[0], sizeof(real_t) * n_class);
	fi->Read(&this->score, sizeof(real_t));
	fi->Read(&this->weight, sizeof(real_t));
	fi->Read(&this->label, sizeof(size_t));
	fi->Read(&this->r, sizeof(real_t));
      }
#endif
      size_t label;
    };

    /*! \brief branch node in decision tree
     */
    template <size_t dim, size_t n_class>
    class _DTBranch : public _DTNode<dim, n_class> {
    public:
      _DTLeaf<dim, n_class>* get_leafnode(const real_t *X) {
	assert(left && right);
	if (X[index]<cutoff) {
	  return left->get_leafnode(X);
	} else {
	  return right->get_leafnode(X);
	}
      }
      real_t get_R() {return left->get_R() + right->get_R();}
#ifdef RABIT_RABIT_H_
      void write(dmlc::Stream *fo) const {
	fo->Write(&this->class_histogram[0], sizeof(real_t) * n_class);
	fo->Write(&this->score, sizeof(real_t));
	fo->Write(&this->weight, sizeof(real_t));
	fo->Write(&this->nleft, sizeof(int));
	fo->Write(&this->nright, sizeof(int));
	fo->Write(&this->index, sizeof(size_t));
	fo->Write(&this->cutoff, sizeof(real_t));
	fo->Write(&this->r, sizeof(real_t));
	fo->Write(&this->R, sizeof(real_t));
      }
      void read(dmlc::Stream *fi) {
	fi->Read(&this->class_histogram[0], sizeof(real_t) * n_class);
	fi->Read(&this->score, sizeof(real_t));
	fi->Read(&this->weight, sizeof(real_t));
	fi->Read(&this->nleft, sizeof(int));
	fi->Read(&this->nright, sizeof(int));
	fi->Read(&this->index, sizeof(size_t));
	fi->Read(&this->cutoff, sizeof(real_t));
	fi->Read(&this->r, sizeof(real_t));
	fi->Read(&this->R, sizeof(real_t));
      }
#endif
      _DTNode<dim, n_class> *left=nullptr, *right=nullptr;
      int nleft = -1, nright = -1;
      size_t index;
      real_t cutoff;
      real_t R;
    };

    /*! \brief node assignment data structure stores
     * the indexes of sample data
     */
    struct node_assignment {
      size_t * ptr; ///< index array
      size_t size; ///< size of index array      
      size_t cache_offset; ///< offset to the cache array head, aka (ptr - cache_offset) should be constant
    };

    /*! \brief data structure for a single sample
     */
    struct sample {
      real_t x;
      size_t y;
      real_t weight;
      size_t index;
    };

    /*! \brief data structure for additional linked list on presort samples
     */
    struct sorted_sample {
      real_t x;
      size_t y;
      real_t weight;
      size_t index;
      sorted_sample *next;
    };

    /*! \brief the whole data structure used in building the decision trees
     */
    template <size_t dim, size_t n_class>    
    struct buf_tree_constructor {      
      std::vector<std::vector<real_t> > X; ///< store data in coordinate-order
      std::vector<size_t> y;
      std::vector<real_t> sample_weight;
      size_t max_depth;
      real_t min_leaf_weight;
      std::vector<sample> sample_cache;
      std::stack<std::tuple<node_assignment, int> > tree_stack;

      // decision tree with presort
      std::vector<std::vector<sorted_sample> > sorted_samples;
      std::vector<std::vector<size_t> > inv_ind_sorted;
      std::vector<char> sample_mask_cache;
    };    

    /*! \brief gini function used in make splits
     */ 
    template <size_t n_class>
    real_t gini(const std::array<real_t, n_class> &proportion) {
      real_t total_weight_sqr;
      total_weight_sqr = std::accumulate(proportion.begin(), proportion.end(), 0.);
      total_weight_sqr = total_weight_sqr * total_weight_sqr;
      if (total_weight_sqr <= 0) return 1.;

      real_t gini = 1.;
      for (size_t i = 0; i<n_class; ++i)
	gini -= (proportion[i] * proportion[i]) / total_weight_sqr ;

      return gini;
    }

    /*! \brief find the best split (cutoff) for a given feature
     */
    template <size_t n_class>
    real_t best_split(sample *sample,
		      size_t n,
		      real_t &cutoff,
		      size_t &left_count,
		      const bool presort) {
      if (!presort) {
	std::sort(sample, sample+n, [](const struct sample &a,
				       const struct sample &b) -> bool {return a.x < b.x;});
      }
      real_t best_goodness = 0;

      std::array<real_t, n_class> proportion_left = {};
      std::array<real_t, n_class> proportion_right = {};
      for (size_t i=0; i<n; ++i) proportion_right[sample[i].y] += sample[i].weight;
      real_t no_split_score =gini(proportion_right);
      for (size_t i=0; i<n; ) {	
	real_t current_x = sample[i].x;
	while (i<n && sample[i].x == current_x) {
	  size_t y=sample[i].y;
	  real_t w=sample[i].weight;
	  proportion_left[y]  += w;
	  proportion_right[y] -= w;
	  i++;
	};
	if (i<n) {
	  real_t goodness = no_split_score -
	    ( gini(proportion_left) * i + gini(proportion_right) * (n-i) ) / n;
	  if (goodness > best_goodness) {
	    best_goodness = goodness;
	    cutoff = sample[i].x;
	    left_count = i;
	  }
	}
      }
      return best_goodness;
    }

    void inplace_split(sample *sample,
		       node_assignment &assignment,
		       real_t cutoff,
		       size_t left_count) {
      struct sample *head, *end;
      head = sample;
      end = sample + assignment.size - 1;
      while (head < end) {
	while (head < end && head->x < cutoff) ++head;
	while (head < end && end->x >= cutoff) --end;
	if (head < end) {
	  struct sample swap_cache;
	  swap_cache = *head;
	  *head      = *end;
	  *end       = swap_cache;
	}
      }
      assert(head - sample == left_count);
      for (size_t i=0; i<assignment.size; ++i)
	assignment.ptr[i] = sample[i].index;      
    }
    
    template <size_t dim, size_t n_class>
    _DTNode<dim, n_class> *build_dtnode(node_assignment &assignment,
					node_assignment &aleft,
					node_assignment &aright,
					buf_tree_constructor<dim, n_class> &buf,
					const bool presort) {
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
      real_t* max_class_w = std::max_element(class_hist.begin(), class_hist.end());
      real_t  all_class_w = std::accumulate(class_hist.begin(), class_hist.end(), 0.);      

      // get the probability score
      real_t prob =  (*max_class_w + _DTNode<dim, n_class>::prior_weight) / (all_class_w + _DTNode<dim, n_class>::prior_weight * n_class);
      real_t r = (1 - *max_class_w / all_class_w);
      if (assignment.size == 1 || buf.tree_stack.size() > buf.max_depth || r < 0.01 || all_class_w < buf.min_leaf_weight) {
	// if the condtion to create a leaf node is satisfied
	_DTLeaf<dim, n_class> *leaf = new _DTLeaf<dim, n_class>();
	leaf->class_histogram = class_hist;
	leaf->label = max_class_w - class_hist.begin();
	leaf->score = - log (prob);
	leaf->weight = all_class_w;
	leaf->r = r * leaf->weight;
	return leaf;
      } else {
	// if it is possible to create a branch node
	std::array<real_t, dim> goodness = {};
	std::array<real_t, dim> cutoff   = {};
	std::array<size_t, dim> left_count = {};
	std::array<size_t, dim> min_index_cache = {};
	// compute goodness split score across different dimensions
	for (size_t ii = 0; ii < dim; ++ii) {
	  sample * sample_cache = &buf.sample_cache[0] + assignment.cache_offset;
	  if (!presort) {
	    for (size_t jj = 0; jj < assignment.size; ++jj) {
	      size_t index = assignment.ptr[jj];
	      sample &sample = sample_cache[jj];
	      sample.x = buf.X[ii][index];
	      sample.y = buf.y[index];
	      sample.weight = buf.sample_weight[index];
	      sample.index = index;
	    }
	  } else {
	    std::vector<size_t> &inv_ind_sorted = buf.inv_ind_sorted[ii];
	    size_t *min_index = std::min_element(assignment.ptr, assignment.ptr + assignment.size,
						 [&](const size_t &a, const size_t &b) -> bool
						 {return inv_ind_sorted[a] < inv_ind_sorted[b];});
	    min_index_cache[ii] = inv_ind_sorted[*min_index];
	    sorted_sample *sorted_sample_ptr = &buf.sorted_samples[ii][min_index_cache[ii]];
	    for (size_t jj = 0; jj < assignment.size; ++jj) {
	      assert(sorted_sample_ptr);
	      sample &sample = sample_cache[jj];
	      sample.x = sorted_sample_ptr->x;
	      sample.y = sorted_sample_ptr->y;
	      sample.weight = sorted_sample_ptr->weight;
	      sample.index  = sorted_sample_ptr->index;
	      sorted_sample_ptr = sorted_sample_ptr->next;
	    }
	  }
	  goodness[ii] = best_split<n_class>(sample_cache,
					     assignment.size,
					     cutoff[ii],
					     left_count[ii],
					     presort);
	}
	// pick the best goodness 
	real_t* best_goodness = std::max_element(goodness.begin(), goodness.end());
	if (*best_goodness < 1E-5) {
	  // if the best goodness is not good enough, a leaf node is still created
	  _DTLeaf<dim, n_class> *leaf = new _DTLeaf<dim, n_class>();
	  leaf->class_histogram = class_hist;
	  leaf->label = max_class_w - class_hist.begin();
	  leaf->score = -log(prob);
	  leaf->weight = all_class_w;
	  leaf->r = r * leaf->weight;
	  return leaf;
	} else {
	  // otherwise, create a branch node subject to the picked dimension/goodness
	  _DTBranch<dim, n_class> *branch = new _DTBranch<dim, n_class>();
	  size_t ii = best_goodness - goodness.begin();
	  branch->index = ii;
	  branch->cutoff = cutoff[ii];
	  branch->score = -log(prob);
	  branch->weight = all_class_w;
	  branch->r = r * branch->weight;
	  sample *sample_cache = &buf.sample_cache[0] + assignment.cache_offset;
	  for (size_t jj=0; jj<assignment.size; ++jj) {
	    sample_cache[jj].x = buf.X[ii][assignment.ptr[jj]];
	    sample_cache[jj].index = assignment.ptr[jj];
	    // note that sample_cache[jj].weight and sample_cache[jj].y are invalid	    
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
	  aright.ptr = assignment.ptr + left_count[ii];
	  aright.size = assignment.size - left_count[ii];
	  aright.cache_offset = assignment.cache_offset + left_count[ii];

	  if (presort) {
	    for (size_t i=0; i<aleft.size; ++i) {
	      buf.sample_mask_cache[aleft.ptr[i]] = 'l';
	    }
	    for (size_t i=0; i<aright.size; ++i){
	      buf.sample_mask_cache[aright.ptr[i]]= 'r';
	    }

	    for (size_t ii=0; ii<dim; ++ii) {
	      sorted_sample *sorted_sample_ptr = &buf.sorted_samples[ii][min_index_cache[ii]];
	      const std::vector<size_t> &inv_ind_sorted = buf.inv_ind_sorted[ii];
	      sorted_sample *left=NULL;
	      sorted_sample *right=NULL;
	      for (size_t i=0; i<assignment.size; ++i) {
		char mask = buf.sample_mask_cache[sorted_sample_ptr->index];
		if (mask == 'l') {
		  if (left) {
		    left->next = sorted_sample_ptr;
		    left = sorted_sample_ptr;
		  } else {
		    left = sorted_sample_ptr;
		  }
		} else if (mask == 'r') {
		  if (right) {
		    right->next = sorted_sample_ptr;
		    right = sorted_sample_ptr;
		  } else {
		    right = sorted_sample_ptr;
		  }
		}
		sorted_sample_ptr = sorted_sample_ptr->next;
	      }
	    }
	  }	  
	  return branch;
	}	
      }
    }


#define BIT_HIGH_POS 30
    template <size_t dim, size_t n_class>
    _DTNode<dim, n_class>*
    post_process_node_arr(std::vector<internal::_DTLeaf<dim, n_class> > &leaf_arr,
			  std::vector<internal::_DTBranch<dim, n_class> > &branch_arr) {
      for (auto iter = branch_arr.begin(); iter < branch_arr.end(); ++iter) {
	if (iter->nleft & 1<<BIT_HIGH_POS) {
	  iter->left = &branch_arr[iter->nleft & ~(1<<BIT_HIGH_POS)];
	} else {
	  iter->left = &leaf_arr[iter->nleft];
	}


	if (iter->nright & 1<<BIT_HIGH_POS) {
	  iter->right = &branch_arr[iter->nright & ~(1<<BIT_HIGH_POS)];
	} else {
	  iter->right = &leaf_arr[iter->nright];
	}
      }
      _DTNode<dim, n_class>* r;
      if (!branch_arr.empty()) {
	r = &branch_arr[0];
	//	printf("%zd\n", static_cast<_DTBranch<dim, n_class> *>(r)->nleft);
      } else {
	r = &leaf_arr[0];
      }
      return r;
    }
    
    template <size_t dim, size_t n_class>
    _DTNode<dim, n_class>* build_tree(size_t sample_size,
				      buf_tree_constructor<dim, n_class> &_buf,
				      std::vector<internal::_DTLeaf<dim, n_class> > &leaf_arr,
				      std::vector<internal::_DTBranch<dim, n_class> > &branch_arr,
				      const bool presort) {
      leaf_arr.clear();
      branch_arr.clear();

      auto &tree_stack = _buf.tree_stack;

      // create index array at root node
      std::vector<size_t> root_index(sample_size);
      for (size_t i=0; i<sample_size; ++i) root_index[i] = i;
      // create the node_assignment at root node and push into stack
      node_assignment root_assignment = {&root_index[0], sample_size, 0};
      tree_stack.push(std::make_tuple(root_assignment, -1));

      // allocate cache memory
      _buf.sample_cache.resize(sample_size);
      // to be returned
      _DTNode<dim, n_class> *root = nullptr;

      // start to travel a tree construction using a stack
      while (!tree_stack.empty()) { 
	auto cur_tree = tree_stack.top(); 
	auto cur_assignment = std::get<0>(cur_tree);
	int cur_parent = std::get<1>(cur_tree);

	node_assignment assignment_left, assignment_right;
	_DTNode<dim, n_class> *node = build_dtnode(cur_assignment,
						   assignment_left,
						   assignment_right,
						   _buf,
						   presort);
	tree_stack.pop();
	bool is_branch;
	if (assignment_left.ptr && assignment_right.ptr) {// spanning the tree
	  is_branch = true;
	  branch_arr.push_back(*std::move(static_cast<_DTBranch<dim, n_class>* > (node)));	  
	  tree_stack.push(std::make_tuple(assignment_left, branch_arr.size()-1));
	  tree_stack.push(std::make_tuple(assignment_right,branch_arr.size()-1));	  
	} else {
	  is_branch = false;
	  leaf_arr.push_back(*std::move(static_cast<_DTLeaf<dim, n_class>* > (node))); 
	}
	if (cur_parent >= 0) {
	  auto &parent = branch_arr[cur_parent];
	  size_t ind = (is_branch)? ((branch_arr.size()-1) | 1<<BIT_HIGH_POS) : (leaf_arr.size()-1);
	  if (parent.nright < 0)
	    parent.nright = ind;
	  else
	    parent.nleft = ind;
	}	
      }

      return post_process_node_arr(leaf_arr, branch_arr);
    }    
  }
  
  /*! \brief the decision tree class that is currently used in marriage learning framework 
   */
  template <size_t dim, size_t n_class>
  class Decision_Tree {
  public:
    static const size_t NUMBER_OF_CLASSES = n_class;

    void init() {}
    void predict(const real_t *X, const size_t n, real_t *y) const {
      const real_t* x = X;
      assert(root);
      for (size_t i=0; i<n; ++i, x+=dim) {
	auto leaf = root->get_leafnode(x);
	y[i] = leaf->label;
      }
    };
    real_t eval(const real_t *X, const real_t y) const {
      LeafNode *leaf = root->get_leafnode(X);
      std::array<real_t, n_class> &histogram = leaf->class_histogram;
      return -log((histogram[(size_t) y] + internal::_DTNode<dim,n_class>::prior_weight) / (leaf->weight + internal::_DTNode<dim, n_class>::prior_weight * n_class));
    }
    void eval_alllabel(const real_t *X, real_t *loss, const size_t stride) const {
      LeafNode *leaf = root->get_leafnode(X);
      std::array<real_t, n_class> &histogram = leaf->class_histogram;
      for (size_t i=0; i<n_class; ++i) {
	loss[i*stride] = -log((histogram[i] + internal::_DTNode<dim,n_class>::prior_weight) / (leaf->weight + internal::_DTNode<dim, n_class>::prior_weight * n_class));
      }
    }
    real_t eval_min(const real_t *X) const {
      return root->get_leafnode(X)->score;
    }
    void evals(const real_t *X, const real_t *y, const size_t n, real_t *loss, const size_t leading, const size_t stride = 1) const {
      for (size_t i=0; i<n; ++i) {
	LeafNode *leaf = root->get_leafnode(X + i*dim);	
	std::array<real_t, n_class> &histogram = leaf->class_histogram;
	loss[i*leading] = -log((histogram[(size_t) y[i*stride]] + internal::_DTNode<dim,n_class>::prior_weight) / (leaf->weight + internal::_DTNode<dim, n_class>::prior_weight * n_class));
      }
    }
    void evals_alllabel(const real_t *X, const size_t n, real_t *loss, const size_t leading, const size_t stride) const {
      for (size_t i=0; i<n; ++i) {
	LeafNode *leaf = root->get_leafnode(X + i*dim);	
	std::array<real_t, n_class> &histogram = leaf->class_histogram;
	for (size_t j=0; j<n_class; ++j) {
	  loss[i*leading + j*stride] = -log((histogram[j] + internal::_DTNode<dim,n_class>::prior_weight) / (leaf->weight + internal::_DTNode<dim, n_class>::prior_weight * n_class));
	}
      }
    }
    void evals_min(const real_t *X, const size_t n, real_t *loss, const size_t leading) const {
      for (size_t i=0; i<n; ++i) {
	LeafNode *leaf = root->get_leafnode(X+i*dim);
	loss[i*leading] = leaf->score;
      }
    }
    
    int fit(const real_t *X, const real_t *y, const real_t *sample_weight, const size_t n,
	    bool sparse = false) {
      assert(X && y && !(sparse && !sample_weight));
      using namespace internal;
      
      // convert sparse data to dense
      const real_t *XX, *yy, *ss;
      size_t sample_size;
      if (sparse) {
	size_t nz = 0;
	for (size_t i = 0; i<n; ++i) nz += sample_weight[i] > 0;
	real_t* XX_ = new real_t [nz * dim];
	real_t* yy_ = new real_t [nz];
	real_t* ss_ = new real_t [nz];
	size_t count = 0;
	for (size_t i = 0; i<n; ++i)
	  if (sample_weight[i] > 0) {	  
	    for (size_t j = 0; j<dim; ++j) XX_[count*dim + j] = X[i*dim + j];
	    yy_[count] = y[i];
	    ss_[count] = sample_weight[i];
	    count ++;
	  }
	XX = XX_;
	yy = yy_;
	ss = ss_;
	sample_size = nz;
      } else {
	XX = X;
	yy = y;
	ss = sample_weight;
	sample_size = n;
      }
      
      buf_tree_constructor<dim, n_class> buf;
      buf.max_depth = max_depth;
      buf.min_leaf_weight = min_leaf_weight;
      {	
	buf.X.resize(dim);
	for (size_t k=0; k<dim; ++k) buf.X[k].resize(sample_size);
	buf.y.resize(sample_size);
	buf.sample_weight.resize(sample_size);
	for (size_t i=0, j=0; i<sample_size; ++i) {
	  for (size_t k=0; k<dim; ++k, ++j) {
	    buf.X[k][i]=XX[j];
	  }
	  buf.y[i]=(size_t) yy[i];
	  if (ss)
	    buf.sample_weight[i]=ss[i];
	  else
	    buf.sample_weight[i]=1.;
	}
      }
      if (presort) {
	buf.sorted_samples.resize(dim);
	buf.inv_ind_sorted.resize(dim);
	buf.sample_mask_cache.resize(sample_size);
	for (size_t k=0; k<dim; ++k) {
	  auto &sorted_samples = buf.sorted_samples[k];
	  auto &inv_ind_sorted = buf.inv_ind_sorted[k];
	  sorted_samples.resize(sample_size);
	  inv_ind_sorted.resize(sample_size);
	  const real_t * XX = &buf.X[k][0];
	  for (size_t i=0; i<sample_size; ++i) {
	    auto &sample = sorted_samples[i];
	    sample.x = XX[i];
	    sample.y = (size_t) yy[i];
	    if (ss)
	      sample.weight = ss[i];
	    else
	      sample.weight = 1.;
	    sample.index = i;
	  }
	  // presort
	  std::sort(sorted_samples.begin(), sorted_samples.end(),
		    [](const struct internal::sorted_sample &a,
		       const struct internal::sorted_sample &b) -> bool {return a.x < b.x;});

	  for (size_t i=0; i<sample_size; ++i) {
	    inv_ind_sorted[sorted_samples[i].index] = i;
	    if (i>0) {
	      sorted_samples[i-1].next = &sorted_samples[i];
	    }
	  }
	}
      }
      root = build_tree(sample_size, buf, leaf_arr, branch_arr, presort);
      if (sparse) {
	delete [] XX;
	delete [] yy;
	delete [] ss;
      }
      return 0;
    }

    inline void set_communicate(bool bval) { communicate = bval; }

    typedef internal::_DTLeaf<dim, n_class> LeafNode;
    typedef internal::_DTBranch<dim, n_class> BranchNode;

#ifdef RABIT_RABIT_H_
    typedef rabit::utils::MemoryBufferStream MemoryBufferStream;

    /*! \brief helper function that caches data to stream */
    inline void save(dmlc::Stream* fo) {
      for (const LeafNode &leaf : leaf_arr) {
	leaf.write(fo);
      }
      for (const BranchNode &branch : branch_arr) {
	branch.write(fo);
      }
    }
    /*! \brief helper function that restores data from stream */
    inline void load(dmlc::Stream* fi) {
      for (LeafNode &leaf : leaf_arr) {
	leaf.read(fi);
      }
      for (BranchNode &branch : branch_arr) {
	branch.read(fi);
      }
    }

    /*! \brief synchronize between multiple processors */
    void sync(size_t rank) {
      bool no_model = false;
      if (rabit::GetRank() == rank) { // check if model exists
	if (leaf_arr.empty()) no_model = true;
      }
      rabit::Broadcast(&no_model, sizeof(bool), rank);
      if (no_model) return;
      
      std::string s_model;
      MemoryBufferStream fs(&s_model);
      size_t n_leaf = leaf_arr.size();
      size_t n_branch = branch_arr.size();

      rabit::Broadcast(&n_leaf, sizeof(size_t), rank);
      rabit::Broadcast(&n_branch, sizeof(size_t), rank);
      if (rabit::GetRank() != rank) {
        leaf_arr.resize(n_leaf);
	branch_arr.resize(n_branch);
      } else if (rabit::GetRank() == rank) {	
	save(&fs);
      }
      fs.Seek(0);
      rabit::Broadcast(&s_model, rank);
      //      if (rabit::GetRank() == rank) printf("%zd: %zd\t %zd\n", rank, n_leaf, n_branch);
      if (rabit::GetRank() != rank) {
	load(&fs);
	//	printf("%zd: load data from %zd\n", rabit::GetRank(), rank);
	this->root = internal::post_process_node_arr(leaf_arr, branch_arr);
	assert(root && !leaf_arr.empty());
      }
      rabit::Barrier();	      
    }
#endif
    
    internal::_DTNode<dim, n_class> *root = nullptr;
  private:
    std::vector<LeafNode> leaf_arr; 
    std::vector<BranchNode> branch_arr;
    size_t max_depth = 12;
    real_t min_leaf_weight = .1;
    bool presort = true;
    bool communicate = true;    
  };    
}

#endif /* _D2_DECISION_TREE_H_ */

