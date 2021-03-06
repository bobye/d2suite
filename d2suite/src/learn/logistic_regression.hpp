#ifndef _D2_LOGISTIC_REGRESSION_H_
#define _D2_LOGISTIC_REGRESSION_H_

#include "../common/common.hpp"
#include "../common/blas_like.h"
#include "../common/cblas.h"
#include "lbfgs.h"
#include <random>
#include <assert.h>

namespace d2 {

  /*! \brief the logistic regression class that is currently used in marriage learning framework
   *
   * Example:
   * \code{.cpp}
   * Logistic_Regression a;
   * a.init() // randomized initialization
   * a.fit(X, y, sample_weight, n) // fit to n vectors X with label y
   * real_t* coeff = a.get_coeff() // get coeffients of fitted LR
   * \endcode
   */
  template <size_t dim, size_t n_class>
  class Logistic_Regression {
  public:
    static const size_t NUMBER_OF_CLASSES = n_class;
    void init() {
      std::random_device rd;
      std::uniform_real_distribution<real_t>  unif(-1., 1.);
      std::mt19937 rnd_gen(rd());
      for (size_t i=0; i<n_class*dim; ++i)
	A[i] = unif(rnd_gen);
      for (size_t i=0; i<n_class; ++i)
	b[i] = 0.;
    }



    int fit(const real_t *X, const real_t *y, const real_t *sample_weight, const size_t n, bool sparse = false) {

      // convert sparse data to dense
      real_t *XX, *yy, *ss;
      if (sparse) {
	size_t nz = 0;
	for (size_t i = 0; i<n; ++i) nz += sample_weight[i] > 0;
	XX = new real_t [nz * dim];
	yy = new real_t [nz];
	ss = new real_t [nz];
	size_t count = 0;
	for (size_t i = 0; i<n; ++i)
	  if (sample_weight[i] > 0) {	  
	    for (size_t j = 0; j<dim; ++j) XX[count*dim + j] = X[i*dim + j];
	    yy[count] = y[i];
	    ss[count] = sample_weight[i];
	    count ++;
	  }
	this->X = XX;
	this->y = yy;
	this->sample_weight = ss;
	sample_size = nz;
      } else {
	this->X = X;
	this->y = y;
	this->sample_weight = sample_weight;
	sample_size = n;
      }
      cache= new real_t [n_class*sample_size + sample_size];
      assert(sizeof(lbfgsfloatval_t) == sizeof(real_t));

      size_t N = n_class*dim+n_class;
      lbfgsfloatval_t fx;      
      lbfgsfloatval_t *x = lbfgs_malloc(N);
      lbfgs_parameter_t param;
      lbfgs_parameter_init(&param);

      std::memcpy(x, coeff, sizeof(real_t) * N);
      A = x;
      b = x + n_class*dim;

      current_lr = this;
      //int ret = lbfgs(N, x, &fx, evaluate_, progress_, NULL, &param);
      int ret = lbfgs(N, x, &fx, evaluate_, NULL, NULL, &param);
      // printf("loss: %lf\n", fx);
      if (ret < 0) printf("L-BFGS optimization terminated with status code = %d\n", ret);
      
      std::memcpy(coeff, x, sizeof(real_t) * N);
      A = coeff;
      b = coeff + n_class*dim;

      lbfgs_free(x);
      current_lr = NULL;
      delete [] cache;
      if (sparse) {
	delete [] XX;
	delete [] yy;
	delete [] ss;
      }
      return ret;
    }
    void predict(const real_t *X, const size_t n, real_t *y) const {
      real_t *v = new real_t[n*n_class];
      real_t *sv= new real_t[n];

      forward_(A, b, X, n, v, sv);
      for (size_t i=0; i<n; ++i) {
	real_t max = -1;
	size_t kk=-1;
	for (size_t k=0; k<n_class; ++k)
	  if (v[i*n_class + k] > max) {
	    max = v[i*n_class+k];
	    kk = k;
	  }
	y[i] = kk;
      }
      delete [] v;
      delete [] sv;
    }
    /*
    real_t eval(const real_t *X, const real_t y) const {
      real_t loss;
      real_t v[n_class];

      _D2_CBLAS_FUNC(gemv)(CblasColMajor, CblasNoTrans, n_class, dim,
			   1.0,
			   A, n_class,
			   X, 1,
			   0,
			   v, 1);
      _D2_CBLAS_FUNC(axpy)(n_class, 1.0, b, 1, v, 1);
      _D2_FUNC(exp)(n_class, v);
      real_t exp_sum=_D2_CBLAS_FUNC(asum)(n_class, v, 1);
      loss = - log(v[(size_t) y]) + log(exp_sum);
      return loss;
    }
    void eval_alllabel(const real_t *X, real_t *loss, const size_t stride) const {

      real_t v[n_class];

      _D2_CBLAS_FUNC(gemv)(CblasColMajor, CblasNoTrans, n_class, dim,
			   1.0,
			   A, n_class,
			   X, 1,
			   0,
			   v, 1);
      _D2_CBLAS_FUNC(axpy)(n_class, 1.0, b, 1, v, 1);
      _D2_FUNC(exp)(n_class, v);
      real_t exp_sum=_D2_CBLAS_FUNC(asum)(n_class, v, 1);
      for (size_t i=0; i<n_class; ++i)
	loss[i*stride] = - log(v[i]) + log(exp_sum);
    }
    
    real_t eval_min(const real_t *X) const {
      real_t loss;
      real_t v[n_class];

      _D2_CBLAS_FUNC(gemv)(CblasColMajor, CblasNoTrans, n_class, dim,
			   1.0,
			   A, n_class,
			   X, 1,
			   0,
			   v, 1);
      _D2_CBLAS_FUNC(axpy)(n_class, 1.0, b, 1, v, 1);
      _D2_FUNC(exp)(n_class, v);
      real_t exp_sum=_D2_CBLAS_FUNC(asum)(n_class, v, 1);
      loss = 0;
      for (size_t j=0; j<n_class; ++j)
	if (loss < v[j]) loss = v[j];
      loss = - log(loss) + log(exp_sum);
      return loss;      
    }
    */
    void evals(const real_t *X, const real_t *y, const size_t n, real_t *loss, const size_t leading, const size_t stride = 1) const {
      real_t *v = new real_t[n*n_class];
      real_t *sv= new real_t[n];

      forward_(A, b, X, n, v, sv);
      for (size_t i=0; i<n; ++i) {
	loss[i*leading] = -log (v[i*n_class + (size_t) y[i*stride]]);
      }
      delete [] v;
      delete [] sv;
    }
    void evals_alllabel(const real_t *X, const size_t n, real_t *loss, const size_t leading, const size_t stride) const {
      real_t *v = new real_t[n*n_class];
      real_t *sv= new real_t[n];

      forward_(A, b, X, n, v, sv);
      for (size_t i=0; i<n; ++i) {
	for (size_t j=0; j<n_class; ++j)
	  loss[i*leading+j*stride] = -log (v[i*n_class + j]);
      }
      delete [] v;
      delete [] sv;
    }
    
    void evals_min(const real_t *X, const size_t n, real_t *loss, const size_t leading) const {
      real_t *v = new real_t[n*n_class];
      real_t *sv= new real_t[n];

      forward_(A, b, X, n, v, sv);
      for (size_t i=0; i<n; ++i) {
	real_t max_prob = 0;
	for (size_t j=i*n_class; j<i*n_class + n_class; ++j)
	  if (max_prob < v[j]) max_prob = v[j];
	loss[i*leading] = -log (max_prob);
      }
      delete [] v;
      delete [] sv;
    }

#ifdef RABIT_RABIT_H_    
    void sync(const size_t rank) {
      rabit::Broadcast(coeff, (n_class*dim+n_class) * sizeof(real_t), rank);      
    }
#endif
    
    inline void set_communicate(bool bval) { communicate = bval; }
  private:
    real_t coeff[n_class*dim+n_class];
    real_t *A = coeff, *b = coeff+n_class*dim;
    real_t *cache;
    const real_t *X, *y, *sample_weight;
    real_t l2_reg = 0.001;
    size_t sample_size;
    bool communicate = true;

    static void forward_(real_t *A, real_t *b,
			 const real_t *X, const size_t n,
			 real_t *v, real_t *sv) {
      // forward
      _D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
			   n_class, n, dim,
			   1.0,
			   A, n_class,
			   X, dim,
			   0.0,
			   v, n_class);
      _D2_FUNC(gcmv)(n_class, n, v, b);
      _D2_FUNC(exp)(n_class * n, v);
      _D2_FUNC(cnorm)(n_class, n, v, sv);
    }
    real_t gradient_(real_t *grad) const {
#ifdef RABIT_RABIT_H_
      using namespace rabit;
#endif
      size_t n = sample_size;
      real_t *v = cache;
      real_t *sv= cache + n*n_class;
      real_t *gradA = grad;
      real_t *gradb = grad + n_class * dim;
      real_t loss = 0.0;

      forward_(A, b, X, n, v, sv);
      real_t sample_wsum;
      if (sample_weight)
	sample_wsum=_D2_CBLAS_FUNC(asum)(n, sample_weight, 1);
      else
	sample_wsum=n;
#ifdef RABIT_RABIT_H_
      if (communicate)
	Allreduce<op::Sum>(&sample_wsum, 1);
#endif

      // compute the regularized loss
      if (sample_weight) {
	for (size_t i=0; i<n; ++i)
	  loss += -log (v[i*n_class + (size_t) y[i]]) * sample_weight[i];
      } else {
	for (size_t i=0; i<n; ++i)
	  loss += -log (v[i*n_class + (size_t) y[i]]);
      }
#ifdef RABIT_RABIT_H_
      if (communicate)
	Allreduce<op::Sum>(&loss, 1);
#endif
      loss /= sample_wsum;

      for (size_t i=0; i<n_class*dim; ++i)
	loss += 0.5 * l2_reg * A[i] * A[i];

      // compute the regularized gradient
      for (size_t i=0; i<n; ++i) {
	v[i*n_class + (size_t) y[i]] -= 1.;
      }

      if (sample_weight) {
	_D2_FUNC(grms)(n_class, n, v, sample_weight);
      }

      // backpropagate
      _D2_CBLAS_FUNC(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
			   n_class, dim, n,
			   1.0,
			   v, n_class,
			   X, dim,
			   0.0,
			   gradA, n_class);      
      _D2_FUNC(rsum)(n_class, n, v, gradb);
#ifdef RABIT_RABIT_H_
      if (communicate)
	Allreduce<op::Sum>(grad, n_class*dim+n_class);
#endif
      _D2_CBLAS_FUNC(scal)(n_class*dim+n_class, 1./sample_wsum, grad, 1);
      _D2_CBLAS_FUNC(axpy)(n_class*dim, l2_reg, A, 1, gradA, 1);
      return loss;
    }

    static lbfgsfloatval_t evaluate_(
			      void *instance,
			      const lbfgsfloatval_t *x,
			      lbfgsfloatval_t *g,
			      const int n,
			      const lbfgsfloatval_t step
			      ) {
      int i;
      lbfgsfloatval_t fx;

      fx = current_lr->gradient_(g);
	
      return fx;
    }
    static int progress_(
			 void *instance,
			 const lbfgsfloatval_t *x,
			 const lbfgsfloatval_t *g,
			 const lbfgsfloatval_t fx,
			 const lbfgsfloatval_t xnorm,
			 const lbfgsfloatval_t gnorm,
			 const lbfgsfloatval_t step,
			 int n,
			 int k,
			 int ls
			 ) {
#ifdef RABIT_RABIT_H_
      if (rabit::GetRank() == 0)
#endif
      {
	printf("Iteration %d:\n", k);
	printf("  fx = %f\n", fx);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
      }
      return 0;
    }

    static Logistic_Regression<dim, n_class> *current_lr;
    
  };

  template <size_t dim, size_t n_class>
  Logistic_Regression<dim, n_class>* Logistic_Regression<dim, n_class>::current_lr = NULL;
  
}
#endif /* _D2_LOGISTIC_REGRESSION_H_ */
