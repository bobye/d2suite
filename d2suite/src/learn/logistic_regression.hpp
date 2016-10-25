#ifndef _D2_LOGISTIC_REGRESSION_H_
#define _D2_LOGISTIC_REGRESSION_H_

#include "../common/common.hpp"
#include "../common/cblas.h"

namespace d2 {

  template <size_t dim, size_t n_class>
  class Logistic_Regression {
  public:
    void init() {
      // to do
    }
    void fit(const real_t *X, const real_t *y, const real_t *sample_weight, size_t n) {
      // to do
    }
    real_t predict(const real_t *X) {
      // to do
    }
    void predicts(const real_t *X, const size_t n, real_t *y) {
      // to do
    }
    real_t eval(const real_t *X, const real_t y) {
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
      loss = - log(v[(size_t) y] / exp_sum);
      return loss;
    }
    void evals(const real_t *X, const real_t *y, const size_t n, real_t *loss) {
      real_t *v = new real_t[n*n_class];
      real_t *sv= new real_t[n];
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
      for (int i=0; i<n; ++i)
	loss[i] = -log (v[i*n_class + (size_t) y[i]]);
      delete [] v;
      delete [] sv;
    }
  protected:
    size_t A[n_class*dim], b[n_class];
  };
}

#endif /* _D2_LOGISTIC_REGRESSION_H_ */
