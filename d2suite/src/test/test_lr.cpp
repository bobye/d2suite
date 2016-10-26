#include "../learn/logistic_regression.hpp"
#include <random>
#define N 1000
#define D 10
using namespace d2;

void sample_naive_data(real_t *X, real_t *y) {
  for (size_t i=0; i<N; ++i) {
    y[i] = rand() % 2;
    if (y[i]) {
      for (size_t j=0; j<D; ++j)
	X[i*D+j]=(real_t) rand() / (real_t) RAND_MAX;
    } else {
      for (size_t j=0; j<D; ++j)
	X[i*D+j]=(real_t) rand() / (real_t) RAND_MAX - 1.;
    }
  }  
}

real_t accuracy(real_t *y_pred, real_t *y_true, size_t n) {
  size_t k=0;
  for (size_t i=0; i<n; ++i)
    if (y_pred[i] == y_true[i]) ++k;
  return (real_t) k / (real_t) n;
}

int main() {
  real_t *X = new real_t[D*N];
  real_t *y = new real_t[N];
  real_t *y_pred = new real_t[N];
  sample_naive_data(X, y);
  auto classifier = new Logistic_Regression<D, 2>();
  classifier->init();
  classifier->fit(X, y, NULL, N);
  sample_naive_data(X, y);
  classifier->predict(X, N, y_pred);
  printf("accuracy: %.3f\n", accuracy(y_pred, y, N) );  
  return 0;
}
