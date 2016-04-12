#ifndef _BLAS_LIKE_H_
#define _BLAS_LIKE_H_


#ifdef __cplusplus
extern "C" {
#endif

  #include <stdlib.h>
  typedef unsigned index_t;

  // assertation
  void _dgzero(size_t n, double *a); //assert (a>0)

  // element-wise op
  void _dadd(size_t, double *a, double b); // a(:) += b;
  void _dvmul(size_t n, double *a, double *b, double *c);// c = a .* b
  void _dexp(size_t n, double *a);//inplace a -> exp(a);

  // column-wise op
  void _dgcmv(size_t m, size_t n, double *a, double *b); // a(:,*) = a(:,*) .+ b
  void _dgcms(size_t m, size_t n, double *a, double *b); // a = diag(b) * a
  void _dicms(size_t m, size_t n, double *a, double *b); // a = diag(1./b) * a
  void _dcsum(size_t m, size_t n, double *a, double *b); // b(*) = sum(a(:,*))
  void _dcsum2(size_t m, size_t n, double *a, double *b); // b(*) += sum(a(:,*))
  void _dcnorm(size_t m, size_t n, double *a, double *sa); // replace a(:,*) -> a(:,*) / sum(a(:,*))
  void _dccenter(size_t m, size_t n, double *a, double *sa); // replace a(:,*) -> a(:,*) - mean(a(:,*))
  void _dcmax(size_t m, size_t n, double *a, double *b);
  void _dcmin(size_t m, size_t n, double *a, double *b);
  // row-wise op
  void _dgrmv(size_t m, size_t n, double *a, double *b); // a(*,:) = a(*,:) .+ b
  void _dgrms(size_t m, size_t n, double *a, double *b); // a = a * diag(b) 
  void _dirms(size_t m, size_t n, double *a, double *b); // a = a * diag(1./b) 
  void _drsum(size_t m, size_t n, double *a, double *b); // b(*) = sum(a(*,:))
  void _drsum2(size_t m, size_t n, double *a, double *b); // b(*) += sum(a(*,:))
  void _drnorm(size_t m, size_t n, double *a, double *sa); // inplace a(*,:) = a(*,:) / sum(a(*,:))
  void _drcenter(size_t m, size_t n, double *a, double *sa); // replace a(*,:) -> a(*,:) - mean(a(*,:))
  void _drmax(size_t m, size_t n, double *a, double *b);
  void _drmin(size_t m, size_t n, double *a, double *b);



  /* compute squared Euclidean distance matrix
   * A: d x n 
   * B: d x m
   * C: n x m 
   * d: dimension of data entry
   * n, m: number of data entry
   */
  void _dpdist2(const size_t d, const size_t n, const size_t m, const double * A, const double * B, double *C);
  void _dpdist2_sym(const size_t d, const size_t n, const size_t m, const double *A, const index_t *Bi, double *C, const double *vocab);
  void _dpdist2_sym2(const size_t d, const size_t n, const size_t m, const index_t *Ai, const index_t *Bi, double *C, const double *vocab);
  void _dpdist2_submat(const size_t m, const int *Bi, double *C, const size_t vocab_size, const double *dist_mat);
  void _dpdist_symbolic(const size_t d, const size_t n, const size_t m, const index_t * A, const index_t * B, double *C, const size_t vocab_size, const double* dist_mat);
  

  // assertation
  void _sgzero(size_t n, float *a); //assert (a>0)

  // element-wise op
  void _sadd(size_t, float *a, float b); // a(:) += b;
  void _svmul(size_t n, float *a, float *b, float *c);// c = a .* b
  void _sexp(size_t n, float *a);//inplace a -> exp(a);

  // column-wise op
  void _sgcmv(size_t m, size_t n, float *a, float *b); // a(:,*) = a(:,*) .+ b
  void _sgcms(size_t m, size_t n, float *a, float *b); // a = diag(b) * a
  void _sicms(size_t m, size_t n, float *a, float *b); // a = diag(1./b) * a
  void _scsum(size_t m, size_t n, float *a, float *b); // b(*) = sum(a(:,*))
  void _scsum2(size_t m, size_t n, float *a, float *b); // b(*) += sum(a(:,*))
  void _scnorm(size_t m, size_t n, float *a, float *sa); // replace a(:,*) -> a(:,*) / sum(a(:,*))
  void _sccenter(size_t m, size_t n, float *a, float *sa); // replace a(:,*) -> a(:,*) - mean(a(:,*))
  void _scmax(size_t m, size_t n, float *a, float *b);
  void _scmin(size_t m, size_t n, float *a, float *b);
  // row-wise op
  void _sgrmv(size_t m, size_t n, float *a, float *b); // a(*,:) = a(*,:) .+ b
  void _sgrms(size_t m, size_t n, float *a, float *b); // a = a * diag(b) 
  void _sirms(size_t m, size_t n, float *a, float *b); // a = a * diag(1./b) 
  void _srsum(size_t m, size_t n, float *a, float *b); // b(*) = sum(a(*,:))
  void _srsum2(size_t m, size_t n, float *a, float *b); // b(*) += sum(a(*,:))
  void _srnorm(size_t m, size_t n, float *a, float *sa); // inplace a(*,:) = a(*,:) / sum(a(*,:))
  void _srcenter(size_t m, size_t n, float *a, float *sa); // replace a(*,:) -> a(*,:) - mean(a(*,:))
  void _srmax(size_t m, size_t n, float *a, float *b);
  void _srmin(size_t m, size_t n, float *a, float *b);


  /* compute squared Euclidean distance matrix
   * A: d x n 
   * B: d x m
   * C: n x m 
   * d: dimension of data entry
   * n, m: number of data entry
   */
  void _spdist2(const size_t d, const size_t n, const size_t m, const float * A, const float * B, float *C);
  void _spdist2_sym(const size_t d, const size_t n, const size_t m, const float *A, const index_t *Bi, float *C, const float *vocab);
  void _spdist2_sym2(const size_t d, const size_t n, const size_t m, const index_t *Ai, const index_t *Bi, float *C, const float *vocab);
  void _spdist2_submat(const size_t m, const size_t *Bi, float *C, const size_t vocab_size, const float *dist_mat);
  void _spdist_symbolic(const size_t d, const size_t n, const size_t m, const index_t * A, const index_t * B, float *C, const size_t vocab_size, const float* dist_mat);
  

#ifdef __cplusplus
}
#endif


#endif /* _BLAS_LIKE_H_ */
