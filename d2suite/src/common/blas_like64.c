#include <math.h>

//#define __USE_C99_MATH
#include <stdbool.h>

#include "blas_like.h"
#include "blas_util.h"
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <float.h>

#define real_t double
#define _D2_MALLOC_SCALAR(n) (real_t*) malloc((n)*sizeof(real_t))
#define _D2_FREE(x) free(x)


void _dgzero(size_t n, real_t *a) {
  size_t i;
  for (i=0; i<n; ++i) assert(a[i] > 1E-10);
}

void _dadd(size_t n, real_t *a, real_t b) {
  size_t i;
  for (i=0; i<n; ++i) a[i] += b;
}

// b = cmax(a)
void _dcmax(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa = a;
  for (j=0; j<n; ++j) b[j] = DBL_MIN;
  for (i=0; i<m; ++i, ++pa) {
    for (j=0; j<n; ++j)
      b[j] = MAX(b[j], pa[j*m]);
  }
}

// b = cmin(a)
void _dcmin(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa = a;
  for (j=0; j<n; ++j) b[j] = DBL_MAX;
  for (i=0; i<m; ++i, ++pa) {
    for (j=0; j<n; ++j)
      b[j] = MIN(b[j], pa[j*m]);
  }
}

// b = rmax(a)
void _drmax(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa =a;
  for (j=0; j<m; ++j) b[j] = DBL_MIN;
  for (i=0; i<n; ++i, pa += m) {
    for (j=0; j<m; ++j)
      b[j] = MAX(b[j], pa[j]);
  }
}

// b = rmin(a)
void _drmin(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa =a;
  for (j=0; j<m; ++j) b[j] = DBL_MAX;
  for (i=0; i<n; ++i, pa += m) {
    for (j=0; j<m; ++j)
      b[j] = MIN(b[j], pa[j]);
  }
}

// a(:,*) = a(:,*) .+ b
void _dgcmv(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa;
  const real_t *pb;
  for (i=0,pa=a; i<n; ++i)
    for (j=0,pb=b; j<m; ++j, ++pa, ++pb)
      *pa += *pb;
}

// a(:,*) = -a(:,*) .+ b
void _dgcmv2(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa;
  const real_t *pb;
  for (i=0,pa=a; i<n; ++i)
    for (j=0,pb=b; j<m; ++j, ++pa, ++pb)
      *pa = -*pa + *pb;
}

// a(*,:) = a(*,:) .+ b
void _dgrmv(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa =a;
  const real_t *pb =b;
  for (i=0; i<n; ++i,++pb)
    for (j=0; j<m; ++j, ++pa)
      *pa += *pb;
}

// a = diag(b) * a
void _dgcms(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa = a;
  const real_t *pb;
  for (i=0; i<n; ++i)
    for (j=0, pb=b; j<m; ++j, ++pa, ++pb)
      *pa *= *pb;
}

// a = a * diag(b) 
void _dgrms(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa = a;
  const real_t *pb = b;
  for (i=0; i<n; ++i,++pb)
    for (j=0; j<m; ++j, ++pa)
      *pa *= *pb;
}

// a = diag(1./b) * a
void _dicms(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa;
  const real_t *pb;
  for (j=0; j<m; ++j) assert(b[j] > 0);
  for (i=0,pa=a; i<n; ++i)
    for (j=0,pb=b; j<m; ++j, ++pa, ++pb)
      *pa /= *pb;
}

// a = a * diag(1./b) 
void _dirms(size_t m, size_t n, real_t *a, const real_t *b) {
  size_t i,j;
  real_t *pa;
  const real_t *pb;
  for (i=0; i<n; ++i) assert(b[i] > 0);
  for (i=0,pa=a,pb=b; i<n; ++i,++pb) {
    for (j=0; j<m; ++j, ++pa)
      *pa /= *pb;
  }
}

// b(*) = sum(a(:,*))
void _dcsum(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa;
  real_t *pb;
  for (i=0,pa=a,pb=b; i<n; ++i, ++pb) {
    *pb = 0;
    for (j=0; j<m; ++j, ++pa)
      *pb += *pa;
  }
}

// b(*) += sum(a(:,*))
void _dcsum2(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa;
  real_t *pb;
  for (i=0,pa=a,pb=b; i<n; ++i, ++pb) {
    for (j=0; j<m; ++j, ++pa)
      *pb += *pa;
  }
}


// b(*) = sum(a(*,:))
void _drsum(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa;
  real_t *pb;
  for (j=0,pb=b; j<m; ++j, ++pb) 
    *pb = 0;
  for (i=0,pa=a; i<n; ++i)
    for (j=0,pb=b; j<m; ++j, ++pa, ++pb)
      *pb += *pa;  
}

// b(*) += sum(a(*,:))
void _drsum2(size_t m, size_t n, const real_t *a, real_t *b) {
  size_t i,j;
  const real_t *pa;
  real_t *pb;
  for (i=0,pa=a; i<n; ++i)
    for (j=0,pb=b; j<m; ++j, ++pa, ++pb)
      *pb += *pa;  
}

// normalize by column
void _dcnorm(size_t m, size_t n, real_t *a, real_t *sa) {
  size_t i, j;
  real_t *pa;
  bool isAllocated = true;
  if (!sa) {
    isAllocated = false;
    sa = _D2_MALLOC_SCALAR(n);
  }    
  _dcsum(m, n, a, sa);
  for (i=0; i<n; ++i) assert(sa[i] > 0);
  for (i=0, pa=sa; i<n; ++i, ++pa) {
    for (j=0; j<m; ++j, ++a) (*a) /= *pa;
  }
  if (!isAllocated) _D2_FREE(sa);
}

// normalize by row
void _drnorm(size_t m, size_t n, real_t *a, real_t *sa) {
  size_t i, j;
  real_t *pa;
  bool isAllocated = true;
  if (!sa) {
    isAllocated = false;
    sa = _D2_MALLOC_SCALAR(m);
  }    
  _drsum(m, n, a, sa);
  for (i=0; i<m; ++i) assert(sa[i] > 0);
  for (i=0; i<n; ++i) {
    pa = sa;
    for (j=0; j<m; ++j, ++a, ++pa) (*a) /= *pa;
  }
  if (!isAllocated) _D2_FREE(sa);
}

// center by column
void _dccenter(size_t m, size_t n, real_t *a, real_t *sa) {
  size_t i, j;
  real_t *pa;
  bool isAllocated = true;
  if (!sa) {
    isAllocated = false;
    sa = _D2_MALLOC_SCALAR(n);
  }    
  _dcsum(m, n, a, sa);
  cblas_dscal(n, 1./m, sa, 1);
  for (i=0, pa=sa; i<n; ++i, ++pa) {
    for (j=0; j<m; ++j, ++a) (*a) -= *pa;
  }
  if (!isAllocated) _D2_FREE(sa);
}

// center by row
void _drcenter(size_t m, size_t n, real_t *a, real_t *sa) {
  size_t i, j;
  real_t *pa;
  bool isAllocated = true;
  if (!sa) {
    isAllocated = false;
    sa = _D2_MALLOC_SCALAR(m);
  }    
  _drsum(m, n, a, sa);
  cblas_dscal(m, 1./n, sa, 1);
  for (i=0; i<n; ++i) {
    pa = sa;
    for (j=0; j<m; ++j, ++a, ++pa) (*a) -= *pa;
  }
  if (!isAllocated) _D2_FREE(sa);
}

// c = a.*b
void _dvmul(size_t n, const real_t *a, const real_t *b, real_t *c) {
  size_t i;
  for (i=0; i<n; ++i, ++c, ++a, ++b)
    *c = (*a) * (*b);
}

void _dpdist2(const size_t d, const size_t n, const size_t m, 
	      const real_t * A, const real_t * B, real_t *C) {
  size_t i, j, ki, kj, k;
  assert(d>0 && n>0 && m>0);

  for (i=0; i<m*n; ++i) C[i] = 0;
  for (i=0; i<m; ++i)
    for (j=0; j<n; ++j)
      for (k=0, kj=j*d, ki=i*d; k<d; ++k, ++kj, ++ki) 
	C[i*n + j] += (A[kj] -  B[ki]) * (A[kj] -  B[ki]);
}

void _dpdist2_sym(const size_t d, const size_t n, const size_t m, const real_t *A, const index_t *Bi, real_t *C, const real_t *vocab) {
  size_t i, j, ki, kj, k;
  for (i=0; i<m*n; ++i) C[i] = 0;
  for (i=0; i<m; ++i)
    for (j=0; j<n; ++j)
      for (k=0, kj=j*d, ki=Bi[i]*d; k<d; ++k, ++kj, ++ki)
	  C[i*n + j] += (A[kj] - vocab[ki]) * (A[kj] - vocab[ki]);
}

void _dpdist2_sym2(const size_t d, const size_t n, const size_t m, const index_t *Ai, const index_t *Bi, real_t *C, const real_t *vocab) {
  size_t i, j, ki, kj, k;
  for (i=0; i<m*n; ++i) C[i] = 0;
  for (i=0; i<m; ++i)
    for (j=0; j<n; ++j)
      for (k=0, kj=Ai[j]*d, ki=Bi[i]*d; k<d; ++k, ++kj, ++ki)
	C[i*n + j] += (vocab[kj] - vocab[ki]) * (vocab[kj] - vocab[ki]);
}

void _dpdist2_submat(const size_t m, const size_t *Bi, real_t *C,
		     const size_t vocab_size, const real_t *dist_mat) {
  size_t i, j;
  assert(m>0);

  for (i=0; i<m; ++i)
    for (j=0; j<vocab_size; ++j)
      C[i*vocab_size + j] = dist_mat[Bi[i]*vocab_size + j];
}

void _dpdist_symbolic(const size_t d, const size_t n, const size_t m, 
		      const index_t * A, const index_t * B, real_t *C, 
		      const size_t vocab_size, const real_t* dist_mat) {
  size_t i,j, k;
  assert(d>0 && n>0 && m>0);
 
  for (i=0; i<m*n; ++i) C[i] = 0;
  for (i=0; i<m; ++i)
    for (j=0; j<n; ++j) 
      for (k=0; k<d; ++k)
	C[i*n+j] += dist_mat[A[j*d + k]*vocab_size + B[i*d + k]];
}

// inplace a -> exp(a)
void _dexp(size_t n, real_t *a) {
  size_t i;
  for (i=0; i<n; ++i, ++a) *a = exp(*a);
}
