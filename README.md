# d2lib
`d2lib` is a C++ library of discrete distribution (d2) based 
large-scale data processing framework. It also contains a collection of
computing tools supporting the analysis of d2 data.

*[under construction]*

## Data Format Specifications
 - discrete distribution over Euclidean space
 - discrete distribution with finite possible supports in Euclidean space (e.g., bag-of-word-vectors and sparsified histograms)
 - n-gram data with cross-term distance
 - dense histogram

## Basic Functions
 - distributed/serial IO 
 - compute Wasserstein distance between a pair of D2.


## Learnings
 - nearest neighbors [TBA]
 - D2-clustering [TBA]
 - Dirichlet process [TBA]

## Builds

Dependencies
 - BLAS
 - [rabit](https://github.com/dmlc/rabit): the use of generic parallel infrastructure
 - [mosek](https://www.mosek.com): fast LP/QP solvers

Make sure you have those re-compiled libraries installed and
configured in the [d2lib/Makefile](d2lib/Makefile).
```bash
cd d2lib && make && make test
```

## Other Tools
 - document analysis: from bag-of-words to .d2s format [TBA]

