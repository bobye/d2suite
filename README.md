# d2suite
`d2suite` is a C++ package for discrete distribution (d2) based 
__large-scale__ data processing framework. It supports distributed data analysis
of distributions at scale, such as nearest neighbors, clustering, and
some other machine learning capability. `d2suite` uses templates and C++11 features 
a lot, aiming to maximize its extensibility for different types of data.

`d2suite` also contains a collection of computing tools supporting the analysis 
of typical d2 data, such as images, sequences, documents.

*[under construction]*

Dependencies
 - BLAS
 - [rabit](https://github.com/dmlc/rabit): the use of generic parallel infrastructure
 - [mosek](https://www.mosek.com): fast LP/QP solvers, academic license available.

Make sure you have those pre-compiled libraries installed and
configured in the [d2suite/make.inc](d2suite/make.inc).
```bash
cd d2suite && make
```
You can run the test cases by first decompressing demo datasets in `d2suite/data/test` directory,
then try
```bash
make test
```

## Introduction
### Data Format Specifications
 - `def::Euclidean`: discrete distribution over Euclidean space
 - `def::WordVec`: discrete distribution with finite possible supports in Euclidean space (aka, embeddings)
 - `def::NGram`: n-gram data with cross-term distance
 - `def::Histogram`: dense histogram with cross-term distance
 - `def::SparseHistogram`: sparse histogram with cross-term distance

### Basic Functions
 - distributed/serial IO 
 - compute distance between a pair of D2: [Wasserstein distance](http://en.wikipedia.org/wiki/Wasserstein_metric) (or EMD).
 - compute lower/upper bounds of Wasserstein distance


### Learnings
 - K nearest neighbors [ongoing]
 - D2-clustering [TBA]

## Other Tools
 - document analysis: from bag-of-words to .d2s format [TBA]
 - [WMD code](http://matthewkusner.com/#page2), ICML 2015
 - [Sinkhorn Distances](http://www.iip.ist.i.kyoto-u.ac.jp/member/cuturi/SI.html): entropic regularized optimal transport, NIPS 2014

