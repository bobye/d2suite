#ifndef _D2_H_
#define _D2_H_
/*!
 * \file d2.hpp
 * \brief Main header file for apps
 *
 * Core data structures in template: 
 *     Elem, ElemMultiPhase, Meta, Block, BlockMultiPhase
 *
 * and distributed data structure: 
 *     DistributedBlockMultiPhase
 *
 */


/**********************************************************************/

#include "common.hpp"

/*! \namespace d2 */
namespace d2 {

  /*!
   * \namespace d2::def
   * \brief The type of discrete distribution decides how the
   * ground distance is computed. 
   * 
   * For example, d2::def::Euclidean
   * means the ground distance is computed from the Euclidean 
   * distance between two arbitrary vectors. Also, the type
   * of discrete distribution also decides how the data is 
   * stored. For example, d2::def::WordVec means the raw 
   * data is stored with index, aka, the word and the meta part 
   * stores the actual vector w.r.t. individual indexed word. 
   * Thus, different types corresponds to different meta data.
   * In the case of d2::def::Euclidean, the meta data is empty.
   */
  namespace def {    
    struct Euclidean;

    struct WordVec;
    
    struct NGram;
    
    struct Histogram;

    struct SparseHistogram;

    /// @tparam FuncType is a classification/regression class 
    template <typename FuncType> 
    struct Function;
  }


  /*!
   * \brief an element type, which has two parameters to specify, one is the
   * type of ground metric space, and the other is the dimension
   * of the ground metric space. 
   */
  template <typename D2Type, size_t dim>
  struct Elem;

  /*!
   * \brief a multi-phase element type
   */
  template <typename... Ts>
  struct ElemMultiPhase;

  /*!
   * \brief a class for storing the meta data associated with
   * an Elem type. It is an empty class for def::Euclidean. 
   */
  template <typename ElemType>
  class Meta;  

  /*!
   * \brief a class is for storing a block data of Elem in a 
   * local node. 
   */
  template <typename ElemType>
  class Block;

  /*!
   * \brief a block of elements with multiple phases
   */
  template <typename... Ts> // a sequence of Elem types
  class BlockMultiPhase;

#ifdef RABIT_RABIT_H_
  /*!
   * \brief a block of elements with 
   * multiple phases subject to a distributed setting.
   */
  template <typename... Ts>
  class DistributedBlockMultiPhase;
#endif 

  /*!
   * \brief compute pairwise (generalized) distance between two sets of vectors.
   * the two sets of vectors can be of different types.
   * \param s1 the first support array
   * \param n1 the count of s1
   * \param s2 the second support array
   * \param n2 the count of s2
   * \param meta the meta data for the second array
   * \param mat the n1 x n2 (distance) cost matrix computed
   */
  template <typename D2Type1, typename D2Type2, size_t dim>
  inline void pdist2 (const typename D2Type1::type *s1, const size_t n1,
		      const typename D2Type2::type *s2, const size_t n2,
		      const Meta<Elem<D2Type2, dim> > &meta,
		      __OUT__ real_t* mat);


  /*!
   * \brief compute EMD between two discrete distributions
   * \param e1 the first element
   * \param e2 the second element
   * \param meta the meta data of the second element
   * \param cache_mat the cached distance matrix, length of e1.len x e2.len 
   * \param cache_primal the primal solution of solved optimal transport, length of e1.len x e2.len 
   * \param cache_dual the dual solution of solved optimal transport, length of e1.len + e2.len
   * \param cost_computed bool variable implying whether the cost matrix 
                          are precomputed and supplied.
   */
  template <typename ElemType1, typename ElemType2>
  inline real_t EMD (const ElemType1 &e1, const ElemType2 &e2,
		     const Meta<ElemType2> &meta,
		     __IN_OUT__ real_t* cache_mat = NULL,
		     __OUT__ real_t* cache_primal = NULL, 
		     __OUT__ real_t* cache_dual = NULL,
		     __IN__ const bool cost_computed= false);

  /*!
   * \brief compute EMD between a (single-phased) discrete distribution and 
   * a block of (single-phased) discrete distributions
   * \param e the querying element
   * \param b the queried block of elements   
   */
  template <typename ElemType1, typename ElemType2>
  void EMD (const ElemType1 &e, const Block<ElemType2> &b,
	    __OUT__ real_t* emds,
	    __IN_OUT__ real_t* cache_mat = NULL,
	    __OUT__ real_t* cache_primal = NULL, 
	    __OUT__ real_t* cache_dual = NULL,
	    __IN__ const bool cost_computed = false);

  /*!
   * \brief compute EMD between a (multi-phased) discrete distribution and 
   * a block of (multi-phased) discrete distributions
   */
  template <template<typename...> class D1, template<typename... > class D2, 
	    typename... Ts1, typename... Ts2>
  void EMD(const D1<Ts1...> &e, const D2<Ts2...> &b,
	   __OUT__ real_t* emds);
  

  /*!
   * \brief compute lower bound of EMD.
   * version 0: extremely cheap, non-iterative
   */
  template <typename ElemType1, typename ElemType2>
  inline real_t LowerThanEMD_v0(const ElemType1 &e1, const ElemType2 &e2,
				const Meta<ElemType2> &meta);

  template <typename ElemType1, typename ElemType2>
  void LowerThanEMD_v0(const ElemType1 &e, const Block<ElemType2> &b,
		       __OUT__ real_t* emds);

  /*!
   * \brief compute lower bound of EMD.
   * version 1: fast, non-iterative and simple
   */
  template <typename ElemType1, typename ElemType2>
  inline real_t LowerThanEMD_v1(const ElemType1 &e1, const ElemType2 &e2,
				const Meta<ElemType2> &meta,
				__IN__ real_t* cache_mat = NULL);

  template <typename ElemType1, typename ElemType2>
  void LowerThanEMD_v1(const ElemType1 &e, const Block<ElemType2> &b,
		       __OUT__ real_t* emds,
		       __IN__ real_t* cache_mat);


  /*!
   * \brief simple linear approach without any prefetching or pruning.
   */
  template <typename ElemType1, typename ElemType2>
  void KNearestNeighbors_Linear(size_t k,
				const ElemType1 &e, const Block<ElemType2> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank);

  template <template<typename...> class D1, template<typename...> class D2,
	    typename... Ts1, typename... Ts2>
  void KNearestNeighbors_Linear(size_t k,
				const D1<Ts1...> &e,
				const D2<Ts2...> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank);


  /*!
   * \brief prefetching and pruning with lowerbounds (return the actual number of EMD computed).
   */
  template <typename ElemType1, typename ElemType2>
  size_t KNearestNeighbors_Simple(size_t k,
				const ElemType1 &e, const Block<ElemType2> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank,
				size_t n = 0);

  template <template<typename...> class D1, template<typename...> class D2,
	    typename... Ts1, typename... Ts2>
  size_t KNearestNeighbors_Simple(size_t k,
				const D1<Ts1...> &e, 
				const D2<Ts2...> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank,
				size_t n);

  /*! \brief initialize the d2 background utilities (including rabit and mosek). */
  inline void Init(int argc, char*argv[]);
  /*! \brief finalize the d2 background utilities. */
  inline void Finalize();

}

#include "d2_data.hpp"
#include "d2_io_impl.hpp"

#ifdef RABIT_RABIT_H_
#include "d2_io_impl_rabit.hpp"
#endif

#include "d2_server.hpp"
#include "d2_sa.hpp"

#endif /* _D2_H_ */
