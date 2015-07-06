#ifndef _D2_H_
#define _D2_H_

/*!
 * Main header file for apps
 */

#include "common.hpp"

namespace d2 {

  namespace def {    
    /*!
     * The type of discrete distribution decides how the
     * ground distance is computed. For example, d2::def::Euclidean
     * means the ground distance is computed from the Euclidean 
     * distance between two arbitray vectors. Also, the type
     * of discrete distribution also decides how the data is 
     * stored. For example, d2::def::WordVec means the raw 
     * data is stored with index, aka, the word and the meta part 
     * stores the actual vector w.r.t. individual indexed word. 
     * Thus, different types corresponds to different meta data.
     * In the case of d2::def::Euclidean, the meta data is empty.
     */
    struct Euclidean;

    struct WordVec;
    
    struct NGram;
    
    struct Histogram;
  }

  /**********************************************************************/
  /* Core data structures in template: 
   *     Elem, ElemMultiPhase, Meta, Block, BlockMultiPhase
   *
   * and distributed data structure: 
   *     DistributedBlockMultiPhase
   */

  /*!
   * Elem defines an element type has two parameters to specify, one is the
   * type of ground metric space, and the other is the dimension
   * of the ground metric space. 
   */
  template <typename D2Type, size_t dim>
  struct Elem;

  /*!
   * ElemMultiPhase defines a multi-phase element type
   * param @Ts: struct Elem<>
   */
  template <typename... Ts>
  struct ElemMultiPhase;

  /*!
   * A Meta class is for storing the meta data associated with
   * an Elem type. It is an empty class for def::Euclidean. 
   */
  template <typename ElemType>
  class Meta;  

  /*!
   * A Block class is for storing a block data of Elem in a 
   * local node. 
   */
  template <typename ElemType>
  class Block;

  /*!
   * BlockMultiPhase defines a block of elements with multiple phases
   * param @Ts: struct Elem<>
   */
  template <typename... Ts> // a sequence of Elem types
  class BlockMultiPhase;

#ifdef RABIT_RABIT_H_
  template <typename... Ts>
  class DistributedBlockMultiPhase;
#endif 

  /**********************************************************************/
  /* Core functions over previous data structures:
   *     pdist2, EMD, LowerThanEMD, KNearestNeighbors_Linear
   *
   */

  template <typename D2Type, size_t dim>
  inline void pdist2 (const typename D2Type::type *s1, const size_t n1,
		      const typename D2Type::type *s2, const size_t n2,
		      const Meta<Elem<D2Type, dim> > &meta,
		      __OUT__ real_t* mat);


  /*!
   * compute EMD between two discrete distributions
   */
  template <typename ElemType, typename MetaType>
  inline real_t EMD (const ElemType &e1, const ElemType &e2, const MetaType &meta,
		     __IN__ real_t* cache_mat = NULL,
		     __OUT__ real_t* cache_primal = NULL, 
		     __OUT__ real_t* cache_dual = NULL);

  /*!
   * compute EMD between a discrete distribution and a block of discrete distributions
   */
  template <typename ElemType>
  void EMD (const ElemType &e, const Block<ElemType> &b,
	    __OUT__ real_t* emds,
	    __IN__ real_t* cache_mat = NULL,
	    __OUT__ real_t* cache_primal = NULL, 
	    __OUT__ real_t* cache_dual = NULL);

  template <typename... Ts>
  void EMD(const ElemMultiPhase<Ts...> &e, const BlockMultiPhase<Ts...> &b,
	   __OUT__ real_t* emds);
  

  /*!
   * compute lower bound of EMD 
   * version 0: extremely cheap, non-iterative
   */
  template <typename ElemType, typename MetaType>
  inline real_t LowerThanEMD_v0(const ElemType &e1, const ElemType &e2, const MetaType &meta);

  /*!
   * compute lower bound of EMD 
   * version 1: fast, non-iterative and simple
   */
  template <typename ElemType, typename MetaType>
  inline real_t LowerThanEMD_v1(const ElemType &e1, const ElemType &e2, const MetaType &meta,
				__IN__ real_t* cache_mat = NULL);


  /*!
   * simple linear approach without any prefetching or pruning
   */
  template <typename ElemType>
  void KNearestNeighbors_Linear(size_t k,
				const ElemType &e, const Block<ElemType> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank);

  template <typename... Ts>
  void KNearestNeighbors_Linear(size_t k,
				const ElemMultiPhase<Ts...> &e,
				const BlockMultiPhase<Ts...> &b,
				__OUT__ real_t* emds_approx,
				__OUT__ index_t* rank);

  inline void Init(int argc, char*argv[]);
  inline void Finalize();

}

#include "d2_data.hpp"
#include "d2_io_impl.hpp"

#ifdef RABIT_RABIT_H_
#include "d2_io_impl_rabit.hpp"
#endif

#include "d2_server.hpp"

#endif /* _D2_H_ */
