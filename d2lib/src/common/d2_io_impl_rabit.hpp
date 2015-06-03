#ifndef _D2_IO_IMPL_RABIT_H_
#define _D2_IO_IMPL_RABIT_H_

#include <rabit.h>


namespace d2 {


  template<typename... Ts>
  void DistributedBlockMultiPhase<Ts...>::read_main(const std::string &filename, const size_t size) {
    using namespace rabit;
    static_cast<BlockMultiPhase<Ts...> > 
      (*this).read_main(filename + ".part" + std::to_string(GetRank()), size);
    global_size = this->size;
    Allreduce<op::Sum>(&global_size, 1);
  }

  template<typename... Ts>
  void DistributedBlockMultiPhase<Ts...>::read(const std::string &filename, const size_t size) {
    this->read_meta(filename);
    this->read_main(filename, (size - 1) / rabit::GetWorldSize() + 1);
  }
  

}

#endif /* _D2_IO_IMPL_RABIT_H_ */
