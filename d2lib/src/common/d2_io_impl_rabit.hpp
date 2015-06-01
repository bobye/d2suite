#ifndef _D2_IO_IMPL_RABIT_H_
#define _D2_IO_IMPL_RABIT_H_

#include <rabit.h>


namespace d2 {

  template<typename... Ts>
  void parallel_md2_block<Ts...>::read_main(const std::string &filename, const size_t size) {
    using namespace rabit;
    static_cast<md2_block<Ts...> > 
      (*this).read_main(filename + ".part" + std::to_string(GetRank()), size);
    global_size = size;
    Allreduce<op::Sum>(&global_size, 1);
    for (size_t i=0; i<this->phase.size(); ++i)
      (*this)[i].get_global_size() = global_size;
  }

  template<typename... Ts>
  void parallel_md2_block<Ts...>::read(const std::string &filename, const size_t size) {
    this->read_meta(filename);
    this->read_main(filename, (size - 1) / rabit::GetWorldSize() + 1);
  }
  

}

#endif /* _D2_IO_IMPL_RABIT_H_ */
