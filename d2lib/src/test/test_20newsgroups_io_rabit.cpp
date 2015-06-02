#include "../common/d2.hpp"
#include "../common/d2_io_impl_rabit.hpp"

/* parallel program, read 20newsgroups dataset of 8 parts */
int main(int argc, char** argv) {

  using namespace d2;
  rabit::Init(argc, argv);

  size_t len = 100, dim = 400, size=20000;
  
  DistributedBlockMultiPhase<def::WordVec> data (size, &dim, &len);
  std::string filename("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s");
  data.read(filename, size);

  rabit::Finalize();
  return 0;
}
