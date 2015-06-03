#include <rabit.h>
#include "../common/d2.hpp"

/* parallel program, read 20newsgroups dataset from 4 parts */
int main(int argc, char** argv) {

  using namespace d2;
  rabit::Init(argc, argv);

  size_t len = 100, size=20000;
  
  DistributedBlockMultiPhase<Elem<def::WordVec, 400> > data (size, &len);
  std::string filename("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s");
  data.read(filename, size);

  rabit::Finalize();
  return 0;
}
