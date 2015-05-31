#include "../common/d2.hpp"
#include "../common/d2_io_impl_rabit.hpp"

/* parallel program, read 20newsgroups dataset of 8 parts */
int main(int argc, char** argv) {

  using namespace d2;
  rabit::Init(argc, argv);

  size_t len[1] = {100}, dim[1] = {400}, size=20000;
  std::string type[1] = {"wordid"}; 
  
  parallel_md2_block data (size, dim, len, type, 1);
  std::string filename("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s");
  data.read(filename, size);

  rabit::Finalize();
  return 0;
}
