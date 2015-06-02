#include "../common/d2.hpp"

/* serial program, split 20newsgroups dataset into 8 parts */
int main(int argc, char** argv) {

  using namespace d2;

  size_t len = 100, dim = 400, size=20000;
  
  BlockMultiPhase<def::WordVec> data (size, &dim, &len);
  std::string filename("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s");
  data.read(filename, size);
  data.split_write(filename, 4);

  return 0;
}
