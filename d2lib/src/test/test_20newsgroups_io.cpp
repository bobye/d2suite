#include "../common/d2.hpp"

/* serial program, split 20newsgroups dataset into 4 parts */
int main(int argc, char** argv) {

  using namespace d2;

  size_t len = 100, size=20000;
  
  BlockMultiPhase<Elem<def::WordVec, 400> > data (size, &len);
  std::string filename("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s");
  data.read(filename, size);
  data.split_write(filename, 4);

  return 0;
}
