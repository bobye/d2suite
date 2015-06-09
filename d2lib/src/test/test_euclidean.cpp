#include "../common/d2.hpp"
#include "time.h"

int main(int argc, char** argv) {
  using namespace d2;

  size_t len[2] = {8, 8}, size=100;  

  BlockMultiPhase<Elem<def::Euclidean, 3>, Elem<def::Euclidean, 3> > data (size, len);
  data.read("data/test/euclidean_testdata.d2", size);
  //  data.write("");

  server::Init(argc, argv);
  srand(time(NULL));
  int i1 = rand() % size, i2 = rand() % size;
  auto & block = data.get_block<0>();
  std::cerr << "squared EMD between #" << i1 << " and #" << i2 
	    << ": "  << EMD(block[i1], block[i2], block.meta, NULL, NULL, NULL) << std::endl;
  server::Finalize();

  return 0;
}
