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
  std::cerr << "squared EMD between #" << i1 << " and #" << i2 
	    << ": "  << EMD(data.head[i1], data.head[i2], &data.head.meta, NULL, NULL, NULL) << std::endl;
  server::Finalize();

  return 0;
}
