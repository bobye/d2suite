#include "../common/d2.hpp"

int main(int argc, char** argv) {

  using namespace d2;

  size_t len = 150, size=100;
  
  Block<Elem<def::SparseHistogram, 0> > data (size, len);
  std::string filename("data/mnist/mnist60k.d2s");
  data.read(filename, size);

  return 0;
}
