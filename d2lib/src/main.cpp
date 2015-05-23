#include "common/d2.hpp"

int main(int argc, char** argv) {

  using namespace d2;

  size_t len[2] = {8, 8}, dim[2] = {3, 3}, size=atoi(argv[1]);
  std::string type[2] = {"euclidean", "euclidean"}; 
  
  mult_d2_block data (size, dim, len, type, 2);
  data.read("mountaindat.d2", size);
  data.write("");
  return 0;
}
