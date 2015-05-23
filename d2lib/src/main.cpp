#include "common/d2.hpp"

int main(int argc, char** argv) {

  using namespace d2;

  size_t len = 10, dim = 3;
  std::vector<std::string> type; type.push_back("euclidean");
  mult_d2_block data (100, &dim, &len, &type[0]);

  return 0;
}
