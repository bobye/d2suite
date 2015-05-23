#include "common/d2.hpp"

int main(int argc, char** argv) {

  using namespace d2;

  int len = 10;
  mult_d2_block<real_t> data (100, &len);

  return 0;
}
