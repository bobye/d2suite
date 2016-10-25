#include "../common/d2.hpp"
#include "../learn/logistic_regression.hpp"

int main(int argc, char** argv) {
  using namespace d2;

  size_t len = 100, size=200;
  
  Block<Elem<def::WordVec, 100> > data (size, len);
  std::string prefix_name("data/20newsgroups/20newsgroups_clean/20newsgroups");
  data.read(prefix_name + ".d2s", size);
  data.read_label(prefix_name + ".label");
  return 0;
}
