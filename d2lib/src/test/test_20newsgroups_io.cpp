#include "../common/d2.hpp"

int main(int argc, char** argv) {

  using namespace d2;

  size_t len[1] = {100}, dim[1] = {400}, size=20000;
  std::string type[1] = {"wordid"}; 
  
  md2_block data (size, dim, len, type, 1);
  data.read("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s", size);
  //  data.write("");
  return 0;
}
