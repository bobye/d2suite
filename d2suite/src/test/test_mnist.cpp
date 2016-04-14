#include "../common/d2.hpp"
#include <string>
#include <sstream>
#include <time.h>

int main(int argc, char** argv) {

  using namespace d2;

  size_t len = 150, size=200;
  
  Block<Elem<def::SparseHistogram, 0> > data (size, len);
  std::string filename("data/mnist/mnist60k.d2s");
  data.read(filename, size);

  Block<Elem<def::SparseHistogram, 0> > &subdata=data.get_subblock(40,10);
  //  std::cerr << subdata[3] << std::endl;


  Block<Elem<def::Histogram, 0> > wm3 (32, 784);
  for (size_t i=0; i<32; ++i) {
    std::string uniform_sample = "0 784 ";
    srand (i);
    for (size_t i=1; i<=784; ++i) {
      uniform_sample += std::to_string(rand()%100+1) + " ";    
    } 
    std::istringstream istr (uniform_sample);
    wm3.append(istr);
  }

  //wm3.write("check_file.txt");

  WM3_SA(wm3, data, 100, .1, 0.997, 0.1);
 
  return 0;
}
