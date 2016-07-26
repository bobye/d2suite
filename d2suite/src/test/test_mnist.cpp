#include "../common/d2.hpp"
#include <string>
#include <sstream>
#include <time.h>

int main(int argc, char** argv) {

  using namespace d2;
  server::Init(argc, argv);

  size_t len = 150, size=200;
  
  Block<Elem<def::SparseHistogram, 0> > data (size, len);
  std::string filename("data/mnist/mnist60k_5.d2s");
  data.read(filename, size);


  size_t k = 40;
  Block<Elem<def::Histogram, 0> > wm3 (k, 784);
  for (size_t i=0; i<k; ++i) {
    std::string uniform_sample = "0 784 ";
    srand (i);
    for (size_t i=1; i<=784; ++i) {
      uniform_sample += std::to_string(rand()%100+1) + " ";    
    } 
    std::istringstream istr (uniform_sample);
    wm3.append(istr);
  }


  WM3_SA(wm3, data, 1000, .1, 0.9, 2., 20);
  wm3.write("data/mnist/mixture_5_" + std::to_string(k) + ".txt");

  std::ofstream f; f.open("data/mnist/real_5.d2s");
  for (size_t i=0; i<size; ++i)
    operator<<= <784> (f, data[i]);
  f.close();
  
  server::Finalize();
  return 0;
}
