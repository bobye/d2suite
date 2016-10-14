#include "../common/d2.hpp"
#include <string>
#include <sstream>
#include <time.h>

int main(int argc, char** argv) {

  using namespace d2;
  server::Init(argc, argv);

  size_t len = 864, size=400;
  
  Block<Elem<def::Histogram, 0> > data (size, len);
  std::string filename("data/orl/orl.d2s");
  data.read(filename, size);


  size_t k = 40;
  Block<Elem<def::Histogram, 0> > wm3 (k, 864);
  for (size_t i=0; i<k; ++i) {
    std::string uniform_sample = "0 864 ";
    srand (i);
    for (size_t i=1; i<=864; ++i) {
      uniform_sample += std::to_string(rand()%100+1) + " ";    
    } 
    std::istringstream istr (uniform_sample);
    wm3.append(istr);
  }


  WM3_SA(wm3, data, 1000, .05, 0.1, 2., 20);
  wm3.write("data/orl/mixture_" + std::to_string(k) + "n.txt");

  std::ofstream f; f.open("data/orl/real.d2s");
  for (size_t i=0; i<size; ++i)
    operator<<(f, data[i]);
  f.close();
  
  server::Finalize();
  return 0;
}
