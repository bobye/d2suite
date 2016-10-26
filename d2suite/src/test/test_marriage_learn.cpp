#include "../common/d2.hpp"
#include "../learn/logistic_regression.hpp"
#include "../learn/marriage_learner.hpp"

#define dim 100
int main(int argc, char** argv) {
  using namespace d2;

  size_t len = 100, size=200;
  
  Block<Elem<def::WordVec, dim> > data (size, len);
  std::string prefix_name("data/20newsgroups/20newsgroups_clean/20newsgroups");
  data.read(prefix_name + ".d2s", size);
  data.read_label(prefix_name + ".label");


  // create and initialize the LR marriage learner 
  Elem<def::Function<Logistic_Regression<dim, 20> >, dim> marriage_learner;
  size_t num_of_classifers = 10;
  marriage_learner.len = num_of_classifers;
  marriage_learner.w = new real_t[num_of_classifers];
  marriage_learner.supp = new Logistic_Regression<dim, 20>[num_of_classifers];
  for (size_t i=0; i<marriage_learner.len; ++i) {
    marriage_learner.w[i] = 1. / num_of_classifers;
    marriage_learner.supp[i].init();
  }


  ML_BADMM(data, marriage_learner, 20);

  delete [] marriage_learner.w;
  delete [] marriage_learner.supp;
  return 0;
}
