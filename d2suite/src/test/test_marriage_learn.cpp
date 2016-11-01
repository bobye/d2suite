#include "../common/d2.hpp"
#include "../learn/logistic_regression.hpp"
#include "../learn/marriage_learner.hpp"

#define dim 100
#define cls 2
int main(int argc, char** argv) {
  using namespace d2;
  std::string prefix_name("data/20newsgroups/20newsgroups_clean/20newsgroups");
  const size_t start = 1;
  {
    size_t len = 100, size=1000;
  
    Block<Elem<def::WordVec, dim> > data (size, len);
    data.read(prefix_name + ".d2s", size);
    data.read_label(prefix_name + ".label", start);

    data.train_test_split_write(prefix_name + ".d2s", 0.7, start);
  }

  Block<Elem<def::WordVec, dim> > train (700, 100);
  train.read(prefix_name + ".d2s.train", 700, prefix_name + ".d2s.meta0");
  train.read_label(prefix_name + ".d2s.train.label", start);

  Block<Elem<def::WordVec, dim> > test (300, 100);
  test.read(prefix_name + ".d2s.test", 300, prefix_name + ".d2s.meta0");
  test.read_label(prefix_name + ".d2s.test.label", start);
  


  // create and initialize the LR marriage learner 
  Elem<def::Function<Logistic_Regression<dim, cls> >, dim> marriage_learner;
  size_t num_of_classifers = 2;
  marriage_learner.len = num_of_classifers;
  marriage_learner.w = new real_t[num_of_classifers];
  marriage_learner.supp = new Logistic_Regression<dim, cls>[num_of_classifers];
  for (size_t i=0; i<marriage_learner.len; ++i) {
    marriage_learner.w[i] = 1. / num_of_classifers;
    marriage_learner.supp[i].init();
  }

  server::Init(argc, argv);
  ML_BADMM(train, marriage_learner, 20, 2.0, &test, 1);
  server::Finalize();

  delete [] marriage_learner.w;
  delete [] marriage_learner.supp;
  return 0;
}
