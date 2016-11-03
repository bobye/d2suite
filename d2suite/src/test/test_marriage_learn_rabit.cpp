#include <rabit/rabit.h>
#include "../common/d2.hpp"
#include "../learn/logistic_regression.hpp"
#include "../learn/marriage_learner.hpp"

#define dim 100
#define cls 2
int main(int argc, char** argv) {
  using namespace d2;
  std::string prefix_name("data/20newsgroups/20newsgroups_clean/20newsgroups");
  const size_t start = 1;
  const real_t propo = 0.5;
  const size_t len = 100, size=1000;    
  server::Init(argc, argv);

#ifdef RABIT_RABIT_H
  if (rabit::GetRank() == 0)
#endif
  {  
    Block<Elem<def::WordVec, dim> > data (size, len);
    data.read(prefix_name + ".d2s", size);
    data.read_label(prefix_name + ".label", start);

    data.train_test_split_write(prefix_name + ".d2s", propo, start);
  }

  auto train_p = new Block<Elem<def::WordVec, dim> > (size*propo, 100);
  train_p->read(prefix_name + ".d2s.train", size*propo, prefix_name + ".d2s.meta0");
  train_p->read_label(prefix_name + ".d2s.train.label", start);

  auto test_p  = new Block<Elem<def::WordVec, dim> > (size*(1-propo), 100);
  test_p-> read(prefix_name + ".d2s.test", size*(1-propo), prefix_name + ".d2s.meta0");
  test_p-> read_label(prefix_name + ".d2s.test.label", start);

#ifdef RABIT_RABIT_H
  size_t subblock_size;
  subblock_size = (train_p->get_size() - 1) / rabit::GetWorldSize() + 1;
  Block<Elem<def::WordVec, dim> > train(*train_p, subblock_size * rabit::GetRank(), subblock_size, false);
  subblock_size = (test_p ->get_size() - 1) / rabit::GetWorldSize() + 1;
  Block<Elem<def::WordVec, dim> > test (*test_p,  subblock_size * rabit::GetRank(), subblock_size, false);
  delete train_p;
  delete test_p;
#else
  Block<Elem<def::WordVec, dim> > &train = *train_p;
  Block<Elem<def::WordVec, dim> > &test  = *test_p;
#endif


  // create and initialize the LR marriage learner 
  Elem<def::Function<Logistic_Regression<dim, cls+1> >, dim> marriage_learner;
  size_t num_of_classifers = 3;
  marriage_learner.len = num_of_classifers;
  marriage_learner.w = new real_t[num_of_classifers];
  marriage_learner.supp = new Logistic_Regression<dim, cls+1>[num_of_classifers];
  for (size_t i=0; i<marriage_learner.len; ++i) {
    marriage_learner.w[i] = 1. / num_of_classifers;
    marriage_learner.supp[i].init();
  }

  ML_BADMM(train, marriage_learner, 40, 2.0, &test, 1);
  server::Finalize();

  delete [] marriage_learner.w;
  delete [] marriage_learner.supp;
#ifndef RABIT_RABIT_H
  delete train_p;
  delete test_p;
#endif
  return 0;
}







