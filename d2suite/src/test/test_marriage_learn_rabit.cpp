#include <rabit/rabit.h>
#include "../common/d2.hpp"
#include "../learn/logistic_regression.hpp"
#include "../learn/decision_tree.hpp"

#define _USE_SPARSE_ACCELERATE_
#include "../learn/marriage_learner.hpp"

#define ClassiferType Decision_Tree
#define dim 100
#define cls 4
int main(int argc, char** argv) {
  using namespace d2;
  std::string prefix_name("data/20newsgroups/20newsgroups_clean/20newsgroups");
  const size_t start = 1;
  const real_t propo = 0.5;
  const size_t len = 100, size=2000;    
  server::Init(argc, argv);


  if (rabit::GetRank() == 0)
  {  
    Block<Elem<def::WordVec, dim> > data (size, len);
    data.read(prefix_name + ".d2s", size);
    data.read_label(prefix_name + ".label", start);

    unsigned int seed = 777;
    data.train_test_split_write(prefix_name + ".d2s", propo, start, seed);
  }
  rabit::Barrier();

  

  Block<Elem<def::WordVec, dim> > train_(size*propo, 100);
  train_.read(prefix_name + ".d2s.train", size*propo, prefix_name + ".d2s.meta0");
  train_.read_label(prefix_name + ".d2s.train.label", start);

  Block<Elem<def::WordVec, dim> > test_(size*(1-propo), 100);
  test_. read(prefix_name + ".d2s.test", size*(1-propo), train_.get_meta());
  test_. read_label(prefix_name + ".d2s.test.label", start);


  size_t subblock_size;
  subblock_size = (train_.get_size() - 1) / rabit::GetWorldSize() + 1;
  Block<Elem<def::WordVec, dim> > train(train_, subblock_size * rabit::GetRank(), subblock_size);
  subblock_size = (test_. get_size() - 1) / rabit::GetWorldSize() + 1;
  Block<Elem<def::WordVec, dim> > test (test_,  subblock_size * rabit::GetRank(), subblock_size);


  // create and initialize the LR marriage learner 
  Elem<def::Function<ClassiferType<dim, cls+1> >, dim> marriage_learner;
  size_t num_of_classifers = 20;
  marriage_learner.len = num_of_classifers;
  marriage_learner.w = new real_t[num_of_classifers];
  //marriage_learner.supp = new Logistic_Regression<dim, cls+1>[num_of_classifers];
  marriage_learner.supp = new ClassiferType<dim, cls+1>[num_of_classifers];

  double startTime = getRealTime();
  std::vector< Block<Elem<def::WordVec, dim> > * > validation;
  validation.push_back(&test);
  ML_BADMM_PARAM param;
  param.bootstrap = true;
  ML_BADMM(train, marriage_learner, param, validation);

  if (rabit::GetRank() == 0)
    std::cerr << getLogHeader() << " logging: marriage learning finished in " 
	      << (getRealTime() - startTime) << " seconds." << std::endl;
  server::Finalize();

  delete [] marriage_learner.w;
  delete [] marriage_learner.supp;
  return 0;
}
