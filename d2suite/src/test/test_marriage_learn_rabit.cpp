#include <rabit/rabit.h>
#include "../common/d2.hpp"
#include "../learn/logistic_regression.hpp"
#include "../learn/decision_tree.hpp"

#define _USE_SPARSE_ACCELERATE_
#include "../learn/marriage_learner.hpp"

#include <tclap/CmdLine.h>

/***********************************************************************************/
// problem setup for specific dataset
#define ClassiferType Decision_Tree ///< type of classifer
#define dim 100 ///< feature dimension
#define cls 4 ///< number of classes
static d2::ML_BADMM_PARAM param;
inline void set_param() {
  param.bootstrap = true; // has to be set true for Decision_Tree<>
};
/***********************************************************************************/

int main(int argc, char** argv) {
/***********************************************************************************/
// command line parameters parsing
  TCLAP::CmdLine cmd("Marriage Learning Client", ' ', "0.1");
  TCLAP::ValueArg<std::string>
    nameArg("f","prefix","prefix of d2s filename to read",true,"","string");
  cmd.add(nameArg);
  TCLAP::ValueArg<size_t>
    sizeArg("n","size","number of samples to read",true,1000,"size_t");
  cmd.add(sizeArg);
  TCLAP::ValueArg<size_t>
    cnumArg("c","cnum","number of classifiers",true,4,"size_t");
  cmd.add(cnumArg);
  try {
    cmd.parse( argc, argv );
  } catch (TCLAP::ArgException &e)  // catch any exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
/***********************************************************************************/
  using namespace d2;
  set_param();
  std::string prefix_name(nameArg.getValue());
  const size_t start = 1;
  const real_t propo = 0.5;
  const size_t len = 100, size=sizeArg.getValue();    
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
  size_t num_of_classifers = cnumArg.getValue();
  marriage_learner.len = num_of_classifers;
  marriage_learner.w = new real_t[num_of_classifers];
  //marriage_learner.supp = new Logistic_Regression<dim, cls+1>[num_of_classifers];
  marriage_learner.supp = new ClassiferType<dim, cls+1>[num_of_classifers];

  double startTime = getRealTime();
  std::vector< Block<Elem<def::WordVec, dim> > * > validation;
  validation.push_back(&test);
  ML_BADMM(train, marriage_learner, param, validation);

  if (rabit::GetRank() == 0)
    std::cerr << getLogHeader() << " logging: marriage learning finished in " 
	      << (getRealTime() - startTime) << " seconds." << std::endl;
  server::Finalize();

  delete [] marriage_learner.w;
  delete [] marriage_learner.supp;
  return 0;
}
