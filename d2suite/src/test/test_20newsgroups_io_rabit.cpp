#include <rabit.h>
#include "../common/d2.hpp"

/* parallel program, read 20newsgroups dataset from multiple parts */
int main(int argc, char** argv) {

  using namespace d2;
  server::Init(argc, argv);

  size_t len = 100, size=20000;
  std::string filename("data/20newsgroups/20newsgroups_clean/20newsgroups.d2s");

  /* load full data */
  BlockMultiPhase<Elem<def::WordVec, 100> > data_retrieve (size, &len);
  data_retrieve.read(filename, size);

  /* load distributed data*/
  DistributedBlockMultiPhase<Elem<def::WordVec, 100> > data_query (size, &len);
  data_query.read(filename, size);

  // test nearest neighbors (simple)
  std::vector<real_t> emds(data_retrieve.get_size());
  std::vector<index_t> ranks(data_retrieve.get_size());

  double startTime = getRealTime();
  //  for (int i=0; i<data_query.get_size(); ++i) {
  int i=0; // select the first entry from query queue
    auto & block0 = data_query.get_block<0>();
    auto & block1 = data_retrieve.get_block<0>();
    std::cout << getLogHeader()  << "number of EMDs: " << KNearestNeighbors_Simple(2, block0[i], block1, &emds[0], &ranks[0]) << std::endl;
    //  }
  double totalTime = getRealTime() - startTime;

  server::Finalize();
  return 0;
}
