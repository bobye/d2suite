#include <string>

namespace d2 {

  /* this defines the float point that will be used to 
   * store real numbers.
   */
  typedef float real_t;

  /* this defines the unsigned integer type that will
   * be used to store index 
   */
  typedef unsigned index_t;

#ifdef RABIT_RABIT_H_
  inline const std::string getLogHeader() 
  {return std::string("@d2lib(") + std::to_string(rabit::GetRank()) + ")";}
#else
  inline const std::string getLogHeader() 
  {return std::string("@d2lib");}
#endif

}
