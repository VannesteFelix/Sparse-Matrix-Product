//#define VIENNACL_BUILD_INFO
#ifndef NDEBUG
 #define NDEBUG
#endif

#define CONST_1 255ULL

#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_WITH_EIGEN 1

#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <vector>

#include "viennacl/linalg/ilu.hpp"


template<class ScalarType>
class BenchMark
{

public:
    int run_benchmark(bool info, bool copyMethod, int benchmarkNbrRun); // copyMethod : TRUE = EIGEN | FALSE = UBLAS

};
