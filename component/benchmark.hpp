//#define VIENNACL_BUILD_INFO
#ifndef NDEBUG
 #define NDEBUG
#endif

#define CONST_1 255ULL

#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_WITH_EIGEN 1

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>


#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/sliced_ell_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/tools/timer.hpp"

#include <sys/types.h>
#include <dirent.h>

//  Eigen Sparse Matrix
#include <Eigen/Sparse>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>
#include <unsupported/Eigen/src/SparseExtra/MatrixMarketIterator.h>

#include <iostream>
#include <vector>

template<class ScalarType>
class BenchMark
{

public:
    int run_benchmark(bool info, bool copyMethod, int benchmarkNbrRun); // copyMethod : TRUE = EIGEN | FALSE = UBLAS

};
