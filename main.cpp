#define BENCHMARK_RUNS          0
#include "component/benchmark.hpp"

int main()
{    
    BenchMark<double> myBenchMark_d;
    BenchMark<float> myBenchMark_f;

    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "               Device Info" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    #ifdef VIENNACL_WITH_OPENCL
        std::cout << viennacl::ocl::current_device().info() << std::endl;
    #endif

    std::cout << std::endl;

    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Benchmark :: Sparse" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    //std::cout << "   -------------------------------" << std::endl;
    //std::cout << "   # benchmarking single-precision" << std::endl;
    //std::cout << "   -------------------------------\n" << std::endl;
    //myBenchMark_f.run_benchmark(true);
    //for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    //{
    //    myBenchMark_f.run_benchmark(false);
    //}

    #ifdef VIENNACL_WITH_OPENCL
    if ( viennacl::ocl::current_device().double_support() )
    #endif
    {
        std::cout << std::endl;
        std::cout << "   -------------------------------" << std::endl;
        std::cout << "   # benchmarking double-precision" << std::endl;
        std::cout << "   -------------------------------\n" << std::endl;
        myBenchMark_d.run_benchmark(true,true,1);
//        myBenchMark_d.run_benchmark(true,false,1);
        for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
        {
            myBenchMark_d.run_benchmark(false,true,1);
        }
    }
    return 0;
}
