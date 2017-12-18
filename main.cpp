#include "component/benchmark.hpp"

int main(int argc, char *argv[])
{    
    int benchMarkRun = 0;
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [BENCHMARK_RUNS_NUMBER]\n", argv[0]);
        exit(1);
    }
    else
    {
        benchMarkRun = atoi(argv[1]);
    }

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
    //for (int runs=0; runs<benchMarkRun; ++runs)
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
        myBenchMark_d.run_benchmark(true,true,100);
//        myBenchMark_d.run_benchmark(true,false,1);
        for (int runs=0; runs<benchMarkRun; ++runs)
        {
            myBenchMark_d.run_benchmark(false,true,1);
        }
    }
    return 0;
}
