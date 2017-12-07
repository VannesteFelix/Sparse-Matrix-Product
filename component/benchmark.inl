#include "benchmark.hpp"


inline double exec_time_ms(double time_sc){
    return time_sc*1000;
}

template<typename ScalarType>
int BenchMark<ScalarType>::run_benchmark(bool info, bool copyMethod, int benchmarkNbrRun)//string matName1,int mat1size1, int mat1size2 ,string matName2, int mat2size1, int mat2size2)
{
    viennacl::tools::timer timer;
    double exec_time_read,exec_time_read_ublas,exec_time_copy,exec_time;

    int sizeK = 46659; //20000;//
    int sizeJ = 31; //30;//


    ///////////////////////////     STEP 1      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                       Init all the variables                               */
    /* -------------------------------------------------------------------------- */

    ////////////////////////////////////////////////////////////////////////////////
    ///  EIGEN SPARSE MATRIX
    Eigen::MatrixMarketIterator<ScalarType> eig_itrr("/home/ros-kinetic/libraries_test/src/testdata");
    Eigen::SparseMatrix<ScalarType,Eigen::RowMajor> eigJtKJ(sizeJ,sizeJ);
    Eigen::SparseMatrix<ScalarType,Eigen::RowMajor> eigJtKJ_test(sizeJ,sizeJ);
    Eigen::SparseMatrix<ScalarType,Eigen::RowMajor> eigK(sizeK,sizeK);
    Eigen::SparseMatrix<ScalarType,Eigen::RowMajor> eigJt(sizeJ,sizeK);
    Eigen::SparseMatrix<ScalarType,Eigen::RowMajor> eigJ(sizeK,sizeJ);

    ////////////////////////////////////////////////////////////////////////////////
    ///  VIENNACL SPARSE MATRIX

    viennacl::compressed_matrix<ScalarType> vcl_compressed_JtKJ;
    viennacl::compressed_matrix<ScalarType> vcl_compressed_K;
    viennacl::compressed_matrix<ScalarType> vcl_compressed_KJ(sizeK,sizeJ);
    viennacl::compressed_matrix<ScalarType> vcl_compressed_J;
    viennacl::compressed_matrix<ScalarType> vcl_compressed_Jt;

    ////////////////////////////////////////////////////////////////////////////////
    ///  UBLAS SPARSE MATRIX

    boost::numeric::ublas::compressed_matrix<ScalarType> ublas_K(sizeK,sizeK);
    boost::numeric::ublas::compressed_matrix<ScalarType> ublas_J(sizeK,sizeJ);
    boost::numeric::ublas::compressed_matrix<ScalarType> ublas_Jt(sizeJ,sizeK);


    ///////////////////////////     STEP 2      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Read from file the sparse matrix                          */
    /*                   (in the Matrix Market format)                            */
    /* -------------------------------------------------------------------------- */


    ////////////////////////////////////////////////////////////////////////////////
    ///  WITH EIGEN
    timer.start();
    eigK = eig_itrr.matrix();
    eig_itrr.operator ++();
    eigJ = eig_itrr.matrix();
    eigJt = eigJ.transpose();
    exec_time_read = timer.get();

    ////////////////////////////////////////////////////////////////////////////////
    ///  WITH UBLAS
    if(!copyMethod)
    {
        timer.start();
        if (!viennacl::io::read_matrix_market_file(ublas_K, "/home/ros-kinetic/libraries_test/src/testdata/matKeig.mtx"))
        {
          std::cout << "Error reading Matrix file" << std::endl;
          return 0;
        }
        //unsigned int cg_mat_size = cg_mat.size();
        //std::cout << "done reading K" << std::endl;

        if (!viennacl::io::read_matrix_market_file(ublas_J, "/home/ros-kinetic/libraries_test/src/testdata/matJ1eig.mtx"))
        {
          std::cout << "Error reading Matrix file" << std::endl;
          return 0;
        }
        //unsigned int cg_mat_size = cg_mat.size();
        //std::cout << "done reading J" << std::endl;
        exec_time_read_ublas = timer.get();
    }


    ///////////////////////////     STEP 3      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Copy matrix from CPU to GPU                               */
    /* -------------------------------------------------------------------------- */
    timer.start();

    //  With Eigen Matrix
    if(copyMethod)
    {
        viennacl::copy(eigK, vcl_compressed_K);
        viennacl::copy(eigJ, vcl_compressed_J);
        viennacl::copy(eigJt, vcl_compressed_Jt);
    }
    // With UBlas Matrix
    else
    {
        ublas_Jt = boost::numeric::ublas::trans(ublas_J);
        viennacl::copy(ublas_K, vcl_compressed_K);
        viennacl::copy(ublas_J, vcl_compressed_J);
        viennacl::copy(ublas_Jt, vcl_compressed_Jt);
    }

    exec_time_copy = timer.get();

    if (info)
    {
        std::cout << "      -------          Matrix info         ---------" << std::endl;
        std::cout << "      eigK = "<< eig_itrr.matname() << std::endl;
        std::cout << "      eigK (rows,cols) : "<< eigK.rows() << " " << eigK.cols() << "\n" << std::endl;
        std::cout << "      eigJt = "<< eig_itrr.matname() << std::endl;
        std::cout << "      eigJ (rows,cols) : "<< eigJ.rows() << " " << eigJ.cols() << std::endl;
        std::cout << "      eigJt (rows,cols) : "<< eigJt.rows() << " " << eigJt.cols() << std::endl;
        if (!copyMethod)
        {
            std::cout << "\n" <<"      ublas_K (rows,cols) : "<< ublas_K.size1() << " " << ublas_K.size2() << std::endl;
            std::cout << "      ublas_J (rows,cols) : "<< ublas_J.size1() << " " << ublas_J.size2() << std::endl;
            std::cout << "      ublas_Jt (rows,cols) : "<< ublas_Jt.size1() << " " << ublas_Jt.size2() << std::endl;
        }
        std::cout << "      ----------------------------------------------\n" << std::endl;

        std::cout << "      Time to read Eigen Matrix : " << exec_time_ms(exec_time_read) << " ms" << std::endl;
        if (!copyMethod) std::cout << "      Time to read UBlas Matrix : " << exec_time_ms(exec_time_read_ublas) << " ms" << std::endl;
        std::cout << "      Time to copy CPU->GPU : " << exec_time_ms(exec_time_copy) << " ms\n" << std::endl;
    }

    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                         Matrix operations                                  */
    /*                             Jt * K J                                       */
    /* -------------------------------------------------------------------------- */

    //  On CPU with Eigen Matrix
    std::cout << "      ------- Jt*K*J product on CPU ----------" << std::endl;
    timer.start();
    for (int runs=0; runs<benchmarkNbrRun; ++runs)
    {
        eigJtKJ = eigJ.transpose() * eigK * eigJ;
    }
    exec_time = timer.get();
    std::cout << "      CPU time: " << exec_time_ms(exec_time) << " ms\n" << std::endl;

    //------------------------------------------------------------------------------

    //  On GPU with ViennaCL Compressed Matrix
    std::cout << "      ------- Jt*K*J product on GPU ----------" << std::endl;
    viennacl::backend::finish();
    timer.start();
    for (int runs=0; runs<benchmarkNbrRun; ++runs)
    {
                std::cout << "     ====> DIFFERENT MATRIX -- FAILURE !!"  << std::endl;
        vcl_compressed_KJ = viennacl::linalg::prod(vcl_compressed_K,vcl_compressed_J);
                std::cout << "     ====> DIFFERENT MATRIX -- FAILURE !!"  << std::endl;
        vcl_compressed_JtKJ = viennacl::linalg::prod(vcl_compressed_Jt,vcl_compressed_KJ);
                std::cout << "     ====> DIFFERENT MATRIX -- FAILURE !!"  << std::endl;
    }
    viennacl::backend::finish();
    exec_time = timer.get();
    std::cout << "      GPU time align1: " << exec_time_ms(exec_time) << " ms\n"<< std::endl;

    //------------------------------------------------------------------------------

    // On GPU with ViennaCL Hybrid Matrix
    //std::cout << "      ------- Jt*K*J product on GPU ----------" << std::endl;
    //viennacl::backend::finish();
    //timer.start();
    //for (int runs=0; runs<benchmarkNbrRun; ++runs)
    //{
    //    vcl_dense_matrix = viennacl::linalg::prod(vcl_hyb_K,vcl_dense_matrix);
    //}
    //viennacl::backend::finish();
    //exec_time = timer.get();
    //std::cout << "      GPU time align1: " << exec_time_ms(exec_time) << " ms\n"<< std::endl;

    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Compare the 2 resulting matrix                            */
    /* -------------------------------------------------------------------------- */
    timer.start();
    viennacl::copy(vcl_compressed_JtKJ,eigJtKJ_test);
    exec_time = timer.get();
    std::cout << "      Time to copy GPU->CPU : " << exec_time_ms(exec_time) << " ms\n"<< std::endl;
    if(!eigJtKJ.isApprox(eigJtKJ_test))
    {
        std::cout << "     ====> DIFFERENT MATRIX -- FAILURE !!"  << std::endl;
    }

    return EXIT_SUCCESS;
}
