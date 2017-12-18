#include <cusparse_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif


extern "C" {
#include "component/libraries/mmio.h"
}

///////////////////////////////////////////

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/sliced_ell_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"

///////////////////////////////////////////

struct MyCSRMat
{
    int * I;        // ROW INDICES OF NZ
    int * J;        // COLUMN INDICES OF NZ
    double * val;   // VALUES OF NZ
    int nz;         // NON-ZERO
    int M;          // ROW
    int N;          // COLUMN
}myMat1,myMat2,myMat3;

// error check macros
#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

char* appendCharToCharArray(char* array, char a)
{
    size_t len = strlen(array);

    char* ret = new char[len+2];

    strcpy(ret, array);
    ret[len] = a;
    ret[len+1] = '\0';

    return ret;
}

MyCSRMat initMatrix(MyCSRMat myMat, const char * myMatName)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int nz;
    int M, N;
    int i, *K, *I, *J;
    double *val;


    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    const char string[] = "/examples/testdata/";
    strcat(buff,string);
    strcat(buff,myMatName);
    printf("matrix ");
    printf(myMatName);
    printf(" read at : \n");
    printf(buff);
    printf("\n");


    if ((f = fopen(buff, "r")) == NULL)
        exit(1);


    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    K = (int *) malloc(nz * sizeof(int));
    I = (int *) malloc(nz+1 * sizeof(int)); // +1 because we put the number of nz in the end
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &K[i], &J[i], &val[i]);
        K[i];  /* adjust from 1-based to 0-based */
        J[i];
    }

    I = K;
    I[nz] = M;

    //printf("I[nz] : %d\n",I[nz]);
    //printf("I[nz-1] : %d\n",I[nz-1]);
    //printf("J[nz-1] : %d\n",J[nz-1]);
    //printf("val[nz-1] : %20.19g\n",val[nz-1]);

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    //for (i=0; i<nz; i++)
    //    fprintf(stdout, "%d %d %20.19g\n", I[i], J[i], val[i]);

    // myMat

    myMat.I = I;
    myMat.J = J;
    myMat.M = M;
    myMat.N = N;
    myMat.nz = nz;
    myMat.val = val;

    return myMat;

}

int compute(MyCSRMat myMat1, MyCSRMat myMat2)
{
    bool test = false; // TRUE => exemple | False => loaded matrix

    int N = 50000;
    // matrix generation and validation depends on these relationships:
    int SCL = 2;
    int K = N;
    int M = SCL*N;
    // A: MxK  B: KxN  C: MxN

    std::clock_t start;
    double duration, computeT;

    cusparseStatus_t stat;
    cusparseHandle_t hndl;
    cusparseMatDescr_t descrA, descrB, descrC;
    int *csrRowPtrA, *csrRowPtrB, *csrRowPtrC, *csrColIndA, *csrColIndB, *csrColIndC;
    int *h_csrRowPtrA, *h_csrRowPtrB, *h_csrRowPtrC, *h_csrColIndA, *h_csrColIndB, *h_csrColIndC;
    float *csrValA, *csrValB, *csrValC, *h_csrValA, *h_csrValB, *h_csrValC;
    double *h_csrValCd;
    int nnzA, nnzB, nnzC;   // number of non-zero
    int m,n,k;
    m = M;
    n = N;
    k = K;

    if (test){
        ///////////////////////////     STEP 1      ////////////////////////////////////
        /* -------------------------------------------------------------------------- */
        /*                           generate A, B=2I                                 */
        /*
                                     A:
                                    |1.0 0.0 0.0 ...|
                                    |1.0 0.0 0.0 ...|
                                    |0.0 1.0 0.0 ...|
                                    |0.0 1.0 0.0 ...|
                                    |0.0 0.0 1.0 ...|
                                    |0.0 0.0 1.0 ...|
                                    ...

                                    B:
                                    |2.0 0.0 0.0 ...|
                                    |0.0 2.0 0.0 ...|
                                    |0.0 0.0 2.0 ...|
                                    ...                                               */
        /* -------------------------------------------------------------------------- */
        start = std::clock();

            nnzA = m;
            nnzB = n;
            h_csrRowPtrA = (int *)malloc((nnzA+1)*sizeof(int));
            h_csrColIndA = (int *)malloc(nnzA*sizeof(int));
            h_csrValA  = (float *)malloc(nnzA*sizeof(float));

            h_csrRowPtrB = (int *)malloc((nnzB+1)*sizeof(int));
            h_csrColIndB = (int *)malloc(nnzB*sizeof(int));
            h_csrValB  = (float *)malloc(nnzB*sizeof(float));

        duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
        printf("Host Malloc :                                   %f ms\n", duration);
        //------------------------------------------------------------------------------
        start = std::clock();

            if ((h_csrRowPtrA == NULL) || (h_csrRowPtrB == NULL) || (h_csrColIndA == NULL) || (h_csrColIndB == NULL) || (h_csrValA == NULL) || (h_csrValB == NULL))
            {printf("malloc fail\n"); return -1;}
            for (int i = 0; i < nnzA; i++){
            h_csrValA[i] = 1.0f;
            h_csrRowPtrA[i] = i;
            h_csrColIndA[i] = i/SCL;
            if (i < nnzB){
              h_csrValB[i] = 2.0f;
              h_csrRowPtrB[i] = i;
              h_csrColIndB[i] = i;}
            }
            h_csrRowPtrA[nnzA] = nnzA;
            h_csrRowPtrB[nnzB] = nnzB;


        duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
        printf("CSR Matrix Generation :                         %f ms\n", duration);
    }
    else{
        nnzA = myMat1.nz;
        nnzB = myMat2.nz;
    }

    ///////////////////////////     STEP 2      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                      Allocate memory on the device                         */
    /*              and return a ptr of its memory emplacement                    */
    /* -------------------------------------------------------------------------- */

    if (test){
        start = std::clock();

            cudaMalloc(&csrRowPtrA, (m+1)*sizeof(int));

        duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
        printf("First cudaMalloc :                              %f ms\n", duration);
        //------------------------------------------------------------------------------
        start = std::clock();

            cudaMalloc(&csrColIndA, nnzA*sizeof(int));
            cudaMalloc(&csrValA, nnzA*sizeof(float));

            cudaMalloc(&csrRowPtrB, (nnzB+1)*sizeof(int));
            cudaMalloc(&csrColIndB, nnzB*sizeof(int));
            cudaMalloc(&csrValB, nnzB*sizeof(float));

        duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
        printf("cudaMalloc csrRowPtrB|csrColIndA/B|csrValA/B :  %f ms\n", duration);
    }
    else {
        start = std::clock();

            cudaMalloc(&csrRowPtrA, (myMat1.nz+1)*sizeof(int));

        duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
        printf("cudaMalloc csrRowPtrA :                         %f ms\n", duration);
        //------------------------------------------------------------------------------
        start = std::clock();

            cudaMalloc(&csrColIndA, myMat1.nz*sizeof(int));
            cudaMalloc(&csrValA, myMat1.nz*sizeof(double));

            cudaMalloc(&csrRowPtrB, (myMat2.nz+1)*sizeof(int));
            cudaMalloc(&csrColIndB, myMat2.nz*sizeof(int));
            cudaMalloc(&csrValB, myMat2.nz*sizeof(double));


        duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
        printf("cudaMalloc csrRowPtrB|csrColIndA/B|csrValA/B :  %f ms\n", duration);
    }

    ///////////////////////////     STEP 3      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Copy the data from the Host (CPU)                         */
    /*                      to the device (GPU)                                   */
    /* -------------------------------------------------------------------------- */
    start = std::clock();
    computeT = start;

    if (test){
        cudaCheckErrors("cudaMalloc fail");
        cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (nnzA+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndA, h_csrColIndA, nnzA*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValA, h_csrValA, nnzA*sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(csrRowPtrB, h_csrRowPtrB, (nnzB+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndB, h_csrColIndB, nnzB*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValB, h_csrValB, nnzB*sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy fail");
    }
    else{
        cudaCheckErrors("cudaMalloc fail");
        cudaMemcpy(csrRowPtrA, myMat1.I, (myMat1.nz+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndA, myMat1.J, myMat1.nz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValA, myMat1.val, myMat1.nz*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(csrRowPtrB, myMat2.I, (myMat2.nz+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndB, myMat2.J, myMat2.nz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValB, myMat2.val, myMat2.nz*sizeof(double), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy fail");

        //printf(myMat2.I);
    }

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("Copy Data from Host to Device :                 %f ms\n", duration);
    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                      set cusparse matrix types                             */
    /*                             ?????                                          */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

        CUSPARSE_CHECK(cusparseCreate(&hndl));

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("cusparseCreate(&hndl) :                         %f ms\n", duration);
    //------------------------------------------------------------------------------
    start = std::clock();

        stat = cusparseCreateMatDescr(&descrA);
        CUSPARSE_CHECK(stat);
        stat = cusparseCreateMatDescr(&descrB);
        CUSPARSE_CHECK(stat);
        stat = cusparseCreateMatDescr(&descrC);
        CUSPARSE_CHECK(stat);

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("cusparseCreateMatDescr(&descrA/B/C) :           %f ms\n", duration);
    //------------------------------------------------------------------------------
    start = std::clock();

        stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUSPARSE_CHECK(stat);
        stat = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUSPARSE_CHECK(stat);
        stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUSPARSE_CHECK(stat);
        stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        CUSPARSE_CHECK(stat);
        stat = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
        CUSPARSE_CHECK(stat);
        stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
        CUSPARSE_CHECK(stat);
        cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;


    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("Set cusparse matrix types :                     %f ms\n", duration);
    ///////////////////////////     STEP 5      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                          Determine csrRowPtrC                              */
    /*                                   &                                        */
    /*                  the total number of nonzero elements                      */
    /*                                                                            */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

    // figure out size of C
    int baseC;

    if (test){
        // nnzTotalDevHostPtr points to host memory
        int *nnzTotalDevHostPtr = &nnzC;
        stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
        CUSPARSE_CHECK(stat);
        cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
        cudaCheckErrors("cudaMalloc fail");

    //------------------------------------------------------------------------------

        // ????
        stat = cusparseXcsrgemmNnz(hndl, transA, transB, m, n, k,
            descrA, nnzA, csrRowPtrA, csrColIndA,
            descrB, nnzB, csrRowPtrB, csrColIndB,
            descrC, csrRowPtrC, nnzTotalDevHostPtr );
        CUSPARSE_CHECK(stat);

    //------------------------------------------------------------------------------

        //  ????
        if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;}
        else{
        cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy fail");
        nnzC -= baseC;}
        cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
        cudaMalloc((void**)&csrValC, sizeof(float)*nnzC);
        cudaCheckErrors("cudaMalloc fail");
    }
    else{
        // nnzTotalDevHostPtr points to host memory
        int *nnzTotalDevHostPtr = &nnzC;
        printf("INFO 1 NNZC:                                    %i\n",nnzC);
        stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
        CUSPARSE_CHECK(stat);
        cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(myMat1.M+1));
        cudaCheckErrors("cudaMalloc fail");

    //------------------------------------------------------------------------------

        // Determine csrRowPtrC
        stat = cusparseXcsrgemmNnz(hndl, transA, transB, myMat1.M, myMat2.N, myMat1.N,
            descrA, nnzA, csrRowPtrA, csrColIndA,
            descrB, nnzB, csrRowPtrB, csrColIndB,
            descrC, csrRowPtrC, nnzTotalDevHostPtr );
        CUSPARSE_CHECK(stat);

    //------------------------------------------------------------------------------

        //  Gathers nnzC
        if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;}
        else
        {
            cudaMemcpy(&nnzC, csrRowPtrC+myMat1.M, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
            cudaCheckErrors("cudaMemcpy fail");
            nnzC -= baseC;
        }
        cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
        cudaMalloc((void**)&csrValC, sizeof(double)*nnzC);
        cudaCheckErrors("cudaMalloc fail");
        printf("INFO 2 NNZC:                                    %i\n",nnzC);

    }

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("nnzTotalDevHostPtr points to host memory :      %f ms\n", duration);
    ///////////////////////////     STEP 6      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                Perform multiplication C = A*B                              */
    /*                                                                            */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

    if (test){
        stat = cusparseScsrgemm(hndl, transA, transB, m, n, k,
            descrA, nnzA,
            csrValA, csrRowPtrA, csrColIndA,
            descrB, nnzB,
            csrValB, csrRowPtrB, csrColIndB,
            descrC,
            csrValC, csrRowPtrC, csrColIndC);
        CUSPARSE_CHECK(stat);
    }
    else{
        stat = cusparseScsrgemm(hndl, transA, transB, myMat1.M, myMat2.N, myMat1.N,
            descrA, nnzA,
            csrValA, csrRowPtrA, csrColIndA,
            descrB, nnzB,
            csrValB, csrRowPtrB, csrColIndB,
            descrC,
            csrValC, csrRowPtrC, csrColIndC);
        CUSPARSE_CHECK(stat);
    }

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("GPU calculation time :                          %f ms\n", duration);
    ///////////////////////////     STEP 7      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Copy result (C) back to host                              */
    /*                       & test & validate it                                 */
    /* -------------------------------------------------------------------------- */
    start = std::clock();
    if (test){
        // copy result (C) back to host
        h_csrRowPtrC = (int *)malloc((m+1)*sizeof(int));
        h_csrColIndC = (int *)malloc(nnzC *sizeof(int));
        h_csrValC  = (float *)malloc(nnzC *sizeof(float));
        if ((h_csrRowPtrC == NULL) || (h_csrColIndC == NULL) || (h_csrValC == NULL))
        {printf("malloc fail\n"); return -1;}
        cudaMemcpy(h_csrRowPtrC, csrRowPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csrColIndC, csrColIndC,  nnzC*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csrValC, csrValC, nnzC*sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy fail");

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("Copy GPU to CPU :                               %f ms\n", duration);
    //------------------------------------------------------------------------------


        // check result, C = 2A
        if (nnzC != m) {printf("invalid matrix size C: %d, should be: %d\n", nnzC, m); return -1;}
        for (int i = 0; i < m; i++){
        if (h_csrRowPtrA[i] != h_csrRowPtrC[i]) {printf("A/C row ptr mismatch at %d, A: %d, C: %d\n", i, h_csrRowPtrA[i], h_csrRowPtrC[i]); return -1;}
        if (h_csrColIndA[i] != h_csrColIndC[i]) {printf("A/C col ind mismatch at %d, A: %d, C: %d\n", i, h_csrColIndA[i], h_csrColIndC[i]); return -1;}
        if ((h_csrValA[i]*2.0f) != h_csrValC[i]) {printf("A/C value mismatch at %d, A: %f, C: %f\n", i, h_csrValA[i]*2.0f, h_csrValC[i]); return -1;}
        }
    }
    else{
        // copy result (C) back to host
        h_csrRowPtrC = (int *)malloc((myMat1.M+1)*sizeof(int));
        h_csrColIndC = (int *)malloc(nnzC *sizeof(int));
        h_csrValCd  = (double *)malloc(nnzC *sizeof(double));
        if ((h_csrRowPtrC == NULL) || (h_csrColIndC == NULL) || (h_csrValCd == NULL))
        {printf("malloc fail\n"); return -1;}
        cudaMemcpy(h_csrRowPtrC, csrRowPtrC, (myMat1.M+1)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csrColIndC, csrColIndC, nnzC*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_csrValCd, csrValC, nnzC*sizeof(double), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy fail");

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("Copy GPU to CPU :                               %f ms\n", duration);
    //------------------------------------------------------------------------------

        //if (nnzC != myMat1.M) {printf("invalid matrix size C: %d, should be: %d\n", nnzC, myMat1.M); return -1;}

    }
    duration = (( std::clock() - computeT ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("===========================================================\n");
    printf("RESULT PRODUCT INFO NZ:                         %i\n",nnzC);
    printf("REAL TIME TO COMPUTE :                          %f ms\n", duration);

    myMat3.I = h_csrRowPtrC;
    myMat3.J = h_csrColIndC;
    myMat3.M = myMat1.M+1;
    myMat3.N = nnzC;
    myMat3.nz = nnzC;
    myMat3.val = h_csrValCd;

//    for (int i=0; i<myMat3.nz; i++)
//        fprintf(stdout, "%d %d %d\n", myMat3.I[i], myMat3.J[i], myMat3.val[i]);


    return 0;
}

inline double exec_time_ms(double time_sc){
    return time_sc*1000;
}

// perform sparse-matrix multiplication C=AxB
int main(int argc, char *argv[]){

//    int ret_code;
//    MM_typecode matcode;
//    FILE *f;
//    int nz;
//    int M, N;
//    int i, *K, *I, *J;
//    double *val;


//    if (argc < 3)
//    {
//        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
//        exit(1);
//    }

//    myMat1 = initMatrix(myMat1,argv[1]);
//    myMat2 = initMatrix(myMat2,argv[2]);

//    //printf("TEST :\n");
//    //printf("I[nz] : %d\n",myMat1.I[myMat1.nz]);
//    //printf("I[nz-1] : %d\n",myMat1.I[myMat1.nz-1]);
//    //printf("J[nz-1] : %d\n",myMat1.J[myMat1.nz-1]);
//    //printf("val[nz-1] : %20.19g\n",myMat1.val[myMat1.nz-1]);

//    //printf("I[nz] : %d\n",myMat2.I[myMat2.nz]);
//    //printf("I[nz-1] : %d\n",myMat2.I[myMat2.nz-1]);
//    //printf("J[nz-1] : %d\n",myMat2.J[myMat2.nz-1]);
//    //printf("val[nz-1] : %20.19g\n",myMat2.val[myMat2.nz-1]);

//    std::clock_t start;
//    double duration;
//    printf("----------          COMPUTE K * K       ---------------------\n\n");
//    start = std::clock();
//    compute(myMat1,myMat2);
//    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
//    printf("COMPLETE PROCESS 1 TIME :                       %f ms\n", duration);

//    printf("\n-----------------------------------------------------------\n\n");

//    start = std::clock();
//    compute(myMat1,myMat2);
//    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
//    printf("COMPLETE PROCESS 2 TIME :                       %f ms\n", duration);


//    return 0;


///////////////////////////////////////////
///////////////////////////////////////////



    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    std::string testdata_dir(buff);
    testdata_dir.append("/examples/testdata");
    std::string myMatrix1 = testdata_dir;
    std::string myMatrix2 = testdata_dir;
    myMatrix1.append("/matKeig.mtx");
    myMatrix2.append("/matJ1eig.mtx");
    bool info = true;
    int benchmarkNbrRun = 100;
    typedef double ScalarType;

    viennacl::tools::timer timer;
    double exec_time_read,exec_time_read_ublas,exec_time_copy,exec_time;

    int sizeK = 46659; //20000;//
    int sizeJ = 31; //30;//


    ///////////////////////////     STEP 1      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                       Init all the variables                               */
    /* -------------------------------------------------------------------------- */

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
    ///  WITH UBLAS

    timer.start();
    if (!viennacl::io::read_matrix_market_file(ublas_K, myMatrix1))
    {
      std::cout << "Error reading Matrix file" << std::endl;
      return 0;
    }
    //unsigned int cg_mat_size = cg_mat.size();
    std::cout << "done reading K" << std::endl;

    if (!viennacl::io::read_matrix_market_file(ublas_J, myMatrix2))
    {
      std::cout << "Error reading Matrix file" << std::endl;
      return 0;
    }
    //unsigned int cg_mat_size = cg_mat.size();
    std::cout << "done reading J" << std::endl;
    exec_time_read_ublas = timer.get();


    ///////////////////////////     STEP 3      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Copy matrix from CPU to GPU                               */
    /* -------------------------------------------------------------------------- */
    timer.start();

    // With UBlas Matrix

    ublas_Jt = boost::numeric::ublas::trans(ublas_J);
    viennacl::copy(ublas_J, vcl_compressed_J);
    viennacl::copy(ublas_Jt, vcl_compressed_Jt);
    timer.start();

    viennacl::copy(ublas_K, vcl_compressed_K);

    exec_time_copy = timer.get();

    if (info)
    {
        std::cout << "\n" <<"      ublas_K (rows,cols) : "<< ublas_K.size1() << " " << ublas_K.size2() << std::endl;
        std::cout << "      ublas_J (rows,cols) : "<< ublas_J.size1() << " " << ublas_J.size2() << std::endl;
        std::cout << "      ublas_Jt (rows,cols) : "<< ublas_Jt.size1() << " " << ublas_Jt.size2() << std::endl;
        std::cout << "      ----------------------------------------------\n" << std::endl;
        std::cout << "      Time to copy CPU->GPU : " << exec_time_ms(exec_time_copy) << " ms\n" << std::endl;
    }

    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                         Matrix operations                                  */
    /*                             Jt * K J                                       */
    /* -------------------------------------------------------------------------- */


    //  On GPU with ViennaCL Compressed Matrix
    std::cout << "      ------- Jt*K*J product on GPU ----------" << std::endl;
    viennacl::backend::finish();
    timer.start();
    for (int runs=0; runs<benchmarkNbrRun; ++runs)
    {
        vcl_compressed_KJ = viennacl::linalg::prod(vcl_compressed_K,vcl_compressed_J);
        vcl_compressed_JtKJ = viennacl::linalg::prod(vcl_compressed_Jt,vcl_compressed_KJ);
    }
    viennacl::backend::finish();
    exec_time = timer.get();
    std::cout << "      GPU time align1: " << exec_time_ms(exec_time) << " ms\n"<< std::endl;

    return 0;

}
