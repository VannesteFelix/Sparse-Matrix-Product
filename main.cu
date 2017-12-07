//#include <cusparse_v2.h>
//#include <stdio.h>
//#include <time.h>
//#include <sys/time.h>

//// error check macros
//#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

//#define cudaCheckErrors(msg) \
//do { \
//    cudaError_t __err = cudaGetLastError(); \
//    if (__err != cudaSuccess) { \
//        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//            msg, cudaGetErrorString(__err), \
//            __FILE__, __LINE__); \
//        fprintf(stderr, "*** FAILED - ABORTING\n"); \
//        exit(1); \
//    } \
//} while (0)


//double timerval()
//{
//    struct timeval st;
//    gettimeofday(&st, NULL);
//    return (st.tv_sec+st.tv_usec*1e-6);
//}

//// perform sparse-matrix multiplication C=AxB
//int main(){
//double avg_time = 0, s_time, e_time;

//cusparseStatus_t stat;
//cusparseHandle_t hndl;
//cusparseMatDescr_t descrA, descrB, descrC;
//int *csrRowPtrA, *csrRowPtrB, *csrRowPtrC, *csrColIndA, *csrColIndB, *csrColIndC;
//int *h_csrRowPtrA, *h_csrRowPtrB, *h_csrRowPtrC, *h_csrColIndA, *h_csrColIndB, *h_csrColIndC,*pos;
//float *csrValA, *csrValB, *csrValC, *h_csrValA, *h_csrValB, *h_csrValC;
//int nnzA, nnzB, nnzC;
//int m=4,n,k,loop;
//int i,j;
//int iterations;
//for (iterations=0;iterations<10;iterations++)
//{
//    m *=2;
//    n = m;
//    k = m;
//    //density of the sparse matrix to be created. Assume 5% density.
//    double dense_const = 0.05;
//    int temp5, temp6,temp3,temp4;
//    int density=(m*n)*(dense_const);
//    nnzA = density;
//    nnzB = density;
//    h_csrRowPtrA = (int *)malloc((m+1)*sizeof(int));
//    h_csrRowPtrB = (int *)malloc((n+1)*sizeof(int));
//    h_csrColIndA = (int *)malloc(density*sizeof(int));
//    h_csrColIndB = (int *)malloc(density*sizeof(int));
//    h_csrValA  = (float *)malloc(density*sizeof(float));
//    h_csrValB  = (float *)malloc(density*sizeof(float));
//    if ((h_csrRowPtrA == NULL) || (h_csrRowPtrB == NULL) || (h_csrColIndA == NULL) || (h_csrColIndB == NULL) || (h_csrValA == NULL) || (h_csrValB == NULL))
//    {printf("malloc fail\n"); return -1;}

//    //position array for random initialisation of positions in input matrix
//    pos= (int *)calloc((m*n), sizeof(int));
//    int temp,temp1;

//    //  printf("the density is %d\n",density);
//    //  printf("check 1:\n");

//    //randomly initialise positions
//    for(i=0;i<density;i++)
//    {
//        temp1=rand()%(m*n);
//        pos[i]=temp1;

//    }
//    //  printf("check 2:\n");

//    //sort the 'pos' array
//    for (i = 0 ; i < density; i++) {
//        int d = i;
//        int t;

//        while ( d > 0 && pos[d] < pos[d-1]) {
//            t          = pos[d];
//            pos[d]   = pos[d-1];
//            pos[d-1] = t;
//            d--;
//        }
//    }
//    // initialise with non zero elements and extract column and row ptr vector
//    j=1;
//    //ja[0]=1;

//    int p=0;
//    int f=0;

//    for(i = 0; i < density; i++)
//    {
//        temp=pos[i];
//         h_csrValA[f] = rand();
//         h_csrValB[f] = rand();
//         h_csrColIndA[f] = temp%m;
//         h_csrColIndB[f] = temp%m;
//        f++;
//        p++;
//        temp5= pos[i];
//        temp6=pos[i+1];
//        temp3=temp5-(temp5%m);
//        temp4=temp6-(temp6%m);

//        if(!(temp3== temp4))
//        {
//            if((temp3+m==temp6))
//            {}
//            else
//            {
//                h_csrRowPtrA[j]=p;
//                h_csrRowPtrB[j]=p;
//                j++;
//            }

//        }
//    }

//    // transfer data to device

//    cudaMalloc(&csrRowPtrA, (m+1)*sizeof(int));
//    cudaMalloc(&csrRowPtrB, (n+1)*sizeof(int));
//    cudaMalloc(&csrColIndA, density*sizeof(int));
//    cudaMalloc(&csrColIndB, density*sizeof(int));
//    cudaMalloc(&csrValA, density*sizeof(float));
//    cudaMalloc(&csrValB, density*sizeof(float));
//    cudaCheckErrors("cudaMalloc fail");
//    cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(csrRowPtrB, h_csrRowPtrB, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(csrColIndA, h_csrColIndA, density*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(csrColIndB, h_csrColIndB, density*sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(csrValA, h_csrValA, density*sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(csrValB, h_csrValB, density*sizeof(float), cudaMemcpyHostToDevice);
//    cudaCheckErrors("cudaMemcpy fail");

//    // set cusparse matrix types
//    CUSPARSE_CHECK(cusparseCreate(&hndl));
//    stat = cusparseCreateMatDescr(&descrA);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseCreateMatDescr(&descrB);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseCreateMatDescr(&descrC);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
//    CUSPARSE_CHECK(stat);
//    stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
//    CUSPARSE_CHECK(stat);
//    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

//    // figure out size of C
//    int baseC;
//    // nnzTotalDevHostPtr points to host memory
//    int *nnzTotalDevHostPtr = &nnzC;
//    stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
//    CUSPARSE_CHECK(stat);
//    cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
//    cudaCheckErrors("cudaMalloc fail");

//    s_time=timerval();

//    stat = cusparseXcsrgemmNnz(hndl, transA, transB, m, n, k,
//    descrA, nnzA, csrRowPtrA, csrColIndA,
//    descrB, nnzB, csrRowPtrB, csrColIndB,
//    descrC, csrRowPtrC, nnzTotalDevHostPtr );
//    CUSPARSE_CHECK(stat);
//    if (NULL != nnzTotalDevHostPtr){
//    nnzC = *nnzTotalDevHostPtr;}
//    else{
//    cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
//    cudaCheckErrors("cudaMemcpy fail");
//    nnzC -= baseC;}
//    cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
//    cudaMalloc((void**)&csrValC, sizeof(float)*nnzC);
//    cudaCheckErrors("cudaMalloc fail");
//    // perform multiplication C = A*B

//    for(loop=0;loop<1000;loop++)
//    {
//        stat = cusparseScsrgemm(hndl, transA, transB, m, n, k,
//        descrA, nnzA,
//        csrValA, csrRowPtrA, csrColIndA,
//        descrB, nnzB,
//        csrValB, csrRowPtrB, csrColIndB,
//        descrC,
//        csrValC, csrRowPtrC, csrColIndC);
//        CUSPARSE_CHECK(stat);
//    }

//    e_time=timerval();

//    avg_time=avg_time/1000;
//    // copy result (C) back to host
//    h_csrRowPtrC = (int *)malloc((m+1)*sizeof(int));
//    h_csrColIndC = (int *)malloc(nnzC *sizeof(int));
//    h_csrValC  = (float *)malloc(nnzC *sizeof(float));
//    if ((h_csrRowPtrC == NULL) || (h_csrColIndC == NULL) || (h_csrValC == NULL))
//    {printf("malloc fail\n"); return -1;}
//    cudaMemcpy(h_csrRowPtrC, csrRowPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_csrColIndC, csrColIndC,  nnzC*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_csrValC, csrValC, nnzC*sizeof(float), cudaMemcpyDeviceToHost);
//    cudaCheckErrors("cudaMemcpy fail");

//    printf ("\n Input size: %d x %d ,Time: %lf and density is %d \n", m,n, avg_time, density);

//    cudaFree(csrRowPtrC);
//    cudaFree(csrColIndC);
//    cudaFree(csrValC);

//    cudaFree(csrRowPtrA);
//    cudaFree(csrColIndA);
//    cudaFree(csrValA);

//    cudaFree(csrRowPtrB);
//    cudaFree(csrColIndB);
//    cudaFree(csrValB);

//    free(h_csrRowPtrC);
//    free(h_csrColIndC);
//    free(h_csrValC);

//    free(h_csrRowPtrA);
//    free(h_csrColIndA);
//    free(h_csrValA);

//    free(h_csrRowPtrB);
//    free(h_csrColIndB);
//    free(h_csrValB);
//}
//return 0;
//}








////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////








//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//#include <assert.h>

//#include "Utilities.cuh"

//#include <cuda_runtime.h>
//#include <cusparse_v2.h>

///********/
///* MAIN */
///********/
//int main()
//{
//    // --- Initialize cuSPARSE
//    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

//    /**************************/
//    /* SETTING UP THE PROBLEM */
//    /**************************/
//    const int N     = 4;                // --- Number of rows and columns

//    // --- Host side dense matrices
//    double *h_A_dense = (double*)malloc(N * N * sizeof(*h_A_dense));
//    double *h_B_dense = (double*)malloc(N * N * sizeof(*h_B_dense));
//    double *h_C_dense = (double*)malloc(N * N * sizeof(*h_C_dense));

//    // --- Column-major ordering
//    h_A_dense[0] = 0.4612;  h_A_dense[4] = -0.0006;     h_A_dense[8]  = 0.3566;     h_A_dense[12] = 0.0;
//    h_A_dense[1] = -0.0006; h_A_dense[5] = 0.4640;      h_A_dense[9]  = 0.0723;     h_A_dense[13] = 0.0;
//    h_A_dense[2] = 0.3566;  h_A_dense[6] = 0.0723;      h_A_dense[10] = 0.7543;     h_A_dense[14] = 0.0;
//    h_A_dense[3] = 0.;      h_A_dense[7] = 0.0;         h_A_dense[11] = 0.0;        h_A_dense[15] = 0.1;

//    // --- Column-major ordering
//    h_B_dense[0] = 0.;      h_B_dense[4] = 0.;          h_B_dense[8]  = 1.;         h_B_dense[12] = 0.;
//    h_B_dense[1] = 1.;      h_B_dense[5] = 0.;          h_B_dense[9]  = 0.;         h_B_dense[13] = 0.;
//    h_B_dense[2] = 0.;      h_B_dense[6] = 1.;          h_B_dense[10] = 0.;         h_B_dense[14] = 0.;
//    h_B_dense[3] = 0.;      h_B_dense[7] = 0.;          h_B_dense[11] = 0.;         h_B_dense[15] = 1.;

//    // --- Create device arrays and copy host arrays to them
//    double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, N * N * sizeof(*d_A_dense)));
//    double *d_B_dense;  gpuErrchk(cudaMalloc(&d_B_dense, N * N * sizeof(*d_B_dense)));
//    double *d_C_dense;  gpuErrchk(cudaMalloc(&d_C_dense, N * N * sizeof(*d_C_dense)));
//    gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, N * N * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense, N * N * sizeof(*d_B_dense), cudaMemcpyHostToDevice));

//    // --- Descriptor for sparse matrix A
//    cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
//    cusparseSafeCall(cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//    cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

//    // --- Descriptor for sparse matrix B
//    cusparseMatDescr_t descrB;      cusparseSafeCall(cusparseCreateMatDescr(&descrB));
//    cusparseSafeCall(cusparseSetMatType     (descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
//    cusparseSafeCall(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE));

//    // --- Descriptor for sparse matrix C
//    cusparseMatDescr_t descrC;      cusparseSafeCall(cusparseCreateMatDescr(&descrC));
//    cusparseSafeCall(cusparseSetMatType     (descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
//    cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

//    int nnzA = 0;                           // --- Number of nonzero elements in dense matrix A
//    int nnzB = 0;                           // --- Number of nonzero elements in dense matrix B

//    const int lda = N;                      // --- Leading dimension of dense matrix

//    // --- Device side number of nonzero elements per row of matrix A
//    int *d_nnzPerVectorA;   gpuErrchk(cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA)));
//    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

//    // --- Device side number of nonzero elements per row of matrix B
//    int *d_nnzPerVectorB;   gpuErrchk(cudaMalloc(&d_nnzPerVectorB, N * sizeof(*d_nnzPerVectorB)));
//    cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrB, d_B_dense, lda, d_nnzPerVectorB, &nnzB));

//    // --- Host side number of nonzero elements per row of matrix A
//    int *h_nnzPerVectorA = (int *)malloc(N * sizeof(*h_nnzPerVectorA));
//    gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));

//    // --- Host side number of nonzero elements per row of matrix B
//    int *h_nnzPerVectorB = (int *)malloc(N * sizeof(*h_nnzPerVectorB));
//    gpuErrchk(cudaMemcpy(h_nnzPerVectorB, d_nnzPerVectorB, N * sizeof(*h_nnzPerVectorB), cudaMemcpyDeviceToHost));

//    printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
//    for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
//    printf("\n");

//    printf("Number of nonzero elements in dense matrix B = %i\n\n", nnzB);
//    for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorB[i]);
//    printf("\n");

//    // --- Device side sparse matrix
//    double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
//    double *d_B;            gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));

//    int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices)));
//    int *d_B_RowIndices;    gpuErrchk(cudaMalloc(&d_B_RowIndices, (N + 1) * sizeof(*d_B_RowIndices)));
//    int *d_C_RowIndices;    gpuErrchk(cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices)));
//    int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
//    int *d_B_ColIndices;    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));

//    cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
//    cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrB, d_B_dense, lda, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

//    // --- Host side sparse matrices
//    double *h_A = (double *)malloc(nnzA * sizeof(*h_A));
//    double *h_B = (double *)malloc(nnzB * sizeof(*h_B));
//    int *h_A_RowIndices = (int *)malloc((N + 1) * sizeof(*h_A_RowIndices));
//    int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));
//    int *h_B_RowIndices = (int *)malloc((N + 1) * sizeof(*h_B_RowIndices));
//    int *h_B_ColIndices = (int *)malloc(nnzB * sizeof(*h_B_ColIndices));
//    int *h_C_RowIndices = (int *)malloc((N + 1) * sizeof(*h_C_RowIndices));
//    gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_B, d_B, nnzB * sizeof(*h_B), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_B_RowIndices, d_B_RowIndices, (N + 1) * sizeof(*h_B_RowIndices), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_B_ColIndices, d_B_ColIndices, nnzB * sizeof(*h_B_ColIndices), cudaMemcpyDeviceToHost));

//    printf("\nOriginal matrix A in CSR format\n\n");
//    for (int i = 0; i < nnzA; ++i) printf("A[%i] = %f ", i, h_A[i]); printf("\n");

//    printf("\nOriginal matrix B in CSR format\n\n");
//    for (int i = 0; i < nnzB; ++i) printf("B[%i] = %f ", i, h_B[i]); printf("\n");

//    printf("\n");
//    for (int i = 0; i < (N + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

//    printf("\n");
//    for (int i = 0; i < (N + 1); ++i) printf("h_B_RowIndices[%i] = %i \n", i, h_B_RowIndices[i]); printf("\n");

//    printf("\n");
//    for (int i = 0; i < nnzA; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

//    printf("\n");
//    for (int i = 0; i < nnzB; ++i) printf("h_B_ColIndices[%i] = %i \n", i, h_B_ColIndices[i]);

//    // --- Performing the matrix - matrix multiplication
//    int baseC, nnzC = 0;
//    // nnzTotalDevHostPtr points to host memory
//    int *nnzTotalDevHostPtr = &nnzC;

//    cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

//    cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, descrB, nnzB,
//                                         d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC, d_C_RowIndices,
//                                         nnzTotalDevHostPtr));
//    if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
//    else {
//        gpuErrchk(cudaMemcpy(&nnzC,  d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
//        gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices,     sizeof(int), cudaMemcpyDeviceToHost));
//        nnzC -= baseC;
//    }
//    int *d_C_ColIndices;    gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
//    double *d_C;            gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(double)));
//    double *h_C = (double *)malloc(nnzC * sizeof(*h_C));
//    int *h_C_ColIndices = (int *)malloc(nnzC * sizeof(*h_C_ColIndices));
//    cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, descrB, nnzB,
//                                      d_B, d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
//                                      d_C, d_C_RowIndices, d_C_ColIndices));

//    cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, N));

//    gpuErrchk(cudaMemcpy(h_C ,           d_C,            nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));

//    printf("\nResult matrix C in CSR format\n\n");
//    for (int i = 0; i < nnzC; ++i) printf("C[%i] = %f ", i, h_C[i]); printf("\n");

//    printf("\n");
//    for (int i = 0; i < (N + 1); ++i) printf("h_C_RowIndices[%i] = %i \n", i, h_C_RowIndices[i]); printf("\n");

//    printf("\n");
//    for (int i = 0; i < nnzC; ++i) printf("h_C_ColIndices[%i] = %i \n", i, h_C_ColIndices[i]);

//    gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));

//    for (int j = 0; j < N; j++) {
//        for (int i = 0; i < N; i++)
//            printf("%f \t", h_C_dense[i * N + j]);
//        printf("\n");
//        }
//}








////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cusparse_v2.h>
#include <stdio.h>

#include <ctime>


//erfredrg

#define N 50000

// matrix generation and validation depends on these relationships:
#define SCL 2
#define K N
#define M (SCL*N)
// A: MxK  B: KxN  C: MxN

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

// perform sparse-matrix multiplication C=AxB
int main(){

    std::clock_t start;
    double duration;

    cusparseStatus_t stat;
    cusparseHandle_t hndl;
    cusparseMatDescr_t descrA, descrB, descrC;
    int *csrRowPtrA, *csrRowPtrB, *csrRowPtrC, *csrColIndA, *csrColIndB, *csrColIndC;
    int *h_csrRowPtrA, *h_csrRowPtrB, *h_csrRowPtrC, *h_csrColIndA, *h_csrColIndB, *h_csrColIndC;
    float *csrValA, *csrValB, *csrValC, *h_csrValA, *h_csrValB, *h_csrValC;
    int nnzA, nnzB, nnzC;
    int m,n,k;
    m = M;
    n = N;
    k = K;


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
        h_csrRowPtrA = (int *)malloc((m+1)*sizeof(int));
        h_csrRowPtrB = (int *)malloc((n+1)*sizeof(int));
        h_csrColIndA = (int *)malloc(m*sizeof(int));
        h_csrColIndB = (int *)malloc(n*sizeof(int));
        h_csrValA  = (float *)malloc(m*sizeof(float));
        h_csrValB  = (float *)malloc(n*sizeof(float));

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("Host Malloc :                                   %f ms\n", duration);
    //------------------------------------------------------------------------------
    start = std::clock();

        if ((h_csrRowPtrA == NULL) || (h_csrRowPtrB == NULL) || (h_csrColIndA == NULL) || (h_csrColIndB == NULL) || (h_csrValA == NULL) || (h_csrValB == NULL))
        {printf("malloc fail\n"); return -1;}
        for (int i = 0; i < m; i++){
        h_csrValA[i] = 1.0f;
        h_csrRowPtrA[i] = i;
        h_csrColIndA[i] = i/SCL;
        if (i < n){
          h_csrValB[i] = 2.0f;
          h_csrRowPtrB[i] = i;
          h_csrColIndB[i] = i;}
        }
        h_csrRowPtrA[m] = m;
        h_csrRowPtrB[n] = n;


    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("CSR Matrix Generation :                         %f ms\n", duration);
    ///////////////////////////     STEP 2      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                      Allocate memory on the device                         */
    /*              and return a ptr of its memory emplacement                    */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

        cudaMalloc(&csrRowPtrA, (m+1)*sizeof(int));

    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("cudaMalloc csrRowPtrA :                         %f ms\n", duration);
    //------------------------------------------------------------------------------
    start = std::clock();

        cudaMalloc(&csrRowPtrB, (n+1)*sizeof(int));
        cudaMalloc(&csrColIndA, m*sizeof(int));
        cudaMalloc(&csrColIndB, n*sizeof(int));
        cudaMalloc(&csrValA, m*sizeof(float));
        cudaMalloc(&csrValB, n*sizeof(float));


    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("cudaMalloc csrRowPtrB|csrColIndA/B|csrValA/B :  %f ms\n", duration);
    ///////////////////////////     STEP 3      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Copy the data from the Host (CPU)                         */
    /*                      to the device (GPU)                                   */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

        cudaCheckErrors("cudaMalloc fail");
        cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrRowPtrB, h_csrRowPtrB, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndA, h_csrColIndA, m*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndB, h_csrColIndB, n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValA, h_csrValA, m*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValB, h_csrValB, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy fail");


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
    /*                             ??????                                         */
    /*                                                                            */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

        // figure out size of C
        int baseC;

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


    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("nnzTotalDevHostPtr points to host memory :      %f ms\n", duration);
    ///////////////////////////     STEP 6      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                Perform multiplication C = A*B                              */
    /*                                                                            */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

        stat = cusparseScsrgemm(hndl, transA, transB, m, n, k,
            descrA, nnzA,
            csrValA, csrRowPtrA, csrColIndA,
            descrB, nnzB,
            csrValB, csrRowPtrB, csrColIndB,
            descrC,
            csrValC, csrRowPtrC, csrColIndC);
        CUSPARSE_CHECK(stat);


    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC) * 1000;
    printf("GPU calculation time :                          %f ms\n", duration);
    ///////////////////////////     STEP 7      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  Copy result (C) back to host                              */
    /*                       & test & validate it                                 */
    /* -------------------------------------------------------------------------- */
    start = std::clock();

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
        printf("Success!\n");

    return 0;
}
