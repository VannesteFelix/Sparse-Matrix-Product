cmake_minimum_required(VERSION 2.8)
project(libraries_test)

set(CMAKE_BUILD_TYPE RELEASE)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstack-protector --param=ssp-buffer-size=4 -D_FORTIFY_SOURCE=2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -W -O3 -std=c++11")

#   Compiler option viennacl (change them on cmake-gui)
option(ENABLE_OPENCL "Use the OpenCL backend")
option(ENABLE_CUDA "Use the CUDA backend")
option(ENABLE_OPENMP "Use the OPENMP backend")

#   Add .exe openCL
set(ENABLE_OPENCL ON)
#   Add .exe CUDA
set(ENABLE_CUDA ON)

#   Info compilation flags
#message("${CMAKE_BUILD_TYPE}")
#message("${CMAKE_CXX_FLAGS_RELEASE}")
#message("${CMAKE_CXX_FLAGS}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(HEADER_FILES

    component/benchmark.hpp
    component/benchmark.inl
    #component/libraries/cuSparse.hpp

        )
set(CUDA_HEADER_FILES

    component/libraries/mmio.h

        )
set(SOURCE_FILES

    main.cpp
    component/benchmark.cpp
    #component/libraries/cuSparse.cpp

        )
set(CUDA_SOURCE_FILES

    component/libraries/mmio.c
    main.cu
        )

####     Subdirectories
add_subdirectory(examples)

####     EIGEN
set(EIGEN_LIBRARY_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/eigen3")

#####    VIENNACL
set(VIENNACL_LIBRARY_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/ViennaCL-1.7.1/")
set(VIENNACL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/ViennaCL-1.7.1/build/libviennacl/libviennacl.so")

####     OPENCL
set(OPENCL_LIBRARY_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/ViennaCL-1.7.1/CL")
set(OPENCL_LIBRARY "/usr/local/cuda-9.0/lib64/libOpenCL.so")

###########     COMPILATION FLAGS        ############################################

# Targets using OpenCL
if (ENABLE_OPENCL)
    add_executable(benchmark ${HEADER_FILES} ${SOURCE_FILES})
    set_target_properties(benchmark PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")

    #####    LIBRARY LINKS               ############################################
    #####    VIENNACL
    target_include_directories(benchmark PUBLIC ${VIENNACL_LIBRARY_DIRECTORY})
    target_link_libraries(benchmark ${VIENNACL_LIBRARY})
    ####     OPENCL
    target_include_directories(benchmark PUBLIC ${OPENCL_LIBRARY_DIRECTORY})
    target_link_libraries(benchmark ${OPENCL_LIBRARY})
    ####     EIGEN
    target_include_directories(benchmark PUBLIC ${EIGEN_LIBRARY_DIRECTORY})

endif(ENABLE_OPENCL)

# Targets using CUDA
if (ENABLE_CUDA)
    find_package(CUDA QUIET REQUIRED)
    #set_source_files_properties(component/libraries/mmio.c PROPERTIES LANGUAGE CUDA)
    cuda_add_executable(benchmark_cu ${CUDA_SOURCE_FILES} ${CUDA_HEADER_FILES})

    #####    LIBRARY LINKS               ############################################
    ####     CUDA
    include_directories(benchmark_cu PUBLIC ${CUDA_INCLUDE_DIRS})                   #  target_include_directories does not work with CUDA
    target_link_libraries(benchmark_cu ${CUDA_LIBRARY} ${CUDA_cusparse_LIBRARY})    #  so we include for all the project
    #####    VIENNACL
    #include_directories(benchmark_cu PUBLIC ${VIENNACL_LIBRARY_DIRECTORY})
    #target_link_libraries(benchmark_cu ${VIENNACL_LIBRARY})


endif(ENABLE_CUDA)

# Targets using OpenMP
if (ENABLE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set_target_properties(benchmark PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENMP")
    endif(OPENMP_FOUND)
endif(ENABLE_OPENMP)
