cmake_minimum_required(VERSION 3.24)

project(zcutlass LANGUAGES CXX CUDA)

option(ZCUTLASS_BUILD_TESTS "Build zcutlass tests" ON)
option(ZCUTLASS_BUILD_BENCHMARKS "Build zcutlass benchmarks" ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(zcutlass STATIC
  src/gemm.cu
)

target_include_directories(zcutlass
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(zcutlass
  PUBLIC
    CUDA::cudart
)

target_compile_options(zcutlass
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(zcutlass PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

if(ZCUTLASS_BUILD_TESTS)
  enable_testing()
  add_executable(zcutlass_tests
    tests/test_gemm.cu
  )
  target_link_libraries(zcutlass_tests
    PRIVATE
      zcutlass
      CUDA::cudart
      CUDA::cublas
  )
  target_compile_options(zcutlass_tests
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  add_test(NAME zcutlass_tests COMMAND zcutlass_tests)
endif()

if(ZCUTLASS_BUILD_BENCHMARKS)
  add_executable(zcutlass_bench
    benchmarks/bench_gemm.cu
  )
  target_link_libraries(zcutlass_bench
    PRIVATE
      zcutlass
      CUDA::cudart
      CUDA::cublas
  )
  target_compile_options(zcutlass_bench
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()
