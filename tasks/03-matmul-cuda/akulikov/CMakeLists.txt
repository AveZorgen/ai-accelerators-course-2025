enable_language(CUDA)

set(target_suffix akulikov)
set(target_name "matmul_cuda_${target_suffix}")

add_executable(${target_name} main.cu)

target_compile_features(${target_name} PRIVATE cxx_std_17)

set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

set(CUTLASS_ROOT "" CACHE PATH "Path to the CUTLASS installation (root directory)")
if(CUTLASS_ROOT)
    message(STATUS "Using CUTLASS from ${CUTLASS_ROOT}")
    target_include_directories(${target_name} PRIVATE "${CUTLASS_ROOT}/include")
endif()
