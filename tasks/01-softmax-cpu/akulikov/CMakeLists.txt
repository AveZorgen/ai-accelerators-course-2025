set(target_suffix akulikov)
set(target_name "softmax_cpu_${target_suffix}")

add_executable(${target_name} main.cpp)

target_compile_features(${target_name} PRIVATE cxx_std_17)

if (MSVC)
  target_compile_options(${target_name} PRIVATE /arch:AVX2)
else ()
  target_compile_options(${target_name} PRIVATE -mavx2)
endif ()

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
