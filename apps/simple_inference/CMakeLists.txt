# apps/simple_inference/CMakeLists.txt

project(simple_inference)

set(SIMPLE_INFERENCE_SOURCES
    main.cpp
)

add_executable(simple_inference ${SIMPLE_INFERENCE_SOURCES})

if(ROCM_FOUND)
    target_link_libraries(simple_inference PRIVATE hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(simple_inference PRIVATE hddt_cuda_shared)
elseif(NEUWARE_FOUND)
    target_link_libraries(simple_inference PRIVATE hddt_neuware_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
endif()

target_include_directories(simple_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(FindMPI)
include_directories(${MPI_INCLUDE_DIR})
target_link_libraries(simple_inference PRIVATE ${MPI_LIBRARY})

# for ROCm, we don't use mpicc.
# mpicc -showme
# -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
# has been added by FindMPI
if(ROCM_FOUND)
    # skip using mpicc while using hipcc as compiler
else()
    find_program(MPICC mpicxx REQUIRED)
    set(CMAKE_C_COMPILER ${MPICC})
    set(CMAKE_CXX_COMPILER ${MPICC})
endif()

# install(TARGETS simple_inference DESTINATION bin)