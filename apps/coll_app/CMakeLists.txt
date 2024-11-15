# apps/coll_app/CMakeLists.txt
project(coll_app)
set(COLL_SOURCES
    main.cpp
)

add_executable(coll_app ${COLL_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(coll_app PRIVATE hddt_rocm_shared)
    target_link_libraries(coll_app PRIVATE hddt_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(coll_app PRIVATE hddt_cuda_shared)
    target_link_libraries(coll_app PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(coll_app PRIVATE hddt_neuware_shared)
    target_link_libraries(coll_app PRIVATE hddt_shared_lib ${NEUWARE_LIBRARIES})
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
endif()
target_include_directories(coll_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(FindMPI)
include_directories(${MPI_INCLUDE_DIR})
target_link_libraries(coll_app PRIVATE ${MPI_LIBRARIES})

# find mpixx
find_program(MPICC mpicxx REQUIRED)
set(CMAKE_C_COMPILER ${MPICC})
set(CMAKE_CXX_COMPILER ${MPICC})

# install(TARGETS rdma_app DESTINATION bin)