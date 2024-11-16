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
target_link_libraries(coll_app PRIVATE ${MPI_LIBRARY})

# message(FATAL_ERROR ${MPI_INCLUDE_DIR})
# message(FATAL_ERROR ${MPI_LIBRARY})
# 当前项目的rocm相关代码需要使用hipcc编译器warpper，和mpicc使用冲突
# mpicc无法识别hipcc中的一些编译选项：解决方案是编译器使用hipcc，手动加入mpicc的包装参数(mpicc -showme)
# -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
# 故在编译rocm时，不使用mpicc
if(ROCM_FOUND)
else()
    find_program(MPICC mpicxx REQUIRED)
    set(CMAKE_C_COMPILER ${MPICC})
    set(CMAKE_CXX_COMPILER ${MPICC})
endif()
# install(TARGETS rdma_app DESTINATION bin)