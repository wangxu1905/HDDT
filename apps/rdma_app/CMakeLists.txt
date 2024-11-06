# apps/rdma_app/CMakeLists.txt
project(rdma_app)
set(RDMA_SERVER_SOURCES
    server.cpp
)
set(RDMA_CLIENT_SOURCES
    client.cpp
)

set(RDMA_MEM_SOURCES
    mem.cpp
)

add_executable(rdma_server ${RDMA_SERVER_SOURCES})
add_executable(rdma_client ${RDMA_CLIENT_SOURCES})
add_executable(mem ${RDMA_MEM_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(rdma_server PRIVATE hddt_rocm_shared)
    target_link_libraries(rdma_client PRIVATE hddt_rocm_shared)
    target_link_libraries(rdma_server PRIVATE hddt_shared_lib ${HIPLIBRARIES})
    target_link_libraries(rdma_client PRIVATE hddt_shared_lib ${HIPLIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(rdma_server PRIVATE hddt_cuda_shared)
    target_link_libraries(rdma_client PRIVATE hddt_cuda_shared)
    target_link_libraries(mem PRIVATE hddt_cuda_shared)
    target_link_libraries(rdma_server PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(rdma_client PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(mem PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()
target_include_directories(rdma_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(rdma_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(mem PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# install(TARGETS rdma_app DESTINATION bin)