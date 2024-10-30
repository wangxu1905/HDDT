# apps/rdma_app/CMakeLists.txt
project(rdma_app)
set(RDMA_APP_SOURCES
    main.cpp
)
add_executable(rdma_app ${RDMA_APP_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(rdma_app PRIVATE hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(rdma_app PRIVATE hddt_cuda_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()
target_include_directories(rdma_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(rdma_app PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})

# install(TARGETS rdma_app DESTINATION bin)