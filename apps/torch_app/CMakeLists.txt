# apps/torch_app/CMakeLists.txt
project(torch_app)
set(TORCH_APP_SOURCES
    main.cpp
)
add_executable(torch_app ${TORCH_APP_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(torch_app PRIVATE hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(torch_app PRIVATE hddt_cuda_shared)
elseif(NEUWARE_FOUND)
    target_link_libraries(torch_app PRIVATE hddt_neuware_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
endif()
target_include_directories(torch_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

include(FindPyTorch)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(ROCM_FOUND)
target_link_libraries(torch_app PRIVATE
    ${TORCH_LIBRARIES}
    ${GLOG_LIBRARY}  # 使用glog的库变量
)
elseif(CUDA_FOUND)
target_link_libraries(torch_app PRIVATE
    ${TORCH_LIBRARIES}
)
endif()

include(FindMPI)
include_directories(${MPI_INCLUDE_DIR})
target_link_libraries(torch_app ${MPI_LIBRARY})

if(ROCM_FOUND)
else()
    find_program(MPICC mpicxx REQUIRED)
    set(CMAKE_C_COMPILER ${MPICC})
    set(CMAKE_CXX_COMPILER ${MPICC})
endif()

# install(TARGETS torch_app DESTINATION bin)
