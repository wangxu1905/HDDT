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
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()
target_include_directories(torch_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
include(FindPyTorch)
include(fetch_mnist)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(torch_app PRIVATE
    ${TORCH_LIBRARIES}
)
# install(TARGETS torch_app DESTINATION bin)
