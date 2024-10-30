# apps/train_cnn/CMakeLists.txt

project(train_cnn)

set(TRAIN_CNN_SOURCES
    main.cpp
)

set(CMAKE_CXX_STANDARD 17)  # 设置 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(train_cnn ${TRAIN_CNN_SOURCES})

if(ROCM_FOUND)
    target_link_libraries(train_cnn PRIVATE hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(train_cnn PRIVATE hddt_cuda_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()

target_include_directories(train_cnn PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# install(TARGETS train_cnn DESTINATION bin)