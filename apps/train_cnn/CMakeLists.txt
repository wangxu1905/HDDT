# apps/train_cnn/CMakeLists.txt

project(train_cnn)

set(TRAIN_CNN_SOURCES
    main.cpp
)

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