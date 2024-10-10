cmake_minimum_required(VERSION 3.21)
project(HDDT)
enable_language(CXX)
enable_language(CUDA)


# # cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


# # option
option(BUILD_STATIC_LIB "Build static library" OFF)


# # build gpu-related module
include(FindRocm)
include(FindCuda)

if(ROCM_FOUND)
    # set(ROCM_SOURCES src/*/rocm/*.cu)
    file(GLOB_RECURSE ROCM_SOURCES "src/*/rocm/*.cu")

    # static
    if(BUILD_STATIC_LIB)
        add_library(hddt_rocm_static STATIC ${ROCM_SOURCES})
        target_include_directories(hddt_rocm_static PUBLIC ${HIP_INCLUDE_DIRS})
        target_link_libraries(hddt_rocm_static PRIVATE rocblas rocrand)
        set_target_properties(hddt_rocm_static PROPERTIES
            CXX_COMPILER "${HIP_HCC}"
        )
        install(TARGETS my_rocm_static DESTINATION lib)
    endif()

    # dynamic
    add_library(hddt_rocm_shared SHARED ${ROCM_SOURCES})
    target_include_directories(hddt_rocm_shared PUBLIC ${HIP_INCLUDE_DIRS})
    target_link_libraries(hddt_rocm_shared PRIVATE rocblas rocrand)
    set_target_properties(hddt_rocm_shared PROPERTIES
        CXX_COMPILER "${HIP_HCC}"
    )
    install(TARGETS my_rocm_shared DESTINATION lib)

elseif(CUDA_FOUND)
    # set(CUDA_SOURCES src/*/cuda/*.cu)
    file(GLOB_RECURSE CUDA_SOURCES "src/*/cuda/*.cu")

    # static
    if(BUILD_STATIC_LIB)
        add_library(hddt_cuda_static STATIC ${CUDA_SOURCES})
        target_include_directories(hddt_cuda_static PUBLIC ${CUDA_INCLUDE_DIRS})
        target_link_libraries(hddt_cuda_static PRIVATE ${CUDA_LIBRARIES})
        install(TARGETS hddt_cuda_static DESTINATION lib)
    endif()

    # dynamic
    add_library(hddt_cuda_shared SHARED ${CUDA_SOURCES})
    target_include_directories(hddt_cuda_shared PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(hddt_cuda_shared PRIVATE ${CUDA_LIBRARIES})
    install(TARGETS hddt_cuda_shared DESTINATION lib)
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()


# # build src
include_directories(include)
# using clang
set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)

file(GLOB_RECURSE SRC_RECURSE "src/*.cpp")
if(NOT SRC_RECURSE)
    message(FATAL_ERROR "No sources files found in the 'src' directory.")
endif()

# static
if(BUILD_STATIC_LIB)
    add_library(hddt_static_lib STATIC ${SRC_RECURSE})
    target_include_directories(hddt_static_lib PUBLIC include)
    if(ROCM_FOUND)
        target_link_libraries(hddt_static_lib PRIVATE hddt_rocm_static)
    elseif(CUDA_FOUND)
        target_link_libraries(hddt_static_lib PRIVATE hddt_cuda_static)
    else()
        message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
    endif()
    install(TARGETS hddt_static_lib DESTINATION lib)
endif()

# dynamic
add_library(hddt_shared_lib SHARED ${SRC_RECURSE})
target_include_directories(hddt_shared_lib PUBLIC include)
if(ROCM_FOUND)
    target_link_libraries(hddt_shared_lib PRIVATE hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(hddt_shared_lib PRIVATE hddt_cuda_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA is required but was not found.")
endif()
install(TARGETS hddt_shared_lib DESTINATION lib)


# # install header
install(DIRECTORY include/ DESTINATION include)


# # build app train_cnn
add_subdirectory(apps/train_cnn)