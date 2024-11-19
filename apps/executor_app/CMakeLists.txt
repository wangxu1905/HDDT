# apps/executor_app/CMakeLists.txt
project(executor_app)
set(EXECUTOR_APP_SOURCES
    main.cpp
)
add_executable(executor_app ${EXECUTOR_APP_SOURCES})

# if(CUDA_FOUND)
#     target_link_libraries(executor_app PRIVATE hddt_cuda_shared)
#     target_link_libraries(executor_app PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
# else()
#     message(FATAL_ERROR "CUDA is required but was not found.")
# endif()

# target_include_directories(executor_app PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
# )

# target_link_libraries(executor_app PUBLIC glog)
# target_link_libraries(executor_app PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})

# install(TARGETS executor_app DESTINATION bin)


# install(TARGETS simple_inference DESTINATION bin)

