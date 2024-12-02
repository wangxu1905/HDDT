# apps/executor_app/CMakeLists.txt
project(executor_app)
set(EXECUTOR_APP_SOURCES
    main.cpp
)
add_executable(executor_app ${EXECUTOR_APP_SOURCES})

target_include_directories(executor_app PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

if(CUDA_FOUND)
  target_link_libraries(executor_app PUBLIC hddt_cuda_shared)
  target_link_libraries(executor_app PUBLIC hddt_shared_lib ${CUDA_LIBRARIES})
else()
  message(FATAL_ERROR "CUDA is required but was not found.")
endif()

target_link_libraries(executor_app PUBLIC glog)
target_link_libraries(executor_app PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})

# install(TARGETS executor_app DESTINATION bin)


# install(TARGETS simple_inference DESTINATION bin)

