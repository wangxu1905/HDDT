# which conda Env
set(CONDA_ENV "py310")
set(PYTHON_ENV "python3.10")
# which Conda
execute_process(
    COMMAND conda run -n ${CONDA_ENV} python -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE CONDA_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# PyTorch headers and libraries
set(TORCH_CMAKE_PATH "${CONDA_PREFIX}/lib/${PYTHON_ENV}/site-packages/torch/share/cmake/Torch" CACHE PATH "Path to PyTorch cmake directory")
message(WARN "-- TORCH_CMAKE_PATH: ${TORCH_CMAKE_PATH}")
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH} ${CMAKE_PREFIX_PATH})
# glog path
set(TORCH_DIR "${CONDA_PREFIX}/lib/${PYTHON_ENV}/site-packages/torch/lib")
set(CMAKE_INSTALL_RPATH "${TORCH_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# cudnn path
if(CUDA_FOUND)
set(CUDNN_LIBRARY_PATH "${CONDA_PREFIX}/lib/${PYTHON_ENV}/site-packages/nvidia/cudnn/lib" CACHE PATH "Path to cudnn library") # ./site-packages/nvidia/cudnn/lib/libcudnn.so.8
set(CUDNN_INCLUDE_PATH "${CONDA_PREFIX}/lib/${PYTHON_ENV}/site-packages/nvidia/cudnn/include" CACHE PATH "Path to cudnn include") 
endif()
if(ROCM_FOUND)
set(GLOG_LIBRARY "${CONDA_PREFIX}/lib/${PYTHON_ENV}/site-packages/torch/lib/libglog.so")
endif()
find_package(Torch REQUIRED)
# "${TORCH_LIBRARIES}"
