// #include<net.h>

// using namespace hddt;

// status_t dummy_mem_alloc(void **addr, size_t length, memory_type_t mem_type){
//     #ifdef ENABLE_CUDA
//     cudaError_t ret;

//     if ((mem_type != memory_type_t::NVIDIA_GPU_MANAGED) && (mem_type !=
//     memory_type_t::NVIDIA_GPU)) {
//         return status_t::UNSUPPORT;
//     }

//     ret = cudaMalloc(addr, length);
//     if (ret != cudaSuccess) {
//         logError("failed to allocate memory %d.", ret);
//         return status_t::ERROR;
//     }

//     return status_t::SUCCESS;
//     #endif

//     return status_t::ERROR;
// }

// status_t dummy_mem_free(void **address_p)
// {
//     #ifdef ENABLE_CUDA
//     cudaError_t ret;
//     ret = cudaFree(address_p);
//     if (ret != cudaSuccess) {
//         log_error("failed to free memory");
//         return status_t::ERROR;
//     }
//     return status_t::SUCCESS;
//     #endif

//     return status_t::ERROR;
// }