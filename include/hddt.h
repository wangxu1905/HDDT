#ifndef HDDT_H
#define HDDT_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
// #include <cuda.h>
#endif
#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <hsa.h>
#include <hsa_ext_amd.h>
#endif

#include <iostream>
#include <pthread.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

namespace hddt {
/* status and log */
enum class status_t { SUCCESS, ERROR, UNSUPPORT };
void logError(const char *format, ...);
void logDebug(const char *format, ...);
void logInfo(const char *format, ...);

/*
gpu driver init
*/
status_t init_gpu_driver(int device_id);
status_t free_gpu_driver();
} // namespace hddt

#endif