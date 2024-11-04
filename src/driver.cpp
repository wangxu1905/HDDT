
#include <hddt.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
// cuda device init
hddt_status_t cuda_init(void) {
  static pthread_mutex_t cuda_init_mutex = PTHREAD_MUTEX_INITIALIZER;
  static volatile int cuda_initialized = 0;
  hddt_status_t status = hddt_status_t::SUCCESS;
  int count;

  cudaGetDeviceCount(&count);

  if (count == 0) {
    logError("There is no device.");
    status = hddt_status_t::ERROR;
    goto end;
  }

  int i;

  for (i = 0; i < count; i++) {
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if (prop.major >= 1) {
        break;
      }
    }
  }

  if (i == count) {
    logError("There is no device supporting CUDA 1.x.");
    status = hddt_status_t::ERROR;
    goto end;
  }

  cudaSetDevice(i);

  cuda_initialized = 1;

end:
  pthread_mutex_unlock(&cuda_init_mutex);
  return status;
}

#endif

#ifdef ENABLE_ROCM
// rocm driver init
#include <hip/hip_runtime.h>
#include <hsa.h>
#include <hsa_ext_amd.h>

hddt_status_t rocm_init(void) {
  static pthread_mutex_t rocm_init_mutex = PTHREAD_MUTEX_INITIALIZER;
  static volatile int rocm_initialized = 0;
  hsa_status_t hsa_status;
  hddt_status_t status = hddt_status_t::SUCCESS;

  if (pthread_mutex_lock(&rocm_init_mutex) == 0) {
    if (rocm_initialized) {
      goto end;
    }
  } else {
    logError("Could not take mutex.");
    status = hddt_status_t::ERROR;
    return status;
  }

  memset(&rocm_agents, 0, sizeof(rocm_agents));

  hsa_status = hsa_init();
  if (hsa_status != HSA_STATUS_SUCCESS) {
    status = hddt_status_t::ERROR;
    logDebug("Failure to open HSA connection: 0x%x.", status);
    goto end;
  }

  hsa_status = hsa_iterate_agents(rocm_hsa_agent_callback, NULL);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    status = hddt_status_t::ERROR;
    logDebug("Failure to iterate HSA agents: 0x%x.", status);
    goto end;
  }

#if ROCM_DMABUF_SUPPERTED
  status = rocmLibraryInit();
  if (status != STATUS_SUCCESS) {
    logDebug("Failure to initialize ROCm library: 0x%x.", status);
    goto end;
  }
#endif

  rocm_initialized = 1;

end:
  pthread_mutex_unlock(&rocm_init_mutex);
  return status;
}

#endif

namespace hddt {
hddt_status_t init_gpu_driver() {
  hddt_status_t ret = hddt_status_t::SUCCESS;
#ifdef ENABLE_CUDA
  ret = cuda_init();
#endif
#ifdef ENABLE_ROCM
  ret = rocm_init();
#endif
  return ret;
}
} // namespace hddt
