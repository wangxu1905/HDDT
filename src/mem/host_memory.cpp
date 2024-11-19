#include <mem.h>

namespace hddt {
memory_type_t memory_supported() {
#ifdef ENABLE_CUDA
  return memory_type_t::NVIDIA_GPU;
#endif

#ifdef ENABLE_ROCM
  return memory_type_t::AMD_GPU;
#endif

  return memory_type_t::CPU;
}

bool memory_dmabuf_supported() {
#ifdef ENABLE_CUDA_DMABUF
  return true;
#else
  return false;
#endif
}

/*
 * host memory
 */
status_t HostMemory::init() { return status_t::SUCCESS; }

status_t HostMemory::free() { return status_t::SUCCESS; }

status_t HostMemory::allocate_buffer(void **addr, size_t size) {
  logInfo("Allocate memory using new malloc.");

  void *buffer = new (std::nothrow) char[size];
  if (buffer == nullptr) {
    logError("HostMemory::allocate_buffer Error.");
    return status_t::ERROR;
  } else {
    *addr = buffer;
    return status_t::SUCCESS;
  }
}

status_t HostMemory::free_buffer(void *addr) {
  if (addr == nullptr) {
    return status_t::ERROR;
  }

  delete[] static_cast<char *>(addr);

  addr = nullptr;

  return status_t::SUCCESS;
}

status_t HostMemory::copy_host_to_device(void *dest, const void *src,
                                         size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copy_host_to_device Error.");
    return status_t::ERROR;
  }

  try {
    memcpy(dest, src, size);
  } catch (const std::exception &e) {
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t HostMemory::copy_device_to_host(void *dest, const void *src,
                                         size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copy_device_to_host Error.");
    return status_t::ERROR;
  }

  try {
    memcpy(dest, src, size);
  } catch (const std::exception &e) {
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t HostMemory::copy_device_to_device(void *dest, const void *src,
                                           size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copy_device_to_device Error.");
    return status_t::ERROR;
  }

  try {
    memcpy(dest, src, size);
  } catch (const std::exception &e) {
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}
} // namespace hddt
