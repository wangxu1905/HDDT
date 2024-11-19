#include <mem.h>

namespace hddt {

#ifdef ENABLE_NEUWARE
/*
 * nvidia gpu memory
 */
status_t NeuwareMemory::init() { return init_gpu_driver(this->device_id); }

status_t NeuwareMemory::free() { return free_gpu_driver(); }

status_t NeuwareMemory::allocate_buffer(void **addr, size_t size) {
  CNresult ret;
  cn_uint64_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);

  if (this->mem_type != memory_type_t::CAMBRICON_MLU) {
    return status_t::UNSUPPORT;
  }
  logInfo("Allocate memory using cnMalloc.");
  ret = cnMalloc(&this->mlu_addr, buf_size);
  if (ret != CN_SUCCESS) {
    logError("failed to allocate memory.");
    return status_t::ERROR;
  }
  *addr = (void *)this->mlu_addr;
  // todo : dmabuf support : cuMemGetHandleForAddressRange()
  return status_t::SUCCESS;
}

status_t NeuwareMemory::free_buffer(void *addr) {
  CNresult ret;
  CNaddr addr_mlu = (CNaddr)addr;
  ret = cnFree(addr_mlu);
  if (ret != CN_SUCCESS) {
    logError("failed to free memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t NeuwareMemory::copy_host_to_device(void *dest, const void *src,
                                            size_t size) {
  CNresult ret;

  if (dest == nullptr || src == nullptr) {
    logError("NeuwareMemory::copy_host_to_device Error.");
    return status_t::ERROR;
  }

  CNaddr dest_mlu = (CNaddr)dest;
  CNaddr src_mlu = (CNaddr)src;
  ret = cnMemcpy(dest_mlu, src_mlu, size);
  if (ret != CN_SUCCESS) {
    logError("failed to copy memory from host to memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t NeuwareMemory::copy_device_to_host(void *dest, const void *src,
                                            size_t size) {
  CNresult ret;

  if (dest == nullptr || src == nullptr) {
    logError("NeuwareMemory::copy_device_to_host Error.");
    return status_t::ERROR;
  }

  CNaddr dest_mlu = (CNaddr)dest;
  CNaddr src_mlu = (CNaddr)src;
  ret = cnMemcpy(dest_mlu, src_mlu, size);
  if (ret != CN_SUCCESS) {
    logError("failed to copy memory from device to host");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t NeuwareMemory::copy_device_to_device(void *dest, const void *src,
                                              size_t size) {
  CNresult ret;

  if (dest == nullptr || src == nullptr) {
    logError("NeuwareMemory::copy_device_to_device Error.");
    return status_t::ERROR;
  }

  CNaddr dest_mlu = (CNaddr)dest;
  CNaddr src_mlu = (CNaddr)src;
  ret = cnMemcpy(dest_mlu, src_mlu, size);
  if (ret != CN_SUCCESS) {
    logError("failed to copy memory from device to device");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

#else
status_t NeuwareMemory::init() { return status_t::UNSUPPORT; }
status_t NeuwareMemory::free() { return status_t::UNSUPPORT; }
status_t NeuwareMemory::allocate_buffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::free_buffer(void *addr) { return status_t::UNSUPPORT; }

status_t NeuwareMemory::copy_host_to_device(void *dest, const void *src,
                                            size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::copy_device_to_host(void *dest, const void *src,
                                            size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::copy_device_to_device(void *dest, const void *src,
                                              size_t size) {
  return status_t::UNSUPPORT;
}

#endif
} // namespace hddt
