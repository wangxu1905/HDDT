#ifndef MEMORY_H
#define MEMORY_H

#include <cstring>
#include <hddt.h>

namespace hddt {

#define ACCEL_PAGE_SIZE (64 * 1024)

enum class memory_type_t {
  CPU,
  NVIDIA_GPU,
  AMD_GPU
}; // todo: NVIDIA_GPU_MANAGED, AMD_GPU_MANAGED

memory_type_t memory_supported();
bool memory_dmabuf_supported();

class Memory {
protected:
  memory_type_t mem_type;
  int device_id;

public:
  Memory(int device_id, memory_type_t mem_type)
      : device_id(device_id), mem_type(mem_type){};
  virtual ~Memory(){};

  virtual status_t init() = 0;
  virtual status_t free() = 0;
  virtual status_t allocate_buffer(void **addr, size_t size) = 0;
  virtual status_t free_buffer(void *addr) = 0;

  virtual status_t copy_host_to_buffer(void *dest, const void *src,
                                       size_t size) = 0;
  virtual status_t copy_buffer_to_host(void *dest, const void *src,
                                       size_t size) = 0;
  virtual status_t copy_buffer_to_buffer(void *dest, const void *src,
                                         size_t size) = 0;
};

class HostMemory : public Memory {
public:
  HostMemory(int device_id, memory_type_t mem_type)
      : Memory(device_id, mem_type) {
    this->init();
  };
  ~HostMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_buffer(void *dest, const void *src, size_t size);
  status_t copy_buffer_to_host(void *dest, const void *src, size_t size);
  status_t copy_buffer_to_buffer(void *dest, const void *src, size_t size);
};

class CudaMemory : public Memory {
public:
  CudaMemory(int device_id, memory_type_t mem_type)
      : Memory(device_id, mem_type) {
    this->init();
  };
  ~CudaMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_buffer(void *dest, const void *src, size_t size);
  status_t copy_buffer_to_host(void *dest, const void *src, size_t size);
  status_t copy_buffer_to_buffer(void *dest, const void *src, size_t size);
};

class RocmMemory : public Memory {
public:
  RocmMemory(int device_id, memory_type_t mem_type)
      : Memory(device_id, mem_type) {
    this->init();
  };
  ~RocmMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_buffer(void *dest, const void *src, size_t size);
  status_t copy_buffer_to_host(void *dest, const void *src, size_t size);
  status_t copy_buffer_to_buffer(void *dest, const void *src, size_t size);
};

} // namespace hddt
#endif