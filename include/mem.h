#ifndef MEMORY_H
#define MEMORY_H

#include <cstring>
#include <hddt.h>
#include <memory>


namespace hddt {

#define ACCEL_PAGE_SIZE (64 * 1024)

enum class memory_type_t {
  DEFAULT,       // 默认情况, 系统决定
  CPU,
  NVIDIA_GPU,
  AMD_GPU,
  CAMBRICON_MLU  
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
        //std::cout << "HostMemory init" << std::endl;
      };
  ~HostMemory() {
    this->free();
  };
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
    //std::cout << "CudaMemory init" << std::endl;
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

/*
* 新增HddtMemory类，可由用户指定设备类型和设备号，并自动创建相应的Memory类实例
* 也可由系统自动识别支持device的类型
* 
*/ 
class HddtMemory {
  private:
    int hddtDeviceId;
    memory_type_t hddtMemoryType;
    std::unique_ptr<Memory> memoryClass;
    status_t initStatus;

  public:
  HddtMemory(int device_id, memory_type_t mem_type = memory_type_t::DEFAULT) {
    this->set_DeviceId_and_MemoryType(device_id, mem_type);
  }

  ~HddtMemory() {
    this->free();
  }

  std::unique_ptr<Memory> createMemoryClass(memory_type_t mem_type);
  status_t init();
  status_t free();

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);

  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t set_DeviceId_and_MemoryType(int device_id, memory_type_t mem_type = memory_type_t::DEFAULT);

  memory_type_t get_MemoryType();
  status_t get_init_Status();
  int get_DeviceId();
};

class NeuwareMemory : public Memory {
public:
  CNaddr mlu_addr;

public:
  NeuwareMemory(int device_id, memory_type_t mem_type)
      : Memory(device_id, mem_type) {
    status_t sret;
    sret = this->init();
    if (sret != status_t::SUCCESS) {
      logError("Fail to init mem_ops");
      exit(1);
    }
  };
  ~NeuwareMemory() { this->free(); };

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