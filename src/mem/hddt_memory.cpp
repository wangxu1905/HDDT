#include <hddt.h>
#include <mem.h>

namespace hddt {

status_t HddtMemory::init() { return this->memoryClass->init(); }

status_t HddtMemory::free() { return this->memoryClass->free(); }

// create memory class according to memory type
std::unique_ptr<Memory> HddtMemory::createMemoryClass(memory_type_t mem_type) {
  switch (mem_type) {
  case memory_type_t::CPU:
    return std::make_unique<HostMemory>(this->hddtDeviceId,
                                        this->hddtMemoryType);
  case memory_type_t::NVIDIA_GPU:
    return std::make_unique<CudaMemory>(this->hddtDeviceId,
                                        this->hddtMemoryType);
  case memory_type_t::AMD_GPU:
    return std::make_unique<RocmMemory>(this->hddtDeviceId,
                                        this->hddtMemoryType);
  case memory_type_t::CAMBRICON_MLU:
    return std::make_unique<NeuwareMemory>(this->hddtDeviceId,
                                           this->hddtMemoryType);
  default:
    return nullptr;
  }
}

// copy data from host to device
status_t HddtMemory::copy_host_to_device(void *dest, const void *src,
                                         size_t size) {
  return this->memoryClass->copy_host_to_device(dest, src, size);
}

// copy data from device to host
status_t HddtMemory::copy_device_to_host(void *dest, const void *src,
                                         size_t size) {
  return this->memoryClass->copy_device_to_host(dest, src, size);
}

// copy data from device to device
status_t HddtMemory::copy_device_to_device(void *dest, const void *src,
                                           size_t size) {
  return this->memoryClass->copy_device_to_device(dest, src, size);
}

status_t HddtMemory::allocate_buffer(void **addr, size_t size) {
  return this->memoryClass->allocate_buffer(addr, size);
}

status_t HddtMemory::free_buffer(void *addr) {
  return this->memoryClass->free_buffer(addr);
}

// get memory type
memory_type_t HddtMemory::get_MemoryType() { return this->hddtMemoryType; }

// get init status
status_t HddtMemory::get_init_Status() { return this->initStatus; }

// get device id
int HddtMemory::get_DeviceId() { return this->hddtDeviceId; }

// reset device id and memory type
status_t HddtMemory::set_DeviceId_and_MemoryType(int device_id,
                                                 memory_type_t mem_type) {
  if (mem_type == memory_type_t::DEFAULT) { // 未指定mem_type, 则根据系统决定
    this->hddtMemoryType = memory_type_t::CPU;

#ifdef ENABLE_CUDA
    this->hddtMemoryType = memory_type_t::NVIDIA_GPU;
#endif

#ifdef ENABLE_ROCM
    this->hddtMemoryType = memory_type_t::AMD_GPU;
#endif

#ifdef ENABLE_NEUWARE
    this->hddtMemoryType = memory_type_t::CAMBRICON_MLU;
#endif

    this->initStatus = status_t::SUCCESS;
  } else {
    this->initStatus = status_t::SUCCESS;
    if (mem_type == memory_type_t::NVIDIA_GPU) {
#ifndef ENABLE_CUDA
      throw std::runtime_error("NVIDIA GPU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    } else if (mem_type == memory_type_t::AMD_GPU) {
#ifndef ENABLE_ROCM
      throw std::runtime_error("AMD GPU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    } else if (mem_type == memory_type_t::CAMBRICON_MLU) {
#ifndef ENABLE_NEUWARE
      throw std::runtime_error("Cambricon MLU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    }
    this->hddtMemoryType = mem_type;
  }
  this->hddtDeviceId = device_id;
  this->memoryClass = this->createMemoryClass(this->hddtMemoryType);
  this->memoryClass->init();

  return this->initStatus;
}

} // namespace hddt