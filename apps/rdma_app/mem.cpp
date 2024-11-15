#include <hddt.h>
#include <mem.h>

using namespace hddt;

int main() {
  /* GPU memory test */
  // Memory *mem_ops = new CudaMemory(1, memory_type_t::NVIDIA_GPU);
  Memory *mem_ops = new RocmMemory(1, memory_type_t::AMD_GPU);
  void *addr;
  mem_ops->allocate_buffer(&addr, 1024);

  uint8_t data[] = "Hello World!\n";
  mem_ops->copy_host_to_device(addr, data, sizeof(data));

  char host_data[1024];
  mem_ops->copy_device_to_host(host_data, addr, sizeof(data));
  printf("Server get Data: %s\n", host_data);

  /* Host memory test */
  // Memory *mem_ops = new HostMemory(1, memory_type_t::CPU);
  // void *addr;
  // mem_ops->allocate_buffer(&addr, 1024);

  // uint8_t data[20] = "Hello World!\n";
  // uint8_t *ptr = (uint8_t *)addr;
  // for (int i = 0; i < 20; i++) {
  //   ptr[i] = data[i];
  // }

  // printf("Server get Data: %s\n", (char *)ptr);

  // free(addr);

  delete mem_ops;

  return 1;
}