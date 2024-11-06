#include <hddt.h>
#include <mem.h>

using namespace hddt;

int main() {
  // CudaMemory *mem_ops = new CudaMemory(1, memory_type_t::NVIDIA_GPU);
  // void *addr;
  // mem_ops->allocate_buffer(&addr, 1024);

  // uint8_t data[] = "Hello World!\n";
  // cudaMemcpy(addr, data, sizeof(data), cudaMemcpyHostToDevice);

  // char host_data[1024];
  // cudaMemcpy(host_data, addr, 1024, cudaMemcpyDeviceToHost);
  // printf("Server get Data: %s\n", host_data);

  HostMemory *mem_ops = new HostMemory(1, memory_type_t::CPU);
  void *addr;
  mem_ops->allocate_buffer(&addr, 1024);

  uint8_t data[20] = "Hello World!\n";

  // 使用正确的指针类型进行赋值
  uint8_t *ptr = (uint8_t *)addr;  // 强制转换 addr 为 uint8_t* 类型指针
  for (int i = 0; i < 20; i++) {
      ptr[i] = data[i];  // 逐个赋值
  }

  printf("Server get Data: %s\n", (char *)ptr);  // 使用 (char *) 进行打印，假设是字符串

  // 释放内存
  free(addr);

  delete mem_ops;

  return 1;
}