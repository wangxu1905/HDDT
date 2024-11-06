#include <hddt.h>
#include <iostream>
#include <net.h>

using namespace hddt;

int main() {
  // google::InitGoogleLogging("HDDT");
  // google::SetLogDestination(google::GLOG_WARNING, "/tmp/today");

  status_t ret;

	char* aim = "192.168.2.245";

  CudaMemory *mem_ops = new CudaMemory(1, memory_type_t::NVIDIA_GPU);
  
  RDMACommunicator *con = new RDMACommunicator(mem_ops, 1024, aim);
  con->Start();

  void *addr = con->share_buffer;
  // 要写入的数据缓冲区
  uint8_t data[] = "Hello World!\n";

  cudaMemcpy(addr, data, sizeof(data), cudaMemcpyHostToDevice);

  // 验证写入的数据是否正确
  char host_data[sizeof(data)];
  cudaMemcpy(host_data, addr, sizeof(data), cudaMemcpyDeviceToHost);
  printf("client Write Data: %s\n", host_data);

  con->Write(addr, sizeof(data));

  return 0;
}