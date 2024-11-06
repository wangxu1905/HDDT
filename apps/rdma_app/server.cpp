#include <hddt.h>
#include <iostream>
#include <net.h>

#include <thread>
#include <chrono>

using namespace hddt;

int main() {
  // google::InitGoogleLogging("HDDT");
  // google::SetLogDestination(google::GLOG_INFO, "/tmp/today");

  status_t ret;

	char* aim = "192.168.2.245";

  Memory *mem_ops = new CudaMemory(0, memory_type_t::NVIDIA_GPU);
  logInfo("%p", mem_ops);
  RDMACommunicator *con = new RDMACommunicator(mem_ops, 1024, aim);
  con->Start();

  void *addr = con->share_buffer;

  while (1)
  {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    char host_data[1024];
    cudaMemcpy(host_data, addr, 1024, cudaMemcpyDeviceToHost);
    printf("Server get Data: %s\n", host_data);
  }

  return 0;
}