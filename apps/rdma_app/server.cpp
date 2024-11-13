#include <hddt.h>
#include <iostream>
#include <net.h>

#include <chrono>
#include <thread>

using namespace hddt;

int main() {
  // google::InitGoogleLogging("HDDT");
  // google::SetLogDestination(google::GLOG_INFO, "/tmp/today");
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  status_t ret;

  // Memory *mem_ops = new CudaMemory(0, memory_type_t::NVIDIA_GPU);
  Memory *mem_ops = new RocmMemory(0, memory_type_t::AMD_GPU);
  // Memory *mem_ops = new HostMemory(1, memory_type_t::CPU);

  logInfo("%p", mem_ops);
  RDMACommunicator *con = new RDMACommunicator(mem_ops, 1024, true);
  con->Start();

  std::this_thread::sleep_for(std::chrono::seconds(1));
  char host_data[1024];
  mem_ops->copy_buffer_to_host(host_data, con->share_buffer, 1024);

  printf("Server get Data: %s\n", host_data);

  return 0;
}