#include <hddt.h>
#include <iostream>
#include <net.h>

using namespace hddt;

int main() {
  // google::InitGoogleLogging("HDDT");
  // google::SetLogDestination(google::GLOG_INFO, "./today");
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  std::string client_ip = "192.168.2.251";

  // Memory *mem_ops = new CudaMemory(1, memory_type_t::NVIDIA_GPU);
  // Memory *mem_ops = new RocmMemory(1, memory_type_t::AMD_GPU);
  Memory *mem_ops = new NeuwareMemory(1, memory_type_t::CAMBRICON_MLU);
  // Memory *mem_ops = new HostMemory(1, memory_type_t::CPU);

  RDMACommunicator *con =
      new RDMACommunicator(mem_ops, 1024, false, true, client_ip);
  con->Start();
  uint8_t data[] = "Hello World!";

  mem_ops->copy_host_to_buffer(con->share_buffer, data, sizeof(data));

  char host_data[sizeof(data)];
  mem_ops->copy_buffer_to_host(host_data, con->share_buffer, sizeof(data));
  printf("client Write Data: %s\n", host_data);

  con->Write(con->share_buffer, 1024);

  return 0;
}