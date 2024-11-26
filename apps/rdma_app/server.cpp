#include <hddt.h>
#include <iostream>
#include <p2p.h>

#include <chrono>
#include <thread>

using namespace hddt;

int main() {
  status_t sret;
  // google::InitGoogleLogging("HDDT");
  // google::SetLogDestination(google::GLOG_INFO, "/tmp/today");
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  std::string client_ip = "192.168.2.251";
  std::string server_ip = "0.0.0.0";

  // Memory *mem_ops = new CudaMemory(0, memory_type_t::NVIDIA_GPU);
  // Memory *mem_ops = new RocmMemory(0, memory_type_t::AMD_GPU);
  // Memory *mem_ops = new NeuwareMemory(1, memory_type_t::CAMBRICON_MLU);
  // Memory *mem_ops = new HostMemory(1, memory_type_t::CPU);
  Memory *mem_ops = new HddtMemory(1);

  // RDMACommunicator *con = new RDMACommunicator(mem_ops, 1024, true, true,
  // client_ip, 2025, server_ip, 2024);
  // RDMACommunicator *con = new RDMACommunicator(mem_ops, 1024, true);
  Communicator *con = new HddtCommunicator(mem_ops, true);
      
  sret = con->Start();
  if (sret != status_t::SUCCESS)
    return 0;

  std::this_thread::sleep_for(std::chrono::seconds(1));
  char host_data[1024];
  mem_ops->copy_device_to_host(host_data, con->share_buffer, 1024);

  printf("Server get Data: %s\n", host_data);
  con->Close();
  return 0;
}