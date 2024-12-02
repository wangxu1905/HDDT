#include <hddt.h>
#include <iostream>
#include <p2p.h>

using namespace hddt;

int main() {
  // google::InitGoogleLogging("HDDT");
  // google::SetLogDestination(google::GLOG_INFO, "./today");
  status_t sret;
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  std::string client_ip = "192.168.2.251";
  std::string server_ip = "0.0.0.0";

  // Memory *mem_ops = new CudaMemory(1, MemoryType::NVIDIA_GPU);
  // Memory *mem_ops = new RocmMemory(1, MemoryType::AMD_GPU);
  // Memory *mem_ops = new NeuwareMemory(1, MemoryType::CAMBRICON_MLU);
  // Memory *mem_ops = new HostMemory(1, MemoryType::CPU);
  Memory *mem_ops = new Memory(1, MemoryType);

  // RDMACommunicator *con =
  //     new RDMACommunicator(mem_ops, 1024, true, true, client_ip, 2024,
  //     server_ip, 2025);
  // RDMACommunicator *con =
  //     new RDMACommunicator(mem_ops, 1024, false, true, client_ip);
  auto con = CreateCommunicator(mem_ops, CommunicatorType::DEFAULT, false, true,
                                client_ip);

  sret = con->Start();
  if (sret != status_t::SUCCESS)
    return 0;

  uint8_t data[] = "Hello World!";

  void *send;
  mem_ops->allocate_buffer(&send, 1024);
  mem_ops->copy_host_to_device(send, data, sizeof(data));

  char host_data[sizeof(data)];
  mem_ops->copy_device_to_host(host_data, send, sizeof(data));
  printf("Client Write Data: %s\n", host_data);

  con->Send(send, 1024, sizeof(data));

  con->Close();
  sleep(5);
  con.reset(); // 将 con 设置为 nullptr 并释放其所管理的资源
  return 0;
}