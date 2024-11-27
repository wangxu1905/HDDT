#include <p2p.h>

bool support_rdma() {
  struct ibv_device **dev_list;
  int num_devices;

  // get device list
  dev_list = ibv_get_device_list(&num_devices);
  if (num_devices == 0) {
    std::cerr << "No RDMA devices found." << std::endl;
    return false;
  }

  // check if any device supports RDMA
  for (int i = 0; i < num_devices; ++i) {
    struct ibv_context *context;
    context = ibv_open_device(dev_list[i]);
    if (context != nullptr) {
      ibv_close_device(context);
      ibv_free_device_list(dev_list); // free device list
      return true;                    // found a device that supports RDMA
    }
  }

  ibv_free_device_list(dev_list); // free device list
  return false;                   // no device supports RDMA
}

namespace hddt {

[[nodiscard]] std::unique_ptr<Communicator>
CreateCommunicator(Memory *mem_op, CommunicatorType comm_type, bool is_server,
                   bool is_client, std::string client_ip, uint16_t client_port,
                   std::string server_ip, uint16_t server_port, int retry_times,
                   int retry_delay_time) {
  if (comm_type == CommunicatorType::DEFAULT) {
    if (support_rdma()) {
      comm_type = CommunicatorType::RDMA;
    } else {
      comm_type = CommunicatorType::TCP;
    }
  }

  switch (comm_type) {
  case CommunicatorType::RDMA:
    return std::make_unique<RDMACommunicator>(
        mem_op, is_server, is_client, client_ip, client_port, server_ip,
        server_port, retry_times, retry_delay_time);
  case CommunicatorType::TCP:
    return std::make_unique<TCPCommunicator>(
        mem_op, is_server, is_client, client_ip, client_port, server_ip,
        server_port, retry_times, retry_delay_time);
  default:
    return nullptr;
  }
}

} // namespace hddt