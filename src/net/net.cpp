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

status_t HddtCommunicator::Send(void *input_buffer, size_t size) {
  return this->communicatorClass->Send(input_buffer, size);
}

status_t HddtCommunicator::Recv(void *recv_buffer, size_t size) {
  return this->communicatorClass->Recv(recv_buffer, size);
}

status_t HddtCommunicator::Start() { return this->communicatorClass->Start(); }

status_t HddtCommunicator::Close() { return this->communicatorClass->Close(); }

} // namespace hddt