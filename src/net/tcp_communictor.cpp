#include <p2p.h>

namespace hddt {

status_t TCPCommunicator::allocate_buffer() {
  status_t sret = status_t::SUCCESS;
  sret =
      this->mem_op->allocate_buffer(&this->client_send_buffer, this->mem_size);
  if (sret != status_t::SUCCESS)
    return sret;
  sret =
      this->mem_op->allocate_buffer(&this->client_recv_buffer, this->mem_size);
  if (sret != status_t::SUCCESS)
    return sret;
  sret =
      this->mem_op->allocate_buffer(&this->server_send_buffer, this->mem_size);
  if (sret != status_t::SUCCESS)
    return sret;
  sret =
      this->mem_op->allocate_buffer(&this->server_recv_buffer, this->mem_size);
  if (sret != status_t::SUCCESS)
    return sret;
  this->is_buffer_ok = true;
  return sret;
};

status_t TCPCommunicator::free_buffer() {
  status_t sret = status_t::SUCCESS;
  if (this->is_buffer_ok) {
    sret = this->mem_op->free_buffer(this->client_send_buffer);
    if (sret != status_t::SUCCESS) {
      logError("RDMACommunicator::allocate_buffer mem_op->free_buffer "
               "client_send_buffer err %s.",
               status_to_string(sret));
      return sret;
    }
    sret = this->mem_op->free_buffer(this->client_recv_buffer);
    if (sret != status_t::SUCCESS) {
      logError("RDMACommunicator::allocate_buffer mem_op->free_buffer "
               "client_recv_buffer err %s.",
               status_to_string(sret));
      return sret;
    }
    sret = this->mem_op->free_buffer(this->server_send_buffer);
    if (sret != status_t::SUCCESS) {
      logError("RDMACommunicator::allocate_buffer mem_op->free_buffer "
               "server_send_buffer err %s.",
               status_to_string(sret));
      return sret;
    }
    sret = this->mem_op->free_buffer(this->server_recv_buffer);
    if (sret != status_t::SUCCESS) {
      logError("RDMACommunicator::allocate_buffer mem_op->free_buffer "
               "server_recv_buffer err %s.",
               status_to_string(sret));
      return sret;
    }
  }
  this->is_buffer_ok = false; // only free once time
  return sret;
};

status_t TCPCommunicator::Send(void *input_buffer, size_t size, size_t flags) {
  return status_t::SUCCESS;
}
status_t TCPCommunicator::Recv(void *input_buffer, size_t size, size_t flags) {
  return status_t::SUCCESS;
}

status_t TCPCommunicator::Start() { return status_t::SUCCESS; }
status_t TCPCommunicator::Close() { return status_t::SUCCESS; }
} // namespace hddt
