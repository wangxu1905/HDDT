#include <p2p.h>

namespace hddt {

status_t TCPCommunicator::allocate_buffer(size_t size) {
  status_t sret = status_t::SUCCESS;
  sret = this->mem_op->allocate_buffer(&this->client_send_buffer, size);
  if (sret != status_t::SUCCESS)
    return sret;
  sret = this->mem_op->allocate_buffer(&this->client_recv_buffer, size);
  if (sret != status_t::SUCCESS)
    return sret;
  sret = this->mem_op->allocate_buffer(&this->server_send_buffer, size);
  if (sret != status_t::SUCCESS)
    return sret;
  sret = this->mem_op->allocate_buffer(&this->server_recv_buffer, size);
  if (sret != status_t::SUCCESS)
    return sret;
  this->is_buffer_ok = true;
  this->mem_size = size;
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
