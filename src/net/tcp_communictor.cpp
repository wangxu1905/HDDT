#include <p2p.h>

namespace hddt {
status_t TCPCommunicator::Send(void *input_buffer, size_t size) { return status_t::SUCCESS; }
status_t TCPCommunicator::Recv(void *input_buffer, size_t size) { return status_t::SUCCESS; }

status_t TCPCommunicator::Start() { return status_t::SUCCESS; }
status_t TCPCommunicator::Close() { return status_t::SUCCESS; }
} // namespace hddt
