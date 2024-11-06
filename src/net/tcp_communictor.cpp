#include <net.h>

namespace hddt {
status_t TCPCommunicator::Send() { return status_t::SUCCESS; }
status_t TCPCommunicator::Recv() { return status_t::SUCCESS; }
status_t TCPCommunicator::Write(void *addr, size_t length) { return status_t::SUCCESS; }
status_t TCPCommunicator::Read(void *addr, size_t length) { return status_t::SUCCESS; }

status_t TCPCommunicator::Start() { return status_t::SUCCESS; }
status_t TCPCommunicator::Close() { return status_t::SUCCESS; }
} // namespace hddt
