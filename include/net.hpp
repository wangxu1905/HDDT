#ifndef HDDT_NET_H
#define HDDT_NET_H

#include <hddt.hpp>
#include <iostream>
#include <string>

namespace hddt {
enum class communicator_type_t { CLIENT, SERVER };

class Queue {};

/*
Communicator: P2P Transport
implemented by TCP and RDMA(libverbs)
*/
class Communicator {
public:
  communicator_type_t role;

protected:
  void **send_buffer;
  void **recv_buffer;
  Queue send_queue;
  Queue recv_queue;

public:
  virtual bool Send();
  virtual bool Recv();
};

class TCPCommunicator : public Communicator {
private:
  std::string ip;
  int32_t port;

public:
  TCPCommunicator(/* args */);
  ~TCPCommunicator();
};

class RDMACommunicator {
private:
  std::string ip;
  int32_t port;

public:
  RDMACommunicator(/* args */);
  ~RDMACommunicator();
};
} // namespace hddt

#endif