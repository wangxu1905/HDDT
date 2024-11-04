#ifndef HDDT_NET_H
#define HDDT_NET_H

#include <hddt.h>
#include <mem.h>
#include <string>

#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

namespace hddt {
enum class communicator_type_t { CLIENT, SERVER };

struct socket_addr {
  uint32_t addr;
  uint16_t port;
};

/*
Communicator: P2P Transport
implemented by TCP and RDMA(libverbs)
*/
class Communicator {
protected:
  communicator_type_t type;
  Memory *mem_op;
  void **send_buffer;
  void **recv_buffer;
  bool is_buffer_ok = false;

public:
  Communicator(communicator_type_t type, Memory *mem_op) : type(type), mem_op(mem_op) {
    this->mem_op->init();
  }
  ~Communicator() {
    status_t ret;
    if(is_buffer_ok){
      this->mem_op->free_buffer(*this->send_buffer);
      this->mem_op->free_buffer(*this->recv_buffer);
    }
    this->mem_op->free();
  }

  status_t alloc_buffer(size_t size) {
    status_t ret = status_t::SUCCESS;
    ret = this->mem_op->allocate_buffer(this->send_buffer, size);
    if (ret != status_t::SUCCESS) return ret;
    ret = this->mem_op->allocate_buffer(this->recv_buffer, size);
    if (ret != status_t::SUCCESS) return ret;
    this->is_buffer_ok = true;
    return ret;
  }

  virtual status_t Send() = 0;
  virtual status_t Recv() = 0;
  virtual status_t Write() = 0;
  virtual status_t Read() = 0;
};

class TCPCommunicator : public Communicator {
private:
  std::string ip;
  int32_t port;

public:
  TCPCommunicator(communicator_type_t type, Memory *mem_op)
      : Communicator(type, mem_op){};

  status_t Send();
  status_t Recv();
  status_t Write();
  status_t Read() = 0;
};

class RDMACommunicator : public Communicator {
private:
  struct sockaddr *server_addr;
  struct sockaddr *client_addr;

  // connection_type : RC, UC，UD : current only support RC
  
  // the RDMA connection identifier
  struct rdma_cm_id *cm_server_id = NULL;
  struct rdma_cm_id *cm_client_id = NULL;

  // Protect Domain
  struct ibv_pd *server_pd = NULL;
  struct ibv_pd *client_pd = NULL;

  // Memory Region
  struct ibv_mr *client_metadata_mr = NULL;
  struct ibv_mr *server_buffer_mr = NULL;
  struct ibv_mr *server_metadata_mr = NULL;

  // Event Channel
  // report asynchronous communication event
  struct rdma_event_channel *client_cm_event_channel = NULL;
  struct rdma_event_channel *server_cm_event_channel = NULL;

  // Completion Channel
  struct ibv_comp_channel *server_completion_channel = NULL;
  struct ibv_comp_channel *client_completion_channel = NULL;
  
  // // queue pair
  // struct ibv_cq *cq = NULL; // notify for receive completion operations
  // struct ibv_qp_init_attr qp_init_attr; // rdma_create_qp(cm_xx_id, pd, &qp_init_attr)
  // struct ibv_qp *client_qp = NULL; // notify for send completion operations

  // // work request
  // struct ibv_recv_wr client_recv_wr, *bad_client_recv_wr = NULL;
  // struct ibv_send_wr server_send_wr, *bad_server_send_wr = NULL;

  // SGE credentials is where we receive the metadata
  // A send request consists of multiple SGE elements.
  // struct ibv_sge client_recv_sge, server_send_sge;

public:
  RDMACommunicator(communicator_type_t type, Memory *mem_op, struct sockaddr *addr)
      : Communicator(type, mem_op), server_addr(addr){
        setup_server();
        setup_client();
      };
  ~RDMACommunicator() {
    Close();
  }

  // IO interface
  status_t Send();
  status_t Recv();
  status_t Write();
  status_t Read();

  // Control interface
  status_t Start();
  status_t Close();

private:
  status_t create_queue_pair();
  status_t create_memory_region();
  status_t create_protection_domain();
  status_t create_completion_queue();
  status_t create_work_request();

  status_t setup_server();
  status_t start_server();

  status_t setup_client();
  status_t start_client();

};

/*
CommunicatorConfigs: Configs
implemented by TCP and RDMA(libverbs)
*/

} // namespace hddt

#endif