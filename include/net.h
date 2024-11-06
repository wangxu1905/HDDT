#ifndef HDDT_NET_H
#define HDDT_NET_H

#include <hddt.h>
#include <mem.h>
#include <string>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <arpa/inet.h> // inet_ntoa

#define CQ_CAPACITY (16)
#define MAX_SGE (2)
#define MAX_WR (8)
#define RDMA_DEFAULT_PORT (2024)

struct __attribute((packed)) rdma_buffer_attr {
  uint64_t address;
  uint32_t length;
  union stag {
    /* if we send, we call it local stags */
    uint32_t local_stag;
    /* if we receive, we call it remote stag */
    uint32_t remote_stag;
  } stag;
};
/* resolves a given destination name to sin_addr */
int get_addr(char *dst, struct sockaddr *addr);

namespace hddt {

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
  Memory *mem_op;

public:
  Communicator(Memory *mem_op) : mem_op(mem_op) { return ; }
  ~Communicator() { this->mem_op->free(); }

  virtual status_t alloc_buffer(size_t size) { return status_t::SUCCESS; };

  virtual status_t Send() = 0;
  virtual status_t Recv() = 0;
  virtual status_t Write(void *addr, size_t length) = 0;
  virtual status_t Read(void *addr, size_t length) = 0;

  virtual status_t Start() = 0;
  virtual status_t Close() = 0;
};

class TCPCommunicator : public Communicator {
private:
  char* ip;
  int32_t port;
  size_t mem_size;

  void *client_send_buffer;
  void *client_recv_buffer;
  void *server_send_buffer;
  void *server_recv_buffer;
  bool is_buffer_ok = false;

public:
  TCPCommunicator(Memory *mem_op) : Communicator(mem_op){};
  ~TCPCommunicator() {
    if (is_buffer_ok) {
      this->mem_op->free_buffer(this->client_send_buffer);
      this->mem_op->free_buffer(this->client_recv_buffer);
      this->mem_op->free_buffer(this->server_send_buffer);
      this->mem_op->free_buffer(this->server_recv_buffer);
    }
  }

  // for TCP, client send data from send_buffer, recv data to recv_buffer
  status_t alloc_buffer(size_t size) {
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

  status_t Send();
  status_t Recv();
  status_t Write(void *addr, size_t length);
  status_t Read(void *addr, size_t length);

  status_t Start();
  status_t Close();
};

class RDMACommunicator : public Communicator {
  // connection_type : RC, UC，UD : current only support RC
private:
  struct sockaddr_in server_addr;
  struct sockaddr_in
      server_newconnection_addr; // remote addr (accepted client)
  struct sockaddr_in client_addr;

  uint8_t initiator_depth = 8; // suggest 2-8
  uint8_t responder_resources = 8;

  // the RDMA connection identifier
  // cm: connection management
  struct rdma_cm_id *server_cm_id = NULL;
  struct rdma_cm_id *server_cm_newconnection_id = NULL;
  struct rdma_cm_id *client_cm_id = NULL;

  // queue pair
  struct ibv_qp *server_newconnection_qp = NULL;
  struct ibv_qp *client_qp = NULL;

  // Protect Domain
  struct ibv_pd *server_newconnection_pd = NULL;
  struct ibv_pd *client_pd = NULL;

  // Memory Region
  struct ibv_mr *server_newconnection_metadata_mr = NULL;
  struct ibv_mr *server_metadata_mr = NULL;
  struct ibv_mr *server_send_buffer_mr = NULL;
  struct ibv_mr *server_recv_buffer_mr = NULL;
  struct ibv_mr *client_newserver_metadata_mr = NULL;
  struct ibv_mr *client_metadata_mr = NULL;
  struct ibv_mr *client_send_buffer_mr = NULL;
  struct ibv_mr *client_recv_buffer_mr = NULL;
  struct rdma_buffer_attr
      server_newconnection_metadata_attr;       // recv from newconnection
  struct rdma_buffer_attr server_metadata_attr; // send to newconnection
  struct rdma_buffer_attr client_newserver_metadata_attr; // remote server
  struct rdma_buffer_attr client_metadata_attr;           // local client

  // Event Channel : report asynchronous communication event
  struct rdma_event_channel *client_cm_event_channel =
      NULL; // 是否可以用同一个channel
  struct rdma_event_channel *server_cm_event_channel = NULL;

  // Completion Channel
  struct ibv_comp_channel *server_completion_channel = NULL; // newconnection
  struct ibv_comp_channel *client_completion_channel = NULL;

  // completion queue
  struct ibv_cq *server_cq = NULL; // notify for receive completion operations
  struct ibv_cq *client_cq = NULL;

  // init attr
  struct ibv_qp_init_attr *server_qp_init_attr;
  struct ibv_qp_init_attr *client_qp_init_attr;

public:
  void *share_buffer;
  bool is_buffer_ok = false;
  size_t mem_size;

  // e.g. connect_to: "192.168.2.134"
  RDMACommunicator(Memory *mem_op, size_t mem_size, char* connect_to)
      : Communicator(mem_op), mem_size(mem_size) {
    status_t sret;
    logDebug("init sockaddr");
    this->init_sockaddr(connect_to);
    logDebug("alloc buffer");
    sret = this->alloc_buffer(this->mem_size);
    if (sret != status_t::SUCCESS) {
      return ;
    }
    logDebug("setup server");
    setup_server();
    logDebug("setup client");
    setup_client();
  };
  ~RDMACommunicator() {
    Close();
    if (is_buffer_ok) {
      this->mem_op->free_buffer(this->share_buffer);
    }
  }

  // for rdma, client and server operate the same buffer
  status_t alloc_buffer(size_t size) {
    status_t sret = status_t::SUCCESS;
    logDebug("mem op alloc buffer");
    sret = this->mem_op->allocate_buffer(&this->share_buffer, size);
    if (sret != status_t::SUCCESS)
      return sret;
    logDebug("mem op alloc buffer ok");
    this->is_buffer_ok = true;
    this->mem_size = size;
    return sret;
  };

  status_t init_sockaddr(char* connect_to) {
    status_t sret = status_t::SUCCESS;
    // server addr
    bzero(&this->server_addr, sizeof this->server_addr);
    this->server_addr.sin_family = AF_INET;
    this->server_addr.sin_port = htons(RDMA_DEFAULT_PORT);
    inet_pton(AF_INET, "0.0.0.0", &this->server_addr.sin_addr);
    // client addr
    bzero(&this->client_addr, sizeof this->client_addr);
    this->client_addr.sin_family = AF_INET;
    this->client_addr.sin_port = htons(RDMA_DEFAULT_PORT);
    inet_pton(AF_INET, connect_to, &this->client_addr.sin_addr);
    return status_t::SUCCESS;
  }

  // IO interface()
  status_t Send();
  status_t Recv();
  status_t Write(void *addr, size_t length);
  status_t Read(void *addr, size_t length);

  // Control interface
  status_t Start();
  status_t Close();

private:
  status_t post_work_request(struct ibv_qp *qp, uint64_t sge_addr,
                             size_t sge_length, uint32_t sge_lkey, int sge_num,
                             ibv_wr_opcode opcode, ibv_send_flags send_flags,
                             uint32_t remote_key, uint64_t remote_addr,
                             bool is_send);
  status_t process_rdma_cm_event(struct rdma_event_channel *echannel,
                                 enum rdma_cm_event_type expected_event,
                                 struct rdma_cm_event **cm_event);
  status_t process_work_completion_events(struct ibv_comp_channel *comp_channel,
                                          struct ibv_wc *wc, int max_wc,
                                          int *wc_count);
  struct ibv_mr *rdma_buffer_register(struct ibv_pd *pd, void *addr,
                                      uint32_t length,
                                      enum ibv_access_flags permission);
  void rdma_buffer_deregister(struct ibv_mr *mr);
  void show_rdma_buffer_attr(struct rdma_buffer_attr *attr);

  status_t server_accept_newconnection();
  status_t server_send_metadata_to_newconnection();

  status_t setup_server();
  status_t start_server();
  status_t close_server();

  status_t setup_client();
  status_t start_client();
  status_t close_client();
};

} // namespace hddt

#endif