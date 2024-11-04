#ifndef HDDT_NET_H
#define HDDT_NET_H

#include <hddt.h>
#include <mem.h>
#include <string>

namespace hddt {
enum class communicator_type_t { CLIENT, SERVER };

class TaskQueue {};
status_t dummy_mem_alloc(void **addr, size_t length, memory_type_t mem_type);
// todo dmabuf status_t dummy_mem_free(void **addr, size_t length,
memory_type_t mem_type;

/*
Communicator: P2P Transport
implemented by TCP and RDMA(libverbs)
*/
class Communicator {
protected:
  communicator_type_t type;
  void **send_buffer;
  void **recv_buffer;
  // TaskQueue send_task_queue;
  // TaskQueue recv_task_queue;

public:
  Communicator(communicator_type_t type, size_t buffer_length) : type(type) {
    dummy_mem_alloc(this->send_buffer, buffer_length,
                    memory_type_t::NVIDIA_GPU);
    dummy_mem_alloc(this->recv_buffer, buffer_length,
                    memory_type_t::NVIDIA_GPU);
  }
  ~Communicator() { std::cout << "Derived destructor" << std::endl; }

  virtual status_t Send();
  virtual status_t Recv();
};

class TCPCommunicator : public Communicator {
private:
  std::string ip;
  int32_t port;

public:
  TCPCommunicator(communicator_type_t type, size_t buffer_length)
      : Communicator(type, buffer_length){};
  ~TCPCommunicator();
  status_t Send();
  status_t Recv();
};

class RDMACommunicator : public Communicator {
private:
  struct sockaddr_in server_sockaddr;

  mr pd connection_type // RC, UC， UD

      // rdma
      struct rdma_event_channel *client_cm_event_channel = NULL;
  struct rdma_event_channel *server_cm_event_channel = NULL;
  struct rdma_cm_id *cm_server_id = NULL, *cm_client_id = NULL;
  struct ibv_pd *pd = NULL;
  struct ibv_comp_channel *io_completion_channel = NULL;
  struct ibv_cq *cq = NULL;
  struct ibv_qp_init_attr qp_init_attr;
  struct ibv_qp *client_qp = NULL;

  /* RDMA memory resources */
  struct ibv_mr *client_metadata_mr = NULL, *server_buffer_mr = NULL,
                *server_metadata_mr = NULL;
  struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
  struct ibv_recv_wr client_recv_wr, *bad_client_recv_wr = NULL;
  struct ibv_send_wr server_send_wr, *bad_server_send_wr = NULL;
  struct ibv_sge client_recv_sge, server_send_sge;

public:
  RDMACommunicator(communicator_type_t type, size_t buffer_length)
      : Communicator(type, buffer_length){

        };
  ~RDMACommunicator();
  status_t Send(); // client_remote_memory_ops();
  status_t Recv();

  int setup_client_resources();
  client_prepare_connection(&server_sockaddr);
  client_pre_post_recv_buffer();
  client_connect_to_server();
  client_xchange_metadata_with_server();
  client_remote_memory_ops();
  client_disconnect_and_clean();
  status_t Connect();
  status_t Close()

      start_rdma_server(&server_sockaddr);
  setup_client_resources();
  accept_client_connection();
  send_server_metadata_to_client();
  disconnect_and_cleanup();
  status_t Server();
  status_t Close();
};

/*
CommunicatorConfigs: Configs
implemented by TCP and RDMA(libverbs)
*/

} // namespace hddt

#endif