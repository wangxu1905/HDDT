#include <p2p.h>

namespace hddt {

/*
 * Public API
 */

status_t RDMACommunicator::allocate_buffer() {
  status_t sret = status_t::SUCCESS;
  sret = this->mem_op->allocate_buffer(&this->share_buffer, this->mem_size);
  if (sret != status_t::SUCCESS) {
    logError(
        "RDMACommunicator::allocate_buffer mem_op->allocate_buffer err %s.",
        status_to_string(sret));
    return sret;
  }
  this->is_buffer_ok = true;
  return sret;
};

status_t RDMACommunicator::free_buffer() {
  status_t sret = status_t::SUCCESS;
  sret = this->mem_op->free_buffer(this->share_buffer);
  if (sret != status_t::SUCCESS) {
    logError("RDMACommunicator::allocate_buffer mem_op->free_buffer err %s.",
             status_to_string(sret));
    return sret;
  }
  this->is_buffer_ok = false; // only free once time
  return sret;
};

status_t RDMACommunicator::init_sockaddr(const char *client_ip,
                                         uint16_t client_port,
                                         const char *server_ip,
                                         uint16_t server_port) {
  // server addr
  bzero(&this->server_addr, sizeof this->server_addr);
  this->server_addr.sin_family = AF_INET;
  this->server_addr.sin_port = htons(server_port);
  inet_pton(AF_INET, server_ip, &this->server_addr.sin_addr);
  // client addr
  bzero(&this->client_addr, sizeof this->client_addr);
  this->client_addr.sin_family = AF_INET;
  this->client_addr.sin_port = htons(client_port);
  inet_pton(AF_INET, client_ip, &this->client_addr.sin_addr);
  return status_t::SUCCESS;
}

status_t RDMACommunicator::Start() {
  status_t sret = status_t::SUCCESS;
  std::atomic<status_t> server_sret(status_t::SUCCESS);

  std::thread server_thread([this, &server_sret] {
    if (this->is_server) {
      server_sret.store(this->start_server());
    }
  });
  if (this->is_client) {
    while (this->retry_count < this->retry_times) {
      sret = this->start_client();
      if (sret == status_t::SUCCESS) {
        break;
      }
      if (this->retry_count >= this->retry_times) {
        sret = status_t::ERROR;
        break;
      }
      this->retry_count++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(this->retry_delay_time));
      logError("Retry to connect server. ...%d.", this->retry_count);
      sret = this->setup_client(); // we need to re-setup the client.
      if (sret != status_t::SUCCESS) {
        break;
      }
    }
  }
  server_thread.join();
  if (server_sret.load() != status_t::SUCCESS) {
    sret = server_sret.load();
  }

  return sret;
}
status_t RDMACommunicator::Close() {
  if (this->is_server)
    this->close_server();
  if (this->is_client)
    this->close_client();
  /* It is uncertain whether the destructor of unique_ptr truly releases memory
   * upon destruction, so we explicitly release the memory each time we close.
   */
  if (this->is_buffer_ok) {
    this->free_buffer();
    logDebug("RDMACommunicator::Close free_buffer success.");
  }
  return status_t::SUCCESS;
}

/*
 * IO API
 * We let the client to manage the single-side memeory operators (write and
 * read)
 * client send notification to server to notify operation status.
 * server recv notification by loop waiting.
 */

status_t RDMACommunicator::Send(void *input_buffer, size_t size, size_t flags) {
  // client send notification to server : Write is done;
  this->mem_op->copy_device_to_device(this->share_buffer, input_buffer, flags);
  this->write(this->share_buffer, flags);
  return status_t::SUCCESS;
}

status_t RDMACommunicator::Recv(void *output_buffer, size_t size,
                                size_t flags) {
  // for loop to check ACK
  this->mem_op->copy_device_to_device(output_buffer, this->share_buffer, flags);
  return status_t::SUCCESS;
}

status_t RDMACommunicator::write(void *addr, size_t length) {
  status_t sret;
  struct ibv_wc wc;
  int wc_count;

  sret = this->post_work_request(
      this->client_qp, (uint64_t)addr, (uint32_t)length,
      this->client_send_buffer_mr->lkey, 1, IBV_WR_RDMA_WRITE,
      IBV_SEND_SIGNALED, this->client_newserver_metadata_attr.stag.remote_stag,
      this->client_newserver_metadata_attr.address, true);
  if (sret != status_t::SUCCESS) {
    logError("Failed to post write WR from client buffer");
    return sret;
  }

  wc_count = this->process_work_completion_events(
      this->client_completion_channel, &wc, 1);
  if (wc_count != 1) {
    logError("We failed to get 1 work completions.");
    return status_t::ERROR;
  }
  logDebug("Client side WRITE is complete.");
  return status_t::SUCCESS;
}

status_t RDMACommunicator::read(void *addr, size_t length) {
  status_t sret;
  struct ibv_wc wc;
  int wc_count;

  sret = this->post_work_request(
      this->client_qp, (uint64_t)addr, (uint32_t)length,
      this->client_recv_buffer_mr->lkey, 1, IBV_WR_RDMA_READ, IBV_SEND_SIGNALED,
      this->client_newserver_metadata_attr.stag.remote_stag,
      this->client_newserver_metadata_attr.address, true);
  if (sret != status_t::SUCCESS) {
    logError("Failed to post read WR from client buffer");
    return sret;
  }

  wc_count = this->process_work_completion_events(
      this->client_completion_channel, &wc, 1);
  if (wc_count != 1) {
    logError("We failed to get 1 work completions.");
    return status_t::ERROR;
  }
  logDebug("Client side READ is complete.");
  return status_t::SUCCESS;
}

status_t RDMACommunicator::setup_server() {
  int ret = -1;

  // 1. event channel
  this->server_cm_event_channel = rdma_create_event_channel();
  if (!this->server_cm_event_channel) {
    logError("Creating cm event channel failed");
    return status_t::ERROR;
  }
  logDebug("Server: RDMA CM event channel is created successfully at %p.",
           this->server_cm_event_channel);

  // 2. create rdma id
  ret = rdma_create_id(this->server_cm_event_channel, &this->server_cm_id, NULL,
                       RDMA_PS_TCP);
  if (ret) {
    logError("Creating server cm id failed");
    return status_t::ERROR;
  }
  logDebug("RDMA connection id for the server is created.");

  // 3. bind server
  ret =
      rdma_bind_addr(this->server_cm_id, (struct sockaddr *)&this->server_addr);
  if (ret) {
    logError("Failed to bind server address");
    return status_t::ERROR;
  }
  logDebug("Server RDMA CM id is successfully binded.");

  return status_t::SUCCESS;
}

status_t RDMACommunicator::start_server() {
  int ret = -1;
  status_t sret;
  struct rdma_cm_event *cm_event = NULL;

  // start server and waiting for acception from client
  // 1. start listening
  ret = rdma_listen(this->server_cm_id,
                    8); /* backlog = 8 clients, same as TCP, see man listen*/
  if (ret) {
    logError("rdma_listen failed to listen on server address.");
    return status_t::ERROR;
  }
  logInfo("Server is listening successfully at: %s , port: %d.",
          inet_ntoa(this->server_addr.sin_addr),
          ntohs(this->server_addr.sin_port));

  // 2. expect a client to connect
  sret = process_rdma_cm_event(this->server_cm_event_channel,
                               RDMA_CM_EVENT_CONNECT_REQUEST, &cm_event);
  if (sret == status_t::ERROR) {
    logError("Failed to get cm event.");
    return sret;
  }
  this->server_cm_newconnection_id = cm_event->id;
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("Failed to acknowledge the cm event errno.");
    return status_t::ERROR;
  }
  logDebug("A new RDMA client(newconnection) connection id is stored at %p.",
           this->server_cm_newconnection_id);

  // 3. setup newconnection client resource
  // 3.1. create pd
  this->server_newconnection_pd =
      ibv_alloc_pd(this->server_cm_newconnection_id->verbs);
  if (!this->server_newconnection_pd) {
    logError("Failed to allocate a protection domain.");
    return status_t::ERROR;
  }
  logDebug("A new protection domain is allocated at %p.",
           this->server_newconnection_pd);

  // 3.2. prepare server's buffer mr
  ibv_access_flags access = static_cast<const ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE);
  this->server_send_buffer_mr = rdma_buffer_register(
      this->server_newconnection_pd,
      this->share_buffer, // for transport : send/recv buffer is the buffer
      this->mem_size, access);
  if (!this->server_send_buffer_mr) {
    logError("Server :send : failed to create a buffer.");
    return status_t::ERROR;
  }
  this->server_recv_buffer_mr = rdma_buffer_register(
      this->server_newconnection_pd,
      this->share_buffer, // for transport : send/recv buffer is the buffer
      this->mem_size, access);
  if (!this->server_recv_buffer_mr) {
    logError("Server :recv : failed to create a buffer.");
    return status_t::ERROR;
  }
  logInfo("Server prepare memory region success.");

  // 3.3. completion channel
  this->server_completion_channel =
      ibv_create_comp_channel(this->server_cm_newconnection_id->verbs);
  if (!this->server_completion_channel) {
    logError("Failed to create an I/O completion event channel.");
    return status_t::ERROR;
  }
  logDebug("An I/O completion event channel is created at %p.",
           this->server_completion_channel);

  // 3.4. completion queue
  this->server_cq = ibv_create_cq(
      this->server_cm_newconnection_id->verbs /* which device*/,
      CQ_CAPACITY /* maximum capacity*/, NULL /* user context, not used here */,
      this->server_completion_channel /* which IO completion channel */,
      0 /* signaling vector, not used here*/);
  if (!this->server_cq) {
    logError("Failed to create a completion queue (cq).");
    return status_t::ERROR;
  }
  logDebug("Completion queue (CQ) is created at %p with %d elements.",
           this->server_cq, this->server_cq->cqe);

  // 3.5. req a event
  ret = ibv_req_notify_cq(this->server_cq, 0);
  if (ret) {
    logError("Failed to request notifications on CQ.");
    return status_t::ERROR;
  }

  // 3.6. setup queue pair
  bzero(&this->server_qp_init_attr, sizeof(this->server_qp_init_attr));
  this->server_qp_init_attr.cap.max_recv_sge = MAX_SGE;
  this->server_qp_init_attr.cap.max_recv_wr = MAX_WR;
  this->server_qp_init_attr.cap.max_send_sge = MAX_SGE;
  this->server_qp_init_attr.cap.max_send_wr = MAX_WR;
  this->server_qp_init_attr.qp_type = IBV_QPT_RC; // current only support RC
  this->server_qp_init_attr.recv_cq =
      this->server_cq; // use same cq for recv and send
  this->server_qp_init_attr.send_cq = this->server_cq;

  ret =
      rdma_create_qp(this->server_cm_newconnection_id,
                     this->server_newconnection_pd, &this->server_qp_init_attr);
  if (ret) {
    logError("Server: Failed to create QP.");
    return status_t::ERROR;
  }
  /* Save the reference for handy typing but is not required */
  this->server_newconnection_qp = this->server_cm_newconnection_id->qp;
  logDebug("Newconnection QP created at %p.", this->server_newconnection_qp);

  // 4 accept new connection
  logDebug("Waiting to accept a new connection.");
  sret = this->server_accept_newconnection();
  if (sret != status_t::SUCCESS) {
    logError("Failed to accept a new connection.");
    return status_t::ERROR;
  }

  // 5. send metadata to client
  logDebug("Start to send metadata to the new connection.");
  sret = this->server_send_metadata_to_newconnection();
  if (sret != status_t::SUCCESS) {
    logError("Failed to send metadata to newconnection.");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t RDMACommunicator::close_server() {
  struct rdma_cm_event *cm_event = NULL;
  int ret = -1;
  status_t sret;
  /* Wait for the client to send a disconnect event */
  logDebug("RDMACommunicator::close_server: Waiting for cm event: "
           "RDMA_CM_EVENT_DISCONNECTED.");
  sret = this->process_rdma_cm_event(this->server_cm_event_channel,
                                     RDMA_CM_EVENT_DISCONNECTED, &cm_event);
  if (sret != status_t::SUCCESS) {
    logError("RDMACommunicator::close_server: Get disconnect event err %s.",
             status_to_string(sret));
    return status_t::ERROR;
  }
  /* Acknowledge the event */
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("RDMACommunicator::close_server: Acknowledge the cm event err %d.",
             ret);
    return status_t::ERROR;
  }
  /* Free all the resources */
  /* Destroy QP */
  rdma_destroy_qp(this->server_cm_newconnection_id);
  /* Destroy client cm id */
  logDebug("RDMACommunicator::close_server: Destory new_connection cm id.");
  ret = rdma_destroy_id(this->server_cm_newconnection_id);
  if (ret) {
    logError("RDMACommunicator::close_server: Destroy client id err %d.", ret);
  }
  /* Destroy CQ */
  ret = ibv_destroy_cq(this->server_cq);
  if (ret) {
    logError("RDMACommunicator::close_server: Destroy completion queue err %d.",
             ret);
  }
  /* Destroy completion channel */
  ret = ibv_destroy_comp_channel(this->server_completion_channel);
  if (ret) {
    logError(
        "RDMACommunicator::close_server: Destroy completion channel err %d.",
        ret);
  }

  /* Destroy memory buffers */
  logDebug("RDMACommunicator::close_server: Destory memory buffers.");
  // to avoid double free, we free the memory buffer by this->Close, so we
  // don't need to free it here.
  this->rdma_buffer_deregister(this->server_send_buffer_mr);
  this->rdma_buffer_deregister(this->server_recv_buffer_mr);
  this->rdma_buffer_deregister(this->server_newconnection_metadata_mr);
  this->rdma_buffer_deregister(this->server_metadata_mr);
  logDebug("RDMACommunicator::close_server: Destory memory buffers success.");

  /* Destroy protection domain */
  ret = ibv_dealloc_pd(this->server_newconnection_pd);
  if (ret) {
    logError("RDMACommunicator::close_server: Destroy client protection domain "
             "err %s.",
             status_to_string(sret));
  }
  /* Destroy rdma server id */
  ret = rdma_destroy_id(this->server_cm_id);
  if (ret) {
    logError("RDMACommunicator::close_server: Destroy server cm id err %d.",
             ret);
  }
  rdma_destroy_event_channel(this->server_cm_event_channel);
  logInfo("RDMACommunicator::close_server: Server shut-down is complete.");

  return status_t::SUCCESS;
}

/**
 * @brief Sets up the RDMA client
 *
 * This function initializes the relevant resources for the RDMA client,
 * including the event channel, connection manager ID, address resolution, route
 * resolution, protection domain, memory region, completion channel, completion
 * queue, and queue pair.
 *
 * @return Returns status_t::SUCCESS if setup is successful; otherwise, returns
 * status_t::ERROR.
 */
status_t RDMACommunicator::setup_client() {
  struct rdma_cm_event *cm_event = NULL;
  status_t sret;
  int ret = -1;
  // alloc resources for client
  // 1. client event channel
  this->client_cm_event_channel = rdma_create_event_channel();
  if (!this->client_cm_event_channel) {
    logError("Creating cm event channel failed.");
    return status_t::ERROR;
  }
  logDebug("Client: RDMA CM event channel is created at : %p.",
           this->client_cm_event_channel);

  // 2. client connection manage id
  ret = rdma_create_id(this->client_cm_event_channel, &this->client_cm_id, NULL,
                       RDMA_PS_TCP);
  if (ret) {
    logError("Creating cm id failed with errno.");
    return status_t::ERROR;
  }

  /* 3. Resolve destination and optional source addresses from IP addresses  to
   * an RDMA address.  If successful, the specified rdma_cm_id will be bound
   * to a local device. */
  ret = rdma_resolve_addr(this->client_cm_id, NULL,
                          (struct sockaddr *)&this->client_addr, 2000);
  if (ret) {
    logError("Failed to resolve address.");
    return status_t::ERROR;
  }
  logDebug("waiting for cm event: RDMA_CM_EVENT_ADDR_RESOLVED.");
  // waiting if recvive the complement event
  sret = process_rdma_cm_event(this->client_cm_event_channel,
                               RDMA_CM_EVENT_ADDR_RESOLVED, &cm_event);
  if (sret != status_t::SUCCESS) {
    logError("Failed to receive a valid event.");
    return sret;
  }
  // ack it
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("Failed to acknowledge the CM event.");
    return status_t::ERROR;
  }
  logDebug("RDMA address is resolved.");

  // resolve RDMA route
  ret = rdma_resolve_route(this->client_cm_id, 2000);
  if (ret) {
    logError("Failed to resolve route.");
    return status_t::ERROR;
  }
  logDebug("waiting for cm event: RDMA_CM_EVENT_ROUTE_RESOLVED.");
  sret = process_rdma_cm_event(this->client_cm_event_channel,
                               RDMA_CM_EVENT_ROUTE_RESOLVED, &cm_event);
  if (sret != status_t::SUCCESS) {
    logError("Failed to receive a valid event.");
    return sret;
  }
  /* ack the event */
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("Failed to acknowledge the CM event.");
    return status_t::ERROR;
  }
  logInfo("Trying to connect to server at : %s port: %d \n",
          inet_ntoa(this->client_addr.sin_addr),
          ntohs(this->client_addr.sin_port));

  // 4. create protection domain
  this->client_pd = ibv_alloc_pd(this->client_cm_id->verbs);
  if (!this->client_pd) {
    logError("Failed to alloc pd.");
    return status_t::ERROR;
  }
  // 5. prepare client's memory region
  ibv_access_flags access = static_cast<const ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE);
  this->client_send_buffer_mr = rdma_buffer_register(
      this->client_pd,
      this->share_buffer, // for transport : send/recv buffer is the buffer
      this->mem_size, access);
  logInfo("rdma_buffer_register client_send_buffer_mr to pd");
  if (!this->client_send_buffer_mr) {
    logError("Client :send : Failed to register the share_buffer.");
    return status_t::ERROR;
  }
  this->client_recv_buffer_mr = rdma_buffer_register(
      this->client_pd,
      this->share_buffer, // for transport : send/recv buffer is the buffer
      this->mem_size, access);
  logInfo("rdma_buffer_register client_recv_buffer_mr to pd");
  if (!this->client_recv_buffer_mr) {
    logError("Client :recv : Failed to register the share_buffer.");
    return status_t::ERROR;
  }
  logInfo("Client prepare memory region success.");

  // 6. create completion channel
  this->client_completion_channel =
      ibv_create_comp_channel(this->client_cm_id->verbs);
  if (!this->client_completion_channel) {
    logError("Failed to create IO completion event channel.");
    return status_t::ERROR;
  }
  logDebug("completion event channel created at : %p \n",
           this->client_completion_channel);

  // 7. create completion queue
  this->client_cq = ibv_create_cq(this->client_cm_id->verbs, CQ_CAPACITY, NULL,
                                  this->client_completion_channel, 0);
  if (!this->client_cq) {
    logError("Failed to create CQ.");
    return status_t::ERROR;
  }
  logDebug("CQ created at %p with %d elements.", client_cq, client_cq->cqe);
  ret = ibv_req_notify_cq(client_cq, 0);
  if (ret) {
    logError("Failed to request notifications.");
    return status_t::ERROR;
  }

  // 8. create qp
  logDebug("Create qp %p.", &this->client_qp_init_attr);
  bzero(&this->client_qp_init_attr, sizeof this->client_qp_init_attr);
  this->client_qp_init_attr.cap.max_recv_sge =
      MAX_SGE; /* Maximum SGE per receive posting */
  this->client_qp_init_attr.cap.max_recv_wr =
      MAX_WR; /* Maximum receive posting capacity */
  this->client_qp_init_attr.cap.max_send_sge =
      MAX_SGE; /* Maximum SGE per send posting */
  this->client_qp_init_attr.cap.max_send_wr =
      MAX_WR; /* Maximum send posting capacity */
  this->client_qp_init_attr.qp_type =
      IBV_QPT_RC; /* QP type, RC = Reliable connection */
  /* use same completion queue */
  this->client_qp_init_attr.recv_cq = this->client_cq;
  this->client_qp_init_attr.send_cq = this->client_cq;
  /*Create a QP */
  logDebug("cap.max_recv_sge is %d.",
           this->client_qp_init_attr.cap.max_recv_sge);
  ret = rdma_create_qp(this->client_cm_id, this->client_pd,
                       &this->client_qp_init_attr);
  if (ret) {
    logError("Client: Failed to create QP.");
    return status_t::ERROR;
  }
  this->client_qp = this->client_cm_id->qp;
  logDebug("QP created at %p \n", this->client_qp);

  // 9. pre post metadata recv buffer
  this->client_newserver_metadata_mr = rdma_buffer_register(
      this->client_pd, &this->client_newserver_metadata_attr,
      sizeof(this->client_newserver_metadata_attr), (IBV_ACCESS_LOCAL_WRITE));
  if (!this->client_newserver_metadata_mr) {
    logError("Failed to setup the newserver metadata mr.");
    return status_t::ERROR;
  }
  logDebug("Setup the newserver metadata mr is successful");

  sret = this->post_work_request(
      this->client_qp, (uint64_t)this->client_newserver_metadata_mr->addr,
      (uint32_t)this->client_newserver_metadata_mr->length,
      (uint32_t)this->client_newserver_metadata_mr->lkey, 1,
      ibv_wr_opcode::IBV_WR_RDMA_READ, ibv_send_flags::IBV_SEND_INLINE, 0, 0,
      false);
  if (sret != status_t::SUCCESS) {
    logError("Failed to pre-post the receive buffer.");
    return status_t::ERROR;
  }
  logDebug("Pre-post receive newserver metadata is successful");

  return status_t::SUCCESS;
}

/**
 * @brief Starts the RDMA client
 *
 * This function is used to start the RDMA client, establish a connection with
 * the RDMA server, and exchange metadata.
 *
 * @return Returns the operation status. Returns status_t::SUCCESS on success,
 * or status_t::ERROR on failure.
 */
status_t RDMACommunicator::start_client() {
  struct rdma_conn_param conn_param;
  struct rdma_cm_event *cm_event = NULL;
  int ret = -1;
  status_t sret;
  bzero(&conn_param, sizeof(conn_param));
  conn_param.initiator_depth = this->initiator_depth;
  conn_param.responder_resources = this->responder_resources;
  conn_param.retry_count = 3;
  ret = rdma_connect(this->client_cm_id, &conn_param);
  if (ret) {
    logError("Failed to connect to remote host.");
    return status_t::ERROR;
  }
  logDebug("Waiting for cm event: RDMA_CM_EVENT_ESTABLISHED.");
  // retry while connect failed
  sret = process_rdma_cm_event(this->client_cm_event_channel,
                               RDMA_CM_EVENT_ESTABLISHED, &cm_event);
  if (sret != status_t::SUCCESS) {
    logError("Failed to connect to server.");
    return status_t::ERROR;
  }
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("Failed to acknowledge cm event.");
    return status_t::ERROR;
  }
  logInfo("The client is connected successfully.");

  // xchange metadata with server
  struct ibv_wc wc[2];
  logInfo("Start xchange");

  // prepare the metadata
  this->client_metadata_attr.address =
      (uint64_t)this->client_send_buffer_mr->addr;
  this->client_metadata_attr.length = this->client_send_buffer_mr->length;
  this->client_metadata_attr.stag.local_stag =
      this->client_send_buffer_mr->lkey;
  this->client_metadata_mr = rdma_buffer_register(
      this->client_pd, &this->client_metadata_attr,
      sizeof(this->client_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
  if (!this->client_metadata_mr) {
    logError("Failed to register the client metadata buffer.");
    return status_t::ERROR;
  }
  // post send work request
  sret = this->post_work_request(this->client_qp,
                                 (uint64_t)this->client_metadata_mr->addr,
                                 (uint32_t)this->client_metadata_mr->length,
                                 this->client_metadata_mr->lkey, 1, IBV_WR_SEND,
                                 IBV_SEND_SIGNALED, 0, 0, true);
  if (sret != status_t::SUCCESS) {
    logError("Failed to send client metadata.");
    return sret;
  }

  // waiting and process work completion event
  // expecting 2 work completion. One for send and one for recv
  int wc_count;
  wc_count = this->process_work_completion_events(
      this->client_completion_channel, wc, 2);
  if (wc_count != 2) {
    logError("We failed to get 2 work completions , wc_count = %d", wc_count);
    return status_t::ERROR;
  }

  logDebug("Server sent us its buffer location and credentials, showing \n");
  show_rdma_buffer_attr(&this->client_newserver_metadata_attr);
  return status_t::SUCCESS;
}

status_t RDMACommunicator::close_client() {
  struct rdma_cm_event *cm_event = NULL;
  int ret = -1;
  status_t sret;
  /* active disconnect from the client side */
  ret = rdma_disconnect(this->client_cm_id);
  if (ret) {
    logError("RDMACommunicator::close_client: Disconnect err: %d.", ret);
    // continuing anyways
  }
  sret = this->process_rdma_cm_event(this->client_cm_event_channel,
                                     RDMA_CM_EVENT_DISCONNECTED, &cm_event);
  if (sret != status_t::SUCCESS) {
    logError("RDMACommunicator::close_client: Get RDMA_CM_EVENT_DISCONNECTED "
             "event err %s.",
             status_to_string(sret));
    // continuing anyways
  }
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("RDMACommunicator::close_client: Acknowledge cm event err: %d.",
             ret);
    // continuing anyways
  }

  /* Free all the resources */
  /* Destroy QP */
  rdma_destroy_qp(this->client_cm_id);
  /* Destroy client cm id */
  logDebug("RDMACommunicator::close_client: Destory new_connection cm id.");
  /* Destroy CQ */
  ret = ibv_destroy_cq(this->client_cq);
  if (ret) {
    logError("RDMACommunicator::close_client: Destroy completion queue err %d.",
             ret);
  }
  /* Destroy completion channel */
  ret = ibv_destroy_comp_channel(this->client_completion_channel);
  if (ret) {
    logError(
        "RDMACommunicator::close_server: Destroy completion channel err %d.",
        ret);
  }

  /* Destroy memory buffers */
  logDebug("RDMACommunicator::close_client: Destory memory buffers.");
  // to avoid double free, we free the memory buffer by this->Close, so we
  // don't need to free it here.
  this->rdma_buffer_deregister(this->client_send_buffer_mr);
  this->rdma_buffer_deregister(this->client_recv_buffer_mr);
  this->rdma_buffer_deregister(this->client_newserver_metadata_mr);
  this->rdma_buffer_deregister(this->client_metadata_mr);
  logDebug("RDMACommunicator::close_client: Destory memory buffers success.");

  /* Destroy protection domain */
  ret = ibv_dealloc_pd(this->client_pd);
  if (ret) {
    logError("RDMACommunicator::close_client: Destroy client protection domain "
             "err %d.",
             ret);
  }
  /* Destroy rdma client id */
  ret = rdma_destroy_id(this->client_cm_id);
  if (ret) {
    logError("RDMACommunicator::close_client: Destroy client cm id err %d.",
             ret);
  }
  rdma_destroy_event_channel(this->client_cm_event_channel);
  logInfo("RDMACommunicator::close_client: Client shut-down is complete.");

  return status_t::SUCCESS;
};

/* Post a work_request to QP
 * : qp: which qp to send
 * : sge: form the work request
 * : sge_num: defaule is 1
 * : opcode: IBV_WR_SEND,
 * : send_flag: IBV_SEND_SIGNALED,
 * : for recv WR, opcode and flags are not needed. set it to
 *        ibv_wr_opcode::IBV_WR_RDMA_READ, ibv_send_flags::IBV_SEND_INLINE
 * : remote_key, remote_addr: for no-data transport, we don't need those, set
 * them to 0 : is_send: ibv_post_send or ibv_post_recv
 */
status_t RDMACommunicator::post_work_request(
    struct ibv_qp *qp, uint64_t sge_addr, size_t sge_length, uint32_t sge_lkey,
    int sge_num, ibv_wr_opcode opcode, ibv_send_flags send_flags,
    uint32_t remote_key, uint64_t remote_addr, bool is_send) {

  int ret = -1;
  struct ibv_sge newconnection_recv_sge;

  if (is_send) {
    struct ibv_send_wr newconnection_wr;
    struct ibv_send_wr *bad_newconnection_wr = NULL;

    newconnection_recv_sge.addr = sge_addr;
    newconnection_recv_sge.length = sge_length;
    newconnection_recv_sge.lkey = sge_lkey;

    bzero(&newconnection_wr, sizeof(newconnection_wr));
    newconnection_wr.sg_list = &newconnection_recv_sge;
    newconnection_wr.num_sge = sge_num;
    newconnection_wr.opcode = opcode;
    newconnection_wr.send_flags = send_flags;
    newconnection_wr.wr.rdma.rkey = remote_key;
    newconnection_wr.wr.rdma.remote_addr = remote_addr;

    ret = ibv_post_send(qp, &newconnection_wr, &bad_newconnection_wr);

  } else {
    struct ibv_recv_wr newconnection_wr;
    struct ibv_recv_wr *bad_newconnection_wr = NULL;

    newconnection_recv_sge.addr = sge_addr;
    newconnection_recv_sge.length = sge_length;
    newconnection_recv_sge.lkey = sge_lkey;

    bzero(&newconnection_wr, sizeof(newconnection_wr));
    newconnection_wr.sg_list = &newconnection_recv_sge;
    newconnection_wr.num_sge = sge_num;

    ret = ibv_post_recv(qp, &newconnection_wr, &bad_newconnection_wr);
  }

  if (ret) {
    logError("Failed to post request work.");
    return status_t::ERROR;
  }
  logDebug("Post request work successful.");
  return status_t::SUCCESS;
}

status_t
RDMACommunicator::process_rdma_cm_event(struct rdma_event_channel *echannel,
                                        enum rdma_cm_event_type expected_event,
                                        struct rdma_cm_event **cm_event) {
  int ret = 1;
  ret = rdma_get_cm_event(echannel, cm_event);
  if (ret) {
    logError(
        "RDMACommunicator::process_rdma_cm_event: Retrieve a cm event err %d.",
        ret);
    return status_t::ERROR;
  }

  if (0 != (*cm_event)->status) {
    logError("RDMACommunicator::process_rdma_cm_event: CM event has non zero "
             "status: %d\n",
             (*cm_event)->status);
    ret = -((*cm_event)->status);
    /* important, we acknowledge the event */
    rdma_ack_cm_event(*cm_event);
    return status_t::ERROR;
  }
  /* good event, was it of the expected type */
  if ((*cm_event)->event != expected_event) {
    logError("RDMACommunicator::process_rdma_cm_event: Unexpected event "
             "received: %s [ expecting: %s ]",
             rdma_event_str((*cm_event)->event),
             rdma_event_str(expected_event));
    /* acknowledge the event */
    rdma_ack_cm_event(*cm_event);
    return status_t::ERROR;
  }
  logDebug("RDMACommunicator::process_rdma_cm_event: A new %s type event is "
           "received \n",
           rdma_event_str((*cm_event)->event));
  /* The caller must acknowledge the event */
  return status_t::SUCCESS;
}

int RDMACommunicator::process_work_completion_events(
    struct ibv_comp_channel *comp_channel, struct ibv_wc *wc, int max_wc) {
  struct ibv_cq *cq_ptr = NULL;
  void *context = NULL;
  int ret = -1, i, total_wc = 0;

  /* We wait for the notification on the CQ channel */
  ret = ibv_get_cq_event(
      comp_channel, /* IO channel where we are expecting the notification */
      &cq_ptr,   /* which CQ has an activity. This should be the same as CQ we
                    created before */
      &context); /* Associated CQ user context, which we did set */
  if (ret) {
    logError("Failed to get next CQ event.");
    return -ret;
  }
  /* Request for more notifications. */
  ret = ibv_req_notify_cq(cq_ptr, 0);
  if (ret) {
    logError("Failed to request further notifications.");
    return -ret;
  }

  // todo. 改成线程模型，控制循环的关闭
  total_wc = 0;
  do {
    ret = ibv_poll_cq(cq_ptr /* the CQ, we got notification for */,
                      max_wc - total_wc /* number of remaining WC elements*/,
                      wc + total_wc /* where to store */);
    if (ret < 0) {
      logError("Failed to poll cq for wc due to %d.", ret);
      /* ret is errno here */
      return ret;
    }
    total_wc += ret;
  } while (total_wc < max_wc);
  logDebug("%d WC are completed.", total_wc);
  /* Check validity and status of I/O work completions */
  for (i = 0; i < total_wc; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      logError("Work completion (WC) has error status: %s at index %d",
               ibv_wc_status_str(wc[i].status), i);
      /* return negative value */
      return -(wc[i].status);
    }
  }
  /* Acknowledge CQ events
   */
  ibv_ack_cq_events(cq_ptr, 
		       1 /* we received one event notification. This is not 
		       number of WC elements */);
  return total_wc;
}

struct ibv_mr *
RDMACommunicator::rdma_buffer_register(struct ibv_pd *pd, void *addr,
                                       uint32_t length,
                                       enum ibv_access_flags permission) {
  struct ibv_mr *mr = NULL;
  if (!pd) {
    logError("Protection domain is NULL, ignoring.");
    return NULL;
  }

  logDebug("rdma_buffer_register: pd is %p, addr is %p, len is %d, permission "
           "is %d.",
           pd, addr, length, permission);
  mr = ibv_reg_mr(pd, addr, length, permission); // todo error
  if (!mr) {
    logError("Failed to create mr on buffer.");
    return NULL;
  }
  logDebug("Registered: %p , len: %u , stag: 0x%x \n", mr->addr,
           (unsigned int)mr->length, mr->lkey);
  return mr;
}

void RDMACommunicator::rdma_buffer_deregister(struct ibv_mr *mr) {
  if (!mr) {
    logError("Passed memory region is NULL, ignoring.");
    return;
  }
  logDebug("Deregistered: %p , len: %u , stag : 0x%x.", mr->addr,
           (unsigned int)mr->length, mr->lkey);
  ibv_dereg_mr(mr);
}

void RDMACommunicator::show_rdma_buffer_attr(struct rdma_buffer_attr *attr) {
  if (!attr) {
    logError("Passed attr is NULL\n");
    return;
  }
  logInfo("---------------------------------------------------------");
  logInfo("buffer attr, addr: %p , len: %u , stag : 0x%x",
          (void *)attr->address, (unsigned int)attr->length,
          attr->stag.local_stag);
  logInfo("---------------------------------------------------------");
}

status_t RDMACommunicator::server_accept_newconnection() {
  status_t sret;
  int ret = -1;
  struct rdma_cm_event *cm_event = NULL;
  // 4. accept a client(newconnection) connection
  // 4.1. memory region
  // prepare the receive buffer in which we will receive the client metadata
  this->server_newconnection_metadata_mr = rdma_buffer_register(
      this->server_newconnection_pd,
      &this->server_newconnection_metadata_attr, // memory info // for metadata
                                                 // : attr is the buffer
      sizeof(this->server_newconnection_metadata_attr),
      (IBV_ACCESS_LOCAL_WRITE));
  if (!this->server_newconnection_metadata_mr) {
    logError("Failed to register new connection attr buffer.");
    return status_t::ERROR;
  }

  // 4.2 post recv WR on the QP
  sret =
      post_work_request(this->server_newconnection_qp,
                        (uint64_t)this->server_newconnection_metadata_mr->addr,
                        this->server_newconnection_metadata_mr->length,
                        this->server_newconnection_metadata_mr->lkey, 1,
                        ibv_wr_opcode::IBV_WR_RDMA_READ,
                        ibv_send_flags::IBV_SEND_INLINE, 0, 0, false);
  // for recv WR, opcode and flags are not needed.
  if (sret != status_t::SUCCESS) {
    logError("Failed to pre-post the receive buffer.");
    return sret;
  }
  logDebug("Receive buffer pre-posting is successful.");

  // 4.3 accept connection
  // 4.3.1 conn_parma
  struct rdma_conn_param conn_param;
  memset(&conn_param, 0, sizeof(conn_param));
  conn_param.initiator_depth = this->initiator_depth;
  conn_param.responder_resources = this->responder_resources;
  ret = rdma_accept(this->server_cm_newconnection_id, &conn_param);
  if (ret) {
    logError("Failed to accept the connection.");
    return status_t::ERROR;
  }
  logDebug("Going to wait for : RDMA_CM_EVENT_ESTABLISHED event.");
  // 4.3.2 expect an RDMA_CM_EVENT_ESTABLISHED
  sret = process_rdma_cm_event(this->server_cm_event_channel,
                               RDMA_CM_EVENT_ESTABLISHED, &cm_event);
  if (sret != status_t::SUCCESS) {
    logError("Failed to get the cm event.");
    return status_t::ERROR;
  }

  // 4.3.3 acknowledge the event
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    logError("Failed to acknowledge the cm event.");
    return status_t::ERROR;
  }
  memcpy(&this->server_newconnection_addr,
         rdma_get_peer_addr(this->server_cm_newconnection_id),
         sizeof(struct sockaddr_in));
  printf("A new connection is accepted from %s \n",
         inet_ntoa(this->server_newconnection_addr.sin_addr));

  return status_t::SUCCESS;
}

status_t RDMACommunicator::server_send_metadata_to_newconnection() {
  struct ibv_wc wc;
  int wc_count;
  status_t sret;

  // Client start a new connection by sending its metadata info.
  wc_count = this->process_work_completion_events(
      this->server_completion_channel, &wc, 1);
  if (wc_count != 1) {
    logError("Failed to receive.");
    return status_t::ERROR;
  }
  // show the attr
  logInfo("Client side buffer information is received...");
  show_rdma_buffer_attr(&this->server_newconnection_metadata_attr);
  logInfo("The client has requested buffer length of : %u bytes \n",
          this->server_newconnection_metadata_attr.length);

  // create server metadata info
  this->server_metadata_attr.address = (uint64_t)server_recv_buffer_mr->addr;
  this->server_metadata_attr.length = (uint32_t)server_recv_buffer_mr->length;
  this->server_metadata_attr.stag.local_stag =
      (uint32_t)server_recv_buffer_mr->lkey;
  this->server_metadata_mr = rdma_buffer_register(
      this->server_newconnection_pd,
      &this->server_metadata_attr, // for metadata : attr is the buffer
      sizeof(this->server_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
  if (!this->server_metadata_mr) {
    logError("Server failed to create to hold server metadata \n");
    return status_t::ERROR;
  }

  // ibv post send : send server metadata mr to newconnection
  sret = post_work_request(
      this->server_newconnection_qp, (uint64_t) & this->server_metadata_attr,
      sizeof(this->server_metadata_attr), this->server_metadata_mr->lkey, 1,
      IBV_WR_SEND, IBV_SEND_SIGNALED, 0, 0, true);
  if (sret != status_t::SUCCESS) {
    logError("Posting of server metdata failed.");
    return sret;
  }

  // check completion notification
  wc_count = this->process_work_completion_events(
      this->server_completion_channel, &wc, 1);
  if (wc_count != 1) {
    logError("Failed to send server metadata.");
    return status_t::ERROR;
  }

  logDebug("Local buffer metadata has been sent to the client.");
  return status_t::SUCCESS;
}

} // namespace hddt