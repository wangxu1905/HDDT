#include<net.h>

namespace hddt {

/*
* Public API
*/
status_t RDMACommunicator::Start() {
    // start server and client in multi-thread
}
status_t RDMACommunicator::Close() {}

// IO
status_t RDMACommunicator::Send(){

}

status_t RDMACommunicator::Recv(){

}

status_t RDMACommunicator::Write(){

}

status_t RDMACommunicator::Read(){

}

/*
* Private API
*/
status_t RDMACommunicator::setup_server() {
    // allco resources for server and setup server
    int ret = -1;

    // 1. event channel
    this->server_cm_event_channel = rdma_create_event_channel();
    if (!this->server_cm_event_channel) {
		logError("Creating cm event channel failed");
		return status_t::ERROR;
	}
    logDebug("RDMA CM event channel is created successfully at %p.", 
			this->server_cm_event_channel);
    
    // 2. create rdma id
    ret = rdma_create_id(this->server_cm_event_channel, &this->cm_server_id, NULL, RDMA_PS_TCP);
    if (ret) {
		logError("Creating server cm id failed");
		return status_t::ERROR;
	}
	logDebug("RDMA connection id for the server is created.");

    // 3. bind server
    ret = rdma_bind_addr(cm_server_id, this->server_addr);
	if (ret) {
		logError("Failed to bind server address");
		return status_t::ERROR;
	}
	logDebug("Server RDMA CM id is successfully binded.");


}

status_t RDMACommunicator::start_server() {
    // start server and waiting for acception from client

    // setup the Queue Pair
}

status_t RDMACommunicator::setup_client() {
    // alloc resources for client

    // setup client

}

status_t RDMACommunicator::start_client() {
    // connect client to server
    // setup the Queue Pair
}

status_t RDMACommunicator::create_queue_pair() {

}
status_t RDMACommunicator::create_memory_region(){

}
status_t RDMACommunicator::create_protection_domain() {

}
status_t RDMACommunicator::create_completion_queue() {

}
status_t RDMACommunicator::create_work_request() {

}


}