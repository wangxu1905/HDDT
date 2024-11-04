#include <mem.h>

namespace hddt {

#ifdef ENABLE_ROCM
// todo : ref
// https://github1s.com/linux-rdma/perftest/blob/master/src/rocm_memory.c
/*
 * amd gpu memory
 */
status_t RocmMemory::init() {}

status_t RocmMemory::free() {}

status_t RocmMemory::allocate_buffer(void **addr, size_t size) {}

status_t RocmMemory::free_buffer(void *addr) {}

status_t RocmMemory::copy_host_to_buffer(void *dest, const void *src,
                                         size_t size) {}

status_t RocmMemory::copy_buffer_to_host(void *dest, const void *src,
                                         size_t size) {}

status_t RocmMemory::copy_buffer_to_buffer(void *dest, const void *src,
                                           size_t size) {}

#endif
} // namespace hddt
