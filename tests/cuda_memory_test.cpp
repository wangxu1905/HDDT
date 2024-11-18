
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <hddt.h>
#include <mem.h>

namespace hddt {
class CudaMemoryTest : public ::testing::Test {
protected:
  memory_type_t mem_type_ = memory_type_t::NVIDIA_GPU;
  CudaMemory *cuda_memory = new CudaMemory(0, mem_type_);

  void SetUp() override {
    cuda_memory->init(); // Initialize CUDA context if necessary
  }

  void TearDown() override {
    cuda_memory->free(); // Clean up any allocated resources
  }
};

// allocte_buffer和free_buffer方法的正常路径
TEST_F(CudaMemoryTest, AllocateHostMemory_And_FreeBuffer_Test) {
  void *addr;
  size_t size = 1024;
  EXPECT_EQ(cuda_memory->allocate_buffer(&addr, size), status_t::SUCCESS);
  EXPECT_NE(addr, nullptr);
  EXPECT_EQ(cuda_memory->free_buffer(addr), status_t::SUCCESS);
  // EXPECT_EQ(addr, nullptr);
}

// free_buffer方法的异常路径，不是被cudaMalloc分配的内存
TEST_F(CudaMemoryTest, FreeBuffer_PtrNotAllocatedByCudaMalloc) {
  char addr[] = "Hello, World!";
  EXPECT_EQ(cuda_memory->free_buffer(addr), status_t::ERROR);
}

// host_to_buffer和buffer_to_host方法的正常路径
TEST_F(CudaMemoryTest, CopyHostToBuffer_HappyPath) {
  char source[] = "Hello, World!";
  char *src;
  char des[20] = {0};
  cuda_memory->allocate_buffer((void **)&src, 20);
  EXPECT_EQ(cuda_memory->copy_host_to_device(src, source, 20),
            status_t::SUCCESS);
  EXPECT_EQ(cuda_memory->copy_device_to_host(des, src, 20), status_t::SUCCESS);
  EXPECT_STREQ(des, source);
}

// host_to_buffer方法的异常路径，dest指针为空
TEST_F(CudaMemoryTest, CopyHostToBuffer_NullDest) {
  const size_t size = 1024;
  void *src;
  cuda_memory->allocate_buffer(&src, size);
  status_t status = cuda_memory->copy_host_to_device(nullptr, src, size);
  EXPECT_EQ(status, status_t::ERROR);
  cuda_memory->free_buffer(src);
}

// host_to_buffer方法的异常路径，src指针为空
TEST_F(CudaMemoryTest, CopyHostToBuffer_NullSrc) {
  const size_t size = 1024;
  void *dest;
  status_t status = cuda_memory->copy_host_to_device(dest, nullptr, size);
  EXPECT_EQ(status, status_t::ERROR);
}

// buffer_to_buffer方法的正常路径
TEST_F(CudaMemoryTest, CopyBufferToBuffer_HappyPath) {
  size_t bufferSize = 1024;
  void *src;
  void *dest;

  // Allocate source and destination buffers on the device
  EXPECT_EQ(cuda_memory->allocate_buffer(&src, bufferSize), status_t::SUCCESS);
  EXPECT_EQ(cuda_memory->allocate_buffer(&dest, bufferSize), status_t::SUCCESS);

  // Initialize source buffer with some data
  int *srcData = new int[bufferSize / sizeof(int)];
  for (size_t i = 0; i < bufferSize / sizeof(int); ++i) {
    srcData[i] = static_cast<int>(i);
  }

  // Copy data to the source device buffer
  EXPECT_EQ(cuda_memory->copy_host_to_device(src, srcData, bufferSize),
            status_t::SUCCESS);

  // Call the copy function
  EXPECT_EQ(cuda_memory->copy_device_to_device(dest, src, bufferSize),
            status_t::SUCCESS);

  // Cleanup
  delete[] srcData;
  cuda_memory->free_buffer(src);
  cuda_memory->free_buffer(dest);
}

// buffer_to_buffer方法的异常路径，dest指针为空
TEST_F(CudaMemoryTest, CopyBufferToBuffer_NullDestination) {
  void *src;
  size_t bufferSize = 1024;

  EXPECT_EQ(cuda_memory->allocate_buffer(&src, bufferSize), status_t::SUCCESS);
  EXPECT_EQ(cuda_memory->copy_device_to_device(nullptr, src, bufferSize),
            status_t::ERROR);
  cuda_memory->free_buffer(src);
}

// buffer_to_buffer方法的异常路径，src指针为空
TEST_F(CudaMemoryTest, CopyBufferToBuffer_NullSource) {
  void *dest;
  size_t bufferSize = 1024;

  EXPECT_EQ(cuda_memory->allocate_buffer(&dest, bufferSize), status_t::SUCCESS);
  EXPECT_EQ(cuda_memory->copy_device_to_device(dest, nullptr, bufferSize),
            status_t::ERROR);
  cuda_memory->free_buffer(dest);
}

} // namespace hddt