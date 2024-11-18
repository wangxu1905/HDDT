#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <hddt.h>
#include <mem.h>

namespace hddt {
class HddtMemoryTest : public ::testing::Test {
protected:
  HddtMemory *hddt_memory = new HddtMemory(0);
  void SetUp() override {
    hddt_memory->init(); // Initialize CUDA context if necessary
  }

  void TearDown() override {
    hddt_memory->free(); // Clean up any allocated resources
  }
};

TEST_F(HddtMemoryTest, init_with_memory_type_Default) {
  HddtMemory *memory_init_test = new HddtMemory(0);
  EXPECT_EQ(memory_init_test->get_init_Status(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->get_MemoryType(), memory_type_t::NVIDIA_GPU);
  EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
}

TEST_F(HddtMemoryTest, init_with_memory_type_CPU) {
  memory_type_t mem_init_type = memory_type_t::CPU;
  HddtMemory *memory_init_test = new HddtMemory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->get_init_Status(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->get_MemoryType(), memory_type_t::CPU);
  EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
}

TEST_F(HddtMemoryTest, init_with_memory_type_NVIDIA_GPU) {
  memory_type_t mem_init_type = memory_type_t::NVIDIA_GPU;
  HddtMemory *memory_init_test = new HddtMemory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->get_init_Status(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->get_MemoryType(), memory_type_t::NVIDIA_GPU);
  EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
}

TEST_F(HddtMemoryTest, init_with_memory_type_AMD_GPU_NOT_SUPPORTED) {
  memory_type_t mem_init_type = memory_type_t::AMD_GPU;
  EXPECT_THROW(HddtMemory *memory_init_test = new HddtMemory(0, mem_init_type),
               std::runtime_error);
  // EXPECT_EQ(memory_init_test->get_init_Status(), status_t::UNSUPPORT);
  // EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
}

// host_to_buffer和buffer_to_host方法的正常路径
TEST_F(HddtMemoryTest, CopyHostToBuffer_HappyPath) {
  char source[] = "Hello, World!";
  char *src;
  char des[20] = {0};
  hddt_memory->allocate_buffer((void **)&src, 20);
  EXPECT_EQ(hddt_memory->copy_host_to_device(src, source, 20),
            status_t::SUCCESS);
  EXPECT_EQ(hddt_memory->copy_device_to_host(des, src, 20), status_t::SUCCESS);
  EXPECT_STREQ(des, source);
}

TEST_F(HddtMemoryTest, CopyBufferToBuffer_NullSource) {
  void *dest;
  size_t bufferSize = 1024;

  EXPECT_EQ(hddt_memory->allocate_buffer(&dest, bufferSize), status_t::SUCCESS);
  EXPECT_EQ(hddt_memory->copy_device_to_device(dest, nullptr, bufferSize),
            status_t::ERROR);
  EXPECT_EQ(hddt_memory->free_buffer(dest), status_t::SUCCESS);
}

TEST_F(HddtMemoryTest, set_new_id_memory_type_NVIDIA_GPUtoCPU) {
  memory_type_t mem_init_type = memory_type_t::NVIDIA_GPU;
  HddtMemory *memory_init_test = new HddtMemory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->get_init_Status(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->get_DeviceId(), 0);
  EXPECT_EQ(memory_init_test->get_MemoryType(), memory_type_t::NVIDIA_GPU);

  memory_init_test->set_DeviceId_and_MemoryType(1, memory_type_t::CPU);
  EXPECT_EQ(memory_init_test->get_init_Status(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->get_DeviceId(), 1);
  EXPECT_EQ(memory_init_test->get_MemoryType(), memory_type_t::CPU);

  EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
}

TEST_F(HddtMemoryTest,
       set_new_id_memory_type_NVIDIA_GPUtoAMD_GPU_NOT_SUPPORTED) {
  memory_type_t mem_init_type = memory_type_t::NVIDIA_GPU;
  HddtMemory *memory_init_test = new HddtMemory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->get_init_Status(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->get_DeviceId(), 0);
  EXPECT_EQ(memory_init_test->get_MemoryType(), memory_type_t::NVIDIA_GPU);

  EXPECT_THROW(
      memory_init_test->set_DeviceId_and_MemoryType(1, memory_type_t::AMD_GPU),
      std::runtime_error);
  EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
}

} // namespace hddt