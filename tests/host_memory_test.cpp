#include <gtest/gtest.h>
#include <mem.h>
#include <hddt.h>

namespace hddt {

class HostMemoryTest : public ::testing::Test {
protected:
    int device_id_src = 0;
    int device_id_des = 1;
    memory_type_t mem_type_ = memory_type_t::CPU;
    HostMemory* src_host_memory = new HostMemory(device_id_src, mem_type_); 

    void SetUp() override {
        // HostMemory* src_host_memory = new HostMemory(device_id_src, mem_type_);  // 源HostMemory对象
        // HostMemory* des_host_memory = new HostMemory(device_id_des, mem_type_);  // 目标HostMemory对象
    }

    void TearDown() override {
        // 在每个测试之后释放资源
    }
};

// allocte_buffer和free_buffer方法的正常路径
TEST_F(HostMemoryTest, AllocateHostMemory_And_FreeBuffer_Test) {
    void* addr;
    size_t size = 1024;
    EXPECT_EQ(src_host_memory->allocate_buffer(&addr, size), status_t::SUCCESS);
    EXPECT_NE(addr, nullptr);
    EXPECT_EQ(src_host_memory->free_buffer(addr), status_t::SUCCESS);
    // EXPECT_EQ(addr, nullptr);
}

// free_buffer方法的异常路径
TEST_F(HostMemoryTest, FreeBuffer_Nullptr) {
    void* addr = nullptr;
    EXPECT_EQ(src_host_memory->free_buffer(addr), status_t::ERROR);
}


// copy_host_to_device方法，正常路径
TEST_F(HostMemoryTest, CopyHostToBuffer_HappyPath) {
    char source[] = "Hello, World!";
    char destination[20] = {0};
    EXPECT_EQ(src_host_memory->copy_host_to_device(destination, source, sizeof(source)), status_t::SUCCESS);
    EXPECT_STREQ(destination, "Hello, World!");
}

// copy_host_to_device方法，异常路径
TEST_F(HostMemoryTest, CopyHostToBuffer_Nullptr) {
    char source[] = "Hello, World!";
    char destination[20] = {0};
    EXPECT_EQ(src_host_memory->copy_host_to_device(nullptr, source, sizeof(source)), status_t::ERROR);
    EXPECT_EQ(src_host_memory->copy_host_to_device(destination, nullptr, sizeof(source)), status_t::ERROR);
}

// copy_device_to_host方法，正常路径
TEST_F(HostMemoryTest, CopyBufferToHost_HappyPath) {
    char source[] = "Hello, World!";
    char destination[20] = {0};
    EXPECT_EQ(src_host_memory->copy_device_to_host(destination, source, sizeof(source)), status_t::SUCCESS);
    EXPECT_STREQ(destination, "Hello, World!");
}

// copy_device_to_host方法，异常路径
TEST_F(HostMemoryTest, CopyBufferToHost_Nullptr) {
    char source[] = "Hello, World!";
    char destination[20] = {0};
    EXPECT_EQ(src_host_memory->copy_device_to_host(nullptr, source, sizeof(source)), status_t::ERROR);
    EXPECT_EQ(src_host_memory->copy_device_to_host(destination, nullptr, sizeof(source)), status_t::ERROR);
}

// copy_device_to_device方法，正常路径
TEST_F(HostMemoryTest, CopyBufferToBuffer_HappyPath) {
    char source[] = "Hello, World!";
    char destination[20] = {0};
    EXPECT_EQ(src_host_memory->copy_device_to_device(destination, source, sizeof(source)), status_t::SUCCESS);
    EXPECT_STREQ(destination, "Hello, World!");
}

// copy_device_to_device方法，异常路径
TEST_F(HostMemoryTest, CopyBufferToBuffer_Nullptr) {
    char source[] = "Hello, World!";
    char destination[20] = {0};
    EXPECT_EQ(src_host_memory->copy_device_to_device(nullptr, source, sizeof(source)), status_t::ERROR);
    EXPECT_EQ(src_host_memory->copy_device_to_device(destination, nullptr, sizeof(source)), status_t::ERROR);
}

}