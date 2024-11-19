#include "executor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cuda.h>
#include <thread>
#include <chrono>

int main() {
    hddt::CudaExecutor executor; 
    // 启动执行器
    hddt::status_t start_status = executor.Start();
    if (start_status != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to start the executor: " << static_cast<int>(start_status) << std::endl;
        return 1;
    }

    // 准备测试任务
    size_t task1_size = 1024;
    size_t task2_size = 2048;

    // 分配主机内存
    char *host_src1 = new char[task1_size];
    char *host_dst1 = new char[task1_size];
    char *host_src2 = new char[task2_size];
    char *host_dst2 = new char[task2_size];

    // 初始化源数据
    for (size_t i = 0; i < task1_size; ++i) {
        host_src1[i] = static_cast<char>(i % 256);
    }
    for (size_t i = 0; i < task2_size; ++i) {
        host_src2[i] = static_cast<char>(i % 256);
    }
    
    std::cout << "data inited ...." << std::endl;


    // 分配设备内存
    char *dev_src1, *dev_dst1, *dev_src2, *dev_dst2;
    cudaMalloc(&dev_src1, task1_size);
    cudaMalloc(&dev_dst1, task1_size);
    cudaMalloc(&dev_src2, task2_size);
    cudaMalloc(&dev_dst2, task2_size);

    std::cout << "memory inited ...." << std::endl;

    // 将数据从主机复制到设备
    cudaMemcpy(dev_src1, host_src1, task1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_src2, host_src2, task2_size, cudaMemcpyHostToDevice);

    std::cout << "data copy to device finfished ...." << std::endl;

    // 创建任务结构体
    hddt::TaskCopy task1 = {dev_src1, dev_dst1, task1_size, hddt::TaskStatus::OPERATION_INITIALIZED};
    hddt::TaskCopy task2 = {dev_src2, dev_dst2, task2_size, hddt::TaskStatus::OPERATION_INITIALIZED};

    hddt::status_t post_status1 = executor.TaskPost(task1);
    hddt::status_t post_status2 = executor.TaskPost(task2);

    std::cout << "tasks post finfished ...." << std::endl;

    if (post_status1 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to post task1: " << static_cast<int>(post_status1) << std::endl;
        return 1;
    }

    if (post_status2 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to post task2: " << static_cast<int>(post_status2) << std::endl;
        return 1;
    }

    // 等待任务完成
    while (executor.TaskTest(&task1) != hddt::TaskStatus::OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    while (executor.TaskTest(&task2) != hddt::TaskStatus::OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 完成任务
    hddt::status_t finalize_status1 = executor.TaskFinalize(&task1);
    hddt::status_t finalize_status2 = executor.TaskFinalize(&task2);

    std::cout << "tasks finfished ...." << std::endl;

    if (finalize_status1 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to finalize task1: " << static_cast<int>(finalize_status1) << std::endl;
        return 1;
    }

    if (finalize_status2 != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to finalize task2: " << static_cast<int>(finalize_status2) << std::endl;
        return 1;
    }

    // 将结果从设备复制回主机
    cudaMemcpy(host_dst1, dev_dst1, task1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dst2, dev_dst2, task2_size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (size_t i = 0; i < task1_size; ++i) {
        assert(host_dst1[i] == host_src1[i]);
    }
    for (size_t i = 0; i < task2_size; ++i) {
        assert(host_dst2[i] == host_src2[i]);
    }

    // 释放内存
    delete[] host_src1;
    delete[] host_dst1;
    delete[] host_src2;
    delete[] host_dst2;
    cudaFree(dev_src1);
    cudaFree(dev_dst1);
    cudaFree(dev_src2);
    cudaFree(dev_dst2);

    // 停止执行器
    hddt::status_t stop_status = executor.Stop();
    if (stop_status != hddt::status_t::SUCCESS) {
        std::cerr << "Failed to stop the executor: " << static_cast<int>(stop_status) << std::endl;
        return 1;
    }

    std::cout << "All tasks completed successfully." << std::endl;

    return 0;
}
